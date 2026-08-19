"""CLI behaviour that costs money if it's wrong.

`resume` is the sharp one: it used to re-run synthesis and critique on a finished run, billing
two model calls to reproduce a report it already had.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest
from rich.console import Console

from deep_research import cli
from deep_research.config import AppConfig
from deep_research.models import (
    McpServerStats,
    ResearchBrief,
    ResearchReport,
    ResearchState,
    RunMetrics,
    SubFinding,
)


@pytest.fixture(autouse=True)
def _no_model_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real provider resolution; `test_preflight_*` exercise it explicitly instead."""
    monkeypatch.setattr(cli, "infer_model", lambda model_string: object())


@pytest.fixture
def finished_state(tmp_path: Path) -> Path:
    """A completed run that hit its breadth budget, with three sub-questions never researched."""
    brief = ResearchBrief(
        question="Q?",
        subquestions=[f"s{i}" for i in range(6)],
        assumptions=[],
        depth_budget=2,
        breadth_budget=3,
    )
    state = ResearchState(
        brief=brief,
        findings={f"s{i}": SubFinding(subquestion=f"s{i}", answer="a", confidence="high") for i in range(3)},
        unresolved_subquestions=["s3", "s4", "s5"],
        round=2,
        status="done",
        critic_rounds=1,
        critic_passed=True,
    )
    path = tmp_path / "run.json"
    state.save(path)
    return path


def resume_args(state_file: Path, **overrides: object) -> argparse.Namespace:
    defaults = {
        "state_file": str(state_file),
        "config": None,
        "breadth_budget": None,
        "depth_budget": None,
        "spend_limit_usd": None,
        "force": False,
    }
    return argparse.Namespace(**{**defaults, **overrides})


# --- preflight --------------------------------------------------------------------------

def test_preflight_rejects_a_model_whose_key_is_missing(config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Agents defer model resolution, so without this the first symptom is a mid-scoping crash."""
    def boom(model_string: str) -> object:
        raise UserWarning("Set the `ANTHROPIC_API_KEY` environment variable")

    monkeypatch.setattr(cli, "infer_model", boom)
    assert cli._check_models_resolvable(config, Console(quiet=True)) is False


def test_preflight_passes_when_every_role_resolves(config: AppConfig) -> None:
    assert cli._check_models_resolvable(config, Console(quiet=True)) is True


# --- resume -----------------------------------------------------------------------------

def test_resume_of_a_finished_run_does_no_work(finished_state: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Re-synthesising a completed run bills real calls to reproduce the report it already had."""
    called = {"n": 0}
    monkeypatch.setattr(cli, "run_research_pipeline", lambda *a, **k: called.__setitem__("n", 1))
    monkeypatch.chdir(tmp_path)

    assert cli.cmd_resume(resume_args(finished_state)) == 0
    assert called["n"] == 0, "resume spent model calls on a run with nothing outstanding"


def stub_pipeline(monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]) -> None:
    """Replace the pipeline with a recorder, so resume tests never touch a model.

    `cmd_resume` wraps the call in `asyncio.run`, so both are stubbed: the coroutine is closed
    to keep Python from warning about it never being awaited.
    """

    async def fake_pipeline(*args: object, **kwargs: object) -> ResearchReport:
        captured["state"] = args[4]
        captured["calls"] = int(captured.get("calls", 0)) + 1
        return ResearchReport(question="Q?", summary="s")

    def fake_run(coro: object) -> tuple[ResearchReport, dict[str, object]]:
        coro.close()  # type: ignore[attr-defined]
        captured["calls"] = int(captured.get("calls", 0)) + 1
        # `(report, mcp_stats)`, matching `_research_with_tool_sources` — the CLI now runs the
        # pipeline inside an MCP session and gets per-server counters back with the report.
        return ResearchReport(question="Q?", summary="s"), {}

    monkeypatch.setattr(cli, "run_research_pipeline", fake_pipeline)
    monkeypatch.setattr(cli.asyncio, "run", fake_run)


def test_resume_force_re_runs_the_write_up(finished_state: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`--force` is the escape hatch for deliberately redoing a finished run."""
    captured: dict[str, object] = {}
    stub_pipeline(monkeypatch, captured)
    monkeypatch.chdir(tmp_path)

    assert cli.cmd_resume(resume_args(finished_state, force=True)) == 0
    assert captured.get("calls") == 1


def test_raising_the_breadth_budget_requeues_skipped_sub_questions(finished_state: Path) -> None:
    """The payoff of recording what a budget cut off: more budget can actually go get it."""
    state = ResearchState.load(finished_state)

    assert cli._raise_brief_budgets(state, resume_args(finished_state, breadth_budget=8)) is True
    cli._requeue_unresolved(state, Console(quiet=True))

    assert state.brief.breadth_budget == 8
    assert state.open_subquestions == ["s3", "s4", "s5"]
    assert state.status == "researching"


def test_budgets_can_only_be_raised_never_lowered(finished_state: Path) -> None:
    """Lowering below what a run already spent can't un-research anything; it just deadlocks."""
    state = ResearchState.load(finished_state)
    original = state.brief.breadth_budget

    assert cli._raise_brief_budgets(state, resume_args(finished_state, breadth_budget=1)) is False
    assert state.brief.breadth_budget == original


def test_requeue_respects_the_new_breadth_budget(finished_state: Path) -> None:
    state = ResearchState.load(finished_state)
    cli._raise_brief_budgets(state, resume_args(finished_state, breadth_budget=4))
    cli._requeue_unresolved(state, Console(quiet=True))

    assert state.open_subquestions == ["s3"], "3 findings + 1 slot = one requeued sub-question"


# --- metrics and stats ------------------------------------------------------------------

def test_metrics_are_appended_and_read_back(config: AppConfig, tmp_path: Path) -> None:
    from deep_research.research import RunBudget

    brief = ResearchBrief(question="Q?", subquestions=["a"], assumptions=[], depth_budget=1, breadth_budget=2)
    st = ResearchState(brief=brief, findings={"a": SubFinding(subquestion="a", answer="x", confidence="high")})
    report = ResearchReport(question="Q?", summary="s", unresolved=["u"])

    cli._record_metrics(config, tmp_path, st, report, 12.3, RunBudget.from_config(config))
    line = (tmp_path / ".deep_research" / "metrics.jsonl").read_text().strip()
    recorded = RunMetrics.model_validate_json(line)

    assert recorded.question == "Q?"
    assert recorded.unresolved_count == 1
    assert recorded.duration_seconds == 12.3


def test_stats_reads_ledger_entries_written_before_spend_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """New metrics fields must not orphan an existing metrics.jsonl."""
    old = {
        "timestamp": "20260101T000000Z",
        "question": "old run",
        "initial_subquestions": 4,
        "total_findings": 4,
        "rounds_used": 1,
        "critic_rounds": 1,
        "critic_passed_first_try": True,
        "budget_exhausted": False,
        "unresolved_count": 0,
        "duration_seconds": 42.0,
    }
    metrics_dir = tmp_path / ".deep_research"
    metrics_dir.mkdir()
    (metrics_dir / "metrics.jsonl").write_text(json.dumps(old) + "\n")
    monkeypatch.chdir(tmp_path)

    assert cli.cmd_stats(argparse.Namespace(config=None)) == 0


def test_stats_is_not_an_error_before_any_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.cmd_stats(argparse.Namespace(config=None)) == 0


def test_stats_reports_per_server_mcp_activity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PRD §11: whether a configured tool source earned its cost is a `stats` question."""
    metrics = RunMetrics(
        timestamp="20260101T000000Z", question="Q?", initial_subquestions=2, total_findings=2,
        rounds_used=1, critic_rounds=1, critic_passed_first_try=True, budget_exhausted=False,
        unresolved_count=0, duration_seconds=10.0,
        mcp={"zotero": McpServerStats(transport="stdio", calls=75, failed_calls=4,
                                      injected_calls=63, citations_kept=10, citations_dropped=1)},
    )
    metrics_dir = tmp_path / ".deep_research"
    metrics_dir.mkdir()
    (metrics_dir / "metrics.jsonl").write_text(metrics.model_dump_json() + "\n")
    monkeypatch.chdir(tmp_path)

    console = Console(record=True, width=200)
    monkeypatch.setattr(cli, "Console", lambda *a, **k: console)
    assert cli.cmd_stats(argparse.Namespace(config=None)) == 0

    output = console.export_text()
    assert "MCP tool sources" in output
    assert "zotero" in output and "75" in output and "10" in output


def test_stats_omits_the_mcp_table_for_a_web_only_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project with no tool sources sees exactly the output it saw before MCP support existed."""
    metrics = RunMetrics(
        timestamp="20260101T000000Z", question="Q?", initial_subquestions=2, total_findings=2,
        rounds_used=1, critic_rounds=1, critic_passed_first_try=True, budget_exhausted=False,
        unresolved_count=0, duration_seconds=10.0,
    )
    metrics_dir = tmp_path / ".deep_research"
    metrics_dir.mkdir()
    (metrics_dir / "metrics.jsonl").write_text(metrics.model_dump_json() + "\n")
    monkeypatch.chdir(tmp_path)

    console = Console(record=True, width=200)
    monkeypatch.setattr(cli, "Console", lambda *a, **k: console)
    cli.cmd_stats(argparse.Namespace(config=None))
    assert "MCP tool sources" not in console.export_text()


# --- init and overrides -----------------------------------------------------------------

def test_init_scaffolds_a_customisable_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.cmd_init(argparse.Namespace(force=False)) == 0

    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "reports").is_dir()


def test_init_does_not_clobber_an_edited_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("model:\n  lead: mine\n  researcher: mine\n  critic: mine\n")

    cli.cmd_init(argparse.Namespace(force=False))
    assert "mine" in (tmp_path / "config.yaml").read_text()


def test_cli_flags_override_configured_budgets(config: AppConfig) -> None:
    args = argparse.Namespace(breadth_budget=11, depth_budget=None, spend_limit_usd=2.5)
    cli._apply_overrides(config, args)

    assert config.budgets.breadth_budget == 11
    assert config.budgets.spend_limit_usd == 2.5
    assert config.budgets.depth_budget == 2, "an unset flag must leave the configured value alone"


# --- resume with a bad checkpoint path --------------------------------------------------

def test_a_missing_checkpoint_is_an_error_message_not_a_traceback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Checkpoint names are long timestamped slugs copied by hand, so a typo is the likely cause."""
    monkeypatch.chdir(tmp_path)
    assert cli.cmd_resume(resume_args(tmp_path / "nope.json")) == 1

    out = capsys.readouterr().out
    assert "No checkpoint at" in out
    assert "Traceback" not in out


def test_a_missing_checkpoint_lists_the_ones_that_do_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    state_dir = tmp_path / ".deep_research"
    state_dir.mkdir()
    (state_dir / "20260101T000000Z-a-real-run.json").write_text("{}")
    monkeypatch.chdir(tmp_path)

    cli.cmd_resume(resume_args(state_dir / "typo.json"))
    assert "a-real-run.json" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("label", "contents"),
    [("not json", "not json at all"), ("json of the wrong shape", '{"foo": 1}')],
)
def test_an_unusable_checkpoint_reports_one_clear_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], label: str, contents: str) -> None:
    path = tmp_path / "broken.json"
    path.write_text(contents)
    monkeypatch.chdir(tmp_path)

    assert cli.cmd_resume(resume_args(path)) == 1
    # Rich wraps to the capture width, so the phrase can be split across lines — compare unwrapped.
    out = " ".join(capsys.readouterr().out.split())
    assert "not a usable checkpoint" in out
    assert "Traceback" not in out
    assert "For further information visit" not in out, (
        f"{label}: the raw Pydantic report leaked instead of one summarised line"
    )


def test_a_directory_instead_of_a_checkpoint_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "adir").mkdir()
    monkeypatch.chdir(tmp_path)

    assert cli.cmd_resume(resume_args(tmp_path / "adir")) == 1
    assert "Could not read" in capsys.readouterr().out


def test_dotenv_is_loaded_from_the_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `.env` in the project folder must be found, as the README promises.

    Regression test. `load_dotenv()` with no argument searches upward from the directory
    of `cli.py`, not from the CWD, so an installed (`pipx`) run silently ignored the user's
    `.env` and every API key came back missing. `tmp_path` is deliberately outside this
    repo: running from inside it would find the repo's own `.env` and mask the bug.
    """
    (tmp_path / ".env").write_text("DEEP_RESEARCH_TEST_TOKEN=from_project_dotenv\n")
    monkeypatch.delenv("DEEP_RESEARCH_TEST_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)

    used = cli.load_project_dotenv()

    assert used is not None, "no .env was located from the working directory"
    assert Path(used) == tmp_path / ".env"
    assert os.environ["DEEP_RESEARCH_TEST_TOKEN"] == "from_project_dotenv"


def test_a_real_environment_variable_beats_the_dotenv_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly exported key must win, so one-off overrides on the command line work."""
    (tmp_path / ".env").write_text("DEEP_RESEARCH_TEST_TOKEN=from_project_dotenv\n")
    monkeypatch.setenv("DEEP_RESEARCH_TEST_TOKEN", "from_real_environment")
    monkeypatch.chdir(tmp_path)

    cli.load_project_dotenv()

    assert os.environ["DEEP_RESEARCH_TEST_TOKEN"] == "from_real_environment"


def test_no_dotenv_anywhere_is_not_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keys exported in the shell are a valid setup; a missing .env must not raise."""
    monkeypatch.chdir(tmp_path)
    assert cli.load_project_dotenv() is None
