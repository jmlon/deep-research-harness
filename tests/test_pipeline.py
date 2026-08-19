"""The research pipeline, end to end and offline.

Most of these assert one property: **a report never claims more coverage than it has.** That
guarantee was absent for a long time - `open_subquestions` is a work queue, emptied as each
round completes, so by synthesis time nothing distinguished "covered everything" from "the
budget cut us off", and the report said the former either way.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from conftest import (
    BriefFactory,
    PipelineRunner,
    critic_model,
    failing_worker_model,
    gap_check_model,
    rejecting_critic_model,
    synthesis_model,
    worker_model,
)
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage, UsageLimits

from deep_research.models import ResearchReport, ResearchState
from deep_research.research import RunBudget


def unresolved_mentioning(report: ResearchReport, needle: str) -> bool:
    return any(needle in item for item in report.unresolved)


# --- coverage honesty -------------------------------------------------------------------

def test_a_clean_run_reports_nothing_unresolved(brief: BriefFactory, run_pipeline: PipelineRunner) -> None:
    state, report, _ = run_pipeline(brief(subquestions=["a", "b"]))
    assert len(state.findings) == 2
    assert report.unresolved == []
    assert state.unresolved_subquestions == []


def test_sub_questions_cut_by_the_breadth_budget_reach_the_report(brief: BriefFactory, run_pipeline: PipelineRunner) -> None:
    state, report, _ = run_pipeline(brief(subquestions=[f"s{i}" for i in range(6)], breadth_budget=3))

    assert len(state.findings) == 3
    assert sorted(state.unresolved_subquestions) == ["s3", "s4", "s5"]
    for skipped in ("s3", "s4", "s5"):
        assert unresolved_mentioning(report, skipped), f"{skipped} was dropped without a trace"


def test_gap_check_follow_ups_that_do_not_fit_are_reported(brief: BriefFactory, run_pipeline: PipelineRunner) -> None:
    state, _report, _ = run_pipeline(
        brief(subquestions=["a", "b"], breadth_budget=3, depth_budget=3),
        gap=gap_check_model(*[f"g{i}" for i in range(5)]),
    )
    # One follow-up fits in the remaining slot; the other four must be accounted for.
    assert len(state.findings) == 3
    assert len(state.unresolved_subquestions) == 4


def test_critic_follow_ups_that_cannot_be_afforded_are_reported(brief: BriefFactory, run_pipeline: PipelineRunner) -> None:
    """The critic named research that would fix its complaint and we couldn't run it.

    Reporting only the complaint, and not the specific research it asked for, loses the most
    actionable part of the critique.
    """
    _, report, _ = run_pipeline(
        brief(subquestions=["a", "b"], breadth_budget=2, depth_budget=2),
        critic=critic_model(passed=False, issues=["missing X"], follow_ups=["research-X"]),
    )
    assert unresolved_mentioning(report, "research-X")
    assert unresolved_mentioning(report, "missing X")


def test_report_unresolved_does_not_depend_on_the_synthesis_model_cooperating(brief: BriefFactory, run_pipeline: PipelineRunner) -> None:
    """Honesty is enforced in code, not requested in a prompt (the PRD's stance on the critic).

    The synthesis model here ignores its instructions and returns an empty `unresolved`.
    """
    _, report, _ = run_pipeline(
        brief(subquestions=[f"s{i}" for i in range(4)], breadth_budget=1),
        synthesis=synthesis_model(unresolved=[]),
    )
    assert len(report.unresolved) == 3


# --- worker failure handling ------------------------------------------------------------

def test_a_permanently_failing_worker_becomes_unresolved_not_an_empty_finding(brief: BriefFactory, run_pipeline: PipelineRunner) -> None:
    """An empty finding would occupy a breadth slot, block retry on resume, and pad the report."""
    state, report, _ = run_pipeline(brief(subquestions=["a", "b"]), worker=failing_worker_model())

    assert state.findings == {}
    assert sorted(state.unresolved_subquestions) == ["a", "b"]
    assert len(report.unresolved) == 2


def test_a_transient_worker_failure_is_retried_once_and_recovers(brief: BriefFactory, run_pipeline: PipelineRunner) -> None:
    model = worker_model(fail_times=1)
    state, report, _ = run_pipeline(brief(subquestions=["a"]), worker=model)

    assert len(state.findings) == 1
    assert model.calls["n"] == 2, "PRD §8 asks for exactly one retry"
    assert report.unresolved == []


def test_a_sub_question_that_later_succeeds_stops_being_reported_as_a_gap(brief: BriefFactory, run_pipeline: PipelineRunner) -> None:
    """Otherwise a question answered on a later round appears as both a finding and a gap."""
    state = ResearchState(
        brief=brief(subquestions=["a"]), open_subquestions=["a"], unresolved_subquestions=["a"]
    )
    state, report, _ = run_pipeline(brief(subquestions=["a"]), state=state)

    assert "a" in state.findings
    assert state.unresolved_subquestions == []
    assert report.unresolved == []


# --- termination and budgets ------------------------------------------------------------

def test_a_critic_that_never_passes_still_terminates(brief: BriefFactory, run_pipeline: PipelineRunner) -> None:
    """PRD §10: "bounded by depth_budget so critique can't loop forever"."""
    critic = rejecting_critic_model()
    state, report, _ = run_pipeline(
        brief(subquestions=["a", "b"], depth_budget=2, breadth_budget=6), critic=critic
    )

    assert state.status == "done"
    assert critic.calls["n"] == 3, "one critique per write-up pass, bounded by depth_budget"
    assert report.unresolved, "an unsatisfied critic's issues must survive into the report"


def test_research_that_exhausts_its_budget_still_produces_a_report(brief: BriefFactory, run_pipeline: PipelineRunner) -> None:
    """The failure this prevents: pay for the research, then get nothing back.

    Reproduces a real run - the research ceiling was blown, and because the write-up was funded
    from a slice of that same ceiling, synthesis had nothing left and the run died with the
    findings gathered and no output.
    """
    starved = RunBudget(
        usage=RunUsage(),
        research_limits=UsageLimits(total_tokens_limit=60),  # blown by the first worker
        report_token_allowance=100_000,
        report_spend_allowance_usd=Decimal(1),
    )
    state, report, _ = run_pipeline(brief(subquestions=[f"s{i}" for i in range(4)]), budget=starved)

    assert report is not None
    assert state.findings == {}
    assert len(report.unresolved) >= 4, "a report built from nothing must say so"


def test_an_unusable_report_allowance_fails_loudly(brief: BriefFactory, run_pipeline: PipelineRunner) -> None:
    """With the write-up guaranteed its own budget, failing to write is a sizing problem.

    It should surface as such rather than being swallowed into an empty report.
    """
    no_allowance = RunBudget(
        usage=RunUsage(),
        research_limits=UsageLimits(total_tokens_limit=100_000),
        report_token_allowance=1,
        report_spend_allowance_usd=Decimal(1),
    )
    with pytest.raises(UsageLimitExceeded):
        run_pipeline(brief(subquestions=["a"]), budget=no_allowance)


def test_every_agent_call_bills_the_same_accumulator(brief: BriefFactory, run_pipeline: PipelineRunner) -> None:
    """3 workers + gap check + synthesis + critique, on one budget rather than six."""
    _, _, budget = run_pipeline(brief(subquestions=["a", "b", "c"]))
    assert budget.usage.requests == 6


# --- checkpointing ----------------------------------------------------------------------

def test_progress_is_checkpointed_for_resume(brief: BriefFactory, run_pipeline: PipelineRunner, tmp_path: Path) -> None:
    state, _, _ = run_pipeline(brief(subquestions=["a", "b"]))
    reloaded = ResearchState.load(tmp_path / "state.json")

    assert reloaded.status == "done"
    assert set(reloaded.findings) == set(state.findings)
    assert reloaded.unresolved_subquestions == state.unresolved_subquestions
