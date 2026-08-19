"""Config loading: the file is hand-edited, so mistakes in it must be loud."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from deep_research.config import (
    MODEL_ENV_VAR,
    ROLE_MODEL_ENV_VARS,
    AppConfig,
    bundled_default_config_text,
    load_config,
)

MINIMAL = """
model:
  lead: "anthropic:claude-fable-5"
  researcher: "anthropic:claude-fable-5"
  critic: "anthropic:claude-fable-5"
"""


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model env vars leak between tests otherwise, and they override config."""
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    for var in ROLE_MODEL_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_bundled_default_is_valid() -> None:
    """The config shipped in the wheel must load - it's the fallback for an un-init-ed project."""
    AppConfig.model_validate_json(load_config().model_dump_json())
    assert "report_token_allowance" in bundled_default_config_text()


def test_project_config_wins_over_bundled_default(tmp_path: Path) -> None:
    write_config(tmp_path, MINIMAL + "budgets:\n  breadth_budget: 42\n")
    assert load_config(project_dir=tmp_path).budgets.breadth_budget == 42


def test_explicit_path_wins_over_project_config(tmp_path: Path) -> None:
    write_config(tmp_path, MINIMAL + "budgets:\n  breadth_budget: 42\n")
    other = tmp_path / "other.yaml"
    other.write_text(textwrap.dedent(MINIMAL + "budgets:\n  breadth_budget: 7\n"))
    assert load_config(other, project_dir=tmp_path).budgets.breadth_budget == 7


@pytest.mark.parametrize("typo", ["breadth_budgets: 40", "worker_concurency: 16"])
def test_a_mistyped_budget_key_is_an_error_not_a_shrug(tmp_path: Path, typo: str) -> None:
    """The failure this prevents is silent: the run just ignores the budget you set.

    Under Pydantic's default `extra='ignore'` these typos left the default in place with
    nothing to explain why the configured value had no effect.
    """
    path = write_config(tmp_path, MINIMAL + f"budgets:\n  {typo}\n")
    with pytest.raises(ValueError) as exc:
        load_config(project_dir=tmp_path)
    assert typo.split(":")[0] in str(exc.value)
    assert str(path) in str(exc.value), "the error must name the file, which may be far from cwd"


def test_missing_required_model_role_is_rejected(tmp_path: Path) -> None:
    write_config(tmp_path, 'model:\n  lead: "x"\n  critic: "y"\n')
    with pytest.raises(ValueError, match="researcher"):
        load_config(project_dir=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("report_token_allowance", 0), ("report_spend_allowance_usd", 0)],
)
def test_report_allowance_must_be_positive(tmp_path: Path, field: str, value: int) -> None:
    """A zero allowance would reintroduce the bug it exists to prevent: no budget to write with."""
    write_config(tmp_path, MINIMAL + f"budgets:\n  {field}: {value}\n")
    with pytest.raises(ValueError, match=field):
        load_config(project_dir=tmp_path)


def test_shared_env_var_overrides_every_role(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_config(tmp_path, MINIMAL)
    monkeypatch.setenv(MODEL_ENV_VAR, "openai:gpt-5-mini")
    cfg = load_config(project_dir=tmp_path)
    assert (cfg.model.lead, cfg.model.researcher, cfg.model.critic) == (
        "openai:gpt-5-mini",
        "openai:gpt-5-mini",
        "openai:gpt-5-mini",
    )


def test_per_role_env_var_wins_over_the_shared_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of the per-role override: a cheaper critic alongside a shared default."""
    write_config(tmp_path, MINIMAL)
    monkeypatch.setenv(MODEL_ENV_VAR, "openai:gpt-5-mini")
    monkeypatch.setenv(ROLE_MODEL_ENV_VARS["critic"], "openai:gpt-5-nano")
    cfg = load_config(project_dir=tmp_path)
    assert cfg.model.researcher == "openai:gpt-5-mini"
    assert cfg.model.critic == "openai:gpt-5-nano"


def test_paths_resolve_against_the_project_folder_not_the_install(tmp_path: Path) -> None:
    """A globally-installed tool must write into the directory it was invoked from (PRD §4a)."""
    write_config(tmp_path, MINIMAL)
    cfg = load_config(project_dir=tmp_path)
    assert cfg.report_dir(tmp_path) == tmp_path / "reports"
    assert cfg.state_dir(tmp_path) == tmp_path / ".deep_research"


def test_absolute_output_paths_are_left_alone(tmp_path: Path) -> None:
    write_config(tmp_path, MINIMAL + f'output:\n  report_dir: "{tmp_path / "elsewhere"}"\n')
    assert load_config(project_dir=tmp_path).report_dir(tmp_path) == tmp_path / "elsewhere"
