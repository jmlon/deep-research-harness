"""Scoping, and specifically `--auto`: an unattended run must terminate without a human.

The interactive path is driven by `console.input`, so these fake it. The auto path is the one
with a real failure mode - a lead agent that keeps asking questions nobody is there to answer.
"""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from rich.console import Console

from deep_research.agents import build_lead_agent
from deep_research.config import AppConfig
from deep_research.scoping import AUTO_MAX_CLARIFICATIONS, run_scoping


def lead_model(*, clarifications: int) -> FunctionModel:
    """A lead agent that asks `clarifications` questions, then proposes a brief."""
    calls = {"n": 0}

    def respond(messages: list[object], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] <= clarifications:
            tool = next(t for t in info.output_tools if "Clarifying" in t.name)
            return ModelResponse(parts=[ToolCallPart(tool.name, {"question": f"q{calls['n']}?"})])
        tool = next(t for t in info.output_tools if "Brief" in t.name)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool.name,
                    {"question": "Refined question", "subquestions": ["s1", "s2"], "assumptions": ["a1"]},
                )
            ]
        )

    model = FunctionModel(respond)
    model.calls = calls  # type: ignore[attr-defined]
    return model


def never_satisfied_lead() -> FunctionModel:
    """A lead agent that ignores the "no human available" nudge and keeps asking."""
    calls = {"n": 0}

    def respond(messages: list[object], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        tool = next(t for t in info.output_tools if "Clarifying" in t.name)
        return ModelResponse(parts=[ToolCallPart(tool.name, {"question": f"q{calls['n']}?"})])

    model = FunctionModel(respond)
    model.calls = calls  # type: ignore[attr-defined]
    return model


@pytest.fixture
def quiet_console() -> Console:
    return Console(quiet=True)


def test_auto_accepts_a_proposed_brief_without_asking(config: AppConfig, quiet_console: Console) -> None:
    agent = build_lead_agent(config)
    with agent.override(model=lead_model(clarifications=0)):
        brief = run_scoping(agent, quiet_console, config, "raw question", auto=True)

    assert brief.question == "Refined question"
    assert brief.subquestions == ["s1", "s2"]
    assert brief.depth_budget == config.budgets.depth_budget
    assert brief.breadth_budget == config.budgets.breadth_budget


def test_auto_answers_clarifying_questions_itself_and_proceeds(config: AppConfig, quiet_console: Console) -> None:
    agent = build_lead_agent(config)
    model = lead_model(clarifications=2)
    with agent.override(model=model):
        brief = run_scoping(agent, quiet_console, config, "raw question", auto=True)

    assert brief.question == "Refined question"
    assert model.calls["n"] == 3, "two nudged clarifications, then the brief"


def test_auto_gives_up_on_a_lead_agent_that_never_stops_asking(config: AppConfig, quiet_console: Console) -> None:
    """The backstop that makes unattended runs safe to schedule.

    Without it this loops forever waiting for input that will never arrive.
    """
    agent = build_lead_agent(config)
    model = never_satisfied_lead()
    with agent.override(model=model):
        brief = run_scoping(agent, quiet_console, config, "raw question", auto=True)

    assert brief.question == "raw question", "falls back to the question the user actually gave"
    assert brief.subquestions == []
    assert any("Unattended run" in a for a in brief.assumptions), (
        "the report must be able to explain why the brief is thin"
    )
    assert model.calls["n"] == AUTO_MAX_CLARIFICATIONS + 1


def test_interactive_confirmation_accepts_on_empty_input(config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    console = Console(quiet=True)
    monkeypatch.setattr(console, "input", lambda *a, **k: "")
    agent = build_lead_agent(config)
    with agent.override(model=lead_model(clarifications=0)):
        brief = run_scoping(agent, console, config, "raw question")

    assert brief.question == "Refined question"


def test_interactive_revision_is_fed_back_to_the_lead_agent(config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    console = Console(quiet=True)
    answers = iter(["focus on performance only", ""])
    monkeypatch.setattr(console, "input", lambda *a, **k: next(answers))

    seen: list[str] = []

    def respond(messages: list[object], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        seen.append(str(getattr(last, "parts", [""])[-1]))
        tool = next(t for t in info.output_tools if "Brief" in t.name)
        return ModelResponse(
            parts=[ToolCallPart(tool.name, {"question": "Refined question", "subquestions": [], "assumptions": []})]
        )

    agent = build_lead_agent(config)
    with agent.override(model=FunctionModel(respond)):
        run_scoping(agent, console, config, "raw question")

    assert any("focus on performance only" in s for s in seen), (
        "a typed revision must reach the model, not just re-prompt the human"
    )
