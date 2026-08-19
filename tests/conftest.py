"""Shared fixtures. Every test here runs fully offline.

No test may make a network call or need an API key. Agents are built normally and then have
their model swapped with `agent.override(model=...)`, so the real model string is never
resolved; `defer_model_check=True` in agents.py is what makes construction work without
credentials in the first place.

One trap worth knowing before writing a test: **`TestModel` cannot drive the worker agent.**
The worker carries `Researcher()`, which brings provider-native tools, and `TestModel` raises
"TestModel does not support built-in tools". A test that uses `TestModel` for the worker
doesn't fail loudly - every worker just errors, gets contained by `_research_one`, and you end
up asserting against a run that silently did no research. Use `worker_model(...)` below, which
returns a `FunctionModel` emitting a real `SubFinding`. The other four agents are fine with
`TestModel`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from rich.console import Console

from deep_research.config import AppConfig, load_config
from deep_research.models import (
    ResearchBrief,
    ResearchReport,
    ResearchState,
    Source,
    SubFinding,
)
from deep_research.research import RunBudget, run_research_pipeline

BriefFactory = Callable[..., ResearchBrief]
"""The `brief` fixture: build a `ResearchBrief`, overriding only the fields a test cares about."""

PipelineRunner = Callable[..., tuple[ResearchState, ResearchReport, RunBudget]]
"""The `run_pipeline` fixture: run the pipeline offline, returning (state, report, budget)."""


@pytest.fixture
def config() -> AppConfig:
    """The bundled default config, with tracing off so tests don't emit spans."""
    cfg = load_config()
    cfg.logging.logfire = False
    return cfg


@pytest.fixture
def console() -> Console:
    """A quiet console: the pipeline prints progress, and tests shouldn't."""
    return Console(quiet=True)


@pytest.fixture
def brief() -> BriefFactory:
    """Build a `ResearchBrief`, overriding only what a test cares about."""

    def _brief(**kwargs: object) -> ResearchBrief:
        defaults: dict[str, object] = {
            "question": "Q?",
            "subquestions": ["a", "b"],
            "assumptions": [],
            "depth_budget": 1,
            "breadth_budget": 8,
        }
        return ResearchBrief(**{**defaults, **kwargs})

    return _brief


def worker_model(
    *, answer: str = "an answer", confidence: str = "high", sources: Sequence[str] = (), fail_times: int = 0
) -> FunctionModel:
    """A worker that returns a valid `SubFinding`, optionally failing the first N calls.

    `fail_times` exercises the retry path in `_research_one` deterministically, which a real
    model can't be relied on to do.
    """
    calls = {"n": 0}

    def respond(messages: list[object], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise RuntimeError(f"simulated worker failure #{calls['n']}")
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "subquestion": "q",
                        "answer": answer,
                        "confidence": confidence,
                        "sources": [
                            {"kind": "web", "identifier": u, "title": "T", "quoted_snippet": "s"} for u in sources
                        ],
                        "contradictions": [],
                    },
                )
            ]
        )

    model = FunctionModel(respond)
    model.calls = calls  # type: ignore[attr-defined]  # so tests can assert attempt counts
    return model


def failing_worker_model() -> FunctionModel:
    """A worker that never succeeds, however many times it is retried."""

    def respond(messages: list[object], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("permanent worker failure")

    return FunctionModel(respond)


def gap_check_model(*follow_ups: str, reasoning: str = "reasoning") -> TestModel:
    return TestModel(
        custom_output_args={"follow_up_subquestions": list(follow_ups), "reasoning": reasoning}
    )


def synthesis_model(*, unresolved: Sequence[str] = (), summary: str = "summary") -> TestModel:
    return TestModel(
        custom_output_args={
            "question": "Q?",
            "summary": summary,
            "sections": [],
            "unresolved": list(unresolved),
            "assumptions": [],
        }
    )


def critic_model(*, passed: bool = True, issues: Sequence[str] = (), follow_ups: Sequence[str] = ()) -> TestModel:
    return TestModel(
        custom_output_args={
            "passed": passed,
            "issues": list(issues),
            "follow_up_subquestions": list(follow_ups),
            "reasoning": "reasoning",
        }
    )


def rejecting_critic_model(follow_up_prefix: str = "follow-up") -> FunctionModel:
    """A critic that never passes and always proposes a *new* follow-up.

    Deliberately adversarial: if termination depended on the critic ever being satisfied, this
    would loop forever. It must be the budget that stops the run.
    """
    calls = {"n": 0}

    def respond(messages: list[object], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "passed": False,
                        "issues": [f"issue-{calls['n']}"],
                        "follow_up_subquestions": [f"{follow_up_prefix}-{calls['n']}"],
                        "reasoning": "not good enough",
                    },
                )
            ]
        )

    model = FunctionModel(respond)
    model.calls = calls  # type: ignore[attr-defined]
    return model


@pytest.fixture
def run_pipeline(config: AppConfig, console: Console, tmp_path: Path) -> PipelineRunner:
    """Run the whole pipeline offline, returning `(state, report, budget)`."""

    from deep_research.agents import (
        build_critic_agent,
        build_gap_check_agent,
        build_synthesis_agent,
        build_worker_agent,
    )

    def _run(
        research_brief: ResearchBrief,
        *,
        worker: object | None = None,
        gap: object | None = None,
        synthesis: object | None = None,
        critic: object | None = None,
        budget: RunBudget | None = None,
        state: ResearchState | None = None,
    ) -> tuple[ResearchState, object, RunBudget]:
        st = state or ResearchState(
            brief=research_brief, open_subquestions=list(research_brief.subquestions)
        )
        w = build_worker_agent(config)
        g = build_gap_check_agent(config)
        s = build_synthesis_agent(config)
        c = build_critic_agent(config)
        b = budget or RunBudget.from_config(config)
        with (
            w.override(model=worker or worker_model()),
            g.override(model=gap or gap_check_model()),
            s.override(model=synthesis or synthesis_model()),
            c.override(model=critic or critic_model()),
        ):
            report = asyncio.run(
                run_research_pipeline(w, g, s, c, st, tmp_path / "state.json", config, console, b)
            )
        return st, report, b

    return _run


# --- message-history builders, for the source-verification tests -------------------------

def local_fetch_history(url: str, *, outcome: str = "success") -> list[object]:
    """The shape a fetch takes when Pydantic AI uses its local web-fetch tool (e.g. OpenAI)."""
    return [
        ModelResponse(parts=[ToolCallPart(tool_name="web_fetch", args={"url": url}, tool_call_id="c1")]),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="web_fetch", content="page", tool_call_id="c1", outcome=outcome)]
        ),
    ]


def native_fetch_history(url: str, *, outcome: str = "success") -> list[object]:
    """The shape a fetch takes when the provider has a native web-fetch tool (e.g. Anthropic)."""
    return [
        ModelResponse(
            parts=[
                NativeToolCallPart(
                    tool_name="web_fetch", tool_kind="web_fetch", args={"url": url}, tool_call_id="c1"
                )
            ]
        ),
        ModelResponse(
            parts=[
                NativeToolReturnPart(
                    tool_name="web_fetch",
                    tool_kind="web_fetch",
                    content="page",
                    tool_call_id="c1",
                    outcome=outcome,
                )
            ]
        ),
    ]


def search_only_history(query: str = "python 3.13") -> list[object]:
    """A search with no fetch: the tool takes `query`, so nothing was actually read."""
    return [
        ModelResponse(
            parts=[ToolCallPart(tool_name="duckduckgo_search", args={"query": query}, tool_call_id="c1")]
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name="duckduckgo_search", content="results", tool_call_id="c1")]),
    ]


def mcp_call_history(
    tool_name: str,
    returned: object,
    *,
    args: dict[str, object] | None = None,
    outcome: str = "success",
) -> list[object]:
    """The shape an MCP tool call takes: a plain tool call whose *return* carries the item IDs.

    The asymmetry with `local_fetch_history` is the whole point of the MCP half of verification
    (PRD §10): a web fetch names its URL in the call's arguments, but a library search names only
    the query — the server decides which item keys come back, so the evidence is in the return.
    """
    return [
        ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args or {"query": "q"}, tool_call_id="m1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name=tool_name, content=returned, tool_call_id="m1", outcome=outcome)]),
    ]


def finding_citing(*urls: str, mcp: Sequence[tuple[str, str]] = ()) -> SubFinding:
    """A finding citing web URLs, and optionally `(server, identifier)` MCP items (PRD §5b)."""
    return SubFinding(
        subquestion="q",
        answer="a",
        confidence="high",
        sources=[Source(identifier=u, title="T", quoted_snippet="s") for u in urls]
        + [
            Source(kind="mcp", server=server, identifier=identifier, title="T", quoted_snippet="s")
            for server, identifier in mcp
        ],
    )
