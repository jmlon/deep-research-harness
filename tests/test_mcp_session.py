"""MCP sessions end to end, against a real in-process FastMCP server (PRD §5b).

This is the one place the whole path runs for real — session lifecycle, tool listing, the
`health_check` probe, `allowed_tools` filtering, argument injection at the dispatch boundary, and
citation attribution — with no network, no subprocess, and no Zotero credential. The fake library
below stands in for `zotero-mcp`: same shape of tool surface (a search returning item keys, a
full-text read with a `max_chars` ceiling), which is what the harness actually depends on.

It also exercises the in-memory transport itself, since a FastMCP instance is exactly what a
bundled server's factory returns.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import FastMCP
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from rich.console import Console

from deep_research import mcp as mcp_module
from deep_research.agents import build_worker_agent
from deep_research.config import AppConfig, McpServerConfig, load_config
from deep_research.mcp import (
    McpConfigError,
    exposed_tools,
    open_servers,
    stats,
    tool_to_server,
    toolsets,
)
from deep_research.verify import fetched_urls, mcp_evidence, verify_finding

KEY = "ABC12345"
LONG_TEXT = "x" * 100_000

pytestmark = pytest.mark.anyio


def fake_library(called: list[str] | None = None) -> FastMCP:
    """A miniature stand-in for `zotero-mcp`: keys come from returns, full text is huge.

    `called` records every tool the server actually served, which is the only way to assert on
    calls the harness makes outside the agent's dispatch path (the health check uses
    `direct_call_tool`, deliberately bypassing the metrics hook — a probe is not research).
    """
    server: FastMCP = FastMCP("fake-library")
    log = called if called is not None else []

    @server.tool
    def search_items(query: str, limit: int = 25, collection_key: str | None = None) -> dict:
        log.append("search_items")
        return {"items": [{"key": KEY, "title": "A Paper", "collection": collection_key}], "limit_used": limit}

    @server.tool
    def get_item_fulltext(item_key: str, max_chars: int = 100_000) -> dict:
        log.append("get_item_fulltext")
        return {"item_key": item_key, "text": LONG_TEXT[:max_chars], "chars": len(LONG_TEXT[:max_chars])}

    @server.tool
    def get_library_info() -> dict:
        log.append("get_library_info")
        return {"items": 1, "writes": False}

    @server.tool
    def delete_everything() -> str:  # never allowlisted; stands in for a tool we don't want exposed
        return "gone"

    return server


@pytest.fixture
def served() -> list[str]:
    """Tools the fake server actually served, in order."""
    return []


@pytest.fixture
def library_config(monkeypatch: pytest.MonkeyPatch, served: list[str]) -> AppConfig:
    """A config with one `in_memory` server resolving to `fake_library`.

    Patching `_build_in_memory` rather than registering a real entry point keeps the test hermetic
    while leaving every other step — session, probe, filtering, injection — exactly as it ships.
    """
    monkeypatch.setattr(mcp_module, "_build_in_memory", lambda cfg: fake_library(served))
    config = load_config()
    config.logging.logfire = False
    config.mcp_servers = [
        McpServerConfig(
            name="library",
            transport="in_memory",
            health_check="get_library_info",
            allowed_tools=["search_items", "get_item_fulltext", "get_library_info"],
            tool_args={"get_item_fulltext": {"max_chars": 500}, "search_items": {"limit": 3}},
            instructions="A curated library of papers.",
        )
    ]
    return config


async def test_session_opens_and_reports_its_tools(library_config: AppConfig) -> None:
    async with open_servers(library_config, Path.cwd(), Console(quiet=True)) as servers:
        assert [s.name for s in servers] == ["library"]
        assert set(servers[0].tool_names) >= {"search_items", "get_item_fulltext", "get_library_info"}


async def test_allowed_tools_filters_what_the_worker_sees(library_config: AppConfig) -> None:
    """A server exposing more than the harness needs contributes only the named subset (PRD §8)."""
    async with open_servers(library_config, Path.cwd(), Console(quiet=True)) as servers:
        assert "delete_everything" not in exposed_tools(servers[0])
        assert "delete_everything" in servers[0].tool_names, "the server does offer it; we chose not to"


async def test_attribution_map_covers_only_exposed_tools(library_config: AppConfig) -> None:
    async with open_servers(library_config, Path.cwd(), Console(quiet=True)) as servers:
        mapping = tool_to_server(servers)
        assert mapping["search_items"] == "library"
        assert "delete_everything" not in mapping


def worker_using_library(tool_args: dict[str, object], *, cite: str = KEY) -> FunctionModel:
    """A worker that calls one library tool, then cites what it got back.

    Driving injection through a real `agent.run` rather than poking the toolset directly is not
    just convenient: `process_tool_call` only fires on the agent's dispatch path, so a test that
    called the toolset by hand would assert against a path production never takes.
    """
    turns = {"n": 0}

    def respond(messages: list[object], info: AgentInfo) -> ModelResponse:
        turns["n"] += 1
        if turns["n"] == 1:
            return ModelResponse(parts=[ToolCallPart("get_item_fulltext", tool_args, tool_call_id="t1")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "subquestion": "q",
                        "answer": "an answer",
                        "confidence": "high",
                        "sources": [
                            {"kind": "mcp", "server": "library", "identifier": cite, "title": "A Paper",
                             "quoted_snippet": "s"}
                        ],
                        "contradictions": [],
                    },
                )
            ]
        )

    return FunctionModel(respond)


async def _run_worker(config: AppConfig, model: FunctionModel):
    """Open the session, run one worker against it, and verify its finding the way the pipeline does."""
    async with open_servers(config, Path.cwd(), Console(quiet=True)) as servers:
        agent = build_worker_agent(config, toolsets(servers))
        with agent.override(model=model):
            result = await agent.run("research this")
        messages = result.all_messages()
        finding = verify_finding(
            result.output, fetched_urls(messages), mcp_evidence(messages, tool_to_server(servers))
        )
        return finding, messages, stats(servers)


def _tool_return(messages: list[object], tool_name: str) -> object:
    for message in messages:
        for part in getattr(message, "parts", []):
            if isinstance(part, ToolReturnPart) and part.tool_name == tool_name:
                return part.content
    raise AssertionError(f"{tool_name} was never called")


async def test_injected_max_chars_actually_bounds_the_payload(library_config: AppConfig) -> None:
    """The §9 hazard, closed at the dispatch boundary rather than by asking the model nicely.

    The tool's own default is 100,000 characters — ~25-30k tokens for one call. This worker passes
    no `max_chars` at all, and still gets 500.
    """
    _, messages, _ = await _run_worker(library_config, worker_using_library({"item_key": KEY}))
    assert _tool_return(messages, "get_item_fulltext")["chars"] == 500


async def test_a_model_supplied_value_cannot_raise_the_ceiling(library_config: AppConfig) -> None:
    """A cap the model can override is not a cap (PRD §5b)."""
    model = worker_using_library({"item_key": KEY, "max_chars": 100_000})
    _, messages, _ = await _run_worker(library_config, model)
    assert _tool_return(messages, "get_item_fulltext")["chars"] == 500


async def test_a_citation_from_a_real_call_survives_and_is_counted(library_config: AppConfig) -> None:
    """The full path: tool call → injected args → return → verified citation → per-server metric."""
    finding, _, counted = await _run_worker(library_config, worker_using_library({"item_key": KEY}))

    assert [s.identifier for s in finding.sources] == [KEY]
    assert finding.sources[0].server == "library"
    assert finding.contradictions == []
    assert counted["library"].citations_kept == 0, "citations are counted by the pipeline, not the session"
    assert counted["library"].calls == 1, "one dispatched tool call; the probe's own call is not research"
    assert counted["library"].transport == "in_memory"


async def test_a_fabricated_key_is_dropped_even_though_a_call_succeeded(library_config: AppConfig) -> None:
    """The worker really did use the library — and still cited an item it was never given."""
    finding, _, _ = await _run_worker(library_config, worker_using_library({"item_key": KEY}, cite="ZZZ99999"))

    assert finding.sources == []
    assert any("ZZZ99999" in note for note in finding.contradictions)


async def test_tools_are_listed_even_when_the_health_check_is_skipped(library_config: AppConfig) -> None:
    """The run's own session passes `health_check=False`, having already been preflighted.

    Regression test for a silent, total failure: an earlier version skipped the whole probe in that
    case, so `tool_names` stayed empty, `tool_to_server` was empty, `verify.py` recognized no MCP
    tool, and **every** library citation was dropped as unretrieved. The run looked fine and cited
    nothing. Attribution has to survive the optimization.
    """
    async with open_servers(library_config, Path.cwd(), Console(quiet=True), health_check=False) as servers:
        assert tool_to_server(servers)["search_items"] == "library"


async def test_the_health_check_is_actually_called_when_asked(
    library_config: AppConfig, served: list[str]
) -> None:
    """Listing tools is not proof a server works — a rejected credential lists tools fine (PRD §5b)."""
    async with open_servers(library_config, Path.cwd(), Console(quiet=True)):
        pass
    assert served == ["get_library_info"], "the preflight must actually call a tool, not just list them"


async def test_the_health_check_is_not_repeated_on_the_run_session(
    library_config: AppConfig, served: list[str]
) -> None:
    """The preflight already proved the credential; re-proving it is a wasted round trip."""
    async with open_servers(library_config, Path.cwd(), Console(quiet=True), health_check=False):
        pass
    assert served == []


async def test_a_missing_health_check_tool_fails_the_probe(library_config: AppConfig) -> None:
    """A name the server doesn't have is a startup error, not a mystery at run time."""
    library_config.mcp_servers[0].health_check = "no_such_tool"
    with pytest.raises(McpConfigError, match="does not provide: no_such_tool"):
        async with open_servers(library_config, Path.cwd(), Console(quiet=True)):
            pass


async def test_an_allowlisted_tool_the_server_lacks_fails_the_probe(library_config: AppConfig) -> None:
    library_config.mcp_servers[0].allowed_tools = ["search_items", "renamed_upstream"]
    with pytest.raises(McpConfigError, match="renamed_upstream"):
        async with open_servers(library_config, Path.cwd(), Console(quiet=True)):
            pass


async def test_tool_args_naming_a_tool_the_server_lacks_fails_the_probe(library_config: AppConfig) -> None:
    """Otherwise the injection silently never happens — the cap is configured and absent."""
    library_config.mcp_servers[0].tool_args["get_fulltext"] = {"max_chars": 10}
    with pytest.raises(McpConfigError, match="get_fulltext"):
        async with open_servers(library_config, Path.cwd(), Console(quiet=True)):
            pass


async def test_a_required_server_that_cannot_start_aborts(library_config: AppConfig, monkeypatch) -> None:
    monkeypatch.setattr(mcp_module, "_build_in_memory", lambda cfg: (_ for _ in ()).throw(RuntimeError("no library")))
    with pytest.raises(McpConfigError, match="failed to start"):
        async with open_servers(library_config, Path.cwd(), Console(quiet=True)):
            pass


async def test_an_optional_server_that_cannot_start_is_dropped(library_config: AppConfig, monkeypatch) -> None:
    library_config.mcp_servers[0].optional = True
    monkeypatch.setattr(mcp_module, "_build_in_memory", lambda cfg: (_ for _ in ()).throw(RuntimeError("no library")))
    async with open_servers(library_config, Path.cwd(), Console(quiet=True)) as servers:
        assert servers == [], "an optional server that failed must not be handed to the worker"


async def test_two_servers_offering_the_same_tool_is_refused(library_config: AppConfig) -> None:
    """Ambiguous tool names would let a citation be verified against the wrong corpus."""
    second = library_config.mcp_servers[0].model_copy(update={"name": "other-library"})
    library_config.mcp_servers.append(second)
    with pytest.raises(McpConfigError, match="provided by both"):
        async with open_servers(library_config, Path.cwd(), Console(quiet=True)):
            pass


async def test_no_configured_servers_yields_nothing_and_touches_nothing() -> None:
    config = load_config()
    async with open_servers(config, Path.cwd(), Console(quiet=True)) as servers:
        assert servers == []
