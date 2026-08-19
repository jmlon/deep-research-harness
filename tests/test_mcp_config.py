"""Configuration and construction of MCP tool sources (PRD §5b, §13).

Everything here runs offline: no server is started, no subprocess spawned, no socket opened.
`build_toolset` deliberately does no I/O — connecting happens in `open_servers` — which is what
makes the whole config surface testable without a Zotero library.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from pydantic import ValidationError

from deep_research import cli
from deep_research import mcp as mcp_module
from deep_research.config import AppConfig, McpServerConfig, load_config
from deep_research.mcp import (
    ENTRY_POINT_GROUP,
    McpConfigError,
    _injector,
    build_server,
    build_toolset,
    expand_env,
    worker_instructions,
)
from deep_research.models import McpServerStats


def server(**overrides: object) -> McpServerConfig:
    defaults: dict[str, object] = {"name": "zotero", "transport": "stdio", "command": ["run-me"]}
    return McpServerConfig(**{**defaults, **overrides})


# --- per-transport validation (PRD §13) -------------------------------------------------------


def test_stdio_needs_a_command() -> None:
    with pytest.raises(ValidationError, match="needs a `command`"):
        McpServerConfig(name="zotero", transport="stdio")


def test_http_needs_a_url() -> None:
    with pytest.raises(ValidationError, match="needs a `url`"):
        McpServerConfig(name="docs", transport="http")


def test_in_memory_needs_nothing_but_a_name() -> None:
    """The bundled case: the server ships with the package, so there is nothing to point at."""
    assert McpServerConfig(name="zotero", transport="in_memory").transport == "in_memory"


def test_a_stale_command_left_on_a_switched_transport_is_an_error() -> None:
    """The migration people will actually perform is stdio → in_memory (PRD §13).

    A leftover `command:` would otherwise sit in the file looking meaningful while nothing read it.
    """
    with pytest.raises(ValidationError, match="does not accept command"):
        McpServerConfig(name="zotero", transport="in_memory", command=["uv", "run", "zotero"])


def test_http_keys_are_rejected_on_a_stdio_server() -> None:
    with pytest.raises(ValidationError, match="does not accept url"):
        server(url="https://example.com/mcp")


def test_unknown_keys_are_rejected_like_everywhere_else_in_the_config() -> None:
    with pytest.raises(ValidationError):
        McpServerConfig(name="zotero", transport="stdio", command=["x"], tool_arg={"oops": {}})


def test_duplicate_server_names_are_rejected() -> None:
    """`Source.server`, the metrics ledger, and `--mcp-args` all key on the name."""
    with pytest.raises(ValidationError, match="Duplicate MCP server name"):
        AppConfig.model_validate(
            {
                "model": {"lead": "m", "researcher": "m", "critic": "m"},
                "mcp_servers": [
                    {"name": "zotero", "command": ["a"]},
                    {"name": "zotero", "command": ["b"]},
                ],
            }
        )


def test_bundled_default_config_has_no_servers() -> None:
    """A fresh install researches the web only — nothing is enabled without the operator asking."""
    assert load_config().mcp_servers == []


# --- secrets are referenced, never stored (PRD §13) -------------------------------------------


def test_env_references_are_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZOTERO_API_KEY", "secret-value")
    assert expand_env("${ZOTERO_API_KEY}", where="test") == "secret-value"


def test_env_reference_default_is_used_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_THING", raising=False)
    assert expand_env("${MISSING_THING:-fallback}", where="test") == "fallback"


def test_unset_env_reference_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The alternative is a server that starts with an empty key and fails every call."""
    monkeypatch.delenv("MISSING_THING", raising=False)
    with pytest.raises(McpConfigError, match="MISSING_THING"):
        expand_env("${MISSING_THING}", where="the test config")


def test_http_server_without_its_token_refuses_to_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCS_TOKEN", raising=False)
    cfg = McpServerConfig(name="docs", transport="http", url="https://x/mcp", auth_token_env="DOCS_TOKEN")
    with pytest.raises(McpConfigError, match="DOCS_TOKEN"):
        build_toolset(cfg, Path.cwd())


# --- stdio paths resolve against the project folder (PRD §4a) ---------------------------------


def transport_of(toolset: object) -> object:
    """The FastMCP transport a built toolset will connect through."""
    return toolset.client.transport  # type: ignore[attr-defined]


def test_relative_cwd_resolves_against_the_project_folder(tmp_path: Path) -> None:
    """Not the install location: this tool is `pipx`-installed, so that would be someone's venv."""
    (tmp_path / "servers" / "zotero").mkdir(parents=True)
    toolset = build_toolset(server(cwd="servers/zotero"), tmp_path)

    assert transport_of(toolset).cwd == str(tmp_path / "servers" / "zotero")


def test_an_absolute_cwd_is_left_alone(tmp_path: Path) -> None:
    toolset = build_toolset(server(cwd=str(tmp_path)), Path("/nowhere"))
    assert transport_of(toolset).cwd == str(tmp_path)


def test_the_command_becomes_the_transports_argv(tmp_path: Path) -> None:
    toolset = build_toolset(server(command=["uv", "run", "python", "-m", "zotero_mcp"]), tmp_path)
    transport = transport_of(toolset)

    assert (transport.command, transport.args) == ("uv", ["run", "python", "-m", "zotero_mcp"])


def test_env_references_in_a_stdio_env_are_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of `${VAR}`: config.yaml names the secret, the environment holds it."""
    monkeypatch.setenv("ZOTERO_API_KEY", "the-real-key")
    toolset = build_toolset(server(env={"ZOTERO_API_KEY": "${ZOTERO_API_KEY}"}), tmp_path)

    assert transport_of(toolset).env["ZOTERO_API_KEY"] == "the-real-key"


def test_a_cwd_that_does_not_exist_is_reported_before_launching(tmp_path: Path) -> None:
    with pytest.raises(McpConfigError, match="cwd does not exist"):
        build_toolset(server(cwd="not/here"), tmp_path)


# --- in-memory discovery (PRD §5b) ------------------------------------------------------------


def test_an_uninstalled_bundled_server_names_the_install_step() -> None:
    """The failure is always "the extra isn't installed", so the message says so rather than
    surfacing an ImportError the operator can do nothing with.

    Uses a name nothing could plausibly register, deliberately: an earlier version of this test
    named `zotero`, which passed only for as long as `zotero-mcp` happened not to be installed in
    the dev environment — and started failing the moment it was. A test whose outcome depends on
    what someone injected into the venv is testing the venv.
    """
    cfg = McpServerConfig(name="definitely-not-a-bundled-server", transport="in_memory")
    with pytest.raises(McpConfigError, match="pipx inject"):
        build_toolset(cfg, Path.cwd())


def test_the_error_lists_the_bundled_servers_that_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"Not installed" is only actionable next to "here is what is"."""
    from importlib.metadata import EntryPoint

    fake = [EntryPoint(name="zotero", value="zotero_mcp:build_server", group=ENTRY_POINT_GROUP)]
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: fake)

    cfg = McpServerConfig(name="some-other-library", transport="in_memory")
    with pytest.raises(McpConfigError, match="currently available: zotero"):
        build_toolset(cfg, Path.cwd())


def test_a_discovered_bundled_server_is_built_through_its_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-memory contract: a zero-argument factory, resolved by name, returning a server.

    Asserted against a stub factory rather than the real `zotero-mcp`, so this holds whether or not
    that package is installed here — the real wiring is verified in that project's own suite.
    """
    from importlib.metadata import EntryPoint

    built = object()

    class StubEntryPoint(EntryPoint):
        def load(self):  # type: ignore[override]
            return lambda: built

    fake = [StubEntryPoint(name="stub", value="x:y", group=ENTRY_POINT_GROUP)]
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: fake)

    assert mcp_module._build_in_memory(McpServerConfig(name="stub", transport="in_memory")) is built


# --- tool_args injection (PRD §5b) ------------------------------------------------------------


async def _call(injector, name: str, args: dict[str, object]) -> dict[str, object]:
    """Run the injector, capturing the arguments it actually dispatched."""
    seen: dict[str, object] = {}

    async def call_tool(tool: str, merged: dict[str, object]) -> str:
        seen.update(merged)
        return "ok"

    await injector(None, call_tool, name, args)
    return seen


@pytest.mark.anyio
async def test_configured_args_are_added_to_the_call() -> None:
    cfg = server(tool_args={"get_item_fulltext": {"max_chars": 20000}})
    dispatched = await _call(_injector(cfg, None), "get_item_fulltext", {"item_key": "ABC12345"})
    assert dispatched == {"item_key": "ABC12345", "max_chars": 20000}


@pytest.mark.anyio
async def test_configured_args_override_the_model() -> None:
    """The asymmetry is the whole point: a cap the model can raise is not a cap (PRD §5b, §9)."""
    cfg = server(tool_args={"get_item_fulltext": {"max_chars": 20000}})
    dispatched = await _call(_injector(cfg, None), "get_item_fulltext", {"max_chars": 100000})
    assert dispatched["max_chars"] == 20000


@pytest.mark.anyio
async def test_other_tools_are_untouched() -> None:
    cfg = server(tool_args={"get_item_fulltext": {"max_chars": 20000}})
    dispatched = await _call(_injector(cfg, None), "search_items", {"query": "attention"})
    assert dispatched == {"query": "attention"}


@pytest.mark.anyio
async def test_calls_are_counted_even_with_no_tool_args() -> None:
    """Counting is why the hook is installed unconditionally (PRD §11)."""
    stats = McpServerStats(transport="stdio")
    await _call(_injector(server(), stats), "search_items", {"query": "q"})
    assert (stats.calls, stats.injected_calls) == (1, 0)


@pytest.mark.anyio
async def test_an_injection_that_changed_nothing_is_not_counted_as_one() -> None:
    """The model already passed the configured value, so nothing was overridden."""
    stats = McpServerStats(transport="stdio")
    cfg = server(tool_args={"search_items": {"limit": 10}})
    await _call(_injector(cfg, stats), "search_items", {"limit": 10})
    assert stats.injected_calls == 0


@pytest.mark.anyio
async def test_a_failed_call_is_counted_as_failed_and_still_raises() -> None:
    stats = McpServerStats(transport="stdio")

    async def boom(tool: str, args: dict[str, object]) -> str:
        raise RuntimeError("server is down")

    with pytest.raises(RuntimeError):
        await _injector(server(), stats)(None, boom, "search_items", {})
    assert (stats.calls, stats.failed_calls) == (1, 1)


# --- worker prompt hint (PRD §5b) -------------------------------------------------------------


def test_instructions_reach_the_worker_prompt() -> None:
    built = build_server(server(instructions="A curated library of ML papers."), Path.cwd())
    rendered = worker_instructions([built])
    assert "zotero" in rendered
    assert "A curated library of ML papers." in rendered
    assert "kind='mcp'" in rendered, "the hint must also state the citation mechanics the harness enforces"


def test_a_server_with_no_instructions_adds_nothing_to_the_prompt() -> None:
    """Web-only runs, and servers the operator didn't describe, leave the prompt as it was."""
    assert worker_instructions([]) == ""
    assert worker_instructions([build_server(server(), Path.cwd())]) == ""


# --- --mcp-args overlay (PRD §5b) -------------------------------------------------------------


def config_with_zotero(**overrides: object) -> AppConfig:
    cfg = load_config()
    cfg.mcp_servers = [server(**overrides)]
    return cfg


def test_mcp_args_merges_over_configured_tool_args() -> None:
    config = config_with_zotero(tool_args={"search_items": {"limit": 10}})
    cli._apply_mcp_args(config, '{"zotero": {"search_items": {"collection_key": "XY99ABCD"}}}')

    assert config.mcp_servers[0].tool_args["search_items"] == {"limit": 10, "collection_key": "XY99ABCD"}


def test_mcp_args_overrides_a_configured_value() -> None:
    config = config_with_zotero(tool_args={"search_items": {"limit": 10}})
    cli._apply_mcp_args(config, '{"zotero": {"search_items": {"limit": 3}}}')
    assert config.mcp_servers[0].tool_args["search_items"]["limit"] == 3


def test_mcp_args_for_an_unknown_server_is_an_error_not_a_no_op() -> None:
    """A run that looks scoped and isn't is worse than one that refuses to start (PRD §5b)."""
    with pytest.raises(ValueError, match="unknown MCP server"):
        cli._apply_mcp_args(config_with_zotero(), '{"nope": {"search_items": {}}}')


def test_malformed_mcp_args_json_is_reported_as_such() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        cli._apply_mcp_args(config_with_zotero(), "{not json")


@pytest.mark.parametrize("raw", ['["zotero"]', '{"zotero": "search_items"}', '{"zotero": {"search_items": 3}}'])
def test_mcp_args_of_the_wrong_shape_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        cli._apply_mcp_args(config_with_zotero(), raw)


def test_no_mcp_args_leaves_the_config_alone() -> None:
    config = config_with_zotero(tool_args={"search_items": {"limit": 10}})
    cli._apply_mcp_args(config, None)
    assert config.mcp_servers[0].tool_args == {"search_items": {"limit": 10}}


def test_run_exits_non_zero_on_bad_mcp_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """End to end through the CLI: a usage error must not become a started run."""
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        question=["anything"], config=None, breadth_budget=None, depth_budget=None,
        spend_limit_usd=None, auto=True, mcp_args="{not json",
    )
    assert cli.cmd_run(args) == 1


# --- preflight (PRD §5b) ----------------------------------------------------------------------


def test_preflight_is_a_no_op_without_configured_servers() -> None:
    """A web-only run must not pay for an event loop it doesn't need."""
    from rich.console import Console

    assert cli._preflight_mcp(load_config(), Path.cwd(), Console(quiet=True)) is True


def test_preflight_fails_the_run_when_a_required_server_cannot_start(tmp_path: Path) -> None:
    """Fail fast: a silently web-only bibliographic report looks complete and isn't."""
    from rich.console import Console

    config = load_config()
    config.mcp_servers = [McpServerConfig(name="zotero", transport="in_memory")]
    assert cli._preflight_mcp(config, tmp_path, Console(quiet=True)) is False


def test_preflight_tolerates_an_optional_server_that_cannot_start(tmp_path: Path) -> None:
    """`optional: true` is the documented opt-out from aborting (PRD §5b)."""
    from rich.console import Console

    config = load_config()
    config.mcp_servers = [McpServerConfig(name="zotero", transport="in_memory", optional=True)]
    assert cli._preflight_mcp(config, tmp_path, Console(quiet=True)) is True
