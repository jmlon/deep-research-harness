"""MCP tool sources for the worker agent (PRD §5b).

`WebSearch`/`WebFetch` reach what the open web indexes, which is the wrong retrieval target for
paywalled literature, a curated bibliography, or an internal knowledge base. This module turns the
`mcp_servers` config key into toolsets on the worker agent — and, just as importantly, bounds what
they can do and proves they work before a run spends anything.

Four things happen here, each answering a failure this harness has already argued about:

1. **Construction** per transport (`in_memory` / `stdio` / `http`), with project-folder-relative
   path resolution (PRD §4a) and `${VAR}` expansion so a secret is *named* in `config.yaml`, never
   stored in it.
2. **Argument injection** (`tool_args`) at the tool-call boundary, so operator-set scoping and
   output caps cannot be dropped by the model. This is the one real lifecycle hook in the system
   (PRD §5, **L**) — a `max_chars` the model is merely *asked* to pass is a `max_chars` that will
   sometimes be a 100,000-character PDF instead (PRD §9).
3. **A preflight probe** that opens each session, checks the tool list, and *calls* a tool. Listing
   is not proof: `zotero-mcp` deliberately accepts a rejected API key at startup so a bad key
   doesn't crash an editor's MCP launch, so a list-only probe passes and then every call fails —
   precisely the silent degradation fail-fast exists to prevent (PRD §5b).
4. **Attribution**, via a tool-name → server-name map built from the advertised tool lists. This is
   what lets `verify.py` check an MCP citation against the server that supposedly returned it, and
   §11 report whether a configured server earned its cost.

Tool names are deliberately **not prefixed** with the server name (Pydantic AI's
`load_mcp_toolsets` does prefix). The names in `tool_args`, `allowed_tools`, `health_check`, and a
server's own documentation are its real tool names; renaming them here would make every one of
those config keys a lie. Collisions between two servers are caught at startup instead.
"""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset
from rich.console import Console

from deep_research.config import AppConfig, McpServerConfig
from deep_research.models import McpServerStats

ENTRY_POINT_GROUP = "deep_research.mcp_servers"
"""Where bundled (`in_memory`) servers are discovered.

A server packaged as an optional extra of this distribution declares an entry point in this group
resolving to a zero-argument factory returning a `FastMCP` instance; `config.yaml` then names it
and nothing else. Deliberately *not* an importable `factory: "module:function"` string in
`config.yaml` — that would make a hand-edited config file a code-execution vector, which cuts
against the blast-radius argument in PRD §8. Credentials reach such a server the way they already
reach everything else: the project's `.env`, which the CLI loads before this runs.
"""

_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class McpConfigError(RuntimeError):
    """A configured server cannot be built or does not work. Carries an operator-actionable message."""


def expand_env(value: str, *, where: str) -> str:
    """Expand `${VAR}` / `${VAR:-default}` references, failing loudly on an unset variable.

    Loudly, because the alternative is a server that starts with an empty API key and fails on
    every call — the same silent-degradation shape the preflight exists to catch.
    """

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.environ.get(name, default)
        if resolved is None:
            raise McpConfigError(
                f"{where} references ${{{name}}}, which is not set. Set it in your project's .env "
                f"file, or give it a default with ${{{name}:-...}}."
            )
        return resolved

    return _ENV_REFERENCE.sub(replace, value)


def _expand_mapping(values: Mapping[str, str], *, where: str) -> dict[str, str]:
    return {key: expand_env(value, where=f"{where} key {key!r}") for key, value in values.items()}


@dataclass
class McpServer:
    """One configured server: its toolset, its advertised tools, and its per-run counters.

    `stats` is created before the toolset, because the injection hook closes over it to count
    calls — so the counter object has to exist first.
    """

    config: McpServerConfig
    stats: McpServerStats
    toolset: AbstractToolset[None]
    tool_names: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.config.name


def _build_in_memory(cfg: McpServerConfig) -> Any:
    """Resolve a bundled server by name through the entry-point group.

    The failure message names the install step rather than the import error, because "not
    installed" is what has actually happened every time this fails: the harness is installed
    without the extra that ships the server.
    """
    from importlib.metadata import entry_points

    available = {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}
    entry = available.get(cfg.name)
    if entry is None:
        known = ", ".join(sorted(available)) or "none"
        raise McpConfigError(
            f"Bundled server {cfg.name!r} is not installed.\n"
            f'  Install it with:  pipx inject deep-research-harness "deep-research-harness[{cfg.name}]"\n'
            f"  Bundled servers currently available: {known}.\n"
            f"  Alternatively, run it as a subprocess instead with `transport: stdio`."
        )
    try:
        factory = entry.load()
        return factory()
    except McpConfigError:
        raise
    except Exception as exc:
        raise McpConfigError(f"Bundled server {cfg.name!r} failed to build: {exc}") from exc


def _build_stdio(cfg: McpServerConfig, project_dir: Path) -> Any:
    from fastmcp.client.transports import StdioTransport

    command = [expand_env(part, where=f"MCP server {cfg.name!r} command") for part in cfg.command or []]
    cwd: str | None = None
    if cfg.cwd:
        # PRD §4a: relative to the researcher's project folder, never the install location. This
        # tool is `pipx`-installed, so "relative to the package" would point at a directory inside
        # someone's pipx venv — a whole class of breakage that in-memory transport avoids entirely.
        path = Path(expand_env(cfg.cwd, where=f"MCP server {cfg.name!r} cwd")).expanduser()
        resolved = path if path.is_absolute() else project_dir / path
        if not resolved.is_dir():
            raise McpConfigError(f"MCP server {cfg.name!r} cwd does not exist: {resolved}")
        cwd = str(resolved)

    return StdioTransport(
        command=command[0],
        args=command[1:],
        env={
            **os.environ,
            **_expand_mapping(cfg.env, where=f"MCP server {cfg.name!r} env"),
        }
        if cfg.env
        else None,
        cwd=cwd,
    )


def build_toolset(
    cfg: McpServerConfig, project_dir: Path, stats: McpServerStats | None = None
) -> MCPToolset[None]:
    """Build (but do not connect) the toolset for one configured server.

    Connecting happens in `open_servers`, so construction is cheap, synchronous, and safe to do
    before deciding whether to run at all.
    """
    kwargs: dict[str, Any] = {}
    if cfg.transport == "in_memory":
        client: Any = _build_in_memory(cfg)
    elif cfg.transport == "stdio":
        client = _build_stdio(cfg, project_dir)
    else:
        client = expand_env(cfg.url or "", where=f"MCP server {cfg.name!r} url")
        if cfg.auth_token_env:
            token = os.environ.get(cfg.auth_token_env)
            if not token:
                raise McpConfigError(
                    f"MCP server {cfg.name!r} needs its bearer token in ${cfg.auth_token_env}, which is not set."
                )
            kwargs["auth"] = token
        if cfg.headers:
            kwargs["headers"] = _expand_mapping(cfg.headers, where=f"MCP server {cfg.name!r} headers")

    toolset: MCPToolset[None] = MCPToolset(
        client,
        id=cfg.name,
        process_tool_call=_injector(cfg, stats),
        read_timeout=cfg.timeout_seconds,
        # 'retry' (the default): hand a tool error back to the model as `ModelRetry` so it can
        # self-correct. An earlier version of this used 'error', reasoning that a failing server
        # should surface immediately and let `_research_one` contain it like a failed fetch. A run
        # against the real Zotero server showed why that is wrong: the worker called `search_items`
        # with an empty query, the server answered with exactly the message it was designed to
        # give an agent — "`query` is empty. Pass search terms, or use `list_recent_items`" — and
        # 'error' threw that away, killing the sub-question twice over (once per worker attempt)
        # and producing a report with no findings at all. `_research_one` contains a failure by
        # *abandoning* the sub-question, which is far too blunt for a recoverable mistake, and the
        # server's error text (Zotero PRD §7.7) exists precisely so the model can act on it.
        # A genuinely dead server still fails every retry and is still contained by §8.
        tool_error_behavior="retry",
    )
    return toolset


def _injector(cfg: McpServerConfig, stats: McpServerStats | None):
    """The `tool_args` hook: merge configured arguments into every matching call (PRD §5b).

    Configured values **win** over model-supplied ones for the same key. That asymmetry is the
    whole point — `tool_args` is a floor the model cannot lower, not a default it can talk its way
    out of. A `max_chars` the model may override is not a cap.

    Installed even when `tool_args` is empty, because this is also where per-server call counting
    happens (§11) — a server whose calls are never counted can't be shown to have earned its cost.
    """

    async def process_tool_call(ctx: Any, call_tool: Any, name: str, args: dict[str, Any]) -> Any:
        injected = cfg.tool_args.get(name)
        merged = {**args, **injected} if injected else args
        if stats is not None:
            stats.calls += 1
            if injected and any(args.get(key) != value for key, value in injected.items()):
                stats.injected_calls += 1
        try:
            return await call_tool(name, merged)
        except Exception:
            if stats is not None:
                stats.failed_calls += 1
            raise

    return process_tool_call


def _allowlist(cfg: McpServerConfig):
    allowed = set(cfg.allowed_tools or ())

    def keep(ctx: Any, tool_def: Any) -> bool:
        return tool_def.name in allowed

    return keep


def build_server(cfg: McpServerConfig, project_dir: Path) -> McpServer:
    """Construct one server, unconnected. Raises `McpConfigError` if it can't be built at all."""
    stats = McpServerStats(transport=cfg.transport)
    return McpServer(config=cfg, stats=stats, toolset=build_toolset(cfg, project_dir, stats))


async def _probe(server: McpServer, toolset: MCPToolset[None], *, call_health_check: bool) -> None:
    """Verify one connected server: its tool list, the names its config references, and a real call.

    **The tool list is always fetched, on every open, even when the health check is skipped** —
    `tool_to_server` is built from it, and that map is what attributes a citation to the server
    that returned it. A run that opened its session without listing tools had an empty map, so
    `verify.py` recognized no MCP tool at all and *every* library citation was dropped as
    unretrieved: the precise failure this module exists to prevent, reintroduced by an
    optimization. It cost one cheap, non-LLM round trip to avoid.

    `call_health_check` is what the preflight/run split actually controls: proving the credential
    works is worth one tool call before the run starts, and pointless to repeat once it has.
    """
    cfg = server.config
    tools = await toolset.list_tools()
    server.tool_names = [tool.name for tool in tools]

    missing = sorted(cfg.referenced_tools() - set(server.tool_names))
    if missing:
        # Most likely a version skew between config and server. Finding out at startup beats
        # finding out as an unexplained absence of citations three minutes into a run.
        raise McpConfigError(
            f"MCP server {cfg.name!r} does not provide: {', '.join(missing)}.\n"
            f"  Tools it does provide: {', '.join(server.tool_names) or 'none'}."
        )

    if cfg.health_check and call_health_check:
        injected = cfg.tool_args.get(cfg.health_check, {})
        await toolset.direct_call_tool(cfg.health_check, dict(injected))


def exposed_tools(server: McpServer) -> list[str]:
    """The tool names the worker will actually see, after `allowed_tools` filtering."""
    if server.config.allowed_tools is None:
        return list(server.tool_names)
    allowed = set(server.config.allowed_tools)
    return [name for name in server.tool_names if name in allowed]


@asynccontextmanager
async def open_servers(
    config: AppConfig,
    project_dir: Path,
    console: Console,
    *,
    health_check: bool = True,
) -> AsyncIterator[list[McpServer]]:
    """Open one session per configured server, shared by every worker, for the whole block.

    One session per *run*, never one per worker: a stdio server launched per worker would multiply
    process startup and blow past the ≤ 4 concurrent upstream requests Zotero asks for once
    `worker_concurrency` workers each held a client, and an in-memory server built per worker would
    duplicate its caches (PRD §5b).

    A server that fails to start aborts, unless it declares `optional: true`, in which case it is
    dropped with a warning — a run that silently fell back to web-only would produce a report that
    looks complete and isn't.

    `health_check=False` skips only the credential-proving tool call, for the run's own session
    after the preflight has already made it. Everything else — including listing tools, which
    citation attribution depends on — happens on every open.
    """
    if not config.mcp_servers:
        yield []
        return

    async with AsyncExitStack() as stack:
        live: list[McpServer] = []
        for cfg in config.mcp_servers:
            try:
                server = build_server(cfg, project_dir)
                connected: MCPToolset[None] = await stack.enter_async_context(server.toolset)  # type: ignore[arg-type]
                await _probe(server, connected, call_health_check=health_check)
            except Exception as exc:
                detail = str(exc) or exc.__class__.__name__
                if not cfg.optional:
                    raise McpConfigError(
                        f"MCP server {cfg.name!r} ({cfg.transport}) failed to start:\n  {detail}\n"
                        f"  (set `optional: true` on the server to run without it)"
                    ) from exc
                console.print(
                    f"[yellow]MCP server {cfg.name!r} ({cfg.transport}) unavailable — continuing without it:[/]\n"
                    f"  [dim]{detail}[/]"
                )
                continue

            if cfg.allowed_tools is not None:
                server.toolset = server.toolset.filtered(_allowlist(cfg))  # type: ignore[assignment]
            live.append(server)

        _check_tool_collisions(live)
        if live:
            console.print(
                "[dim]Tool sources: "
                + ", ".join(f"{s.name} ({s.config.transport}, {len(exposed_tools(s))} tools)" for s in live)
                + "[/]"
            )
        yield live


def _check_tool_collisions(servers: list[McpServer]) -> None:
    """Two servers offering the same tool name would break attribution, so refuse to run.

    Attribution is not cosmetic here: `verify.py` checks an MCP citation against the evidence of
    *the server that returned it*, so an ambiguous tool name means a citation could be verified
    against the wrong corpus.
    """
    owners: dict[str, str] = {}
    for server in servers:
        for tool in exposed_tools(server):
            if tool in owners:
                raise McpConfigError(
                    f"Tool name {tool!r} is provided by both {owners[tool]!r} and {server.name!r}. "
                    f"Use `allowed_tools` on one of them to keep tool names unambiguous."
                )
            owners[tool] = server.name


def tool_to_server(servers: list[McpServer]) -> dict[str, str]:
    """Tool name → server name, for citation attribution in `verify.py` and metrics in §11."""
    return {tool: server.name for server in servers for tool in exposed_tools(server)}


def toolsets(servers: list[McpServer]) -> list[AbstractToolset[None]]:
    return [server.toolset for server in servers]


def worker_instructions(servers: list[McpServer]) -> str:
    """The `instructions` hints, rendered for the worker prompt (PRD §5b).

    Injection alone leaves a worker holding a tool it has no idea when to use, so the operator's
    prose about what each server is *for* is passed through verbatim, with the citation mechanics
    the harness enforces stated once for all of them.
    """
    described = [s for s in servers if s.config.instructions]
    if not described:
        return ""

    lines = [
        "",
        "Additional tool sources are available beyond web search and fetch:",
        "",
    ]
    for server in described:
        lines.append(f"- **{server.name}**: {(server.config.instructions or '').strip()}")
    lines.append("")
    lines.append(
        "When you answer from one of these sources, cite the item with kind='mcp', server set to "
        "the source's name above, and identifier set to the item's own ID exactly as the tool "
        "returned it — not a URL, and not an ID you inferred or reconstructed."
    )
    return "\n".join(lines)


def stats(servers: list[McpServer]) -> dict[str, McpServerStats]:
    return {server.name: server.stats for server in servers}
