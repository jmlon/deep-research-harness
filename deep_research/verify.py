"""Deterministic, non-LLM verification (PRD §8, §10 item 1).

"A worker's SubFinding is only accepted if every source in it was actually fetched this run" —
this runs after every worker call, inspecting the run's own message history for tool calls that
actually returned content, no extra agent call and no extra cost. This is the "per-finding" half
of verification; the critic agent (agents.py/research.py) is the LLM-based "per-report" half.

Matching is deliberately by tool-call *shape* (a successful tool call with a `url` argument)
rather than by a specific tool name: the worker's `Researcher()` capability may resolve to a
provider-native web-fetch tool or the local DuckDuckGo/markdownify fallback depending on the
model, and both use `url` as the argument name, but native tool names vary by provider.

Which makes the base classes below load-bearing, because *which* shape you get depends on the
provider. `WebFetch` drops its local fallback whenever the model supports the provider-native
tool, and the two differ on exactly this: Anthropic models support a native `WebFetchTool`, so
fetches arrive as `NativeToolCallPart`/`NativeToolReturnPart`; OpenAI models don't, so `WebFetch`
falls back to the local markdownify tool and fetches arrive as `ToolCallPart`/`ToolReturnPart`.
Those are *siblings*, not subclasses. Matching either concrete class alone therefore works on one
provider and silently sees nothing on the other — and "no URL was fetched" is indistinguishable
here from "every citation is hallucinated", so every real source would be dropped from every
finding. Match `BaseToolCallPart`/`BaseToolReturnPart`, the common ancestors, so the check is
provider-agnostic; both expose the `args_as_dict()` and `outcome` used below.

Scope note: this checks URLs that were *fetched*. A URL that only ever appeared in a search
result the worker never opened is not accepted — reading the source is the bar (PRD §3:
"the source was actually fetched and read").

**MCP sources verify from the other end of the call** (PRD §5b, §10). A URL is evidence taken from
the tool call's *arguments* — the worker chose to fetch it. A Zotero item key generally never
appears in any argument: the worker asks for "attention interpretability" and the *server* decides
which keys come back. So MCP identifiers are matched against successful tool *returns*, attributed
to the server that produced them. Reading arguments alone — the natural extension of the web check
— would verify nothing at all about MCP sources while appearing to work.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic_ai.messages import BaseToolCallPart, BaseToolReturnPart, ModelMessage

from deep_research.models import Source, SubFinding

MIN_MCP_IDENTIFIER_LENGTH = 4
"""Below this, a substring match against a tool return proves nothing.

MCP returns are matched by searching the serialized return text, because an identifier can arrive
as a JSON field (`{"key": "ABC12345"}`), inside a nested structure, or in a `text/plain` body with
no structure at all — and the harness deliberately knows nothing about any particular server's
schema. That works because real identifiers are long and distinctive (a Zotero key is 8
alphanumerics). A 1-3 character "identifier" would match almost any text by accident, so it is
rejected rather than accepted on weak evidence.
"""


def fetched_urls(messages: list[ModelMessage]) -> set[str]:
    """URLs a successful tool call actually fetched during this run's message history."""
    returns_by_call_id = {
        part.tool_call_id: part
        for message in messages
        for part in getattr(message, "parts", [])
        if isinstance(part, BaseToolReturnPart)
    }

    urls: set[str] = set()
    for message in messages:
        for part in getattr(message, "parts", []):
            if not isinstance(part, BaseToolCallPart):
                continue
            args = part.args_as_dict() if part.args else {}
            url = args.get("url")
            if not isinstance(url, str):
                continue
            tool_return = returns_by_call_id.get(part.tool_call_id)
            if tool_return is not None and tool_return.outcome == "success":
                urls.add(url)
    return urls


def _return_text(content: Any) -> str:
    """Serialize a tool return for substring searching, whatever shape it arrived in."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, default=str)
    except (TypeError, ValueError):
        return str(content)


def mcp_evidence(messages: list[ModelMessage], tool_to_server: Mapping[str, str]) -> dict[str, str]:
    """Per-server text of everything its tools successfully returned this run.

    `tool_to_server` maps a tool name to the server that provides it, built at startup from each
    server's advertised tool list (see `mcp.py`). A tool the harness doesn't recognize contributes
    no evidence — notably `Researcher()`'s own web tools, which are verified by URL instead.
    """
    if not tool_to_server:
        return {}

    server_by_call_id: dict[str, str] = {}
    for message in messages:
        for part in getattr(message, "parts", []):
            if isinstance(part, BaseToolCallPart) and part.tool_name in tool_to_server:
                server_by_call_id[part.tool_call_id] = tool_to_server[part.tool_name]

    evidence: dict[str, list[str]] = {}
    for message in messages:
        for part in getattr(message, "parts", []):
            if not isinstance(part, BaseToolReturnPart) or part.outcome != "success":
                continue
            server = server_by_call_id.get(part.tool_call_id)
            if server is not None:
                evidence.setdefault(server, []).append(_return_text(part.content))
    return {server: "\n".join(chunks) for server, chunks in evidence.items()}


def source_retrieved(source: Source, fetched: set[str], evidence: Mapping[str, str]) -> bool:
    """Whether this run actually retrieved what `source` claims to cite.

    Polymorphic in `source.kind`, per PRD §10 — the two kinds carry different evidence:

    - `web`: the exact URL was fetched successfully (checked against tool-call arguments).
    - `mcp`: the identifier appears in what that server successfully returned. Well-formedness is
      not evidence: a plausible-looking 8-character key no call ever returned is still dropped.
    """
    if source.kind == "web":
        return source.identifier in fetched

    if source.server is None or len(source.identifier) < MIN_MCP_IDENTIFIER_LENGTH:
        return False
    return source.identifier in evidence.get(source.server, "")


def verify_finding(
    finding: SubFinding,
    fetched: set[str],
    evidence: Mapping[str, str] | None = None,
) -> SubFinding:
    """Drop any cited source this run didn't actually retrieve.

    Closes the "agent cites a source it never really read" failure mode (PRD §8), for both source
    kinds. Dropped sources are noted in `contradictions` rather than silently disappearing, so a
    human reviewing the final report can still see that something was claimed but not
    substantiated.

    `evidence` defaults to empty, which means *no* MCP citation verifies. That is the correct
    default rather than an inconvenient one: an empty map means no recognized MCP tool returned
    anything this run, so a finding citing library items didn't get them from here.
    """
    evidence = evidence or {}
    dropped = [s for s in finding.sources if not source_retrieved(s, fetched, evidence)]
    if not dropped:
        return finding

    kept = [s for s in finding.sources if source_retrieved(s, fetched, evidence)]
    note = "Dropped unverified source(s) not confirmed retrieved this run: " + ", ".join(s.label() for s in dropped)
    return finding.model_copy(update={"sources": kept, "contradictions": [*finding.contradictions, note]})
