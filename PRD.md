# PRD: Deep Research Harness (Pydantic AI)

**Status:** Draft
**Owner:** TBD
**Related:** [`pydantic-zotero-mcp`](https://github.com/jmlon/pydantic-zotero-mcp) (the first MCP
tool source, §5b)

> Historical note: this PRD was written while the harness lived inside a larger `PydanticAI`
> monorepo. Relative paths such as `../../MCP/ZoteroMCP/` and references to
> `implementation-guidelines.md` describe that layout; the harness now lives in its own
> repository and the Zotero server is published as `pydantic-zotero-mcp` on PyPI.

## 1. Problem statement

A single `Researcher()` capability stack (`WebSearch` + `WebFetch` + `SubAgents` +
`ToolOutputLimits`) is enough for shallow, single-hop questions ("what changed in the last three
Django releases?"). It is not enough for **deep research** tasks — open-ended questions that
require decomposing into sub-questions, running many searches in parallel, reconciling
contradictory sources, and producing a long, structured, fully-cited report. Those tasks need an
explicit multi-phase harness on top of the base `Researcher` primitive, not just a bigger prompt.

## 2. Goals

- Given a broad research question (and optional constraints: depth, time horizon, output format),
  produce a structured, **fully source-attributed** report — every non-trivial claim traceable to
  a fetched source.
- Handle **breadth**: decompose into sub-questions and research them in parallel via sub-agents,
  rather than one agent serially searching everything.
- Handle **depth**: allow follow-up research when a sub-answer is thin, contradictory, or
  contested, up to a bounded budget.
- Research **beyond the open web**: accept additional tool sources — bibliographic databases,
  internal knowledge bases, specialized search services — as configured MCP servers, so a run can
  draw on a curated corpus the web does not index. See §5b.
- Be **resumable**: a long research run must survive a process restart / context reset without
  re-doing already-completed sub-research.
- Be **bounded and observable**: explicit cost/iteration/time caps, and full trajectory tracing
  (which queries ran, which pages were fetched, which claims came from where).

## 3. Non-goals

- Not a general-purpose `Coder`-style agent — no file editing of a target codebase, no shell
  execution against arbitrary systems.
- The chat interface (§4) is a **control surface for scoping and HITL checkpoints**, not a
  turn-by-turn research conversation — once scoping is confirmed, research itself still runs as a
  bounded, mostly-unattended job (minutes, not turns) that reports progress back into the chat.
- Not responsible for hosting/serving the report anywhere beyond the CLI (Slack bot, web UI,
  etc.) — the harness produces a Markdown artifact; other delivery channels are out of scope.
- Not attempting perfect fact-checking — verification here means "every claim has a source link
  and the source was actually fetched and read," not independent ground-truth validation.
- Not building MCP **servers**. The harness is an MCP *client/host*; a server such as
  `zotero-mcp` is a separate deliverable with its own PRD, and the harness's job is to configure,
  launch, bound, and cite it — not to reimplement its retrieval.
- Not a general MCP host. Servers are **operator-configured and expected to be read-only** (§5b).
  There is no discovery mechanism, no per-run server installation, and no path by which a research
  agent acquires a write- or shell-capable tool — see §8's blast-radius argument.

## 4. Users / entry points

- **Primary:** an internal engineer or analyst using a **CLI chat application** built on top of
  this harness. The chat interface is the mechanism for:
  - submitting the initial research question in natural language;
  - the **human-in-the-loop touchpoint** for phase 1 (clarify & scope) — the lead agent asks
    clarifying questions directly in the chat when the brief is ambiguous, and the human answers
    inline before research proceeds;
  - optional mid-run checkpoints (e.g. surfacing a low-confidence/contradictory finding for a
    human call) without needing a separate approval system;
  - receiving the final report, rendered as Markdown directly in the terminal.
- **Secondary (future):** scheduled/triggered runs (e.g. "research competitor X weekly") with no
  human present — see §14, out of scope for v1. Because phase 1 depends on the chat interface for
  ambiguity resolution, an unattended mode will need a fallback (proceed on reasonable assumptions,
  as already described in §6) rather than blocking on a human who isn't there.

### 4a. Distribution model

The tool ships as a **standalone, `pipx`-installable package** (its own `pyproject.toml`,
independent of any surrounding monorepo) rather than something run out of a checked-out source
tree. The install is global and one-time; each researcher then works from their own **project
folder** — a plain, otherwise-unremarkable directory with no required structure — and runs
`deep-research` from inside it. This has two concrete implications for the design:

- **Config discovery is project-folder-relative, not install-location-relative.** The tool looks
  for `./config.yaml` in the current working directory first, falling back to a bundled default
  shipped inside the package if the project hasn't been initialized. A `deep-research init`
  command scaffolds `config.yaml` + the report directory in the current directory so a researcher
  can start customizing without hand-copying anything out of the install location.
- **All relative config paths — especially `output.report_dir` (§13) — resolve against the
  project folder (the CWD the tool was invoked from), never against wherever `pipx` put the
  package.** This is what makes "each researcher works in their own folder, output lands in their
  own `reports/`" true regardless of how many researchers share the same globally-installed tool.

MCP support (§5b) adds a third implication, because one of its three transport modes puts a server
*inside* this package:

- **In-memory MCP servers ship as optional dependency extras of this package**; local-process and
  external servers are described entirely in the project's `config.yaml` and ship nothing.
  `pipx install "deep-research-harness[zotero]"` installs the harness and a bundled Zotero server
  together; `pipx inject` adds one to an existing install. Extras rather than hard dependencies
  because each bundled server drags its own transitive tree (`fastmcp`, `pyzotero`, …) that a
  web-only researcher has no use for, and because a server release should not force a harness
  release for people not using it. The trade-off is explicit: **bundling couples the server's
  version to an install step, while stdio/HTTP servers can be upgraded independently of the
  harness.** For a server whose release cadence you don't control, that argues for stdio.
- **This is the one case where "config in the project folder" is not the whole story.** A bundled
  server's *code* comes from the install; its *behaviour* (`tool_args`, `allowed_tools`,
  `instructions`) still comes from the project's `config.yaml`, and its *credentials* still come
  from the project's `.env`, which the CLI already loads. So a bundled server is enabled per
  project, not per install, and two researchers sharing one global install can point the same
  bundled Zotero server at two different libraries. Nothing about bundling moves configuration into
  the install location — that would undo the property §4a exists to guarantee.

## 5. Architecture overview

Mapped onto `H = (E, T, C, S, L, V)` and the Pydantic AI capability model from
`implementation-guidelines.md`. **This table describes the system as built.** Where the design
settled somewhere other than the capability originally named, the reason is given — several of
those choices are load-bearing and were re-litigated once already.

| Component | Implementation |
| --- | --- |
| **E** — Execution loop | Plain async Python in `research.py`: `asyncio.gather` over a `worker_concurrency` semaphore, with round counting and budget arithmetic in code. Not `Dynamic Workflow` — that capability exists for *model-decided* fan-out, whereas this fan-out is fully determined (one worker per open sub-question), and it would add a dependency (`pydantic-ai-harness[dynamic-workflow]`) to buy nothing. A deterministic blueprint around agent calls is what the guidelines prescribe for exactly this shape. |
| **T** — Tool registry | `Researcher()` on the worker agent only, which supplies `WebSearch` + `WebFetch` + `ToolOutputLimits`, **plus one MCP toolset per configured server** (§5b) — also worker-only, so the "exactly one agent has tools" invariant that §9 rests on survives MCP support. Search and fetch resolve **per provider**: Pydantic AI uses a provider-native tool where the model has one and its local fallback where it doesn't (Anthropic has native web fetch, OpenAI does not). That is not an implementation detail — it changes the message-part classes a fetch produces, and assuming one shape silently broke source verification for a whole provider (§10). No `Shell`/`FileSystem` on any agent — see §8. |
| **C** — Context manager | `ToolOutputLimits` on the worker (the only agent with tools; the other four take a rendered prompt and return structured output). MCP tool returns are the largest single payloads in the system — a Zotero PDF full text defaults to 100,000 characters, ~25–30k tokens — so they are bounded twice: by `ToolOutputLimits` and by injected `tool_args` caps (§5b, §9). **No `Compaction` anywhere, deliberately:** no agent's context grows across the run. See §9. |
| **S** — State store | A `ResearchState` artifact (Pydantic model, persisted as JSON) tracking sub-questions, findings, what a budget cut off, and phase status — see §7. It is also the plan, which is why no `Planning` capability is used: `open_subquestions`/`unresolved_subquestions` on disk outlive the process and drive `deep-research resume`, where an in-context todo list would not. **Not built:** the `Memory` capability for cross-run knowledge ("we already researched competitor X last week") — see §14. |
| **L** — Lifecycle hooks | Two hooks, at opposite ends of a tool call. **After** the run: source validation as a deterministic post-run check (`verify.py`), not as a tool hook — a citation is chosen in the worker's final structured output, not in a tool call, so there is no tool boundary to intercept. Every cited **identifier** (a URL for web sources, a server-native ID such as a Zotero item key for MCP ones) is checked against the identifiers that appeared in successful tool calls/returns in that run's message history, and any the worker didn't actually retrieve is dropped and noted (§10). **Before** an MCP tool call: argument injection, which *is* a real tool boundary — configured `tool_args` are merged into the call so operator-set scoping and output caps cannot be dropped by the model (§5b). **Not built:** the pre-search hook that dedupes queries or fetches across workers — see §14. |
| **V** — Evaluation interface | A separate critic agent (generator/evaluator separation) checking source attribution and coverage before the report ships (§10). Stopping is enforced in code, not by any agent's judgment: `budget_exhausted()` bounds research rounds, and the critic's verdict — not the generator's self-report — decides whether another round happens. |

### 5a. Agent roles

Five narrow agents, each a separate `Agent` instance with its own instructions. Note that **no
agent orchestrates**: `research.py` does, and every agent call is stateless with respect to the
others (a fresh `agent.run(prompt)` rendered from `ResearchState`).

1. **Lead agent** — scoping only (phase 1): alternates `ClarifyingQuestion` / `BriefDraft` with
   the human, and produces the initial sub-question list. This is the only agent that keeps a
   conversation, and only within that phase.
2. **Worker agent** (`Researcher()`-based) — one run per sub-question, fresh context each,
   returning a structured `SubFinding` (§7). Parallel up to `worker_concurrency`. Spawned as
   ordinary concurrent `agent.run` calls by the orchestrator, not via `SubAgents` delegation —
   though `Researcher()` does carry a `SubAgents` toolset the worker may use internally.
3. **Gap-check agent** — no tools; reads the findings summary and proposes follow-up
   sub-questions or signals that coverage is sufficient (phase 4).
4. **Synthesis agent** — no tools; merges findings into the `ResearchReport` (phase 5).
5. **Critic agent** — no tools; reviews the draft report against the brief and the findings, and
   returns pass/fail plus concrete issues and any research that would fix them (phase 6).

The gap-check and synthesis agents share `model.lead` with the lead agent by configuration, but
they are distinct instances with non-overlapping instructions, so scoping guidance cannot bleed
into synthesis. The critic has its own `model.critic` so it can be a cheaper model.

### 5b. MCP tool sources

`WebSearch` + `WebFetch` reach what the open web indexes. That is the wrong retrieval target for a
large class of research: peer-reviewed literature behind paywalls, a researcher's own curated
corpus, an internal knowledge base, a specialized search API. MCP is the standard interface for
those (`implementation-guidelines.md` §2), and the `mcp_servers` config key has been reserved for
it since the first draft of §13 without ever being wired up. This section specifies the wiring.

**The first server is `zotero-mcp`** (`../../MCP/ZoteroMCP/PRD.md`) — read-only access to a Zotero
library: metadata and full-text search, item detail, the researcher's own notes, collections and
tags, and the indexed full text of attached PDFs. Its own bet is directly complementary to this
harness's: "generic web search returns what exists; a Zotero library returns what *this researcher
already decided was worth keeping*." A deep-research run that can consult both is strictly better
sourced than one that can only search the web.

#### Where they attach

**Worker agents only**, for v1. This is the load-bearing constraint of the whole design: exactly
one agent in the system has tools, and §9's "no agent's context grows across the run" argument —
which is why there is no `Compaction` anywhere — depends on that staying true. MCP servers return
the largest payloads the harness will ever see, so this is the wrong place to relax it.

Each configured server becomes one MCP toolset on the worker agent, alongside `Researcher()`'s
`WebSearch`/`WebFetch`. The worker therefore chooses *per sub-question* whether the library, the
web, or both are the right source, which is the decision that actually needs model judgment.

A **survey agent** — a tool-using agent sitting below the lead and above the workers, which
orients in the corpus before sub-questions are fixed ("the library holds 40 items on this topic,
across these three collections") — is a genuinely valuable future extension and is recorded as a
line of work in §14a. It is deliberately *not* the lead agent: the lead holds the only
conversation in the system (§9), and pouring corpus survey output into that context is exactly the
saturation this architecture avoids everywhere else.

#### Transport modes

A server reaches the harness one of three ways, and the choice is not merely a connection detail —
it determines where the server's configuration lives, whether it ships inside this package, and
what isolation the harness has from it. All three are supported; none is a fallback for another.

| Mode | Transport | Where the server runs | Where its config lives |
| --- | --- | --- | --- |
| **In-memory** | `FastMCPTransport` | Inside the harness process, as an imported library | **Bundled with the package**; `config.yaml` enables it and sets `tool_args`, credentials come from the environment/`.env` |
| **Local process** | `stdio` | A subprocess the harness launches | Entirely in the project's `config.yaml` — command, `cwd`, `env` |
| **External** | Streamable HTTP | Another host, over the open internet | Entirely in the project's `config.yaml` — `url`, `auth_token_env` |

SSE is not supported, matching the Zotero server's own decision (Zotero PRD §7.2): it is
deprecated upstream and there is no legacy client here to accommodate.

**In-memory: bundled, so it must be packaged.** A server used in-memory is a Python dependency of
this package — the harness imports its `create_server()` factory, builds it with a settings object,
and talks to it over `FastMCPTransport` with no subprocess, no socket, and no serialization hop.
That means it must ship *with* the tool for `pipx` distribution to work at all: a researcher who
runs `pipx install deep-research-harness` and writes `transport: in_memory` in `config.yaml` must
get a working server without cloning anything. Concretely (§4a, §14):

- Bundled servers are declared as **optional dependency extras**, not hard dependencies:
  `pipx install "deep-research-harness[zotero]"`. Every bundled server pulls its own transitive
  tree (`fastmcp`, `pyzotero`, …), and a researcher who only wants web research should not install
  a Zotero client, nor should a Zotero release force a harness release for people not using it.
- **`zotero-mcp` is not currently a distributable package** — `MCP/ZoteroMCP/` has no
  `pyproject.toml` and depends on the monorepo root environment. Packaging it is therefore a
  prerequisite for in-memory support, not a consequence of it, and is called out as its own step in
  §14 v4.
- Because the factory is *imported*, an arbitrary `factory: "module:function"` string in
  `config.yaml` would make that file a code-execution vector — the same posture problem §8 spends
  effort avoiding. Bundled servers are instead **discovered**, via a `deep_research.mcp_servers`
  entry point that a server package declares; `config.yaml` names one by `name` and nothing else.
  This keeps the standard Python plugin pattern (a new bundled server is `pipx inject` plus one
  config line, no harness code change) without config.yaml naming importable code.

**What in-memory buys, and what it doesn't.** It removes process startup, the socket, and JSON
serialization on every tool call, and — worth more in practice — it removes the whole class of
`cwd`/`env`/interpreter-path breakage that stdio invites in a `pipx`-installed tool where the
harness and the server live in different environments (§4a). It does **not** reduce token cost by
one token: a tool return enters the model's context identically whichever transport carried it, so
every bound in §9 applies unchanged. In-memory is a plumbing optimization, not a budget one.

**Trust and containment differ per mode, in opposite directions.** This is the honest version of
the read-only caveat at the end of this section:

- **In-memory** has *no* isolation — the server shares the harness's process, memory, environment,
  event loop, and filesystem privileges — but the strongest provenance, since it is a pinned,
  versioned dependency the operator installed deliberately. The Zotero PRD says exactly this of its
  own in-memory mode: "a trust boundary, not a sandbox" (§7.2, §11). A crash or a hang in an
  in-memory server can take the whole run with it, which is the one place the "a dead server does
  not fail the run" containment below does not hold.
- **Local process** has real isolation for free: its own environment (so it can be handed exactly
  the credentials it needs and nothing else), and a crash that the harness survives.
- **External** has a network boundary and mandatory auth, but the weakest provenance — the code is
  operated by someone else and changes without notice — and one property the other two lack
  entirely: **the research question and any corpus content leave the machine.** For a personal
  bibliography that is a privacy decision the operator must make knowingly, which is why the
  Zotero server refuses to boot over HTTP without a token and never binds beyond `127.0.0.1`
  unasked (Zotero PRD §8).

For `zotero-mcp` specifically the recommendation is **in-memory** once it is packaged (simplest
install, no path resolution to get wrong, and the corpus never leaves the process), **stdio** while
it is still a sibling directory in this monorepo, and **HTTP** only for a genuinely shared group
library.

#### Passing extra information to tools

Tools frequently need information the sub-question doesn't carry: which collection to scope a
search to, which sites are in scope, how many characters of a PDF to return, which citation style
to use. Two mechanisms, deliberately both, because they answer different questions:

1. **`tool_args` — deterministic argument injection (what the call must contain).** A per-server,
   per-tool mapping of default arguments that the harness **merges into every matching MCP tool
   call before it is dispatched**. This is a `Lifecycle hook` at a real tool boundary (§5, **L**),
   not prompt text, so the model cannot forget, reinterpret, or drop it. This is the mechanism for
   anything the operator needs to be *true of every call*: a `collection_key` that scopes a run to
   one part of a library, a `max_chars` that keeps a full-text return inside the context budget.
   Model-supplied arguments for keys not present in `tool_args` pass through untouched; where both
   supply a key, **`tool_args` wins** — the point is a floor the model cannot lower.

2. **`instructions` — a free-text hint (when to reach for this server at all).** Appended to the
   worker's rendered prompt, naming the server and what it is good for. Injection alone leaves a
   worker with a tool it has no idea when to use; guidance alone leaves an operator with caps a
   model may ignore. Neither is sufficient, so both ship.

Per-run overrides come from a single CLI flag taking a **JSON string**, keyed by server then tool,
merged over the configured `tool_args` (§13):

```bash
deep-research run "..." --mcp-args '{"zotero": {"search_items": {"collection_key": "XY99ABCD"}}}'
```

A key in `--mcp-args` naming a server or tool that isn't configured is a **startup error**, not a
silent no-op — same reasoning as `extra="forbid"` on the config (§13): an override that quietly
did nothing is worse than one that refuses to start, because the run looks scoped and isn't.

Injected arguments are **recorded in the trajectory** (§11) and, where they narrow what was
searched, surfaced in the report's `assumptions` — a report produced from one collection of a
library is a materially different claim than one produced from the whole of it, and the reader
cannot tell from the citations alone.

#### Lifecycle, concurrency, and failure

- **One server session per run, shared by all workers** — never one per worker, in any transport
  mode. (As built, the session brackets the research *pipeline* rather than literally phases 1–7,
  for an event-loop reason recorded in §14b; it still covers every phase that can use a tool.) A stdio server launched per worker would multiply process startup and, for Zotero, blow
  past the ≤ 4 concurrent upstream requests its API asks for (Zotero PRD §7.4) once
  `worker_concurrency` workers each held their own client; an in-memory server built per worker
  would duplicate its caches and do the same. Sessions open before phase 1 and close after phase 7,
  around the whole run.
- **In-memory sessions must run on the harness's own event loop.** The server is constructed with
  `create_server(settings)` and entered as `async with Client(server)`; its `lifespan` — which is
  where the upstream client and caches are built (Zotero PRD §7.2) — therefore runs under the
  harness's loop, and cache lifetime is the run, not the process. `mcp.run()` must never be called:
  it would try to start a second event loop inside a process that already has one.
- **`worker_concurrency` already bounds in-flight tool calls** at 4 by default, which coincides
  with Zotero's own recommended ceiling. Servers with tighter limits must bound themselves; the
  harness does not model per-server rate limits. One exception worth stating: an **external** HTTP
  server may be serving other clients too, so `worker_concurrency` bounds this run's load on it,
  not the server's total load — a shared endpoint can rate-limit a run that is individually
  well-behaved.
- **Fail fast at startup.** Every configured server is probed during the existing preflight that
  resolves model strings, *before* any scoping call: the session is opened and its tool list
  fetched. A server that cannot start exits non-zero, naming the server and surfacing its own
  error text. What "cannot start" means differs by mode, and so should the remedy in the message:

  ```
  ✗ MCP server 'zotero' (in_memory) failed to start:
    Bundled server 'zotero' is not installed.
    Install it with:  pipx inject deep-research-harness deep-research-harness[zotero]

  ✗ MCP server 'zotero' (stdio) failed to start:
    Command not found: uv (cwd: /home/me/research/../MCP/ZoteroMCP)

  ✗ MCP server 'zotero' (http) failed to start:
    Connection refused: https://mcp.internal.example/mcp
  ```

  The operator configured that server deliberately. A run that silently fell back to web-only
  would produce a bibliographic report that *looks* complete and isn't — a wrong answer that
  reads like a right one, which is the failure mode this harness spends §10 defending against.
  A server may opt out with `optional: true`, in which case its failure is a warning, the server
  is dropped, and the degradation is recorded in the report's `assumptions` so it appears in the
  deliverable rather than only in the console scrollback.
- **The probe must call a tool, not just open the session.** Listing tools is not proof the server
  works. `zotero-mcp` is the concrete case: `create_server()` raises on malformed settings, but a
  *rejected* API key only logs a warning in its lifespan — deliberately, so a bad key doesn't crash
  an editor's MCP launch (Zotero PRD §7.2, and the code confirms it). Under a list-only probe, a
  server with an invalid credential starts clean and then fails every call, which is precisely the
  silent-degradation outcome fail-fast exists to prevent. The preflight therefore issues one cheap
  read — `get_library_info` for Zotero, a configurable `health_check` tool name in general — and
  treats its failure as a failure to start.
- **A tool call failing mid-run is contained**, exactly like a failed fetch: the worker retries
  once and then lowers confidence (§8). A dead server does not fail the run — with one asymmetry
  named in the transport table above: an in-memory server shares the harness's process, so an
  unhandled crash or a deadlock in it is *not* contained the way a subprocess or a remote endpoint
  is. Bundling a server is a decision to trust its stability as well as its intent.
- **Read-only expectation.** §8 forbids `Shell`/`FileSystem` on research agents to keep the blast
  radius small, and an MCP server is a general-purpose escape hatch from that guarantee — code the
  operator chose, free to expose any tool at all, and in the in-memory case running with the
  harness's own privileges. The harness cannot enforce read-only-ness across a protocol boundary
  (and has no boundary at all in-memory), so it does two things it *can* do: a per-server
  `allowed_tools` allowlist, so a server exposing more than the harness needs contributes only the
  named subset; and treating the choice of server as an operator trust decision, documented as
  such. For Zotero specifically this composes with the server's own posture — writes off by
  default, no delete tools at any flag setting (Zotero PRD §5.5) — and the harness should be
  pointed at a **read-only API key**.

## 6. Workflow phases

1. **Clarify & scope** (lead agent, via the CLI chat interface): turn the raw question into a
   `ResearchBrief` — restated question, explicit sub-questions, depth/breadth budget. The lead
   agent asks clarifying questions directly in the chat when the brief is ambiguous, and proceeds
   once the human confirms scope (or, if run unattended, falls back to reasonable defaults and
   notes the assumptions in the final report).
2. **Plan** (folded into phase 1, no separate call): the sub-question list arrives as
   `BriefDraft.subquestions` and becomes `ResearchState.open_subquestions`, each a unit of work
   for a worker. Asking the lead agent to decompose in the same turn it restates the question
   costs one call instead of two and keeps the plan visible in the brief the human confirms. The
   `Planning` capability is not used — the plan lives in the state store (§5, **S**).
3. **Parallel research** (orchestrated in `research.py`): fan out one worker run per open
   sub-question, each producing a `SubFinding` (answer, confidence, sources). Concurrent
   `agent.run` calls bounded by a `worker_concurrency` semaphore — not `SubAgents` delegation or
   `Dynamic Workflow`, for the reasons in §5 (**E**). Each worker chooses per sub-question among
   web search/fetch and any configured MCP tool sources (§5b); which source a citation came from is
   recorded on the `Source` itself, not inferred later.
4. **Gap check & follow-up** (gap-check agent): review findings for thinness/contradiction; spawn a
   bounded number of follow-up sub-questions (depth budget in §8 governs how many rounds). Repeat
   phase 3 for new sub-questions only — this is a `pipeline`-shaped loop, not a full re-run.
5. **Synthesis** (synthesis agent): merge all `SubFinding`s into a structured report
   (`ResearchReport`), reconciling contradictions explicitly rather than silently picking a side,
   and funded by the report allowance in §8 so it is affordable even when research overran. No
   `Compaction` — nothing accumulates to compact, see §9.
6. **Critique & verify** (critic agent, §10): check every claim in the draft has a matching
   `SubFinding` source; check sub-questions from the brief were actually covered. On failure,
   return to phase 3/5 with the specific gaps identified, bounded by the retry budget.
7. **Deliver**: render the final `ResearchReport` as **Markdown** in the chat interface and persist
   it to disk; persist the full `ResearchState` trajectory (JSON) separately for observability/
   resumability — the state artifact is internal bookkeeping, not a user-facing output format.

## 7. Data model (state & handoff artifacts)

```python
class SubFinding(BaseModel):
    subquestion: str
    answer: str
    confidence: Literal["high", "medium", "low"]
    sources: list[Source]          # url, title, fetched_at, quoted_snippet
    contradictions: list[str] = [] # notes on conflicting sources found

class Source(BaseModel):
    kind: Literal["web", "mcp"] = "web"   # how it was retrieved — governs verification & rendering
    identifier: str                       # a URL for web; a server-native ID (Zotero item key) for mcp
    server: str | None = None             # which configured MCP server returned it; None for web
    title: str
    fetched_at: datetime
    quoted_snippet: str

class ResearchBrief(BaseModel):
    question: str
    subquestions: list[str]
    depth_budget: int              # max follow-up rounds
    breadth_budget: int            # max concurrent/total sub-questions
    assumptions: list[str] = []

class ResearchState(BaseModel):
    brief: ResearchBrief
    findings: dict[str, SubFinding]     # keyed by subquestion
    open_subquestions: list[str]
    round: int
    status: Literal["planning", "researching", "synthesizing", "critiquing", "done", "failed"]

class ResearchReport(BaseModel):
    question: str
    summary: str
    sections: list[ReportSection]       # each citing SubFinding sources
    unresolved: list[str]               # gaps the critic flagged and budget ran out on
    assumptions: list[str]

    def to_markdown(self) -> str:
        """The one user-facing rendering of a report — see §7a."""
```

`ResearchState` is the **handoff artifact**: persisted as JSON after every phase so a crashed or
context-reset run resumes from the last completed phase/round instead of restarting (§5, State
store). This is the concrete instance of "context resets over endless compaction" from the
guidelines — a fresh lead-agent run rehydrates from `ResearchState`, not from conversation history.
`ResearchState` (and `SubFinding`/`Source`) are internal bookkeeping, never shown to the user
directly.

**Why `Source` gained provenance** (`kind`/`identifier`/`server`, replacing a bare `url`): a
Zotero item is identified by an 8-character library key, not a URL, and many items have no useful
public URL at all. Under the old shape a library citation had nowhere to go, and the consequence
was not cosmetic — the verification check in §10 accepts only URLs confirmed fetched this run, so
**every Zotero-sourced citation would have been silently deleted from every finding**, and a
worker that correctly answered a sub-question from the library would have reported it as
unsourced. Provenance makes verification and rendering polymorphic in the one place that actually
differs (what counts as "the same source I retrieved") while keeping a single `Source` type
everywhere else. `server` is what lets the report attribute a claim to *which* corpus it came
from, and lets §11 measure whether a configured server earns its cost.

**Checkpoint compatibility:** `ResearchState` is persisted JSON that `deep-research resume` loads
back through `model_validate_json`. Renaming `url` → `identifier` invalidates every existing
checkpoint on disk. Accept old checkpoints by giving `identifier` a validation alias for `url`
and defaulting `kind` to `"web"`, so a pre-MCP checkpoint rehydrates as exactly what it was.
Failing to resume a long, expensive, already-paid-for run is the most annoying possible way to
ship this change.

### 7a. Output format

Output is **strictly Markdown** — `ResearchReport.to_markdown()` is the one user-facing rendering,
shown in the CLI chat and written to disk as the deliverable (e.g. `report.md`). There is no
separate JSON deliverable for consumers; the JSON `ResearchState` trajectory (§11) is an
observability/debugging artifact, not an alternate output format. Citations render as standard
Markdown links (`[title](url)`) inline in each section.

**Rendering an MCP source.** A `kind="mcp"` source has no URL to link, so it renders as the title
plus an attributed identifier — the server's name and the item's native ID — rather than a broken
or invented link:

```markdown
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — "..."     # kind="web"
- Attention Is All You Need — Zotero item `ABC12345` — "..."               # kind="mcp"
```

This is deliberately plain rather than clever. Two alternatives were considered and rejected: a
`zotero://items/ABC12345` link (the URI is real — Zotero PRD §5.6 — but it resolves only inside
that server's own address space, so in a report shared with anyone else it is a link that goes
nowhere), and silently substituting the item's `url` field where it has one (which would render a
citation pointing at a page the worker never read, reintroducing §8's failure mode through the
renderer). Attributing the identifier honestly costs the reader one lookup and claims nothing
false. A properly formatted bibliography — via the Zotero server's own `format_bibliography`
(Zotero PRD §5.4, M3) — is the natural upgrade once that milestone lands, and is noted in §14a.

## 8. Guardrails and budgets

- **`breadth_budget`**: hard cap on total sub-questions ever created (initial + follow-ups)
  across the whole run — prevents unbounded fan-out. **Default: 8** (roughly: 4–6 initial
  sub-questions from phase 2, leaving headroom for a couple of follow-up rounds).
- **`depth_budget`**: hard cap on follow-up rounds (phase 4) — prevents infinite gap-chasing.
  **Default: 2** follow-up rounds after the initial pass.
- **`Spend Limits`**: token/cost ceiling on the whole run, accumulated across every agent call
  rather than applied per call. Split into two pots: the ceiling bounds **research**, and the
  **write-up** (synthesis + critique) gets a separate guaranteed allowance
  (`report_token_allowance` / `report_spend_allowance_usd`) granted on top of whatever research
  actually cost. Reserving a *fraction* of the ceiling instead does not work — research can
  overshoot the ceiling (see below), consuming the reserve before synthesis starts and leaving a
  run that paid for research and produced no report.
- Both ceilings are **soft**. `worker_concurrency` requests are in flight at once and a single
  worker turn can add ~100k tokens, so a run can pass the ceiling by roughly the concurrency
  factor before anything notices (measured: 303k against a 170k ceiling). Size the number with
  that in mind; exceeding it is contained, not fatal.
- **Concurrency cap** on simultaneous worker sub-agents (protects rate limits on search/fetch
  providers, keeps cost predictable). **Default: 4** concurrent workers.

These defaults are starting points, not tuned values — see §13 (they live in the YAML config, not
hardcoded) and §12 for the plan to revisit them empirically once real research runs accumulate.
- **No `Shell`/`FileSystem`** on any research agent — this harness only reads the web; it has no
  business writing to disk or executing commands. Keeps the tool registry minimal and the blast
  radius small.
- **No `Shell`/`FileSystem` via MCP either.** An MCP server is a general escape hatch from the
  bullet above — an operator-launched process free to expose any tool. The harness's enforceable
  half is a per-server `allowed_tools` allowlist (§5b): a server contributes only the tools named,
  so a server that grows a new capability doesn't silently hand it to a research worker. The
  unenforceable half — that the server is read-only at all — is an operator trust decision, stated
  as one in §5b rather than pretended away.
- **MCP tool output is bounded twice**: by `ToolOutputLimits` on the worker, and by injected
  `tool_args` caps (§5b) on the tools known to return large payloads. Zotero's
  `get_item_fulltext` defaults to 100,000 characters — roughly 25–30k tokens for a *single* call,
  the largest payload anywhere in this harness — so a configured `max_chars` is not a nicety;
  without one a handful of full-text reads is the run's entire token ceiling. §9 quantifies this.
- **Source-retrieval verification hook**: reject any citation whose identifier isn't in the set of
  identifiers this run actually retrieved content for — a URL that `WebFetch` returned content
  for, or a server-native ID (a Zotero item key) that a successful MCP tool call returned. Closes
  the "agent cites a source it never really read" failure mode for both source kinds; a library
  item named in a search result but never opened is no more citable than an unfetched URL.
- Treat a stalled/failing worker as a **recoverable** error (retry once, then mark that
  sub-question `low confidence` / unresolved) rather than failing the whole run — one bad source
  shouldn't sink the report.

## 9. Context management specifics

**No agent's context grows across the run, so there is nothing to compact.** An earlier draft of
this section assumed a long-lived lead agent accumulating `SubFinding`s across dozens of
sub-questions, and prescribed `Compaction` for it. That agent does not exist. The state store
took its place, which is the same fix the guidelines recommend ("context resets over endless
compaction") applied at the architecture level instead of the capability level.

How it actually works:

- **Every phase call is stateless.** `research.py` calls `agent.run(prompt)` with no
  `message_history`; the prompt is rendered fresh from `ResearchState` by the `render_*_prompt`
  helpers. The only conversation in the system is inside phase 1, where the lead agent and the
  human genuinely are talking to each other.
- **`ResearchState` is the source of truth between phases**, not any context window. Findings are
  summarised into it as they complete, and re-rendered compactly for whoever needs them next.
- **Worker agents:** fresh context each, with `ToolOutputLimits` capping any single fetched page's
  contribution — the one place a single tool return can be large enough to matter. **MCP servers
  raise the stakes here and nowhere else.** A web page fetched through `WebFetch` is markdownified
  prose; a Zotero `get_item_fulltext` return is an entire research paper, defaulting to 100,000
  characters (~25–30k tokens). Three or four of those in one worker is a six-figure token context
  from tool returns alone, against a `total_tokens_limit` of 1.5M shared by the whole run. Hence
  the `tool_args` caps in §5b: the recommended Zotero configuration sets `max_chars` explicitly
  (20,000 is a reasonable starting point — a substantial excerpt, an order of magnitude off the
  ceiling), and the cap is *injected* rather than suggested precisely because a model asked
  politely to pass `max_chars` will sometimes not.
- **Recitation** is a property of the rendered prompt, not of memory: each phase is *given* the
  current open and unresolved sub-questions, so it cannot misremember what is left to do.

Measured headroom, at the default `breadth_budget` of 8 with verbose answers, five sources per
finding, contradictions, and four unresolved items: the critique prompt — the largest of them —
comes to roughly **3,900 tokens against a 250,000-token report allowance (§8), a ~65× margin.**
Compaction would be solving a problem two orders of magnitude away.

The thing that *does* need watching is the opposite end: `_findings_summary` renders each
finding's full `answer`, so prompt size grows linearly with `breadth_budget`. At a breadth of
several dozen, revisit this section rather than assuming the margin still holds.

## 10. Verification strategy

Generator/evaluator separation, applied twice:

1. **Per-finding**: a worker's `SubFinding` is only accepted if every source in it was actually
   retrieved this run (deterministic hook, §8) — cheap, automatic, no extra agent call. **The check
   is polymorphic in `Source.kind`**, because "the same source I retrieved" means different things
   per transport:
   - `kind="web"` — the identifier is a URL, matched against URLs from successful tool calls
     carrying a `url` argument. Matching stays by tool-call *shape*, not tool name, for the
     provider-native-vs-local-fallback reason already documented in `verify.py`.
   - `kind="mcp"` — the identifier is a server-native ID, matched against the IDs that appear in
     the **returns** of successful MCP tool calls attributed to that `server`. Note the asymmetry
     with the web case: a URL is verified from the *call* (the worker chose to fetch it), but a
     Zotero key is verified from the *return* (the worker asked for "attention interpretability"
     and the server decided which keys came back). Reading the call arguments alone would verify
     nothing about MCP sources, since the key generally isn't in them.

   The failure mode this closes is specific and was live: with a URL-only check, a correct
   library-sourced finding is indistinguishable from a fabricated one, and every real citation
   gets dropped (§7). It is worth an explicit test per source kind, plus one asserting a
   *fabricated* Zotero key — well-formed, 8 characters, never returned by any call — is still
   dropped. Well-formedness is not evidence of retrieval.
2. **Per-report**: the critic agent (no search tools, reads only the draft report + `ResearchState`)
   checks: (a) every claim traces to a `SubFinding`/source, (b) every brief sub-question is
   addressed or explicitly listed under `unresolved`. This is the `/goal` maker/checker split
   applied to "is the report done," not just "is the report accurate."

Critic failure feeds back into the workflow (targeted follow-up sub-questions or a resynthesis
pass), bounded by `depth_budget` so critique can't loop forever — if the budget is exhausted, the
report ships with an honest `unresolved` list rather than looping indefinitely.

## 11. Observability

- Logfire instrumentation on every agent run (lead, workers, critic) so a single trace covers the
  whole research job, not just one phase.
- Trajectory to capture per sub-question: query strings used, URLs fetched, tokens spent,
  confidence assigned, whether it triggered a follow-up round.
- Track aggregate metrics across runs: average sub-questions per brief, average rounds to
  convergence, critic pass rate on first pass — useful signal for tuning `depth_budget`/
  `breadth_budget` defaults over time.
- **Per MCP server, per run**: transport mode, tool calls made, calls that failed, tokens returned,
  and citations that survived verification — broken down by server, with `web` as one of the
  buckets. Recording the transport matters for reading the numbers back: a call-latency or
  failure-rate comparison between an in-memory server and one across the internet is not a
  comparison of the servers. This is
  the only way to answer the question an operator will actually ask, which is whether a configured
  server is earning its cost: a server that is called often and cited never is either wrongly
  scoped (`tool_args` too narrow), wrongly described (`instructions` misleading), or wrong for the
  corpus. Injected `tool_args` are recorded on the run too, since a scoped run's numbers are not
  comparable with an unscoped one's.

## 12. Open questions (resolved)

- **Human-in-the-loop for scoping** → resolved: the harness ships as a **CLI chat application**
  (§4). The chat is the natural surface for phase 1 clarification — the lead agent asks, the human
  answers inline, no separate approval system needed. Unattended/scheduled use (§13) remains a
  later extension with a no-human fallback.
- **`depth_budget`/`breadth_budget` defaults** → resolved: ship the sensible defaults in §8
  (breadth 8, depth 2, concurrency 4), configurable via `config.yaml` (§13), and tune them
  experimentally as real research runs accumulate (tracked via §11 metrics).
- **Swappable search backend / MCP servers** → resolved: made configurable rather than fixed —
  see §13. The harness should not hardcode a specific search provider or tool source. **MCP
  support is now specified in full (§5b) and scheduled (§14, v4)**, with `zotero-mcp` as the first
  server; `search.backend` remains forward-compatibility only. Four sub-decisions, each of which
  had a defensible alternative:
  - *Which transports?* → **all three** (§5b): in-memory for servers bundled with this package as
    optional extras, stdio for a local subprocess, streamable HTTP for a remote endpoint. Not a
    fallback chain — they differ in where configuration lives, what ships in the `pipx` install,
    and what isolation the harness retains, and the last of those runs *opposite* to convenience.
  - *Which agents get MCP tools?* → **worker only.** Preserves the one-agent-has-tools invariant
    §9 depends on. A survey agent is future work (§14a), and explicitly not the lead agent.
  - *How does extra information reach a tool?* → **both** deterministic `tool_args` injection and
    a free-text `instructions` hint (§5b). Injection alone leaves the worker not knowing when to
    use the server; hints alone leave the operator's caps optional.
  - *How are non-URL sources cited and verified?* → **`Source` gains provenance**
    (`kind`/`identifier`/`server`), and verification becomes polymorphic in `kind` (§7, §10).
  - *What happens when a configured server won't start?* → **fail fast**, with a per-server
    `optional: true` opt-out (§5b). A silently web-only bibliographic report is a wrong answer
    that reads like a right one.
- **Output format** → resolved: **strictly Markdown** (§7a). No JSON deliverable for consumers;
  `ResearchState` JSON remains internal/observability-only.

## 13. Configuration (`config.yaml`)

Configurable, deployment-level options live in a YAML file rather than code, so operators can
retune the harness without touching the implementation:

This block is the schema as implemented — `config.py` rejects unknown keys (`extra="forbid"`),
so a key that isn't here is a startup error rather than a silently ignored line. Keep this
example and `deep_research/default_config.yaml` in step. **The one exception is `mcp_servers`,
shown below at its full specified schema (§5b) but implemented today only as
`{name, transport, command}` and never read by the harness** — that gap is v4 (§14).

```yaml
model:
  lead: "anthropic:claude-fable-5"       # scoping, gap-check, synthesis
  researcher: "anthropic:claude-fable-5" # per-sub-question worker
  critic: "anthropic:claude-fable-5"     # a smaller/cheaper model is fine here

search:
  backend: "duckduckgo"      # forward-compatibility only; not yet wired up (§12)

# Additional tool sources for the WORKER agent, beyond WebSearch/WebFetch — see §5b.
# Empty by default: a fresh install researches the web only.
#
# Three transports (§5b "Transport modes"). Keys common to all: name, transport, optional,
# timeout_seconds, allowed_tools, health_check, tool_args, instructions. The rest are
# transport-specific, and a key belonging to a different transport is a startup error rather
# than a line that quietly does nothing.
mcp_servers:
  # (1) IN-MEMORY — a server bundled with this package, imported and run in-process.
  #     Requires the matching extra:  pipx install "deep-research-harness[zotero]"
  - name: "zotero"                 # names a discovered bundled server; also used in citations,
    transport: "in_memory"         #   metrics, and --mcp-args. No command/url/cwd: nothing to
    optional: false                #   launch and nothing to reach. Credentials come from the
    timeout_seconds: 60            #   project's .env, as they already do for everything else.
    health_check: "get_library_info"   # one cheap read the preflight calls to prove it works (§5b)
    allowed_tools:                 # omit to expose every tool the server offers
      - search_items
      - get_item
      - get_item_fulltext
      - get_item_notes
      - list_collections
    tool_args:                     # merged into every matching call; wins over model-supplied keys
      search_items:
        limit: 10
      get_item_fulltext:
        max_chars: 20000           # see §9 — the server's own default is 100,000 (~25-30k tokens)
    instructions: |                # appended to the worker prompt: when to reach for this server
      A curated Zotero library of peer-reviewed literature is available. Prefer it over web
      search for published academic work, and cite items by their Zotero key. Fall back to the
      web for recent, non-academic, or fast-moving material the library will not hold.

  # (2) LOCAL PROCESS — a stdio subprocess the harness launches. Nothing is bundled; the whole
  #     server description lives here. Use this for a server you have checked out locally, or
  #     whose version you want to control independently of the harness install (§4a).
  # - name: "zotero"
  #   transport: "stdio"
  #   command: ["uv", "run", "python", "-m", "zotero_mcp"]
  #   cwd: "../MCP/ZoteroMCP"      # relative paths resolve against the PROJECT folder (§4a)
  #   env:                         # the subprocess's entire environment; ${VAR} references only
  #     ZOTERO_API_KEY: "${ZOTERO_API_KEY}"
  #     ZOTERO_LIBRARY_ID: "${ZOTERO_LIBRARY_ID}"
  #   health_check: "get_library_info"

  # (3) EXTERNAL — a streamable-HTTP server on another host. Note that the research question and
  #     any corpus content leave this machine (§5b): a privacy decision, not just a config one.
  # - name: "internal-docs"
  #   transport: "http"
  #   url: "https://mcp.internal.example/mcp"
  #   auth_token_env: "INTERNAL_DOCS_TOKEN"   # the env var NAME, never the token itself

budgets:
  breadth_budget: 8
  depth_budget: 2
  worker_concurrency: 4
  spend_limit_usd: 5.0            # ceiling on RESEARCH for the whole run, not per agent call
  total_tokens_limit: 1500000     # the same ceiling for models with no pricing data
  report_token_allowance: 250000  # guaranteed to the write-up, on top of research's actual cost
  report_spend_allowance_usd: 1.0 # the same guarantee in USD

output:
  report_dir: "reports"
  state_dir: ".deep_research"

logging:
  logfire: true
```

Two corrections against earlier drafts of this section, now that the code is the authority:
`model.worker` is `model.researcher`; and `search.api_key_env` / `output.format` were never
implemented — output format is fixed to Markdown (§7a), so a key to configure it would only
ever have had one legal value.

Guidelines for using this file:

- The CLI loads `config.yaml` at startup and constructs the capability stack (`WebSearch`,
  `WebFetch`, any additional MCP-backed tools, `Spend Limits`, model choices) from it — no budget
  or backend value should be hardcoded in the harness code.
- `mcp_servers` lets the deployment add tool sources (e.g. a bibliographic database or an internal
  knowledge base) to worker agents without code changes, consistent with treating MCP as the
  standard tool interface (see `implementation-guidelines.md` §2).
- A per-run override (CLI flags) should be able to shadow individual config values (e.g.
  `--breadth-budget 12` for one experimental run) without editing the file.

Notes specific to `mcp_servers`, all following rules established elsewhere in this PRD:

- **`extra="forbid"` applies here too**, per-server. A mistyped `tool_arg` or `allow_tools` is a
  startup error, for the same reason as everywhere else in this file: an ignored key looks like a
  configured one.
- **`cwd` and any relative path resolve against the project folder** (the CWD `deep-research` was
  invoked from), never the install location — the §4a rule, which matters more here than anywhere
  else, because `pipx`-installed tooling launching a sibling repository's server is exactly the
  case where "relative to the package" would silently point at the wrong filesystem. This class of
  breakage is a large part of why in-memory is the recommended mode for a bundled server: there is
  no path to resolve.
- **Transport-specific keys are validated per transport**, not merely accepted. `command`/`cwd`/
  `env` are stdio-only; `url`/`auth_token_env` are http-only; `in_memory` takes none of them and
  requires that a bundled server of that `name` is installed. A stdio `command` left behind on a
  server switched to `in_memory` is a startup error — the migration people will actually perform is
  exactly the one where a stale key would otherwise sit there looking meaningful.
- **Secrets are referenced, never stored.** `env` takes `${VAR}` references and `auth_token_env`
  takes a variable *name*; `config.yaml` is a hand-edited, checked-in-adjacent file, and a literal
  key in it is a key in someone's git history. The CLI already loads a project `.env`, which is
  where the actual values belong.
- **`--mcp-args '<json>'`** overlays `tool_args` for one run (§5b). Naming an unconfigured server
  or tool is an error, not a no-op.
- **`allowed_tools`** is an allowlist, applied after the server's tool list is fetched at startup.
  A name in it that the server doesn't offer is a startup error — the likeliest cause is a server
  version skew, and discovering it at startup beats discovering it as an unexplained absence of
  library citations three minutes into a run.

## 14. Milestones (suggested)

1. **v0**: CLI chat skeleton + single-phase research — wrap bare `Researcher()` behind the
   `ResearchBrief`/`ResearchReport` data model and `config.yaml` loading, no parallel
   sub-questions, no critic. Validates the chat-driven scoping flow and Markdown report rendering
   end-to-end. *(done — developed in-tree, run via `uv run python -m deep_research.cli`.)*
1a. **v0 packaging**: repackaged as a standalone, `pipx`-installable tool with its own
   `pyproject.toml`, decoupled from the parent monorepo's environment; added `deep-research
   init`/`run` subcommands and project-folder-relative config/report-path resolution (§4a).
   *(done.)*
2. **v1**: Add phases 2–4 (plan → parallel workers → gap check), `ResearchState` persistence for
   resumability, budgets/defaults from §8 sourced from config. *(done — phase 2 folded into
   scoping's `BriefDraft.subquestions` rather than a separate call; parallel workers bounded by
   `worker_concurrency`; gap-check follow-up rounds bounded by `depth_budget`/`breadth_budget`,
   enforced in code per §8; `ResearchState` checkpointed after every round/phase to
   `output.state_dir`, resumable via `deep-research resume <state-file>`.)*
3. **v2**: Add the critic agent (phase 6) and the feedback loop back into research. *(done — added
   a deterministic, non-LLM per-finding check (`verify.py`) that drops any cited source a worker
   didn't actually fetch that run (§8/§10 item 1), plus an independent critic agent (§10 item 2)
   that reviews the draft report against the brief and findings; failed critiques feed back into
   another research round via the same `depth_budget`/`breadth_budget` accounting as gap-check,
   per §10's "bounded by depth_budget so critique can't loop forever" — verified budget-bounded
   termination even when the critic never passes. When the budget runs out first, the report
   ships with the critic's remaining issues folded into `unresolved` rather than silently dropped.)*
4. **v3**: Observability polish (§11) and, if warranted, the autonomy-loop wrapper from
   `implementation-guidelines.md` (§ "execution loop vs. autonomy loop") for scheduled/triggered
   research runs with a no-human scoping fallback. *(done — observability: every worker call,
   research round, gap-check, synthesis, and critique attempt now opens a `logfire` span (via
   `tracing.py`, a no-op when `logging.logfire` is off) grouped under one top-level per-run trace
   (opened in `cli.py`, covering scoping through the final report) rather than a flat list of
   agent calls; a local `output.state_dir/metrics.jsonl` ledger (`RunMetrics`) records per-run
   sub-question counts, rounds to convergence, first-try critic pass/fail, and budget-exhaustion,
   surfaced via a new `deep-research stats` command — a lightweight, always-available substitute
   for the "solve rate / step ratio" framing in §11 that doesn't depend on logfire access.
   No-human scoping fallback: added `--auto` to `deep-research run`, per §4a/§4's "Secondary
   (future)" entry deliberately deferred here — clarifying questions get a canned "no human
   available" nudge with a hard-capped retry count (a real unattended run must terminate on its
   own), and a proposed brief is auto-confirmed. Per this section's own "don't start by building a
   loop" guidance and the loop-vs-harness distinction (§ "execution loop vs. autonomy loop"), this
   deliberately stops at "can run unattended" — no scheduler/cron/webhook wrapper was built; wiring
   `--auto` to an external trigger is left as the outer autonomy loop, layered on top, for whoever
   operates this once a manual `--auto` run has proven the task works unattended.)*

5. **v4 — MCP tool sources (§5b).** *(v4.1–v4.6 done.)* The `mcp_servers` config key had existed since the first draft and been inert
   throughout; this milestone wires it up, with `zotero-mcp` as the first server. Order as planned:

   | Step | Contents | Exit criterion |
   | --- | --- | --- |
   | **v4.1 — Source provenance** *(done)* | `Source` gains `kind`/`identifier`/`server`; `to_markdown` renders both kinds (§7a); verification generalized to match on `kind` (§10); a before-validator maps the old `url` field so pre-MCP checkpoints still load | Existing web-only runs behave identically; old checkpoints still resume; a fabricated MCP identifier is dropped |
   | **v4.2 — Server lifecycle** *(done)* | Config schema (§13) with per-transport validation, **stdio and HTTP transports**, startup probe in the existing preflight (session opened *and* `health_check` called), fail-fast with `optional` opt-out, one shared session per run, `allowed_tools` filtering | `deep-research run` with a misconfigured server exits non-zero naming it and its transport; a server whose credential is rejected fails the preflight rather than every later call; with a working one, the worker's tool list includes exactly the allowlisted tools |
   | **v4.3 — Extra tool info** *(done)* | `tool_args` injection at the call boundary, `instructions` appended to the worker prompt, `--mcp-args` per-run overlay, injected args recorded in the trajectory and surfaced in `assumptions` | An injected `max_chars` bounds a full-text return even when the model omits it; an unconfigured `--mcp-args` key errors at startup |
   | **v4.4 — Zotero over stdio, end to end** *(done)* | Worker instructions generalized beyond "fetch the URL first"; per-server metrics in the §11 ledger and `deep-research stats` | A run against a real library produces a report citing verified Zotero items alongside web sources, and `stats` shows per-server call/citation counts |
   | **v4.5 — Packaging `zotero-mcp`** *(done)* | Gave `MCP/ZoteroMCP/` its own `pyproject.toml`, lock file, `.venv`, console script (`zotero-mcp`), and bounded dependency ranges — the same treatment this harness got in v0 packaging (§14.1a); `pyzotero` dropped from the root project and its tests removed from root collection | `pipx install` / `uv pip install` of the Zotero server alone succeeds outside the monorepo; its 80-test suite runs against the installed package from its own environment |
   | **v4.6 — In-memory transport** *(done)* | `deep_research.mcp_servers` entry-point discovery (declared by `zotero-mcp`, resolving to its zero-argument `build_server`); `transport: in_memory` built on the harness's own event loop; `[zotero]` extra declared on this package (§4a) | A measured run over `in_memory`: 28 tool calls, 19 argument-injected, 0 failed, 8 citations kept, 0 dropped. Install is `pipx inject` by path until `zotero-mcp` is published, at which point `pipx install "deep-research-harness[zotero]"` is the documented form |

   One thing the extra turned out not to be able to do: `zotero-mcp` is not published to PyPI, so
   `pipx install "deep-research-harness[zotero]"` cannot resolve today. The extra is declared
   anyway, and the install documented as `pipx inject deep-research-harness <path>` until
   publication — which also surfaces a **pre-publication check nobody has done**: `zotero-mcp` is a
   plausible name for someone else's package, and the distribution name may need changing.

   Sequencing note: **stdio comes first and in-memory last**, even though in-memory is the
   recommended end state (§5b). Two reasons. It is the only ordering that lets the interesting work
   — provenance, verification, injection, prompt changes — be validated against a real library
   *before* taking on a packaging dependency, since stdio needs nothing of `zotero-mcp` but a
   working checkout. And v4.6 is genuinely blocked on v4.5: a server with no `pyproject.toml`
   cannot be an extra of anything. Shipping in-memory first would mean doing the packaging work
   under time pressure to unblock the features, which is how the coupling in §4a's trade-off note
   becomes a problem rather than a choice.

   Three things to watch, all argued above rather than discovered late, and all now handled: the
   worker instructions stated a URL-shaped citation rule ("a URL may only appear in `sources` if you
   fetched that exact URL this run"), rewritten in v4.4 as a two-kind rule so the worker doesn't
   believe library citations are forbidden; `get_item_fulltext` is the largest payload in the
   system, so v4.3's injection is what makes v4.4 affordable (§9); and the preflight calls a tool
   rather than only listing tools, because `zotero-mcp` accepts a rejected API key at startup by
   design (§5b).

#### 14b. What the build changed about the design above

Four corrections, each forced by something the implementation or a real run showed. Recorded here
because three of them contradict a claim stated confidently earlier in this document.

- **The session brackets the pipeline, not "phase 1 through 7."** §5b says sessions open before
  phase 1 and close after phase 7. They cannot: scoping and the research pipeline run under two
  separate `asyncio.run` calls, and an MCP session cannot outlive the event loop it was opened on.
  Since MCP servers attach to the worker agent only, and workers run only in the pipeline, the
  session brackets the pipeline — which covers every phase that can actually use a tool. The
  preflight probe therefore opens its own short-lived session first and closes it; the cost is one
  extra server startup, and what it buys is fail-fast *before* any model call.
- **Tool errors are handed back to the model, not surfaced as failures.** The first implementation
  set `tool_error_behavior='error'`, reasoning that a failing server should surface immediately and
  be contained by §8's "retry once, then mark the sub-question unresolved". A run against the real
  Zotero library refuted this in the most direct way available: the worker called `search_items`
  with an empty query, the server replied with exactly the message it was designed to give an agent
  ("`query` is empty. Pass search terms, or use `list_recent_items`"), and that message was thrown
  away — killing the sub-question on both attempts and producing **a report with no findings at
  all.** `_research_one` contains a failure by *abandoning* the sub-question, which is far too blunt
  for a recoverable mistake. The Zotero PRD's actionable error text (§7.7) exists precisely so the
  model can act on it, so the default `'retry'` is correct and a genuinely dead server is still
  bounded by the same retry count.
- **Listing tools is not an optimization you can skip.** The run's own session skipped the whole
  probe, having been preflighted. But `tool_to_server` — the map that attributes a citation to the
  server that returned it — is built from the tool list, so an unprobed session had an empty map,
  `verify.py` recognized no MCP tool, and **every library citation was dropped as unretrieved**:
  the exact failure §7 introduced provenance to prevent, reintroduced one layer down, and silent
  (a run that cites nothing looks like a library with nothing in it). Only the health-check *call*
  is skippable; the tool list is fetched on every open.
- **A first real run, for the record.** 2 sub-questions against a live Zotero library over stdio:
  75 tool calls, 63 of them argument-injected, 4 recoverable failures the model self-corrected
  from, 10 citations kept and **0 dropped**, 338k tokens. The report correctly concluded the library
  holds no direct work on the topic and named the adjacent items instead — which is the answer a
  curated corpus should give, and one the web could not have given.

### 14a. Specified but not built

Items that are specified, or deliberately deferred, and **not scheduled** — recorded here rather
than left implied by their absence. The decision on each is open. (MCP support was in this list
until it was specified in §5b and scheduled as v4 above.)

- **`Memory` for cross-run knowledge** (§5, **S**). The use case is the §4 "Secondary (future)"
  one: recurring research on the same subject, where a run should know what a previous run
  found. It has no data to work with until scheduled runs exist, and `--auto` deliberately
  stopped short of a scheduler. It also carries a hazard specific to research: injecting a
  previous run's findings as current context is how a report becomes confidently out of date,
  and recency is much of the point of the tool. Whoever builds it needs a staleness design —
  findings timestamped and cited as "a run on date Y found X", not merged into the present.
- **Query/fetch deduplication across workers** (§5, **L**). Workers research different
  sub-questions in fresh contexts, so overlap is not guaranteed, but it does happen: in one
  measured run, two of five citations were the same URL fetched by two different workers, and a
  topic with a single authoritative source (a docs page, a PEP, a changelog) should overlap more
  as `breadth_budget` grows. Each duplicate fetch costs real tokens against a ceiling that has
  already been observed to overrun. The next step is measurement, not construction: record
  distinct-versus-total fetched URLs per run in the §11 metrics ledger, and build a shared cache
  only if the waste turns out to be material. A cache also trades away some source diversity
  between workers, which is worth knowing before paying for it.
- **A survey agent, between the lead and the workers** (§5b). Once a corpus is reachable, the
  best sub-questions depend on what is *in* it — "the library holds 40 items on this topic across
  three collections, and nothing at all on that one" changes the decomposition, and today's lead
  agent decomposes blind. The design constraint is where that context lands: it must be a
  **separate agent below the lead**, not a capability added to the lead, because the lead holds the
  only conversation in the system (§9) and a corpus survey is exactly the kind of bulk tool output
  that would saturate it. A survey agent runs once, tool-enabled, stateless like every other phase
  call, and returns a *compact* structured orientation (collection names, counts, coverage gaps)
  that the lead reads as rendered prompt text — the same pattern by which findings reach synthesis
  without synthesis ever holding a tool. This is the only reason the worker-only rule in §5b was
  framed as "for v1" rather than permanent. It should be built after v4 has proven a real library
  improves reports, not before: it optimizes a decomposition step whose payoff is unmeasured.
- **Properly formatted bibliographies for MCP sources** (§7a). Library citations currently render
  as title + native identifier, which is honest but not publication-ready. The Zotero server's
  `format_bibliography` / `format_citation` (Zotero PRD §5.4) render real CSL styles server-side,
  which would let a report carry a correctly styled reference list instead of a list of item keys.
  Blocked on that server's M3, and on a decision this PRD has so far avoided: a bibliography is a
  second user-facing rendering, and §7a's "strictly Markdown, one rendering" is a constraint worth
  re-opening deliberately rather than by accident.
