# Deep Research Harness

CLI chat implementation of the deep-research harness described in `PRD.md`, built on top of
Pydantic AI's `Researcher()` capability.

v3 scope (PRD §14): chat-driven scoping, then parallel per-sub-question research, gap-check
follow-up rounds, synthesis, and an independent critic that reviews the draft report and can send
research back for another round — all bounded by config budgets, all resumable via persisted
`ResearchState` checkpoints, all traced under a single logfire trace per run, with a local
`deep-research stats` command for tracking efficiency over time, and an `--auto` flag for
unattended (no-human) invocations. See the PRD for the full plan.

This is a **standalone package**, published to PyPI as
[`deep-research-harness`](https://pypi.org/project/deep-research-harness/), so it can be
installed once, globally, via `pipx`, and then run from any directory — a researcher's own
project folder — without needing this source tree.

## Installation (pipx)

### Prerequisites

- Python 3.12+.
- [`pipx`](https://pipx.pypa.io/) itself. If you don't have it yet:

  ```bash
  # via pip
  python3 -m pip install --user pipx
  python3 -m pipx ensurepath

  # or via a system package manager
  brew install pipx        # macOS
  sudo apt install pipx    # Debian/Ubuntu
  ```

  `pipx ensurepath` adds pipx's bin directory to your shell `PATH` — open a new terminal (or
  `source` your shell rc file) afterwards if `deep-research` isn't found once installed.

### Install from PyPI

```bash
pipx install deep-research-harness
# with the bundled Zotero MCP tool source (see "MCP tool sources" below):
pipx install "deep-research-harness[zotero]"
```

Or, from a checkout of this repository:

```bash
pipx install /path/to/deep-research-harness
```

Either way this installs a single `deep-research` command, isolated in its own virtual
environment. Verify it landed:

```bash
deep-research --help
pipx list             # shows "deep-research-harness" among installed packages
```

### Editable install (for developing the tool itself)

```bash
pipx install --editable /path/to/deep-research-harness
```

Changes to the source under `deep_research/` take effect immediately — no reinstall needed. Only
changes to `pyproject.toml` (e.g. a new dependency) require re-running the command above.

### Install from a built wheel (e.g. to hand off to someone without this source tree)

```bash
cd /path/to/deep-research-harness
uv build                              # produces dist/deep_research_harness-<version>-py3-none-any.whl
pipx install dist/deep_research_harness-*.whl
```

The wheel is self-contained (it bundles `default_config.yaml` as package data — see PRD §13) and
can be copied anywhere; the recipient only needs `pipx` and Python, not this repo.

### Upgrading / uninstalling

```bash
pipx upgrade deep-research-harness       # from PyPI
pipx install --force /path/to/deep-research-harness   # from source, after pulling changes
pipx uninstall deep-research-harness
```

### After installing

Set the API key for whichever provider your config points at:

```bash
export ANTHROPIC_API_KEY=...      # if your models are anthropic:*
export OPENAI_API_KEY=...         # if your models are openai:*
export OPENROUTER_API_KEY=...     # if your models are openrouter:*
```

A `.env` file works too, and is usually easier: put it in the same project folder as your
`config.yaml`. It is discovered from the working directory upward, so running
`deep-research` from a project folder — or any subdirectory of it — picks it up. An
already-exported variable always wins over the file, so a one-off override on the command
line still works.

Only variables the harness or Pydantic AI actually reads have any effect. Provider keys
(`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) and the
`DEEP_RESEARCH_MODEL*` variables below are read; anything else (a `MODEL=` line, say) is
ignored — here the model comes from `config.yaml` or `DEEP_RESEARCH_MODEL*`.

The harness isn't tied to a provider: `model.lead`/`researcher`/`critic` are Pydantic AI model
strings (`"<provider>:<model-name>"`) and the three roles don't even have to share a provider.
The bundled default names Anthropic models; to run against something else without editing any
file, set `DEEP_RESEARCH_MODEL` (all three roles) or `DEEP_RESEARCH_MODEL_LEAD` /
`DEEP_RESEARCH_MODEL_RESEARCHER` / `DEEP_RESEARCH_MODEL_CRITIC` (per role, and they win over the
shared one):

```bash
DEEP_RESEARCH_MODEL=openai:gpt-5-mini deep-research run "your question"
```

Every configured model is resolved before any work starts, so a missing or wrong-provider key is
reported immediately — with the name of the variable to set — rather than surfacing partway
through scoping once a call has already been billed.

One provider difference worth knowing, since it's invisible in config: web *search* is native on
both Anthropic and OpenAI, but web *fetch* is native only on Anthropic — on OpenAI the harness
uses Pydantic AI's local fetch tool instead (hence the `pydantic-ai-slim[duckduckgo,web-fetch]`
dependency). Source verification handles both shapes identically, so citations are checked the
same way either way.

## Use it from your own project folder

The tool has no notion of "the repo" — it operates entirely on your current working directory,
which we call the **project folder**. There's nothing special about it; any empty directory works.

```bash
mkdir ~/research/django-history && cd ~/research/django-history
deep-research init      # scaffolds ./config.yaml and ./reports/
deep-research run "What changed in the last three major releases of Django?"
```

`deep-research init` copies the tool's bundled default config into `./config.yaml` so you can
customize it (models, budgets, output location — see PRD §13) without touching the installed
package. If you skip `init` and just run, the tool falls back to its built-in defaults.

`deep-research run` will ask clarifying questions in the chat if your question is ambiguous —
answer inline. Once it proposes a brief, confirm it (`Enter`/`y`), reject it (`n`), or type a
free-form revision. From there the pipeline runs with no further input needed:

1. Each sub-question from the brief is researched **in parallel** (bounded by
   `budgets.worker_concurrency`), each producing its own source-attributed finding. Every cited
   source is checked against what the worker actually fetched that run — a source the model
   only claims to have read, but never really did, is dropped and noted rather than trusted.
2. A gap-check pass reviews the findings and proposes follow-up sub-questions for anything
   thin, contradictory, or uncovered — bounded by `budgets.depth_budget` (max follow-up rounds)
   and `budgets.breadth_budget` (max total sub-questions across every round). These are enforced
   in code, not left to model judgment. Any sub-question a budget cuts off — from the original
   brief, from a gap-check, or from the critic — is recorded and listed under "Unresolved" in the
   final report, so a truncated run reads as truncated rather than as complete coverage.
3. Everything gathered is synthesized into a draft report.
4. An independent critic — deliberately a separate agent/call, not the synthesis agent grading
   its own work — reviews the draft against the brief and findings: is every claim source-backed,
   does the report cover every sub-question, were contradictions surfaced rather than papered
   over? If it finds a fixable gap, that becomes another research round (same budget as gap-check
   above); if the budget runs out first, the report still ships, with the critic's remaining
   issues listed under "Unresolved" rather than silently dropped.

The report prints as rendered Markdown and is saved under `./reports/<timestamp>-<slug>.md` **in
your project folder** — like `output.state_dir` below, `output.report_dir` in `config.yaml`
(default `"reports"`) is always resolved relative to the directory you ran `deep-research` from,
never relative to wherever the tool is installed.

Per-run overrides (no need to edit the file): `--breadth-budget`, `--depth-budget`,
`--spend-limit-usd`, `--config <path>`.

#### Budgets: where they live and how to size them

All of them live under `budgets:` in **your project folder's `config.yaml`** (run
`deep-research init` to get a copy you can edit). Every one is a ceiling on the **whole run**,
not on each agent call: all agents accumulate into one counter, and research stops when any
ceiling is reached.

| Setting | Default | Bounds | Raise it when |
|---|---|---|---|
| `breadth_budget` | 8 | Total sub-questions across all rounds | The report has too many "unresolved" entries |
| `depth_budget` | 2 | Follow-up rounds after the initial pass | Gaps keep going unexamined |
| `worker_concurrency` | 4 | Sub-questions researched in parallel | Runs are slow and your rate limits allow more |
| `request_limit` | 300 | **Model requests for research** | A run stops with "would exceed the request_limit" |
| `total_tokens_limit` | 1,500,000 | Research tokens | Research stops early on an unpriced model |
| `spend_limit_usd` | 5.0 | Research spend, for priced models | Research stops early and you want to spend more |
| `report_token_allowance` | 250,000 | Tokens for synthesis + critique | The report can't be written |
| `report_spend_allowance_usd` | 1.0 | USD for synthesis + critique | Same, on a priced model |
| `report_request_allowance` | 30 | Requests for synthesis + critique | Same, if it names the request limit |

**`request_limit` is usually the one that bites**, because one sub-question is not one request:
the worker takes several turns and delegates to a researcher that takes several more. Size it as

```
requests ≈ 1 (scoping) + breadth_budget × ~8 + (depth_budget + 1) (gap checks)
```

so the default breadth of 8 needs about 70 with no headroom at all. Left unset, Pydantic AI
applies its own default of **50**, which is below what a single round of workers needs — that is
what stopped a real run mid-research with tokens and USD barely touched, so it is now configured
explicitly rather than inherited.

Writing the report is funded separately, by `report_token_allowance`,
`report_spend_allowance_usd`, and `report_request_allowance`. Those are granted **on top of
whatever research actually used**, so a run can never spend its research budget and then be
unable to afford the report — the one failure mode where you pay in full and get nothing back.
If a run says the allowance wasn't enough, it names the exact setting to raise; raise it and
resume, since findings are checkpointed and are not re-researched.

Each run prints what it actually spent when it finishes, and records it in `metrics.jsonl` for
`deep-research stats`.

#### When the run warns that costs can't be enforced

```
CostNotFoundWarning: A `cost_limit` is set but cannot be enforced because no cost was
calculated for this run.
```

This means `spend_limit_usd` and `report_spend_allowance_usd` are inert for that model, and only
the token and request ceilings are holding. Pricing is not part of this tool: Pydantic AI
computes cost from the [`genai-prices`](https://github.com/pydantic/genai-prices) package, whose
data is bundled with the installed version. So there is nothing to edit in `config.yaml` — check
and fix it there instead:

```bash
# Is your model priced? Split "<provider>:<model>" — the model ref must not repeat the provider.
# Prints the price of 1M in + 1M out tokens; LookupError means no data, so USD limits are inert.
python - <<'PY'
from genai_prices import calc_price, Usage
provider, model = "openai", "gpt-4o-mini"        # e.g. "openrouter", "openai/gpt-5.6-luna"
print(calc_price(Usage(input_tokens=1_000_000, output_tokens=1_000_000),
                 model, provider_id=provider).total_price)
PY

# Pick up newer pricing data in the installed tool
pipx runpip deep-research-harness install --upgrade genai-prices
```

Common causes: a model too new to be in the bundled snapshot, or a provider-prefixed name the
snapshot doesn't carry (OpenRouter-hosted names are frequently missing even when the underlying
model is priced under its native provider).

#### Supplying your own prices

Rather than wait for the pricing data to catch up, put the model's published rates in the
`prices:` section of `config.yaml`. Keys are the model strings from `model:`, matched exactly:

```yaml
model:
  lead: "openrouter:openai/gpt-5.6-luna"
  researcher: "openrouter:openai/gpt-5.6-luna"
  critic: "openrouter:openai/gpt-5.6-luna"

prices:
  "openrouter:openai/gpt-5.6-luna":
    input_usd_per_1m: 1.25     # copy both numbers straight off the provider's pricing page
    output_usd_per_1m: 10.0
```

With that in place `spend_limit_usd` and `report_spend_allowance_usd` are enforced again, and the
run reports `Spend: $0.4213 (estimated from config.yaml prices)` so an estimate is never mistaken
for a measured cost. Three rules keep it predictable:

- **Measured pricing always wins.** An entry for a model `genai-prices` already knows is ignored,
  so a stale hand-entered rate can't override what the provider actually charged.
- **Only what you list is billed.** A model with no entry contributes nothing to the estimate, so
  a partial `prices:` block under-reports rather than guessing.
- **Cost is attributed per call site**, using the role that made the call — a worker's delegated
  researcher turns bill at the `researcher` rate.

Accuracy is bounded by the rates you enter: cached-input discounts, batch pricing, and per-request
surcharges are not modelled, so treat the figure as a close estimate rather than an invoice. If
you would rather not maintain rates, `total_tokens_limit` and `request_limit` remain enforceable
without any pricing data at all.

Both research ceilings are soft rather than hard: `worker_concurrency` requests are in flight at
once and one worker turn that fetches a large page can add ~100k tokens by itself, so a run can
pass the ceiling by roughly the concurrency factor before anything notices. Overshoot is
contained — workers stop, the gap check falls through, and the pipeline writes up what it has.

Unknown keys in `config.yaml` are rejected at startup rather than ignored, so a mistyped budget
fails loudly instead of quietly leaving the default in place.

Note: `mcp_servers` and `search.backend` are present in the config schema for forward
compatibility but not yet wired up — see PRD §12.

### Researching beyond the web: MCP tool sources

By default a run searches and fetches the open web. You can give the **worker agent** additional
tool sources — a bibliography, an internal knowledge base, a specialized search API — by
configuring MCP servers (PRD §5b). The first supported one is
[`pydantic-zotero-mcp`](https://pypi.org/project/pydantic-zotero-mcp/), read-only access to a
Zotero library.

Servers attach to the worker agent and nowhere else, so the worker chooses *per sub-question*
whether the library, the web, or both are the right source.

#### Three transports

| `transport` | The server runs | Configure |
| --- | --- | --- |
| `in_memory` | inside this process, as a bundled library | just `name` — install the extra, credentials from `.env` |
| `stdio` | as a subprocess this tool launches | `command`, `cwd`, `env` |
| `http` | on another host | `url`, `auth_token_env` — **your question and retrieved content leave this machine** |

#### In-memory (recommended for a bundled server)

`in_memory` imports the server and runs it inside this process — no subprocess, no socket, and no
`cwd`/`env`/interpreter path to get wrong, which is the failure mode `stdio` invites in a
`pipx`-installed tool. Install the server into the harness's environment, then name it:

```bash
pipx install "deep-research-harness[zotero]"
# or into an existing install:
pipx inject deep-research-harness pydantic-zotero-mcp
```

```yaml
mcp_servers:
  - name: "zotero"                  # resolved through the `deep_research.mcp_servers` entry point
    transport: "in_memory"
    health_check: "get_library_info"
    allowed_tools: [search_items, get_item, get_item_fulltext, list_collections]
    tool_args:
      search_items: {limit: 10}
      get_item_fulltext: {max_chars: 20000}
    instructions: |
      A curated Zotero library of peer-reviewed literature is available. Prefer it over web
      search for published academic work.
```

Credentials go in your project's `.env` alongside your model API keys (`ZOTERO_API_KEY`,
`ZOTERO_LIBRARY_ID`) — the server reads the working directory the harness was launched from, so
two project folders sharing one install can point at two different libraries.

Note that in-memory removes the process boundary entirely: the server runs with this tool's
privileges and an unhandled crash in it takes the run with it. That is a trust decision about the
server you install, not something the harness can enforce.

#### Local subprocess (`stdio`)

Use this for a server you have checked out locally, or one whose version you want to upgrade
independently of this tool's install:

```yaml
mcp_servers:
  - name: "zotero"
    transport: "stdio"
    command: ["uv", "run", "python", "-m", "zotero_mcp"]
    cwd: "../pydantic-zotero-mcp"    # relative to YOUR project folder, not the install location
    env:
      ZOTERO_API_KEY: "${ZOTERO_API_KEY}"   # a reference; never paste a secret into config.yaml
      ZOTERO_LIBRARY_ID: "${ZOTERO_LIBRARY_ID}"
    health_check: "get_library_info"  # one cheap read, called at startup to prove it works
    allowed_tools: [search_items, get_item, get_item_fulltext, list_collections]
    tool_args:
      search_items: {limit: 10}
      get_item_fulltext: {max_chars: 20000}
    instructions: |
      A curated Zotero library of peer-reviewed literature is available. Prefer it over web
      search for published academic work. Fall back to the web for recent or non-academic material.
```

#### Passing extra information to tools

Two mechanisms, because they answer different questions:

- **`tool_args`** — arguments the harness merges into every matching call *before dispatch*, and
  which **win over whatever the model passed**. This is how you scope a run to one collection or
  cap a payload. Set `get_item_fulltext.max_chars`: the server's own default is 100,000 characters,
  roughly 25–30k tokens for a **single call**, so a few full-text reads is otherwise your whole
  token budget.
- **`instructions`** — free text appended to the worker's prompt, telling it when to reach for this
  source at all. Without it a worker holds a tool it doesn't know when to use.

Override `tool_args` for one run without editing the file:

```bash
deep-research run "..." --mcp-args '{"zotero": {"search_items": {"collection_key": "XY99ABCD"}}}'
```

Naming a server or tool that isn't configured is a startup error, not a silent no-op — an override
that quietly did nothing would leave the run looking scoped when it isn't.

#### What happens when a server is broken

Every configured server is **probed before the run spends anything**: the session is opened, its
tool list checked against every name your config references, and `health_check` actually called.
A failure exits non-zero, naming the server and its transport. This is deliberate — a
bibliographic report that silently fell back to web-only looks complete and isn't. Set
`optional: true` on a server to downgrade its failure to a warning; the degradation is then
recorded in the report's assumptions.

A tool call that fails *during* a run is handed back to the model to self-correct (servers like
`zotero-mcp` write their errors for exactly that), and a worker that still can't proceed is
contained the way a failed fetch already is.

#### Citations from a tool source

A library item has no URL, so it is cited by its own identifier and verified differently: the
harness checks that the identifier appears in what that server actually **returned** this run.
A well-formed key the server never sent back is dropped from the finding, same as an unfetched
URL. In the report, web sources render as links and tool-source items render as attributions:

```markdown
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — "..."
- Detecting hallucinations using semantic entropy — Zotero item `IB3P2GSL` — "..."
```

`deep-research stats` reports per-server calls, failures, injected calls, and citations kept vs.
dropped — a server called often and cited never is wrongly scoped or wrongly described.

### Resuming an interrupted run

Progress is checkpointed to `output.state_dir` (default `.deep_research/`, in your project
folder) after every research round, gap-check, synthesis, and critique — a crash, a `Ctrl-C`, or a
budget error doesn't lose completed work. If a run is interrupted, it prints the checkpoint path;
pick back up with:

```bash
deep-research resume .deep_research/20260101T000000Z-your-question-slug.json
```

Resume only redoes what wasn't finished — already-completed sub-questions aren't re-researched.
State files are kept even after a run finishes, as a durable trace of what was searched, found,
and decided — not just the final report.

If the original run stopped because it hit `breadth_budget`/`depth_budget`, resuming as-is has
nothing to do — the budget is recorded in the brief, so it's still exhausted. Raise it on the
resume instead, and the sub-questions that were skipped get requeued:

```bash
deep-research resume .deep_research/<file>.json --breadth-budget 16
```

Resuming a run that is already complete with no outstanding work does nothing and costs nothing
— it tells you what was skipped rather than paying for a second synthesis and critique of the
same findings. Pass `--force` if you do want those re-run.

### Unattended runs (`--auto`)

The default flow is a chat — the lead agent asks clarifying questions and you confirm the brief.
For scripted or scheduled invocations where no one's there to answer, pass `--auto` (the question
must then come in as a CLI argument, since there's no one to prompt for it):

```bash
deep-research run --auto "What changed in the last three major releases of Django?"
```

In `--auto` mode, clarifying questions get a canned "no human available, use your best judgment"
nudge instead of a real answer, and a proposed brief is accepted immediately rather than asking
for confirmation. If the lead agent still won't stop asking (more than a few rounds), the tool
gives up on it and builds a minimal brief directly from your raw question instead — an unattended
run must terminate on its own, it can't ever block waiting for input that will never come.

This flag intentionally stops at "can run without a human" — it does not add a scheduler, cron
job, or webhook listener of its own. Wiring `deep-research run --auto` up to `cron`, a CI job, or
another trigger is an *outer autonomy loop*, layered on top of this harness rather than built
into it — and that's worth doing only once a manual, recurring `--auto` invocation has already
proven the task works unattended.

### Tracking efficiency over time (`deep-research stats`)

Every completed run appends a summary line to `output.state_dir/metrics.jsonl` (default
`.deep_research/metrics.jsonl`, in your project folder): question, sub-question counts, rounds
used, whether the critic passed on the first try, whether a budget limit was hit, and duration.

```bash
deep-research stats
```

prints an aggregate table across all recorded runs — useful for judging whether your
`depth_budget`/`breadth_budget` defaults are well-calibrated (e.g. a low critic first-try-pass
rate suggests synthesis needs tighter instructions; runs that constantly hit the budget suggest
raising it).

If `logging.logfire` is enabled in `config.yaml` (the default) and a `LOGFIRE_TOKEN` is set, each
run also produces one logfire trace covering the *entire* job — scoping, every worker call, every
gap-check, every synthesis/critique attempt — grouped by round and sub-question, not just a flat
list of agent calls.

## Developing this tool

From a checkout of this repository:

```bash
uv sync
uv run deep-research init     # this directory doubles as a scratch project folder in dev
uv run deep-research run "your question"
```

To build/verify the distributable artifact: `uv build` (produces `dist/*.whl`).

To work on the `in_memory` MCP transport, install a bundled server into this dev environment:

```bash
uv sync --extra zotero     # pulls pydantic-zotero-mcp from PyPI
```

Note that a plain `uv sync` removes it again — it is an optional extra, not a declared dependency,
and deliberately so (§4a). The test suite doesn't need it: `tests/test_mcp_session.py` runs the
whole in-memory path against a FastMCP server defined in the test file, so it passes either way.

### Tests

```bash
uv sync --group dev
uv run pytest
```

The suite runs entirely offline — no API key, no network. Agents are constructed normally and
then have their model swapped via `agent.override(model=...)`, so a test never resolves a real
model string. A test that reaches the network is a bug in the test.

One trap is documented in `tests/conftest.py` and worth repeating: **`TestModel` cannot drive
the worker agent.** The worker carries `Researcher()`, which brings provider-native tools, and
`TestModel` raises "does not support built-in tools". That failure is contained by
`_research_one`, so a test using it doesn't fail loudly — every worker just errors and you
assert against a run that silently researched nothing. Use `conftest.worker_model(...)`, which
returns a `FunctionModel` emitting a real `SubFinding`. The other four agents are fine with
`TestModel`.

Most tests here exist because something broke, and each names the failure it guards. They have
been checked by reintroducing each of those bugs and confirming the suite goes red — a test
that passes with and without the bug it names isn't protecting anything.

### Dependency policy

`pipx install` resolves from the ranges in `pyproject.toml` and **never reads `uv.lock`**. The
lockfile pins the development environment (`uv sync`) so the test suite is reproducible; the
ranges are what end users actually get.

So every dependency carries an upper bound at the next breaking version. Floors alone are an
open-ended promise about releases that don't exist yet, and it had already gone wrong: `rich>=14`
was resolving 15 and `logfire>=3.14.1` was resolving 4, each a major version beyond anything
this code had run against. Bounds are ranges rather than exact pins because an installed CLI
that demands one exact version conflicts with everything else and blocks security updates.
`pydantic-ai-harness` is 0.x, where the minor is the breaking position, hence `<0.22`.

`tests/test_dependencies.py` closes the loop: whatever the dev environment installs and passes
the suite against must satisfy the published ranges, and every dependency must have an upper
bound. Raising the lockfile past a declared bound fails there rather than shipping an untested
combination. Widening a bound is therefore a deliberate act: raise it, run the suite, commit.

## Layout

```
deep_research/
  config.py             # config.yaml discovery (project folder → bundled default) + AppConfig
  default_config.yaml    # bundled default config, packaged as data (see config.py)
  models.py             # PRD §7 data model: BriefDraft/ResearchBrief/SubFinding/GapCheckResult/
                         # CriticVerdict/ResearchState (with save/load)/ResearchReport/RunMetrics
  agents.py              # lead (scoping), worker (Researcher()+SubFinding), gap-check, synthesis,
                         # critic — five narrow agents, each with its own instructions
  verify.py              # deterministic per-finding source check (PRD §8/§10): drops any cited
                         # source a worker didn't actually retrieve that run — no LLM call. Web
                         # sources verify by fetched URL, MCP ones by what the server returned
  mcp.py                 # MCP tool sources (PRD §5b): builds one toolset per configured server
                         # (in_memory/stdio/http), injects `tool_args` at the call boundary,
                         # probes every server before the run, and maps tools → server for
                         # citation attribution and per-server metrics
  tracing.py             # logfire span() helper (PRD §11) — a no-op when logging.logfire is off
  scoping.py             # phase 1: clarify/confirm loop over the CLI chat (PRD §6.1), plus the
                         # --auto no-human fallback (PRD §14 v3)
  research.py            # phases 3–6: parallel research, gap-check rounds, synthesis, critique
                         # (PRD §6) — the critic's follow-ups share the same budget as gap-check's
  util.py                # slug()/run_timestamp() shared by cli.py and research.py
  cli.py                 # entry point: `init`/`run`/`resume`/`stats` subcommands, chat loop,
                         # Markdown save, metrics.jsonl recording
```
