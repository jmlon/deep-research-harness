"""Phases 3–6 (PRD §6): parallel research, gap-check follow-up rounds, synthesis, and critique.

Phase 2 ("Plan") is folded into scoping (`BriefDraft.subquestions`) rather than a separate call —
see agents.py's module docstring. Orchestration here is plain async Python (a deterministic
"blueprint" around the agent calls, per implementation-guidelines.md §8) — round-counting and
budget enforcement are code, not model judgment; `ResearchState` is saved after every step so a
crashed run can resume via `deep-research resume <state-file>`.

The critic (phase 6, PRD §10 item 2) reuses the *same* round/breadth budget as gap-check follow-
ups — a critic-driven follow-up round counts against `depth_budget` exactly like a gap-check one,
per PRD §10: "bounded by depth_budget so critique can't loop forever."
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage, UsageLimits
from rich.console import Console

from deep_research.agents import (
    render_critique_prompt,
    render_gap_check_prompt,
    render_synthesis_prompt,
    render_worker_prompt,
)
from deep_research.config import AppConfig, ModelPrice
from deep_research.mcp import McpServer, tool_to_server, worker_instructions
from deep_research.models import (
    CriticVerdict,
    GapCheckResult,
    ResearchReport,
    ResearchState,
    Source,
    SubFinding,
)
from deep_research.tracing import span
from deep_research.verify import fetched_urls, mcp_evidence, verify_finding

WORKER_ATTEMPTS = 2
"""One retry for a failed worker, per PRD §8 ("retry once, then mark that sub-question
unresolved rather than failing the whole run")."""


@dataclass(frozen=True)
class ToolSources:
    """The MCP tool sources available to workers this run (PRD §5b), pre-rendered.

    Derived once from the open servers rather than recomputed per worker: the tool→server map and
    the prompt hint are the same for every sub-question, and both are needed on every worker call —
    the hint to render the prompt, the map to attribute citations in `verify.py`.
    """

    servers: tuple[McpServer, ...] = ()
    instructions: str = ""
    tool_to_server: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_servers(cls, servers: Sequence[McpServer]) -> ToolSources:
        return cls(
            servers=tuple(servers),
            instructions=worker_instructions(list(servers)),
            tool_to_server=tool_to_server(list(servers)),
        )

    def stats_for(self, server_name: str | None) -> Any:
        return next((s.stats for s in self.servers if s.name == server_name), None)


NO_TOOL_SOURCES = ToolSources()
"""The web-only default: no MCP servers configured, so no MCP citation can verify (see `verify.py`)."""


def _report_allowance_knob(exc: UsageLimitExceeded) -> str:
    """Which `budgets.report_*` setting the write-up actually exhausted.

    Pydantic AI reports which limit was hit only in the message text, so that is what we read.
    """
    message = str(exc)
    if "request_limit" in message:
        return "report_request_allowance"
    if "cost_limit" in message:
        return "report_spend_allowance_usd"
    return "report_token_allowance"


@dataclass
class RunBudget:
    """The cost/token ceilings for one pipeline run, shared by every agent call in it.

    `budgets.spend_limit_usd` / `total_tokens_limit` bound the *run*, not each call (PRD §8:
    "token/cost ceiling on the whole run"). That only works if every `agent.run(...)` accumulates
    into one `RunUsage`: a fresh `UsageLimits` per call, as this used to build, gives each of the
    ~12+ calls in a default run its own full allowance, so a "$5 limit" could bill many times it.

    Two pots over one accumulator:

    - **Research** (workers, gap checks) is capped by the configured run ceiling.
    - **The write-up** (synthesis, critique) gets `budgets.report_token_allowance` /
      `report_spend_allowance_usd` *on top of whatever research actually cost*, granted by
      `open_report_allowance()` when the write-up begins.

    That "on top of actual usage" is the whole point, and it replaces an earlier design that
    reserved a fixed fraction of the ceiling. A fraction cannot survive overshoot, and research
    overshoots: measured on a real run, 303k tokens against a 170k research ceiling, so the
    reserve was already spent before synthesis started and the run ended with the research paid
    for and no report written — the exact outcome the reserve existed to prevent. An allowance
    measured from actual usage cannot be eaten by an overrun that precedes it.

    Overshoot itself is not fixable by tightening numbers, so both ceilings are **soft**:

    - Concurrency. `worker_concurrency` requests are in flight at once, each checked against the
      accumulator as it goes, so the ceiling can be passed by roughly the concurrency factor
      before any of them notices.
    - Granularity. One worker turn that fetches a large page can add ~100k tokens by itself, so
      the limit is crossed *within* a request, not between two.

    Read the research ceiling as "stop researching soon after this", not "never exceed this".
    Exceeding it is contained, not fatal: workers return `None`, the gap check falls through,
    and the pipeline proceeds to write up what it has — which is now affordable by construction.

    Worst-case total spend is therefore research-ceiling + overshoot + report allowance. That is
    a deliberate trade: a slightly fuzzier ceiling in exchange for never paying for research and
    getting nothing back.

    Scope is one process. `deep-research resume` starts a fresh budget for the continuation,
    which is the intended reading of "per run" — the resumed leg is a new run.
    """

    usage: RunUsage = field(default_factory=RunUsage)
    research_limits: UsageLimits = field(default_factory=UsageLimits)
    report_token_allowance: int = 0
    report_spend_allowance_usd: Decimal = Decimal(0)
    report_request_allowance: int = 0
    prices: dict[str, ModelPrice] = field(default_factory=dict)

    _report_baseline_tokens: int = field(default=0, repr=False)
    _report_baseline_usd: Decimal = field(default=Decimal(0), repr=False)
    _report_baseline_requests: int = field(default=0, repr=False)
    _estimated_cost_usd: Decimal = field(default=Decimal(0), repr=False)
    _report_baseline_estimated_usd: Decimal = field(default=Decimal(0), repr=False)
    _configured_spend_limit_usd: Decimal = field(default=Decimal(0), repr=False)

    @classmethod
    def from_config(cls, config: AppConfig) -> RunBudget:
        budgets = config.budgets
        return cls(
            usage=RunUsage(),
            research_limits=UsageLimits(
                cost_limit=Decimal(str(budgets.spend_limit_usd)),
                total_tokens_limit=budgets.total_tokens_limit,
                # Explicit, because Pydantic AI defaults this to 50. Since the whole run shares
                # one accumulator, that default is a run-wide cap and it binds long before the
                # token ceiling: one worker costs several requests (its own turns plus the
                # researcher it delegates to), so a single round of workers can exhaust it.
                request_limit=budgets.request_limit,
            ),
            report_token_allowance=budgets.report_token_allowance,
            report_spend_allowance_usd=Decimal(str(budgets.report_spend_allowance_usd)),
            report_request_allowance=budgets.report_request_allowance,
            prices=dict(config.prices),
            _configured_spend_limit_usd=Decimal(str(budgets.spend_limit_usd)),
        )

    def open_report_allowance(self) -> None:
        """Grant a fresh write-up allowance, measured from everything spent so far.

        Called immediately before each synthesis. Re-granting it per write-up pass is
        deliberate: when the critic sends the run back for another research round, that round
        bills against the research ceiling, and the report it then produces needs its own
        allowance rather than the remains of the previous pass's.
        """
        self._report_baseline_tokens = self.usage.total_tokens
        self._report_baseline_usd = Decimal(str(self.usage.cost or 0))
        self._report_baseline_requests = self.usage.requests
        self._report_baseline_estimated_usd = self._estimated_cost_usd

    @property
    def report_limits(self) -> UsageLimits:
        """Ceilings for synthesis/critique: the baseline at `open_report_allowance()` + allowance."""
        return UsageLimits(
            total_tokens_limit=self._report_baseline_tokens + self.report_token_allowance,
            cost_limit=self._report_baseline_usd + self.report_spend_allowance_usd,
            # Requests need the same baseline treatment as tokens and USD. Omitting it left the
            # write-up on Pydantic AI's default of 50 counted across the *whole* run, so a run
            # that had already spent 50 requests on research could not issue one for the report -
            # and the failure was reported as the token allowance being too small, which it was
            # not.
            #
            # An allowance of 0 means "not configured", not "no requests": a budget built directly
            # rather than through `from_config` leaves requests unconstrained instead of being
            # unable to issue a single one. `BudgetsConfig` requires > 0, so a real config can
            # never land here.
            request_limit=(
                self._report_baseline_requests + self.report_request_allowance
                if self.report_request_allowance > 0
                else None
            ),
        )

    # ------------------------------------------------------------------ configured prices
    def charge(self, model: str, before_input: int, before_output: int) -> None:
        """Bill one completed call against `prices`, if that model has a configured rate.

        Called with the token counts read *before* the call, because every agent shares one
        accumulator: the delta is this call's usage. Attribution is per call site, so a worker's
        delegated researcher turns are billed at the researcher rate - which is the model that
        actually ran them.

        A no-op when genai-prices already priced the run (`usage.cost` set) or the model has no
        configured rate: a measured cost is never overwritten by a hand-entered one.
        """
        if self.usage.cost is not None:
            return
        price = self.prices.get(model)
        if price is None:
            return
        self._estimated_cost_usd += price.cost_usd(
            max(0, self.usage.input_tokens - before_input),
            max(0, self.usage.output_tokens - before_output),
        )

    @property
    def cost_is_estimated(self) -> bool:
        """True when the figure came from `prices`, not from measured provider pricing."""
        return self.usage.cost is None and self._estimated_cost_usd > 0

    def enforce_configured_spend(self, *, report: bool) -> None:
        """Stop the run when configured prices say the ceiling is spent.

        Pydantic AI enforces `cost_limit` from its own pricing data, so with no data it enforces
        nothing - which is exactly when `prices` applies. Raising `UsageLimitExceeded` matches
        what the rest of the pipeline already handles: workers give up, the gap check falls
        through, and the write-up reports a sizing problem.
        """
        if self.usage.cost is not None or not self.prices:
            return

        if report:
            ceiling = self._report_baseline_estimated_usd + self.report_spend_allowance_usd
            knob = "budgets.report_spend_allowance_usd"
        else:
            ceiling = self._configured_spend_limit_usd
            knob = "budgets.spend_limit_usd"

        if ceiling > 0 and self._estimated_cost_usd >= ceiling:
            raise UsageLimitExceeded(
                f"Estimated cost ${self._estimated_cost_usd:.4f} reached the {knob} of "
                f"${ceiling:.4f}. This estimate comes from the `prices` section of config.yaml, "
                "because genai-prices has no data for this model."
            )

    def spent_usd(self) -> float:
        """What this run has cost so far.

        Measured cost when the provider is priced, otherwise the estimate from configured
        `prices`, otherwise 0.0. Check `cost_is_estimated` before presenting it as measured.
        """
        if self.usage.cost is not None:
            return float(self.usage.cost)
        return float(self._estimated_cost_usd)


async def _run_metered(
    agent: Agent[None, Any],
    prompt: str,
    *,
    budget: RunBudget,
    model: str,
    limits: UsageLimits,
    report: bool = False,
) -> Any:
    """Run an agent, bill the call against configured prices, then enforce the spend ceiling.

    Charging happens in `finally` so a call that raises still counts the tokens it burned;
    enforcement happens only on the way out, so a configured-price stop never masks the real
    exception that ended the call.
    """
    before_input, before_output = budget.usage.input_tokens, budget.usage.output_tokens
    try:
        result = await agent.run(prompt, usage=budget.usage, usage_limits=limits)
    finally:
        budget.charge(model, before_input, before_output)
    budget.enforce_configured_spend(report=report)
    return result


def _count_citations(finding: SubFinding, dropped: list[Source], sources: ToolSources) -> None:
    """Attribute surviving and dropped citations to their server, for the §11 ledger.

    Both halves matter: a server that is called often and cited never is wrongly scoped or wrongly
    described, and one whose citations are routinely *dropped* is being hallucinated against.
    """
    for source in finding.sources:
        stats = sources.stats_for(source.server) if source.kind == "mcp" else None
        if stats is not None:
            stats.citations_kept += 1
    for source in dropped:
        stats = sources.stats_for(source.server) if source.kind == "mcp" else None
        if stats is not None:
            stats.citations_dropped += 1


async def _research_one(
    worker_agent: Agent[None, SubFinding],
    subquestion: str,
    state: ResearchState,
    config: AppConfig,
    budget: RunBudget,
    console: Console,
    sources: ToolSources = NO_TOOL_SOURCES,
) -> SubFinding | None:
    """Research one sub-question. Returns `None` if it couldn't be answered.

    A failure is contained — one bad worker shouldn't sink the round — but it is *not* recorded
    as an empty finding: that would occupy a breadth-budget slot, block the sub-question from
    ever being retried on resume, and pad the report with a finding that says nothing. The
    caller records failures as unresolved instead.
    """
    prompt = render_worker_prompt(subquestion, state.brief, sources.instructions)
    with span(config, "worker: {subquestion}", subquestion=subquestion, round=state.round) as sp:
        for attempt in range(1, WORKER_ATTEMPTS + 1):
            try:
                result = await _run_metered(
                    worker_agent,
                    prompt,
                    budget=budget,
                    model=config.model.researcher,
                    limits=budget.research_limits,
                )
                messages = result.all_messages()
                claimed = result.output.sources
                finding = verify_finding(
                    result.output,
                    fetched_urls(messages),
                    mcp_evidence(messages, sources.tool_to_server),
                )
                kept = {(s.kind, s.identifier) for s in finding.sources}
                dropped = [s for s in claimed if (s.kind, s.identifier) not in kept]
                _count_citations(finding, dropped, sources)
                sp.set_attributes(
                    {
                        "confidence": finding.confidence,
                        "sources_kept": len(finding.sources),
                        "sources_dropped": len(dropped),
                        "mcp_sources_kept": sum(1 for s in finding.sources if s.kind == "mcp"),
                        "attempts": attempt,
                    }
                )
                return finding
            except UsageLimitExceeded as exc:
                # The run's budget is spent; a retry would fail identically and every sibling
                # worker is about to hit the same wall. Give up on this one immediately.
                sp.set_attributes({"error": str(exc), "attempts": attempt, "budget_exhausted": True})
                console.print(f"[yellow]Budget reached while researching:[/] {subquestion}")
                return None
            except Exception as exc:  # noqa: BLE001 — contained on purpose, see the docstring
                if attempt < WORKER_ATTEMPTS:
                    console.print(f"[dim]Worker failed on '{subquestion}' ({exc}) — retrying once...[/]")
                    continue
                sp.set_attributes({"error": str(exc), "attempts": attempt})
                console.print(f"[yellow]Worker failed on:[/] {subquestion} [dim]({exc})[/]")
                return None
    return None


async def research_round(
    worker_agent: Agent[None, SubFinding],
    subquestions: list[str],
    state: ResearchState,
    config: AppConfig,
    budget: RunBudget,
    console: Console,
    sources: ToolSources = NO_TOOL_SOURCES,
) -> tuple[dict[str, SubFinding], list[str]]:
    """Research every sub-question concurrently. Returns `(findings, failed_subquestions)`."""
    semaphore = asyncio.Semaphore(max(1, config.budgets.worker_concurrency))

    async def bounded(subquestion: str) -> tuple[str, SubFinding | None]:
        async with semaphore:
            return subquestion, await _research_one(
                worker_agent, subquestion, state, config, budget, console, sources
            )

    with span(config, "research round {round}", round=state.round, subquestion_count=len(subquestions)):
        results = await asyncio.gather(*(bounded(q) for q in subquestions))

    findings = {q: f for q, f in results if f is not None}
    failed = [q for q, f in results if f is None]
    return findings, failed


async def run_gap_check(
    gap_check_agent: Agent[None, GapCheckResult], state: ResearchState, config: AppConfig, budget: RunBudget
) -> GapCheckResult:
    prompt = render_gap_check_prompt(state)
    with span(config, "gap-check round {round}", round=state.round) as sp:
        result = await _run_metered(
            gap_check_agent, prompt, budget=budget, model=config.model.lead, limits=budget.research_limits
        )
        sp.set_attribute("follow_up_count", len(result.output.follow_up_subquestions))
        return result.output


async def run_synthesis(
    synthesis_agent: Agent[None, ResearchReport],
    state: ResearchState,
    config: AppConfig,
    budget: RunBudget,
    critic_issues: list[str] | None = None,
) -> ResearchReport:
    """Write (or rewrite) the report. `critic_issues` carries a failed critique into the prompt."""
    prompt = render_synthesis_prompt(state, critic_issues or ())
    with span(
        config,
        "synthesize",
        round=state.round,
        findings_count=len(state.findings),
        revision=bool(critic_issues),
    ):
        result = await _run_metered(
            synthesis_agent, prompt, budget=budget, model=config.model.lead,
            limits=budget.report_limits, report=True,
        )
        return result.output


async def run_critique(
    critic_agent: Agent[None, CriticVerdict],
    state: ResearchState,
    report: ResearchReport,
    config: AppConfig,
    budget: RunBudget,
) -> CriticVerdict:
    prompt = render_critique_prompt(state, report)
    with span(config, "critique attempt {attempt}", attempt=state.critic_rounds + 1) as sp:
        result = await _run_metered(
            critic_agent, prompt, budget=budget, model=config.model.critic,
            limits=budget.report_limits, report=True,
        )
        sp.set_attributes({"passed": result.output.passed, "issue_count": len(result.output.issues)})
        return result.output


MAX_REPORT_REVISIONS = 2
"""Rewrite-only passes the critic loop may spend on writing problems (PRD §10 item 2).

The critic distinguishes issues that need more research (it proposes follow-ups) from issues
with the writing itself (it doesn't — "more research can't fix a writing problem"). The loop
below researches the former, but for the latter its only remedy used to be shipping as-is: a
real run shipped a 4KB draft that ignored most of its 8 findings because the critic's 5 issues
were all writing problems. A rewrite pass — re-running synthesis with the critic's issues in
the prompt — is what actually fixes those. Bounded, because a synthesis model and a critic
model can genuinely disagree, and ping-ponging drafts between them burns the report allowance
without converging; after this many rewrites the draft ships with the critic's issues noted,
same as before.
"""

UNVERIFIED_NOTE = (
    "The report was not independently reviewed: the run's budget was spent before the critic "
    "could check it. Treat its claims as unverified."
)


async def _critique_or_skip(
    critic_agent: Agent[None, CriticVerdict],
    state: ResearchState,
    report: ResearchReport,
    config: AppConfig,
    budget: RunBudget,
    console: Console,
) -> tuple[CriticVerdict, bool]:
    """Critique the draft. Returns `(verdict, budget_ran_out)`.

    Losing the whole report because there wasn't enough budget left to *review* it would be the
    worst of both outcomes — the expensive part is already paid for. An unverified report that
    says so is more useful than no report, and saying so is the honest half (PRD §10).
    """
    try:
        return await run_critique(critic_agent, state, report, config, budget), False
    except UsageLimitExceeded as exc:
        console.print(f"[yellow]Budget reached before the report could be critiqued:[/] {exc}")
        verdict = CriticVerdict(
            passed=False, issues=[UNVERIFIED_NOTE], follow_up_subquestions=[], reasoning=str(exc)
        )
        return verdict, True


def budget_exhausted(state: ResearchState) -> bool:
    """Whether either budget in the brief leaves no room for another round of research.

    Public because the CLI reports it in run metrics and uses it to decide whether a resume can
    make progress — one definition of "out of budget", not a copy per caller.
    """
    return state.round >= state.brief.depth_budget or len(state.findings) >= state.brief.breadth_budget


def _record_unresolved(state: ResearchState, subquestions: list[str]) -> None:
    """Remember sub-questions a budget cut off, so the report can admit to them.

    Called at every point where the pipeline drops planned work: the breadth-truncated tail of
    a research batch, and the follow-ups a gap-check or a critique proposed that didn't fit.
    Without this the drop is invisible — `open_subquestions` is emptied each round, so by
    synthesis time nothing distinguishes "covered everything" from "ran out of budget".
    """
    for question in subquestions:
        if question not in state.unresolved_subquestions and question not in state.findings:
            state.unresolved_subquestions.append(question)


def _clear_resolved(state: ResearchState) -> None:
    """Drop anything from the unresolved list that has since been answered.

    A sub-question can land in `unresolved_subquestions` (a worker failed, or a budget skipped
    it) and then be researched successfully on a later round or a resumed run. Without this it
    would be reported as both a finding and a gap.
    """
    state.unresolved_subquestions = [q for q in state.unresolved_subquestions if q not in state.findings]


def _fit_follow_ups(state: ResearchState, proposed: list[str]) -> list[str]:
    """The follow-ups that fit in the remaining breadth budget; the rest are recorded as unresolved."""
    candidates = [q for q in proposed if q not in state.findings]
    remaining_slots = max(0, state.brief.breadth_budget - len(state.findings))
    _record_unresolved(state, candidates[remaining_slots:])
    return candidates[:remaining_slots]


async def run_research_pipeline(
    worker_agent: Agent[None, SubFinding],
    gap_check_agent: Agent[None, GapCheckResult],
    synthesis_agent: Agent[None, ResearchReport],
    critic_agent: Agent[None, CriticVerdict],
    state: ResearchState,
    state_path: Path,
    config: AppConfig,
    console: Console,
    budget: RunBudget | None = None,
    sources: ToolSources = NO_TOOL_SOURCES,
) -> ResearchReport:
    """Drive `state` to `"done"`, checkpointing after every step (PRD §7, §9 — recitation).

    Budget note: `open_subquestions` is a work *queue*, emptied as each round completes, so it
    can't answer "what did we never get to". Anything a budget cuts off — a breadth-truncated
    batch, follow-ups a gap-check or critique proposed that didn't fit — is recorded in
    `state.unresolved_subquestions` instead, handed to synthesis, and folded into the report's
    `unresolved` list deterministically at the end (PRD §10(b): every brief sub-question is
    either addressed or explicitly listed as unresolved).

    Resume note: if a crash lands between an LLM call finishing and its downstream state being
    saved (a gap-check's follow-ups, a critique's verdict), the resumed run treats that step as
    not having happened and moves on to the next phase in sequence — a deliberate, documented
    "lose at most one round of cheap LLM judgment, never re-do finished research" tradeoff.
    """
    budget = budget if budget is not None else RunBudget.from_config(config)

    while state.open_subquestions and state.status != "done":
        remaining_slots = max(0, state.brief.breadth_budget - len(state.findings))
        batch = state.open_subquestions[:remaining_slots]
        skipped = state.open_subquestions[remaining_slots:]
        if skipped:
            console.print(f"[dim]Breadth budget reached — skipping {len(skipped)} sub-question(s) this round.[/]")
            _record_unresolved(state, skipped)

        console.print(f"[bold]Round {state.round}[/]: researching {len(batch)} sub-question(s)...")
        state.status = "researching"
        state.save(state_path)

        findings, failed = await research_round(worker_agent, batch, state, config, budget, console, sources)
        state.findings.update(findings)
        _record_unresolved(state, failed)
        _clear_resolved(state)
        state.open_subquestions = []
        state.save(state_path)

        if budget_exhausted(state):
            break

        console.print("[dim]Checking for gaps...[/]")
        state.status = "gap_checking"
        state.save(state_path)

        try:
            gap = await run_gap_check(gap_check_agent, state, config, budget)
        except UsageLimitExceeded as exc:
            # Out of research budget. That's a reason to stop looking for more work, not to
            # throw away the findings — fall through to synthesis on what we have.
            console.print(f"[yellow]Budget reached before the gap check could run:[/] {exc}")
            break
        follow_ups = _fit_follow_ups(state, gap.follow_up_subquestions)

        if not follow_ups:
            state.save(state_path)
            break

        console.print(f"[dim]{gap.reasoning}[/]")
        state.round += 1
        state.open_subquestions = follow_ups
        state.save(state_path)

    console.print("[bold]Synthesizing report...[/]")
    state.status = "synthesizing"
    state.save(state_path)
    budget.open_report_allowance()
    try:
        report = await run_synthesis(synthesis_agent, state, config, budget)
    except UsageLimitExceeded as exc:
        # The write-up had a guaranteed allowance and still could not finish, so this is a
        # sizing problem, not research having eaten the budget. Say which knob to turn.
        # Name the allowance that actually ran out. Pointing at the token knob when the run hit
        # the request ceiling sends you tuning a number that was never the constraint.
        knob = _report_allowance_knob(exc)
        console.print(
            f"[red]The report allowance was not enough to write the report:[/] {exc}\n"
            f"[dim]Raise budgets.{knob} in config.yaml, then resume — findings are checkpointed "
            "and will not be re-researched.[/]"
        )
        raise

    console.print("[bold]Critiquing draft report...[/]")
    state.status = "critiquing"
    state.save(state_path)
    verdict, out_of_budget = await _critique_or_skip(critic_agent, state, report, config, budget, console)
    state.critic_rounds += 1
    state.critic_passed = verdict.passed
    state.save(state_path)

    revisions_left = MAX_REPORT_REVISIONS
    while not verdict.passed and not out_of_budget:
        console.print(f"[yellow]Critic found issues:[/] {'; '.join(verdict.issues)}")

        follow_ups = _fit_follow_ups(state, verdict.follow_up_subquestions) if not budget_exhausted(state) else []
        if follow_ups:
            state.round += 1
            state.status = "researching"
            state.open_subquestions = follow_ups
            state.save(state_path)

            findings, failed = await research_round(
                worker_agent, follow_ups, state, config, budget, console, sources
            )
            state.findings.update(findings)
            _record_unresolved(state, failed)
            _clear_resolved(state)
            state.open_subquestions = []
            state.save(state_path)
            console.print("[bold]Re-synthesizing report...[/]")
        elif revisions_left > 0:
            # Nothing to research — the issues are with the writing, or the budget can't afford
            # more research. Either way a rewrite against the critic's issues is still cheap
            # (report allowance, not research budget) and is the only remedy that can help.
            if budget_exhausted(state):
                # Record what the critic wanted researched and couldn't be, so the report names
                # the specific gap rather than only the complaint about it.
                _record_unresolved(state, verdict.follow_up_subquestions)
            revisions_left -= 1
            state.save(state_path)
            console.print("[bold]Revising the report against the critic's issues...[/]")
        else:
            console.print(
                f"[dim]Critic still unsatisfied after {MAX_REPORT_REVISIONS} revision(s) — "
                f"shipping with its issues noted.[/]"
            )
            _record_unresolved(state, verdict.follow_up_subquestions)
            state.save(state_path)
            break

        state.status = "synthesizing"
        state.save(state_path)
        budget.open_report_allowance()
        try:
            # The failed critique feeds the rewrite on both paths: after a research round it
            # says what the previous draft got wrong; on a pure revision it is the entire
            # reason synthesis is running again.
            report = await run_synthesis(synthesis_agent, state, config, budget, critic_issues=verdict.issues)
        except UsageLimitExceeded as exc:
            # Keep the previous draft rather than losing the run: it's a report built from
            # fewer findings, which is exactly what `unresolved` is for.
            console.print(f"[yellow]Budget reached before the report could be rewritten:[/] {exc}")
            break

        console.print("[bold]Re-critiquing draft report...[/]")
        state.status = "critiquing"
        state.save(state_path)
        verdict, out_of_budget = await _critique_or_skip(critic_agent, state, report, config, budget, console)
        state.critic_rounds += 1
        state.critic_passed = verdict.passed
        state.save(state_path)

    # Fold everything the run couldn't finish into `unresolved` in code rather than trusting the
    # synthesis agent to have honoured the prompt. Same reasoning as the critic (PRD §10): if the
    # honesty of the report matters, don't leave it to model judgment. Budget-truncated
    # sub-questions are reported whether or not the critic passed.
    unresolved = list(report.unresolved)

    def _add(item: str) -> None:
        if item not in unresolved:
            unresolved.append(item)

    # The synthesis prompt tells the model *not* to list these, precisely so this merge can be
    # an unconditional append: the two lists have disjoint jobs (the model reports gaps in what
    # was researched, the harness reports what was never researched at all). Trying instead to
    # detect and skip duplicates by substring was worse than useless — a one-word sub-question
    # like "b" matches the "b" in "budget" and gets silently dropped from the report.
    for question in state.unresolved_subquestions:
        _add(f"Not researched (budget exhausted): {question}")
    if not verdict.passed:
        for issue in verdict.issues:
            _add(issue)
    if unresolved != report.unresolved:
        report = report.model_copy(update={"unresolved": unresolved})

    state.status = "done"
    state.save(state_path)
    return report
