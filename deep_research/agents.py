"""Agent construction for the deep-research harness (PRD §5).

Five agent roles, all sharing `config.model.*` but kept as separate `Agent` instances rather than
one multi-purpose agent — each has a narrow, non-overlapping job and its own instructions, so
there's no risk of e.g. scoping instructions bleeding into synthesis:

- **lead agent** — negotiates scope with the human (`ClarifyingQuestion | BriefDraft`). Also
  produces the initial sub-question list (phase 2, "Plan" — folded into the brief itself).
- **worker agent** — the bare `Researcher()` capability, one call per sub-question, each producing
  a structured `SubFinding` (phase 3, parallel research). Its output also passes through
  `verify.py`'s deterministic per-finding check before being trusted (PRD §10 item 1).
- **gap-check agent** — no tools; reviews accumulated findings and proposes follow-up
  sub-questions, or signals coverage is sufficient (phase 4).
- **synthesis agent** — no tools; merges all findings into the final `ResearchReport` (phase 5).
- **critic agent** — no tools; independently reviews the *draft report* against the brief and
  findings (phase 6, PRD §10 item 2 — generator/evaluator separation applied to the report
  itself, not just the raw findings). Deliberately a separate agent/call from synthesis so it
  isn't grading its own work.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai import Agent
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai_harness.researcher import Researcher

from deep_research.config import AppConfig
from deep_research.models import (
    BriefDraft,
    ClarifyingQuestion,
    CriticVerdict,
    GapCheckResult,
    ResearchBrief,
    ResearchReport,
    ResearchState,
    SubFinding,
)

LEAD_INSTRUCTIONS = """\
You are the scoping assistant for a deep-research harness. Your only job is to turn the human's \
question into a clear, well-scoped research brief before any research begins — you do not \
research anything yourself.

- If the question is ambiguous, underspecified, or could reasonably be answered several very \
different ways, ask ONE focused clarifying question (respond with a ClarifyingQuestion).
- Don't ask more than two or three clarifying questions in total. Once the scope is reasonably \
clear — or you've asked enough — respond with a BriefDraft: a crisply restated question, a short \
list of concrete sub-questions the research should cover, and explicit assumptions for anything \
you decided not to ask about.
- Prefer proceeding with a stated assumption over asking a low-value clarifying question.
"""

WORKER_INSTRUCTIONS = """\
You are a research worker. You will be given ONE specific sub-question — research only that \
sub-question, not the broader topic around it. Produce a SubFinding: a direct answer, a \
confidence level, the sources you actually retrieved, and any contradictions between sources. If \
sources disagree, say so in `contradictions` rather than silently picking a side.

Citation rule, enforced automatically after you answer: you may only cite a source you actually \
retrieved and read this run. How that is checked depends on where the source came from:

- **A web page** — set kind='web' and identifier to the exact URL you fetched. Seeing a URL in \
search results is NOT enough: search gives you candidates, and you must fetch a candidate before \
you can cite it.
- **An item from another tool source** (a bibliography, a document store — any such source \
available to you is described in your prompt) — set kind='mcp', server to that source's name, and \
identifier to the item's own ID exactly as the tool returned it. Do not invent, guess, correct, or \
reconstruct an ID, and do not put a URL there.

Any source you list without having retrieved it is deleted from your finding by a deterministic \
check, so a plausible-looking citation you never opened does not make your answer better-sourced; \
it makes it emptier. A well-formed ID is not evidence — only one a tool actually returned to you \
counts. So: search to find candidates, then retrieve the ones you intend to rely on, then cite \
only those. If a retrieval fails, either use an alternative source or lower your confidence and \
say why — do not cite what you failed to read.
"""

GAP_CHECK_INSTRUCTIONS = """\
You review research findings gathered so far against the original brief and decide whether more \
research is needed. You do not research anything yourself.

- Propose follow-up sub-questions ONLY for concrete gaps: a sub-question with low confidence, \
unresolved contradictions, or a part of the original brief nothing yet addresses.
- Each follow-up must be a new, specific, researchable question — not a rephrasing of one already \
answered.
- If coverage is already sufficient, return an empty `follow_up_subquestions` list. Prefer \
stopping over asking for marginal completeness — a bounded budget means every follow-up round has \
a real cost.
"""

SYNTHESIS_INSTRUCTIONS = """\
You write the final research report from findings already gathered — you do not research \
anything yourself and must not introduce claims beyond what the findings support.

- One section per sub-question (or per theme, if that reads better), citing sources inline as \
Markdown links: [title](url).
- Surface contradictions explicitly rather than silently picking a side.
- Any sub-question still unanswered or low-confidence when the research budget ran out belongs in \
`unresolved`, not papered over in a section.
"""

CRITIC_INSTRUCTIONS = """\
You are an adversarial, skeptical reviewer of a draft research report — you did not write it and \
you do not research anything yourself. Your only job is to catch problems, not to praise the report.

Check specifically for:
- Any claim in the report that isn't backed by a source in the findings you're shown. Don't take \
the report's own citations at face value — cross-check them against the findings.
- Any sub-question from the original brief that the report neither answers nor lists under \
`unresolved`.
- Contradictions the findings flagged that the report silently resolved instead of surfacing.

Set `passed=false` and list every concrete issue if you find any of the above — vague dissatisfaction \
doesn't count, each issue must be specific enough to act on. For each issue that stems from missing \
research (not just missing-from-report), propose a concrete follow-up sub-question that would fix \
it. If an issue is purely about how the report is written (not missing research), don't add a \
follow-up for it — more research can't fix a writing problem. Set `passed=true` only if you find \
nothing to flag.
"""

WORKER_PROMPT_TEMPLATE = """\
Research question (for context only): {question}

Sub-question to research: {subquestion}
{tool_sources}"""

GAP_CHECK_PROMPT_TEMPLATE = """\
Original research question: {question}

Sub-questions already covered:
{findings_summary}

Budget remaining: {rounds_remaining} follow-up round(s), {breadth_remaining} more sub-question(s) \
total across all remaining rounds.

Decide whether follow-up sub-questions are needed, respecting the remaining budget.
"""

SYNTHESIS_PROMPT_TEMPLATE = """\
Research question: {question}

Assumptions agreed with the requester:
{assumptions}

Findings gathered:
{findings_summary}

Sub-questions never researched, because a budget cut them off. Do not fabricate answers for \
these, and do not list them under `unresolved` — the harness appends them itself, so listing \
them here would only duplicate them. They are shown to you so you can write the report around \
the gap honestly:
{open_subquestions}

Use `unresolved` for gaps you notice in the findings themselves: a sub-question that was \
researched but came back too thin or too contradictory to answer.

Produce the final ResearchReport.
"""

WORKER_RETRIES = {"tools": 3, "output": 2}
"""Retry budgets for the worker's built-in tools and its structured output.

Pydantic AI defaults both to 1, and real runs died on each: two consecutive paywalled fetches
killed a worker with "Tool 'web_fetch' exceeded max retries count of 1" (a 403 is routine web
reality, not a reason to abandon a sub-question), and a weaker model producing one malformed
SubFinding died with "Exceeded maximum output retries (1)". MCP tools are unaffected — each
server's `max_tool_retries` config (default 3) overrides this agent-level default.
"""

SYNTHESIS_REVISION_ADDENDUM = """\

An independent reviewer rejected the previous draft of this report for the issues listed below. \
Most such issues are findings content the draft left out — fix every issue that can be fixed \
from the findings above. Do not invent claims beyond the findings to satisfy the reviewer: if an \
issue asks for something the findings genuinely do not contain, leave it unfixed and it will be \
reported as unresolved.

Reviewer issues with the previous draft:
{issues}
"""

CRITIQUE_PROMPT_TEMPLATE = """\
Original research question: {question}

Sub-questions the brief asked for:
{subquestions}

Findings gathered (the only source of truth — cross-check the draft report's claims against these):
{findings_summary}

Draft report to review:
---
{report_markdown}
---

Review the draft against the findings and the brief's sub-questions per your instructions.
"""


def build_lead_agent(config: AppConfig) -> Agent[None, ClarifyingQuestion | BriefDraft]:
    return Agent(
        config.model.lead,
        name="lead",
        instructions=LEAD_INSTRUCTIONS,
        output_type=ClarifyingQuestion | BriefDraft,
        # Defers provider/API-key resolution to first run, so building the agent (e.g. in tests
        # that override the model) doesn't require credentials for the configured provider.
        defer_model_check=True,
    )


def build_worker_agent(
    config: AppConfig,
    toolsets: Sequence[AbstractToolset[None]] | None = None,
) -> Agent[None, SubFinding]:
    """The one agent in the system with tools (PRD §5, **T**).

    `toolsets` carries any configured MCP tool sources (PRD §5b). They attach here and nowhere
    else: §9's "no agent's context grows across the run" argument — the reason there is no
    `Compaction` anywhere — depends on exactly one agent holding tools, and MCP servers return the
    largest payloads in the system, so this is the wrong invariant to relax.
    """
    return Agent(
        config.model.researcher,
        name="worker",
        instructions=WORKER_INSTRUCTIONS,
        # instructions=None: WORKER_INSTRUCTIONS above already covers sourcing/citation guidance
        # for this narrower, single-sub-question role — avoid stacking Researcher()'s own default.
        capabilities=[Researcher(instructions=None)],
        toolsets=list(toolsets or ()),
        output_type=SubFinding,
        retries=WORKER_RETRIES,
        defer_model_check=True,
    )


def build_gap_check_agent(config: AppConfig) -> Agent[None, GapCheckResult]:
    return Agent(
        config.model.lead,
        name="gap-check",
        instructions=GAP_CHECK_INSTRUCTIONS,
        output_type=GapCheckResult,
        defer_model_check=True,
    )


def build_synthesis_agent(config: AppConfig) -> Agent[None, ResearchReport]:
    return Agent(
        config.model.lead,
        name="synthesis",
        instructions=SYNTHESIS_INSTRUCTIONS,
        output_type=ResearchReport,
        defer_model_check=True,
    )


def build_critic_agent(config: AppConfig) -> Agent[None, CriticVerdict]:
    return Agent(
        config.model.critic,
        name="critic",
        instructions=CRITIC_INSTRUCTIONS,
        output_type=CriticVerdict,
        defer_model_check=True,
    )


def render_worker_prompt(subquestion: str, brief: ResearchBrief, tool_sources: str = "") -> str:
    """Render the worker's prompt, including any operator guidance on MCP tool sources.

    `tool_sources` comes from `mcp.worker_instructions` — the per-server `instructions` hints. It
    belongs in the prompt rather than the agent's static instructions because it is configuration,
    not role: the same worker agent is correct for a web-only run and a library-backed one.
    """
    return WORKER_PROMPT_TEMPLATE.format(
        question=brief.question,
        subquestion=subquestion,
        tool_sources=tool_sources,
    )


def _findings_summary(state: ResearchState) -> str:
    if not state.findings:
        return "(no findings yet)"
    lines: list[str] = []
    for subquestion, finding in state.findings.items():
        lines.append(f"- [{finding.confidence}] {subquestion} → {finding.answer}")
        if finding.contradictions:
            lines.append(f"  contradictions: {'; '.join(finding.contradictions)}")
    return "\n".join(lines)


def render_gap_check_prompt(state: ResearchState) -> str:
    rounds_remaining = max(0, state.brief.depth_budget - state.round)
    breadth_remaining = max(0, state.brief.breadth_budget - len(state.findings))
    return GAP_CHECK_PROMPT_TEMPLATE.format(
        question=state.brief.question,
        findings_summary=_findings_summary(state),
        rounds_remaining=rounds_remaining,
        breadth_remaining=breadth_remaining,
    )


def render_synthesis_prompt(state: ResearchState, critic_issues: Sequence[str] = ()) -> str:
    assumptions = "\n".join(f"- {a}" for a in state.brief.assumptions) or "(none)"
    # Both lists matter: `unresolved_subquestions` is everything a budget cut off, and
    # `open_subquestions` is anything still queued if synthesis is running early (a resumed
    # run). Reading only the latter always yielded "(none)", since the pipeline empties the
    # queue before it ever synthesizes.
    unresolved = list(state.unresolved_subquestions)
    for question in state.open_subquestions:
        if question not in unresolved:
            unresolved.append(question)
    open_subquestions = "\n".join(f"- {q}" for q in unresolved) or "(none — everything was covered)"
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        question=state.brief.question,
        assumptions=assumptions,
        findings_summary=_findings_summary(state),
        open_subquestions=open_subquestions,
    )
    # A failed critique feeds back into the rewrite, whichever path led here: after a critic-
    # driven research round the issues say what the previous draft got wrong, and on a pure
    # revision they are the entire reason the synthesis is running again. A real run shipped a
    # 4KB draft that ignored most of its findings precisely because the re-synthesis prompt
    # never told the model what the critic had objected to.
    if critic_issues:
        prompt += SYNTHESIS_REVISION_ADDENDUM.format(issues="\n".join(f"- {i}" for i in critic_issues))
    return prompt


def render_critique_prompt(state: ResearchState, report: ResearchReport) -> str:
    subquestions = "\n".join(f"- {q}" for q in state.brief.subquestions) or "(none stated — just the question itself)"
    return CRITIQUE_PROMPT_TEMPLATE.format(
        question=state.brief.question,
        subquestions=subquestions,
        findings_summary=_findings_summary(state),
        report_markdown=report.to_markdown(),
    )
