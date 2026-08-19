"""The data model of PRD §7: what the phases hand each other, and what is persisted.

Three groups. The *scoping* models (`ClarifyingQuestion`, `BriefDraft`, `ResearchBrief`) carry
the question from raw text to a budgeted brief. The *research* models (`SubFinding`, `Source`,
`GapCheckResult`, `CriticVerdict`) are what agents return, each shaped so the harness can check
it rather than take it on trust. `ResearchState` is the handoff and checkpoint artifact, and
`ResearchReport` the single user-facing output (§7a) — everything else is internal bookkeeping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ClarifyingQuestion(BaseModel):
    """Emitted by the lead agent when the brief is still ambiguous."""

    question: str = Field(description="A single, focused clarifying question to ask the user.")


class BriefDraft(BaseModel):
    """Emitted by the lead agent once it has enough information to scope the research.

    Budgets are deliberately not part of this model — they come from config (PRD §13), not
    model judgment, and are merged in by `ResearchBrief.from_draft` (see §8: "not model-decided").
    """

    question: str = Field(description="The research question, restated clearly and precisely.")
    subquestions: list[str] = Field(
        default_factory=list,
        description="Concrete sub-questions the research should cover.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made to resolve ambiguity the user wasn't asked about.",
    )


class ResearchBrief(BaseModel):
    """The scoped, human-confirmed research request handed to the researcher agent."""

    question: str
    subquestions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    depth_budget: int
    breadth_budget: int

    @classmethod
    def from_draft(cls, draft: BriefDraft, *, depth_budget: int, breadth_budget: int) -> ResearchBrief:
        return cls(
            question=draft.question,
            subquestions=draft.subquestions,
            assumptions=draft.assumptions,
            depth_budget=depth_budget,
            breadth_budget=breadth_budget,
        )


class Source(BaseModel):
    """One thing a worker actually retrieved and is citing (PRD §7).

    `kind`/`identifier`/`server` replaced a bare `url` when MCP tool sources arrived (PRD §5b).
    The reason is not cosmetic: a Zotero item is identified by an 8-character library key and many
    items have no useful public URL at all, so under the old shape every library citation would
    have been dropped by the fetch check in `verify.py` — a correct, library-sourced answer would
    have been reported as unsourced. Verification and rendering are polymorphic in `kind`; nothing
    else in the harness needs to know the difference.
    """

    kind: Literal["web", "mcp"] = Field(
        default="web",
        description="'web' for something fetched from a URL; 'mcp' for an item retrieved from a "
        "configured tool source such as a bibliography.",
    )
    identifier: str = Field(
        description="For kind='web', the exact URL fetched. For kind='mcp', the item's "
        "server-native ID (e.g. a Zotero item key) exactly as the tool returned it."
    )
    server: str | None = Field(
        default=None,
        description="For kind='mcp', the name of the tool source that returned this item. Null for web.",
    )
    title: str
    quoted_snippet: str = Field(description="A short quote from the source supporting the claim.")

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_url(cls, data: Any) -> Any:
        """Load pre-MCP `ResearchState` checkpoints, which named this field `url`.

        `deep-research resume` reads persisted JSON, so renaming the field would otherwise strand
        every checkpoint on disk — failing to resume a long, already-paid-for run is the most
        annoying possible way to ship this change (PRD §7). Handled here rather than with a
        `validation_alias` so the field name the model sees in its output schema stays clean.
        """
        if isinstance(data, dict) and "identifier" not in data and "url" in data:
            data = {**data, "identifier": data["url"]}
            data.pop("url", None)
        return data

    def render_markdown(self) -> str:
        """One citation line. A web source links; an MCP source attributes its identifier.

        A `kind="mcp"` source deliberately does not become a link (PRD §7a): a `zotero://` URI
        resolves only inside that server's own address space, and substituting the item's `url`
        field would point the reader at a page the worker never read — reintroducing through the
        renderer exactly the failure mode §8 exists to close.
        """
        cited = f"[{self.title}]({self.identifier})" if self.kind == "web" else f"{self.title} — {self.label()}"
        return f'{cited} — "{self.quoted_snippet}"'

    def label(self) -> str:
        """How this source is named outside a citation line (prompts, critique, logs)."""
        if self.kind == "web":
            return self.identifier
        source = self.server or "tool source"
        return f"{source} item `{self.identifier}`"


class ReportSection(BaseModel):
    heading: str
    content: str = Field(
        description="Markdown body. Cite web sources inline as [title](url); cite items from a "
        "tool source by name and identifier, e.g. `ABC12345`, since they have no public URL."
    )
    sources: list[Source] = Field(default_factory=list)


class SubFinding(BaseModel):
    """A worker agent's answer to one sub-question (PRD §7 — phase 3, parallel research)."""

    subquestion: str
    answer: str
    confidence: Literal["high", "medium", "low"]
    sources: list[Source] = Field(default_factory=list)
    contradictions: list[str] = Field(
        default_factory=list,
        description="Notes on conflicting sources found while researching this sub-question.",
    )


class GapCheckResult(BaseModel):
    """The lead agent's phase-4 verdict: are we done, or is there a specific gap to chase?"""

    follow_up_subquestions: list[str] = Field(
        default_factory=list,
        description="New, specific sub-questions to research next. Empty means coverage is sufficient.",
    )
    reasoning: str = Field(description="Brief justification for the follow-ups (or for stopping).")


class CriticVerdict(BaseModel):
    """The critic's phase-6 verdict on a draft `ResearchReport` (PRD §6/§10 — generator/evaluator
    separation applied to the report itself, not just the findings).
    """

    passed: bool = Field(description="True if every claim is source-backed and the brief is covered.")
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems found: an unsupported claim, a missing citation, a brief "
        "sub-question the report never addresses, etc. Empty iff passed.",
    )
    follow_up_subquestions: list[str] = Field(
        default_factory=list,
        description="New, specific sub-questions that would resolve the issues above, if any research "
        "can fix them. Empty if the issues are about the report's writing, not missing research.",
    )
    reasoning: str = Field(description="Brief justification for the verdict.")


class ResearchState(BaseModel):
    """The handoff artifact (PRD §7): internal bookkeeping, persisted after every phase/round so a
    crashed or interrupted run can resume via `deep-research resume <state-file>` instead of
    restarting. Never shown to the user directly — see §7a.
    """

    brief: ResearchBrief
    findings: dict[str, SubFinding] = Field(default_factory=dict)
    open_subquestions: list[str] = Field(default_factory=list)
    unresolved_subquestions: list[str] = Field(
        default_factory=list,
        description="Sub-questions that were planned or proposed but never researched, because a "
        "budget cut them off. Distinct from `open_subquestions`, which is the queue for the next "
        "round and is emptied as work completes: this list only grows, and is what the report's "
        "`unresolved` section is built from, so budget-truncated work is reported rather than "
        "silently forgotten.",
    )
    round: int = 0
    status: Literal[
        "researching", "gap_checking", "synthesizing", "critiquing", "done", "failed"
    ] = "researching"
    critic_rounds: int = Field(default=0, description="How many times the critic has been called this run.")
    critic_passed: bool | None = Field(default=None, description="The most recent critique's verdict, if any.")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> ResearchState:
        return cls.model_validate_json(path.read_text())


class ResearchReport(BaseModel):
    question: str
    summary: str
    sections: list[ReportSection] = Field(default_factory=list)
    unresolved: list[str] = Field(
        default_factory=list,
        description="Sub-questions or gaps that couldn't be resolved within budget.",
    )
    assumptions: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """The one user-facing rendering of a report — see PRD §7a."""
        lines: list[str] = [f"# {self.question}", "", self.summary, ""]

        for section in self.sections:
            lines.append(f"## {section.heading}")
            lines.append("")
            lines.append(section.content)
            if section.sources:
                lines.append("")
                lines.append("**Sources:**")
                for source in section.sources:
                    lines.append(f"- {source.render_markdown()}")
            lines.append("")

        if self.assumptions:
            lines.append("## Assumptions")
            lines.append("")
            for assumption in self.assumptions:
                lines.append(f"- {assumption}")
            lines.append("")

        if self.unresolved:
            lines.append("## Unresolved")
            lines.append("")
            for item in self.unresolved:
                lines.append(f"- {item}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


class McpServerStats(BaseModel):
    """One MCP server's contribution to one run (PRD §11).

    The question an operator actually asks is whether a configured server earns its cost, and
    `calls` alone can't answer it: a server called often and cited never is wrongly scoped,
    wrongly described, or wrong for the corpus. `transport` is recorded because a latency or
    failure-rate comparison between an in-process server and one across the internet is not a
    comparison of the servers.
    """

    transport: str
    calls: int = 0
    failed_calls: int = 0
    injected_calls: int = Field(default=0, description="Calls where configured `tool_args` overrode or added an argument.")
    citations_kept: int = Field(default=0, description="Citations from this server that survived verification.")
    citations_dropped: int = Field(default=0, description="Citations claimed against this server it never returned.")


class RunMetrics(BaseModel):
    """One completed run's summary, appended to `output.state_dir/metrics.jsonl` (PRD §11).

    Not part of the ResearchState/ResearchReport handoff — a separate, append-only local ledger
    purely for tracking efficiency across runs over time (`deep-research stats` reads it back),
    since the PRD's "solve rate, step ratio, tool-call ratio" framing needs more than one run to
    be meaningful.
    """

    timestamp: str
    question: str
    initial_subquestions: int
    total_findings: int
    rounds_used: int
    critic_rounds: int
    critic_passed_first_try: bool
    budget_exhausted: bool
    unresolved_count: int
    duration_seconds: float
    spend_usd: float = Field(
        default=0.0,
        description="What the run cost against `budgets.spend_limit_usd`. 0.0 when the model has "
        "no pricing data, in which case `total_tokens` is the meaningful figure.",
    )
    total_tokens: int = Field(default=0, description="Tokens the run consumed across every agent call.")
    mcp: dict[str, McpServerStats] = Field(
        default_factory=dict,
        description="Per-MCP-server activity, keyed by server name (PRD §11). Empty for a "
        "web-only run, which is what makes an old metrics ledger still readable.",
    )

