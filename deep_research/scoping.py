"""Phase 1: clarify & scope, driven by the CLI chat interface (PRD §4, §6.1).

The lead agent alternates between asking clarifying questions and proposing a `BriefDraft`. Once
it proposes a draft, the human confirms or asks for a revision before research starts — this is
the human-in-the-loop touchpoint the PRD resolves in favor of a chat interface rather than a
separate approval system.

`auto=True` (the `--auto` CLI flag, PRD §4a/§14 v3 — a no-human scoping fallback for unattended
runs) skips both of those human touchpoints: clarifying questions get a canned "no one's here,
proceed on your best assumption" nudge instead of a real answer, and a proposed brief is accepted
immediately. `AUTO_MAX_CLARIFICATIONS` is a hard backstop — an unattended run must terminate even
if the lead agent doesn't respect the nudge, so after enough rounds we bypass it entirely and
build a minimal brief straight from the raw question.
"""

from __future__ import annotations

from pydantic_ai import Agent
from rich.console import Console
from rich.panel import Panel

from deep_research.config import AppConfig
from deep_research.models import BriefDraft, ClarifyingQuestion, ResearchBrief

LeadAgent = Agent[None, ClarifyingQuestion | BriefDraft]
"""The lead agent's output type is the whole protocol of this phase: it either asks one more
question or hands over a draft brief, and `run_scoping` branches on which it got."""

AUTO_MAX_CLARIFICATIONS = 3

AUTO_FALLBACK_PROMPT = (
    "No human is available to answer right now (unattended run) — don't ask another clarifying "
    "question. Make your best reasonable assumption, state it explicitly in `assumptions`, and "
    "respond with a BriefDraft now."
)


def _print_brief(console: Console, draft: BriefDraft) -> None:
    lines = [f"[bold]Question:[/] {draft.question}"]
    if draft.subquestions:
        lines.append("\n[bold]Sub-questions:[/]")
        lines.extend(f"  - {q}" for q in draft.subquestions)
    if draft.assumptions:
        lines.append("\n[bold]Assumptions:[/]")
        lines.extend(f"  - {a}" for a in draft.assumptions)
    console.print(Panel("\n".join(lines), title="Proposed research brief", border_style="cyan"))


def run_scoping(
    lead_agent: LeadAgent,
    console: Console,
    config: AppConfig,
    initial_question: str,
    *,
    auto: bool = False,
) -> ResearchBrief:
    """Run the clarify/confirm loop to completion and return a confirmed `ResearchBrief`."""
    message_history = None
    user_prompt = initial_question
    clarifications = 0

    while True:
        result = lead_agent.run_sync(user_prompt, message_history=message_history)
        message_history = result.all_messages()
        output = result.output

        if isinstance(output, ClarifyingQuestion):
            if auto:
                clarifications += 1
                console.print(f"[dim]Lead agent asked (no human available): {output.question}[/]")
                if clarifications > AUTO_MAX_CLARIFICATIONS:
                    console.print(
                        "[yellow]Auto mode: too many clarifying questions — proceeding directly "
                        "from the raw question instead.[/]"
                    )
                    return ResearchBrief(
                        question=initial_question,
                        subquestions=[],
                        assumptions=[
                            (
                                "Unattended run: proceeded directly from the raw question "
                                "after the lead agent's clarifying questions went unanswered."
                            )
                        ],
                        depth_budget=config.budgets.depth_budget,
                        breadth_budget=config.budgets.breadth_budget,
                    )
                user_prompt = AUTO_FALLBACK_PROMPT
                continue

            console.print(f"[bold cyan]Lead agent:[/] {output.question}")
            user_prompt = console.input("[bold green]You:[/] ")
            continue

        # output is a BriefDraft — confirm with the human before research starts (skipped in auto mode).
        _print_brief(console, output)

        if auto:
            console.print("[dim]Auto mode: proceeding without confirmation.[/]")
            return ResearchBrief.from_draft(
                output,
                depth_budget=config.budgets.depth_budget,
                breadth_budget=config.budgets.breadth_budget,
            )

        answer = console.input(
            "[bold yellow]Proceed with this brief? \\[Y/n, or type a revision][/] "
        ).strip()
        if answer.lower() in ("", "y", "yes"):
            return ResearchBrief.from_draft(
                output,
                depth_budget=config.budgets.depth_budget,
                breadth_budget=config.budgets.breadth_budget,
            )
        if answer.lower() in ("n", "no"):
            user_prompt = "The brief isn't right yet — please ask me what needs to change."
        else:
            user_prompt = f"Please revise the brief: {answer}"
