"""CLI entry point for the deep-research harness (PRD §4).

Four subcommands:
- `deep-research init` — scaffold `config.yaml` + `reports/` in the current directory (the
  researcher's project folder).
- `deep-research run <question>` — run the scoping chat, then the research pipeline: parallel
  research, gap-check rounds, synthesis, and critique (PRD §6). `--auto` skips both human
  touchpoints in scoping for unattended runs (PRD §14 v3).
- `deep-research resume <state-file>` — continue an interrupted research pipeline from its last
  saved `ResearchState` checkpoint (PRD §7, §14 — resumability).
- `deep-research stats` — summarize `output.state_dir/metrics.jsonl` across past runs (PRD §11).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import ValidationError
from pydantic_ai.models import infer_model
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from deep_research.agents import (
    build_critic_agent,
    build_gap_check_agent,
    build_lead_agent,
    build_synthesis_agent,
    build_worker_agent,
)
from deep_research.config import (
    MODEL_ENV_VAR,
    AppConfig,
    bundled_default_config_text,
    find_project_config,
    load_config,
)
from deep_research.mcp import McpConfigError, open_servers, toolsets
from deep_research.mcp import stats as mcp_stats
from deep_research.models import (
    McpServerStats,
    ResearchReport,
    ResearchState,
    RunMetrics,
)
from deep_research.research import (
    RunBudget,
    ToolSources,
    budget_exhausted,
    run_research_pipeline,
)
from deep_research.scoping import run_scoping
from deep_research.tracing import span
from deep_research.util import run_timestamp, slug


def _check_models_resolvable(config: AppConfig, console: Console) -> bool:
    """Resolve every configured model up front, reporting a missing key before any work starts.

    Agents are built with `defer_model_check=True`, so without this the first sign of a missing
    or wrong-provider API key is an exception partway through scoping — after a model call has
    already been paid for. `infer_model` does the same resolution the agent would, and its error
    already names the exact environment variable, whichever provider is configured.
    """
    for role, model_string in (
        ("lead", config.model.lead),
        ("researcher", config.model.researcher),
        ("critic", config.model.critic),
    ):
        try:
            infer_model(model_string)
        except Exception as exc:  # noqa: BLE001 — any resolution failure is a config problem
            console.print(f"[red]Cannot use the configured '{role}' model ({model_string}):[/] {exc}")
            console.print(
                "[dim]Set the key it names, point this role at a provider you do have a key for "
                "in config.yaml, or override it for one run:\n"
                f"  {MODEL_ENV_VAR}=openai:gpt-5-mini deep-research ...[/]"
            )
            return False
    return True


def _configure_logfire(config: AppConfig) -> None:
    if not config.logging.logfire:
        return
    import logfire

    logfire.configure(send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()


def _metrics_path(config: AppConfig, project_dir: Path) -> Path:
    return config.state_dir(project_dir) / "metrics.jsonl"


def _print_spend(budget: RunBudget, config: AppConfig, console: Console) -> None:
    """Report what the run actually cost against the ceiling it was given."""
    spent = budget.spent_usd()
    limit = config.budgets.spend_limit_usd
    if spent:
        # Say which kind of number this is. An estimate from `prices:` is only as good as the
        # rates that were typed in, and reporting it identically to a provider-measured cost
        # would hide that distinction at exactly the moment it matters.
        source = " (estimated from config.yaml `prices`)" if budget.cost_is_estimated else ""
        console.print(f"[dim]Spend: ${spent:.4f}{source} of the ${limit:.2f} run limit "
                      f"({budget.usage.requests} model requests).[/]")
    else:
        # No pricing data for this model — the token ceiling is what actually bounded the run.
        console.print(f"[dim]Spend: not priced for this model; {budget.usage.total_tokens:,} tokens "
                      f"of the {config.budgets.total_tokens_limit:,} run limit "
                      f"({budget.usage.requests} model requests).[/]")


def _record_metrics(
    config: AppConfig,
    project_dir: Path,
    state: ResearchState,
    report: ResearchReport,
    duration_seconds: float,
    budget: RunBudget,
    mcp: dict[str, McpServerStats] | None = None,
) -> None:
    metrics = RunMetrics(
        mcp=mcp or {},
        timestamp=run_timestamp(),
        question=state.brief.question,
        initial_subquestions=len(state.brief.subquestions) or 1,
        total_findings=len(state.findings),
        rounds_used=state.round,
        critic_rounds=state.critic_rounds,
        critic_passed_first_try=state.critic_rounds == 1 and state.critic_passed is True,
        budget_exhausted=budget_exhausted(state),
        unresolved_count=len(report.unresolved),
        duration_seconds=round(duration_seconds, 2),
        spend_usd=round(budget.spent_usd(), 4),
        total_tokens=budget.usage.total_tokens,
    )
    metrics_path = _metrics_path(config, project_dir)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a") as f:
        f.write(metrics.model_dump_json() + "\n")


def _apply_mcp_args(config: AppConfig, raw: str | None) -> None:
    """Overlay `--mcp-args '<json>'` onto the configured `tool_args` (PRD §5b).

    Shape: `{"<server>": {"<tool>": {"<arg>": value}}}`, merged per tool over what `config.yaml`
    set. Naming a server or tool that isn't configured is an **error, not a no-op**: an override
    that quietly did nothing leaves a run looking scoped when it isn't, which is worse than one
    that refuses to start. (Tool names are checked against the server's real tool list later, in
    the preflight probe — that needs a connection.)

    Every problem here raises `ValueError`, including the "wrong JSON shape" ones a type-focused
    linter would rather see as `TypeError`: this is malformed *operator input*, not a programming
    error, and the caller reports any `ValueError` as a usage message and exits 1.
    """
    if not raw:
        return
    try:
        overlay = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--mcp-args is not valid JSON: {exc}") from exc
    if not isinstance(overlay, dict):
        raise ValueError('--mcp-args must be a JSON object keyed by server name, e.g. \'{"zotero": {...}}\'')  # noqa: TRY004 — malformed operator input, not a type error; see the docstring

    for server_name, tools in overlay.items():
        server = config.mcp_server(server_name)
        if server is None:
            known = ", ".join(s.name for s in config.mcp_servers) or "none configured"
            raise ValueError(f"--mcp-args names unknown MCP server {server_name!r} (configured: {known}).")
        if not isinstance(tools, dict):
            raise ValueError(f"--mcp-args entry for {server_name!r} must be an object keyed by tool name.")  # noqa: TRY004 — malformed operator input, not a type error; see the docstring
        for tool_name, tool_args in tools.items():
            if not isinstance(tool_args, dict):
                raise ValueError(f"--mcp-args entry for {server_name}.{tool_name} must be an object of arguments.")  # noqa: TRY004 — malformed operator input, not a type error; see the docstring
            server.tool_args[tool_name] = {**server.tool_args.get(tool_name, {}), **tool_args}


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if getattr(args, "breadth_budget", None) is not None:
        config.budgets.breadth_budget = args.breadth_budget
    if getattr(args, "depth_budget", None) is not None:
        config.budgets.depth_budget = args.depth_budget
    if getattr(args, "spend_limit_usd", None) is not None:
        config.budgets.spend_limit_usd = args.spend_limit_usd
    _apply_mcp_args(config, getattr(args, "mcp_args", None))
    return config


async def _research_with_tool_sources(
    config: AppConfig,
    project_dir: Path,
    state: ResearchState,
    state_path: Path,
    console: Console,
    budget: RunBudget,
) -> tuple[ResearchReport, dict[str, McpServerStats]]:
    """Open one MCP session per configured server, then run the pipeline inside it.

    The session brackets the *pipeline*, not the whole command, because that is where tools are
    used: MCP servers attach to the worker agent only (PRD §5b), and workers run only here.
    Scoping happens on a separate event loop before this, and an MCP session cannot outlive the
    loop it was opened on — so bracketing "phase 1 through 7" literally would not work.

    Only the *health check* is skipped here: `_preflight_mcp` already proved the credential works,
    and calling a tool again to re-prove it buys nothing. Tools are still listed on this session,
    because citation attribution is built from that list — skipping it left the attribution map
    empty and silently dropped every library citation.
    """
    async with open_servers(config, project_dir, console, health_check=False) as servers:
        sources = ToolSources.from_servers(servers)
        worker_agent = build_worker_agent(config, toolsets(servers))
        report = await run_research_pipeline(
            worker_agent,
            build_gap_check_agent(config),
            build_synthesis_agent(config),
            build_critic_agent(config),
            state,
            state_path,
            config,
            console,
            budget,
            sources,
        )
        return report, mcp_stats(servers)


def _preflight_mcp(config: AppConfig, project_dir: Path, console: Console) -> bool:
    """Prove every configured server works before the run spends anything (PRD §5b).

    Deliberately its own short-lived session, opened and closed before scoping: the run's real
    session belongs to the pipeline's event loop, and scoping runs on a different one. The cost is
    one extra server startup; what it buys is that a misconfigured server is discovered before any
    model call rather than after the scoping conversation.
    """
    if not config.mcp_servers:
        return True

    async def probe() -> None:
        async with open_servers(config, project_dir, console):
            pass

    try:
        asyncio.run(probe())
    except McpConfigError as exc:
        console.print(f"[red]✗ {exc}[/]")
        return False
    except Exception as exc:  # noqa: BLE001 — anything here is a startup failure, reported not raised
        console.print(f"[red]✗ MCP preflight failed:[/] {exc}")
        return False
    return True


def _save_report(
    config: AppConfig, project_dir: Path, report: ResearchReport, console: Console
) -> None:
    markdown = report.to_markdown()
    console.print(Markdown(markdown))

    report_dir = config.report_dir(project_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{run_timestamp()}-{slug(report.question)}.md"
    report_path.write_text(markdown)
    console.print(f"\n[dim]Report saved to {report_path}[/]")


def cmd_init(args: argparse.Namespace) -> int:
    console = Console()
    project_dir = Path.cwd()
    config_path = project_dir / "config.yaml"

    if config_path.exists() and not args.force:
        console.print(f"[yellow]{config_path} already exists — leaving it as-is (use --force to overwrite).[/]")
    else:
        config_path.write_text(bundled_default_config_text())
        console.print(f"[green]Wrote {config_path}[/]")

    config = load_config(project_dir=project_dir)
    report_dir = config.report_dir(project_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Created {report_dir}[/]")

    console.print(
        "\nEdit config.yaml to customize models/budgets/output, then run:\n"
        '  deep-research run "your research question"'
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    project_dir = Path.cwd()
    config = load_config(args.config, project_dir=project_dir)
    console = Console()
    console.print("[bold]Deep Research harness[/]\n")
    try:
        _apply_overrides(config, args)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        return 1

    if not _check_models_resolvable(config, console):
        return 1
    if not _preflight_mcp(config, project_dir, console):
        return 1
    _configure_logfire(config)
    if find_project_config(project_dir) is None and args.config is None:
        console.print(
            "[dim]No config.yaml found in this directory — using built-in defaults. "
            'Run "deep-research init" to create one you can customize.[/]\n'
        )

    initial_question = " ".join(args.question).strip()
    if not initial_question:
        if args.auto:
            console.print("[red]--auto requires the question as a CLI argument — there's no human to prompt.[/]")
            return 1
        initial_question = console.input("[bold green]Research question:[/] ")
    if not initial_question:
        console.print("[red]No question provided.[/]")
        return 1

    start_time = time.monotonic()
    with span(config, "deep research: {question}", question=initial_question, auto=args.auto):
        lead_agent = build_lead_agent(config)
        brief = run_scoping(lead_agent, console, config, initial_question, auto=args.auto)

        state = ResearchState(
            brief=brief,
            open_subquestions=list(brief.subquestions) or [brief.question],
        )
        state_path = config.state_dir(project_dir) / f"{run_timestamp()}-{slug(brief.question)}.json"
        console.print(f"\n[dim]Checkpointing progress to {state_path}[/]")

        console.print("\n[bold]Researching...[/] this may take a few minutes.\n")
        budget = RunBudget.from_config(config)
        mcp_stats: dict[str, McpServerStats] = {}
        try:
            report, mcp_stats = asyncio.run(
                _research_with_tool_sources(config, project_dir, state, state_path, console, budget)
            )
        except Exception as exc:  # noqa: BLE001 — surface any failure (incl. budget exceeded) to the chat
            console.print(f"[red]Research failed:[/] {exc}")
            console.print(f'[dim]Progress was saved — resume with: deep-research resume "{state_path}"[/]')
            return 1

        _record_metrics(config, project_dir, state, report, time.monotonic() - start_time, budget, mcp_stats)

    _save_report(config, project_dir, report, console)
    _print_spend(budget, config, console)
    return 0


def _raise_brief_budgets(state: ResearchState, args: argparse.Namespace) -> bool:
    """Apply `--breadth-budget`/`--depth-budget` to a resumed run's brief. True if either grew.

    Budgets live in the brief (frozen at scoping time), not in config, so overriding config on a
    resume would have no effect — the brief is what `budget_exhausted` reads. Only *raising* is
    allowed: lowering a budget below what a run already spent can't un-research anything, it
    would just deadlock the resume.
    """
    raised = False
    if args.breadth_budget is not None and args.breadth_budget > state.brief.breadth_budget:
        state.brief.breadth_budget = args.breadth_budget
        raised = True
    if args.depth_budget is not None and args.depth_budget > state.brief.depth_budget:
        state.brief.depth_budget = args.depth_budget
        raised = True
    return raised


def _requeue_unresolved(state: ResearchState, console: Console) -> None:
    """Put budget-skipped sub-questions back on the queue, now that there's budget for them."""
    remaining_slots = max(0, state.brief.breadth_budget - len(state.findings))
    requeued = state.unresolved_subquestions[:remaining_slots]
    if not requeued:
        return
    state.open_subquestions = [q for q in requeued if q not in state.open_subquestions]
    state.status = "researching"
    console.print(f"[green]Budget raised — requeuing {len(state.open_subquestions)} skipped sub-question(s).[/]")


def _load_checkpoint(
    path: Path, config: AppConfig, project_dir: Path, console: Console
) -> ResearchState | None:
    """Load a checkpoint, or explain why it couldn't be loaded and return `None`.

    Checkpoint filenames are long timestamped slugs that get copied by hand from a previous
    run's output, so a typo is the likely reason for getting here - which makes a traceback
    both alarming and unhelpful. List what's actually there instead.
    """
    try:
        return ResearchState.load(path)
    except FileNotFoundError:
        console.print(f"[red]No checkpoint at[/] {path}")
        available = _available_checkpoints(config, project_dir)
        if available:
            console.print("[dim]Checkpoints in this project folder:[/]")
            for candidate in available[-5:]:
                console.print(f"[dim]  {candidate}[/]")
        else:
            console.print(
                "[dim]No checkpoints found in this project folder — a run writes one to "
                f"{config.state_dir(project_dir)} as it goes. Did you mean to start a new "
                'run with "deep-research run"?[/]'
            )
        return None
    except OSError as exc:
        console.print(f"[red]Could not read[/] {path}: {exc.strerror or exc}")
        return None
    except ValidationError as exc:
        console.print(f"[red]{path} is not a usable checkpoint.[/]")
        console.print(f"[dim]{_first_validation_problem(exc)}[/]")
        console.print(
            "[dim]These files are written by deep-research itself; a hand-edited or truncated "
            "one can't be resumed. Start a new run instead.[/]"
        )
        return None


def _available_checkpoints(config: AppConfig, project_dir: Path) -> list[Path]:
    state_dir = config.state_dir(project_dir)
    return sorted(state_dir.glob("*.json")) if state_dir.is_dir() else []


def _first_validation_problem(exc: ValidationError) -> str:
    """One line describing the first thing wrong, rather than the full Pydantic report."""
    errors = exc.errors()
    if not errors:
        return str(exc)
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "(root)"
    return f"{location}: {first.get('msg', 'invalid')}"


def cmd_resume(args: argparse.Namespace) -> int:
    project_dir = Path.cwd()
    config = load_config(args.config, project_dir=project_dir)
    console = Console()
    try:
        _apply_overrides(config, args)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        return 1

    if not _check_models_resolvable(config, console):
        return 1
    if not _preflight_mcp(config, project_dir, console):
        return 1
    _configure_logfire(config)

    state_path = Path(args.state_file)
    state = _load_checkpoint(state_path, config, project_dir, console)
    if state is None:
        return 1

    console.print(f"[bold]Resuming research[/]: {state.brief.question}")

    raised = _raise_brief_budgets(state, args)
    if raised:
        _requeue_unresolved(state, console)

    # Resuming a finished run re-runs synthesis and critique for the same findings and the same
    # conclusion — real model calls, real money, no new information. Only worth it if there is
    # actually more work to do (a raised budget) or the user explicitly asks for a redo.
    if state.status == "done" and not state.open_subquestions and not args.force:
        console.print("[yellow]This run already completed and no work is outstanding.[/]")
        console.print(
            "[dim]Nothing to resume. Raise a budget to research what was skipped "
            "(--breadth-budget/--depth-budget), pass --force to re-run synthesis and critique "
            "on the existing findings, or start a new run.[/]"
        )
        if state.unresolved_subquestions:
            console.print(f"[dim]{len(state.unresolved_subquestions)} sub-question(s) were never researched:[/]")
            for question in state.unresolved_subquestions:
                console.print(f"[dim]  - {question}[/]")
        return 0

    start_time = time.monotonic()
    budget = RunBudget.from_config(config)
    mcp_server_stats: dict[str, McpServerStats] = {}
    with span(config, "deep research (resumed): {question}", question=state.brief.question):
        try:
            report, mcp_server_stats = asyncio.run(
                _research_with_tool_sources(config, project_dir, state, state_path, console, budget)
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Research failed:[/] {exc}")
            console.print(f'[dim]Progress was saved — resume with: deep-research resume "{state_path}"[/]')
            return 1

        _record_metrics(
            config, project_dir, state, report, time.monotonic() - start_time, budget, mcp_server_stats
        )

    _save_report(config, project_dir, report, console)
    _print_spend(budget, config, console)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    console = Console()
    project_dir = Path.cwd()
    config = load_config(args.config, project_dir=project_dir)
    metrics_path = _metrics_path(config, project_dir)

    if not metrics_path.exists():
        console.print(f"[yellow]No metrics yet at {metrics_path} — run some research first.[/]")
        return 0

    runs = [RunMetrics.model_validate_json(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not runs:
        console.print(f"[yellow]{metrics_path} exists but has no recorded runs yet.[/]")
        return 0

    n = len(runs)
    avg = lambda values: sum(values) / n
    table = Table(title=f"deep-research stats — {n} run(s)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Avg. initial sub-questions per brief", f"{avg([r.initial_subquestions for r in runs]):.1f}")
    table.add_row("Avg. rounds to convergence", f"{avg([r.rounds_used for r in runs]):.1f}")
    table.add_row("Critic pass-on-first-try rate", f"{avg([r.critic_passed_first_try for r in runs]) * 100:.0f}%")
    table.add_row("Runs that hit a budget limit", f"{sum(r.budget_exhausted for r in runs)}/{n}")
    table.add_row("Avg. unresolved items per report", f"{avg([r.unresolved_count for r in runs]):.1f}")
    table.add_row("Avg. duration", f"{avg([r.duration_seconds for r in runs]):.0f}s")
    table.add_row("Avg. spend per run", f"${avg([r.spend_usd for r in runs]):.4f}")
    table.add_row("Avg. tokens per run", f"{avg([r.total_tokens for r in runs]):,.0f}")
    console.print(table)
    _print_mcp_stats(runs, console)
    return 0


def _print_mcp_stats(runs: list[RunMetrics], console: Console) -> None:
    """Per-MCP-server activity across recorded runs (PRD §11).

    Printed as its own table, and only when there is something to print, so a web-only project's
    `stats` output is unchanged and an older metrics ledger (no `mcp` key) still reads cleanly.

    The column that answers the operator's actual question is **Cited**: a server called often and
    cited never is wrongly scoped (`tool_args` too narrow), wrongly described (`instructions`
    misleading), or wrong for the corpus. A non-zero **Dropped** is the other signal worth acting
    on — the model is citing items that server never returned.
    """
    servers = sorted({name for run in runs for name in run.mcp})
    if not servers:
        return

    table = Table(title="MCP tool sources")
    table.add_column("Server")
    table.add_column("Transport")
    table.add_column("Runs", justify="right")
    table.add_column("Calls", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Injected", justify="right")
    table.add_column("Cited", justify="right")
    table.add_column("Dropped", justify="right")

    for name in servers:
        entries = [run.mcp[name] for run in runs if name in run.mcp]
        transports = sorted({e.transport for e in entries})
        table.add_row(
            name,
            "/".join(transports),
            str(len(entries)),
            str(sum(e.calls for e in entries)),
            str(sum(e.failed_calls for e in entries)),
            str(sum(e.injected_calls for e in entries)),
            str(sum(e.citations_kept for e in entries)),
            str(sum(e.citations_dropped for e in entries)),
        )
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deep-research", description="Deep research harness CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Scaffold config.yaml and reports/ in the current directory")
    init_parser.add_argument("--force", action="store_true", help="Overwrite an existing config.yaml")
    init_parser.set_defaults(func=cmd_init)

    run_parser = subparsers.add_parser("run", help="Run the scoping chat + research pipeline")
    run_parser.add_argument("question", nargs="*", help="Initial research question (omit to be prompted)")
    run_parser.add_argument("--config", type=str, default=None, help="Path to a config.yaml (default: project/bundled)")
    run_parser.add_argument("--breadth-budget", type=int, default=None, help="Override budgets.breadth_budget")
    run_parser.add_argument("--depth-budget", type=int, default=None, help="Override budgets.depth_budget")
    run_parser.add_argument("--spend-limit-usd", type=float, default=None, help="Override budgets.spend_limit_usd")
    run_parser.add_argument(
        "--mcp-args",
        type=str,
        default=None,
        metavar="JSON",
        help='Per-run MCP tool arguments, merged over config tool_args: \'{"zotero": '
        '{"search_items": {"collection_key": "XY99ABCD"}}}\'',
    )
    run_parser.add_argument(
        "--auto",
        action="store_true",
        help="Unattended mode: no clarifying-question or brief-confirmation prompts (requires the "
        "question as a CLI argument). For scripted/scheduled invocations with no human present.",
    )
    run_parser.set_defaults(func=cmd_run)

    resume_parser = subparsers.add_parser("resume", help="Continue an interrupted run from a saved ResearchState")
    resume_parser.add_argument("state_file", help="Path to a .deep_research/<...>.json checkpoint")
    resume_parser.add_argument("--config", type=str, default=None, help="Path to a config.yaml (default: project/bundled)")
    resume_parser.add_argument(
        "--mcp-args", type=str, default=None, metavar="JSON", help="Per-run MCP tool arguments (see `run --mcp-args`)"
    )
    resume_parser.add_argument(
        "--breadth-budget", type=int, default=None,
        help="Raise the resumed run's breadth budget, requeuing sub-questions the original run skipped",
    )
    resume_parser.add_argument(
        "--depth-budget", type=int, default=None, help="Raise the resumed run's depth budget (follow-up rounds)"
    )
    resume_parser.add_argument(
        "--spend-limit-usd", type=float, default=None, help="Spend ceiling for the resumed leg of the run"
    )
    resume_parser.add_argument(
        "--force", action="store_true",
        help="Re-run synthesis and critique even when the run is already complete and nothing is outstanding",
    )
    resume_parser.set_defaults(func=cmd_resume)

    stats_parser = subparsers.add_parser("stats", help="Summarize past runs from output.state_dir/metrics.jsonl")
    stats_parser.add_argument("--config", type=str, default=None, help="Path to a config.yaml (default: project/bundled)")
    stats_parser.set_defaults(func=cmd_stats)

    return parser


def load_project_dotenv() -> str | None:
    """Load `.env` from the working directory upward, returning the file used.

    `usecwd=True` is the whole point. Bare `load_dotenv()` searches upward from the
    directory of *this file*, which is the source tree during development and
    `site-packages` once installed - so a `pipx`-installed run never sees the `.env`
    sitting next to the user's `config.yaml`, and every key silently goes missing.
    Development hid this: walking up from the source file happens to find the repo's own
    `.env`.

    Searching from the CWD matches how `config.yaml` is resolved (`load_config` starts at
    `Path.cwd()`), so both halves of a project folder are discovered the same way.
    Existing environment variables still win - `load_dotenv` does not override.
    """
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)
        return path
    return None


def main(argv: list[str] | None = None) -> int:
    # Load .env before anything reads config: model strings and API keys can both come from it,
    # and every subcommand resolves config, not just `run`.
    load_project_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
