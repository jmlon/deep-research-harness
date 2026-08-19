"""The data model: checkpoint round-tripping and the one user-facing rendering (PRD §7/§7a)."""

from __future__ import annotations

import json
from pathlib import Path

from deep_research.models import (
    BriefDraft,
    ReportSection,
    ResearchBrief,
    ResearchReport,
    ResearchState,
    Source,
    SubFinding,
)
from deep_research.util import slug


def make_state(**kwargs: object) -> ResearchState:
    brief = ResearchBrief(
        question="Q?", subquestions=["a"], assumptions=[], depth_budget=2, breadth_budget=8
    )
    return ResearchState(brief=brief, **kwargs)


def test_checkpoint_round_trips(tmp_path: Path) -> None:
    state = make_state(
        findings={"a": SubFinding(subquestion="a", answer="ans", confidence="high")},
        unresolved_subquestions=["b"],
        round=2,
        status="critiquing",
        critic_rounds=1,
        critic_passed=False,
    )
    path = tmp_path / "s.json"
    state.save(path)
    reloaded = ResearchState.load(path)

    assert reloaded.model_dump() == state.model_dump()


def test_a_checkpoint_written_before_unresolved_tracking_still_loads(tmp_path: Path) -> None:
    """Resumability is the point of these files; a schema addition must not orphan old ones."""
    state = make_state()
    raw = json.loads(state.model_dump_json())
    del raw["unresolved_subquestions"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(raw))

    assert ResearchState.load(path).unresolved_subquestions == []


def test_save_creates_missing_parent_directories(tmp_path: Path) -> None:
    """`state_dir` usually doesn't exist yet on a fresh project folder."""
    path = tmp_path / "nested" / "deeper" / "s.json"
    make_state().save(path)
    assert path.is_file()


def test_brief_takes_budgets_from_config_not_from_the_model() -> None:
    """PRD §8: budgets are configuration, not something the lead agent gets to choose."""
    draft = BriefDraft(question="Q?", subquestions=["a"], assumptions=["x"])
    brief = ResearchBrief.from_draft(draft, depth_budget=3, breadth_budget=9)

    assert (brief.depth_budget, brief.breadth_budget) == (3, 9)
    assert brief.subquestions == ["a"] and brief.assumptions == ["x"]


def test_markdown_renders_sections_sources_assumptions_and_unresolved() -> None:
    report = ResearchReport(
        question="What changed?",
        summary="A summary.",
        sections=[
            ReportSection(
                heading="Section one",
                content="Body text.",
                sources=[Source(url="https://e.com", title="Ref", quoted_snippet="quote")],
            )
        ],
        assumptions=["assumed a thing"],
        unresolved=["never got to this"],
    )
    md = report.to_markdown()

    assert md.startswith("# What changed?")
    assert "## Section one" in md
    assert "[Ref](https://e.com)" in md and "quote" in md
    assert "## Assumptions" in md and "assumed a thing" in md
    assert "## Unresolved" in md and "never got to this" in md
    assert md.endswith("\n") and not md.endswith("\n\n\n")


def test_a_checkpoint_written_before_source_provenance_still_loads(tmp_path: Path) -> None:
    """`Source.url` became `kind`/`identifier`/`server` when MCP sources arrived (PRD §7).

    These files are what `deep-research resume` reads, so renaming the field without a
    compatibility path would strand every checkpoint on disk — failing to resume a long,
    already-paid-for run.
    """
    state = make_state(findings={"a": SubFinding(subquestion="a", answer="ans", confidence="high")})
    raw = json.loads(state.model_dump_json())
    raw["findings"]["a"]["sources"] = [{"url": "https://old.example", "title": "T", "quoted_snippet": "s"}]
    path = tmp_path / "pre-mcp.json"
    path.write_text(json.dumps(raw))

    loaded = next(iter(ResearchState.load(path).findings.values())).sources[0]
    assert (loaded.kind, loaded.identifier, loaded.server) == ("web", "https://old.example", None)


def test_an_mcp_source_is_attributed_rather_than_linked() -> None:
    """A `zotero://` URI resolves only inside that server, and the item's own `url` is a page the
    worker never read — so a library citation names its identifier instead (PRD §7a)."""
    source = Source(
        kind="mcp", server="zotero", identifier="ABC12345", title="A Paper", quoted_snippet="quote"
    )
    rendered = source.render_markdown()

    assert "ABC12345" in rendered and "zotero" in rendered
    assert "](" not in rendered, "an MCP source must not render as a link"


def test_report_renders_web_and_mcp_citations_side_by_side() -> None:
    report = ResearchReport(
        question="Q",
        summary="S",
        sections=[
            ReportSection(
                heading="Both",
                content="Body.",
                sources=[
                    Source(identifier="https://e.com", title="Web Ref", quoted_snippet="w"),
                    Source(kind="mcp", server="zotero", identifier="ABC12345", title="Lib Ref", quoted_snippet="l"),
                ],
            )
        ],
    )
    md = report.to_markdown()

    assert "[Web Ref](https://e.com)" in md
    assert "Lib Ref — zotero item `ABC12345`" in md


def test_markdown_omits_empty_optional_sections() -> None:
    md = ResearchReport(question="Q", summary="S").to_markdown()
    assert "## Assumptions" not in md and "## Unresolved" not in md


def test_slug_is_filesystem_safe_and_bounded() -> None:
    assert slug("What changed in Django 5.0?") == "what-changed-in-django-5-0"
    assert len(slug("x" * 200)) <= 60
    assert slug("???") == "report", "a question with no usable characters still needs a filename"


def test_package_version_comes_from_installed_metadata() -> None:
    """The one place a version is stated.

    Module docstrings used to carry milestone stamps ("v1", "v2") that nothing verified, so they
    drifted — `models.py` claimed "no critic yet" while `CriticVerdict` was defined below it.
    """
    import deep_research

    assert deep_research.__version__ != "0.0.0+unknown", "package metadata should be readable"
    assert deep_research.__version__[0].isdigit()
