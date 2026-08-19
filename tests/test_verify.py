"""Source verification: only URLs actually fetched this run may be cited (PRD §8/§10 item 1).

The provider-shape cases exist because this check silently inverted once already. `WebFetch`
resolves to a provider-native tool where the model supports one (Anthropic) and to Pydantic AI's
local tool where it doesn't (OpenAI), and the two emit sibling part classes rather than
subclasses. Matching only the concrete `ToolCallPart`/`ToolReturnPart` therefore saw nothing on
the native path, and "no URL was fetched" is indistinguishable here from "every citation is
invented" - so every real source was dropped from every finding, on the provider the shipped
default config named. Both shapes are asserted here so that can't regress.
"""

from __future__ import annotations

import pytest
from conftest import (
    finding_citing,
    local_fetch_history,
    native_fetch_history,
    search_only_history,
)

from deep_research.verify import fetched_urls, verify_finding

URL = "https://example.com/doc"
OTHER = "https://example.com/never-opened"


@pytest.mark.parametrize(
    ("label", "history"),
    [("local tool", local_fetch_history(URL)), ("provider-native tool", native_fetch_history(URL))],
)
def test_fetch_is_recognised_in_both_provider_shapes(label: str, history: list[object]) -> None:
    assert fetched_urls(history) == {URL}, f"{label} fetch went unnoticed"


@pytest.mark.parametrize(
    ("label", "history"),
    [("local tool", local_fetch_history(URL, outcome="failed")),
     ("provider-native tool", native_fetch_history(URL, outcome="failed"))],
)
def test_failed_fetch_does_not_count_as_read(label: str, history: list[object]) -> None:
    assert fetched_urls(history) == set(), f"{label}: a failed fetch was treated as a read"


def test_search_without_fetch_is_not_a_read() -> None:
    """Seeing a URL in search results is not reading it - the bar is a successful fetch.

    This is what made every source vanish on a real run: the worker cited search hits it never
    opened. The harness was right; the worker's instructions now say so explicitly.
    """
    assert fetched_urls(search_only_history()) == set()


def test_unfetched_citation_is_dropped_and_noted() -> None:
    finding = finding_citing(URL, OTHER)
    verified = verify_finding(finding, fetched_urls(local_fetch_history(URL)))

    assert [s.identifier for s in verified.sources] == [URL]
    assert any(OTHER in note for note in verified.contradictions), (
        "a dropped source must leave a trace - silently shrinking the list hides the problem"
    )


def test_fully_verified_finding_is_returned_untouched() -> None:
    finding = finding_citing(URL)
    verified = verify_finding(finding, {URL})

    assert verified is finding, "nothing was dropped, so the finding should not be rebuilt"
    assert verified.contradictions == []


def test_finding_with_no_sources_is_left_alone() -> None:
    finding = finding_citing()
    assert verify_finding(finding, set()).contradictions == []
