"""Verification of MCP-sourced citations (PRD §5b, §10 item 1).

This file exists because of a specific, live bug rather than for symmetry. Before `Source` carried
provenance, the check accepted only URLs confirmed fetched — so a Zotero item key, which is never a
URL, matched nothing, and **every library citation was deleted from every finding**. A worker that
correctly answered its sub-question from the library would have reported it as unsourced, and the
report would have said so. The `test_library_citation_*` pair is what keeps that from coming back.

The other half is the opposite failure: an identifier no call ever returned must still be dropped.
A Zotero key is 8 well-formed alphanumerics whether it is real or invented, so well-formedness
cannot be the test — only having been returned counts.
"""

from __future__ import annotations

from conftest import finding_citing, local_fetch_history, mcp_call_history

from deep_research.verify import fetched_urls, mcp_evidence, verify_finding

KEY = "ABC12345"
GHOST = "ZZZ99999"
URL = "https://example.com/doc"
TOOLS = {"search_items": "zotero", "get_item_fulltext": "zotero"}

SEARCH_RESULT = {"items": [{"key": KEY, "title": "Attention Is All You Need", "year": 2017}], "total_matched": 1}


def test_evidence_comes_from_the_return_not_the_call() -> None:
    """The identifier is in what the server sent back; the call only carried a query."""
    history = mcp_call_history("search_items", SEARCH_RESULT, args={"query": "attention"})
    assert mcp_evidence(history, TOOLS) == {"zotero": '{"items": [{"key": "ABC12345", '
                                                     '"title": "Attention Is All You Need", "year": 2017}], '
                                                     '"total_matched": 1}'}


def test_library_citation_survives_verification() -> None:
    """The regression this module exists for: a real library citation must not be dropped."""
    history = mcp_call_history("search_items", SEARCH_RESULT)
    finding = finding_citing(mcp=[("zotero", KEY)])

    verified = verify_finding(finding, fetched_urls(history), mcp_evidence(history, TOOLS))

    assert [s.identifier for s in verified.sources] == [KEY]
    assert verified.contradictions == [], "a verified library citation must leave no complaint"


def test_library_citation_is_dropped_when_no_call_returned_it() -> None:
    """Well-formed is not the same as retrieved — the point of checking at all."""
    history = mcp_call_history("search_items", SEARCH_RESULT)
    finding = finding_citing(mcp=[("zotero", GHOST)])

    verified = verify_finding(finding, fetched_urls(history), mcp_evidence(history, TOOLS))

    assert verified.sources == []
    assert any(GHOST in note for note in verified.contradictions)


def test_plain_text_return_is_searched_too() -> None:
    """`get_item_fulltext` returns a document body, not JSON — the identifier is still findable."""
    history = mcp_call_history("get_item_fulltext", f"Full text of item {KEY}: ...", args={"item_key": KEY})
    finding = finding_citing(mcp=[("zotero", KEY)])

    verified = verify_finding(finding, fetched_urls(history), mcp_evidence(history, TOOLS))
    assert [s.identifier for s in verified.sources] == [KEY]


def test_failed_mcp_call_is_not_evidence() -> None:
    history = mcp_call_history("search_items", SEARCH_RESULT, outcome="failed")
    assert mcp_evidence(history, TOOLS) == {}

    finding = finding_citing(mcp=[("zotero", KEY)])
    verified = verify_finding(finding, fetched_urls(history), mcp_evidence(history, TOOLS))
    assert verified.sources == []


def test_citation_is_checked_against_its_own_server() -> None:
    """Attribution matters: evidence from one corpus cannot verify a claim about another."""
    history = mcp_call_history("search_items", SEARCH_RESULT)
    finding = finding_citing(mcp=[("some-other-library", KEY)])

    verified = verify_finding(finding, fetched_urls(history), mcp_evidence(history, TOOLS))
    assert verified.sources == [], "a key returned by zotero must not verify a claim against another server"


def test_unrecognised_tool_contributes_no_evidence() -> None:
    """Only tools the harness mapped to a server count — a web search return is not MCP evidence."""
    history = mcp_call_history("duckduckgo_search", {"key": KEY}, args={"query": "q"})
    assert mcp_evidence(history, TOOLS) == {}


def test_web_and_mcp_sources_verify_independently_in_one_finding() -> None:
    """A worker may legitimately use both in one sub-question, and each is judged on its own evidence."""
    history = [*local_fetch_history(URL), *mcp_call_history("search_items", SEARCH_RESULT)]
    finding = finding_citing(URL, "https://example.com/never-opened", mcp=[("zotero", KEY), ("zotero", GHOST)])

    verified = verify_finding(finding, fetched_urls(history), mcp_evidence(history, TOOLS))

    assert {s.identifier for s in verified.sources} == {URL, KEY}


def test_no_configured_servers_means_no_mcp_citation_verifies() -> None:
    """The default is correct, not merely convenient: nothing returned it, so nothing verifies it."""
    finding = finding_citing(mcp=[("zotero", KEY)])
    assert verify_finding(finding, set()).sources == []


def test_short_identifier_is_refused_rather_than_matched_loosely() -> None:
    """Evidence is a substring search, so a 1-3 character 'identifier' would match anything."""
    history = mcp_call_history("search_items", {"items": [{"key": "AB"}]})
    finding = finding_citing(mcp=[("zotero", "AB")])

    verified = verify_finding(finding, fetched_urls(history), mcp_evidence(history, TOOLS))
    assert verified.sources == [], "a 2-character identifier is not evidence of anything"
