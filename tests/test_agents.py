"""Agent construction (PRD §5) — the retry budgets the worker is built with.

Pydantic AI defaults every retry budget to 1, and real runs died on each half of it: a worker
was killed by two consecutive paywalled `web_fetch` 403s, and another by a single malformed
SubFinding. `WORKER_RETRIES` is the fix; these tests pin it to the constructed agent so a
refactor of `build_worker_agent` can't silently drop it back to the library default.

Asserts against the agent's private `_max_*` fields because Pydantic AI exposes no public
accessor for the resolved budgets; if these break on an upgrade, re-check how `retries=` is
normalized rather than deleting the assertion.
"""

from __future__ import annotations

from deep_research.agents import WORKER_RETRIES, build_worker_agent
from deep_research.config import load_config


def test_worker_tools_get_more_than_one_retry() -> None:
    """A 403 on `web_fetch` is routine web reality, not grounds to abandon a sub-question."""
    agent = build_worker_agent(load_config())
    assert agent._max_tool_retries == WORKER_RETRIES["tools"] > 1  # noqa: SLF001


def test_worker_output_validation_gets_a_second_attempt() -> None:
    """One malformed SubFinding from a weak model shouldn't kill the whole worker."""
    agent = build_worker_agent(load_config())
    assert agent._max_output_retries == WORKER_RETRIES["output"] > 1  # noqa: SLF001
