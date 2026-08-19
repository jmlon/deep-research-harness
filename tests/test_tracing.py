"""Tracing must be inert when it's off, and must not be the reason a run fails.

Development runs with logfire on, so the disabled path is the one that goes untested and then
breaks in the configuration someone actually deploys.
"""

from __future__ import annotations

from deep_research.config import AppConfig
from deep_research.tracing import NullSpan, Span, span


def test_null_span_satisfies_the_span_protocol() -> None:
    """The point of the Protocol: the no-op can't drift from what callers use."""
    assert isinstance(NullSpan(), Span)


def test_span_is_a_no_op_when_tracing_is_disabled(config: AppConfig) -> None:
    config.logging.logfire = False
    with span(config, "worker: {subquestion}", subquestion="a", round=0) as sp:
        assert isinstance(sp, NullSpan)
        # Exactly the calls the pipeline makes; none may raise with tracing off.
        sp.set_attribute("error", "boom")
        sp.set_attributes({"confidence": "high", "sources_kept": 2})


def test_span_does_not_swallow_exceptions(config: AppConfig) -> None:
    """A span is for observing work, not for changing whether it failed."""
    config.logging.logfire = False
    try:
        with span(config, "work"):
            raise RuntimeError("boom")
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("the exception was swallowed")
