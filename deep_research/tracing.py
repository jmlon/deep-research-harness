"""Tracing helper (PRD §11): a single logfire trace should cover the whole research job, not just
one phase. `logfire.instrument_pydantic_ai()` (wired up in cli.py) already gives a detailed span
per agent call; what's missing without this module is the *grouping* — which round a worker call
belongs to, which sub-question it answered, which critique attempt found what. `span()` adds that
layer; it's a no-op when `logging.logfire` is off in config, so callers don't need to branch.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

from deep_research.config import AppConfig


@runtime_checkable
class Span(Protocol):
    """The slice of `logfire.LogfireSpan` this harness actually uses.

    Spelling it out as a Protocol is what makes the no-op below verifiable. `NullSpan` used to
    claim it had the "same interface" as a real span while implementing two methods with
    `*args, **kwargs` — so a caller reaching for anything else would work with tracing on and
    fail only with tracing off, which is the configuration nobody runs in development.

    `runtime_checkable` so a test can assert the no-op conforms; it only checks that the method
    names exist, which is enough to catch the drift this guards against.
    """

    def set_attribute(self, key: str, value: Any) -> None: ...

    def set_attributes(self, attributes: Mapping[str, Any]) -> None: ...


class NullSpan:
    """Stand-in for a real span when tracing is disabled: satisfies `Span`, records nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        return None


@contextmanager
def span(config: AppConfig, msg_template: str, **attributes: Any) -> Iterator[Span]:
    """Group the work inside into one logfire span, or do nothing if tracing is off.

    `attributes` is genuinely open-ended — it carries whatever identifies this unit of work
    (round number, sub-question, attempt) into the trace — so `Any` is the honest annotation.
    """
    if not config.logging.logfire:
        yield NullSpan()
        return

    import logfire

    with logfire.span(msg_template, **attributes) as active_span:
        yield active_span
