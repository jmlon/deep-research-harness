"""`RunBudget`: one accumulator for the run, plus a guaranteed allowance for the write-up.

Two distinct bugs have lived here, and both are asserted against below:

1. A fresh `UsageLimits` was built per call, so every one of the ~12 calls in a run got the
   full budget and a "$5 limit" could bill many times that.
2. The write-up was funded by reserving a *fraction* of the ceiling. Research overshoots
   (measured: 303k tokens against a 170k ceiling), so the reserve was already gone by the time
   synthesis started - the run paid for the research and produced no report at all.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage, UsageLimits

from deep_research.config import AppConfig, ModelPrice
from deep_research.research import RunBudget


def spent(budget: RunBudget, tokens: int, usd: str = "0") -> None:
    """Simulate having burned `tokens` (and optionally `usd`) so far."""
    budget.usage = RunUsage(input_tokens=tokens, output_tokens=0, cost=Decimal(usd) if usd != "0" else None)


def test_research_is_capped_by_the_configured_ceiling(config: AppConfig) -> None:
    budget = RunBudget.from_config(config)
    assert budget.research_limits.total_tokens_limit == config.budgets.total_tokens_limit
    assert budget.research_limits.cost_limit == Decimal(str(config.budgets.spend_limit_usd))


def test_report_allowance_is_granted_on_top_of_actual_usage(config: AppConfig) -> None:
    budget = RunBudget.from_config(config)
    spent(budget, 400_000)
    budget.open_report_allowance()

    headroom = budget.report_limits.total_tokens_limit - budget.usage.total_tokens
    assert headroom == config.budgets.report_token_allowance


def test_allowance_survives_research_overshooting_the_ceiling(config: AppConfig) -> None:
    """The regression that matters: an overrun *before* the write-up must not consume it.

    A percentage-of-ceiling reserve fails this - which is how a real run ended up with 303k
    tokens spent and no report written.
    """
    budget = RunBudget.from_config(config)
    overshoot = config.budgets.total_tokens_limit + 300_000
    spent(budget, overshoot)
    budget.open_report_allowance()

    headroom = budget.report_limits.total_tokens_limit - budget.usage.total_tokens
    assert headroom == config.budgets.report_token_allowance, (
        "the write-up allowance was eaten by a research overrun that preceded it"
    )


def test_each_write_up_pass_gets_its_own_allowance(config: AppConfig) -> None:
    """The critic can send the run back for more research; the next report needs funding too."""
    budget = RunBudget.from_config(config)
    spent(budget, 100_000)
    budget.open_report_allowance()
    first = budget.report_limits.total_tokens_limit

    spent(budget, 500_000)  # another research round happened
    budget.open_report_allowance()
    second = budget.report_limits.total_tokens_limit

    assert second > first
    assert second - budget.usage.total_tokens == config.budgets.report_token_allowance


def test_usd_allowance_tracks_actual_cost(config: AppConfig) -> None:
    budget = RunBudget.from_config(config)
    spent(budget, 1_000, usd="3.50")
    budget.open_report_allowance()

    assert budget.report_limits.cost_limit == Decimal("3.50") + Decimal(
        str(config.budgets.report_spend_allowance_usd)
    )


def test_spent_usd_is_zero_for_an_unpriced_model() -> None:
    """`genai-prices` doesn't know every model; an unpriceable run must not crash the report."""
    assert RunBudget(usage=RunUsage(input_tokens=10, output_tokens=10)).spent_usd() == 0.0


def test_a_shared_budget_accumulates_across_calls() -> None:
    """One `RunUsage` for the whole run is what makes the ceiling a run ceiling at all."""
    budget = RunBudget(
        usage=RunUsage(), research_limits=UsageLimits(total_tokens_limit=1_000), report_token_allowance=10
    )
    budget.usage.incr(RunUsage(input_tokens=100, output_tokens=50))
    budget.usage.incr(RunUsage(input_tokens=200, output_tokens=50))
    assert budget.usage.total_tokens == 400


def test_research_request_ceiling_comes_from_config_not_the_library_default(config: AppConfig) -> None:
    """Regression: an unset `request_limit` silently inherits Pydantic AI's default of 50.

    Every call in a run shares one accumulator, so that default is a run-wide cap. One
    sub-question costs several requests (worker turns plus the researcher it delegates to), so
    a single round of workers exhausted it while tokens and USD were nearly untouched - and the
    run died before it could write anything.
    """
    budget = RunBudget.from_config(config)

    assert budget.research_limits.request_limit == config.budgets.request_limit
    assert budget.research_limits.request_limit != UsageLimits().request_limit, (
        "the configured ceiling is indistinguishable from the library default"
    )
    assert budget.research_limits.request_limit >= 70, (
        "a default breadth of 8 needs ~70 requests before any headroom"
    )


def test_report_request_allowance_is_granted_on_top_of_research_requests(config: AppConfig) -> None:
    """The write-up must be able to make requests even after research spent the ceiling.

    This is the failure the token/USD allowances already guarded against, reproduced for
    requests: research burned the run-wide request cap, so synthesis could not issue one call
    and the run reported the *token* allowance as too small.
    """
    budget = RunBudget.from_config(config)
    budget.usage = RunUsage(input_tokens=10, output_tokens=10, requests=config.budgets.request_limit)
    budget.open_report_allowance()

    headroom = budget.report_limits.request_limit - budget.usage.requests
    assert headroom == config.budgets.report_request_allowance


def test_each_write_up_pass_gets_its_own_request_allowance(config: AppConfig) -> None:
    """A critic-triggered extra round bills research, then the next report needs a fresh allowance."""
    budget = RunBudget.from_config(config)
    budget.usage = RunUsage(input_tokens=10, output_tokens=10, requests=40)
    budget.open_report_allowance()
    first = budget.report_limits.request_limit

    budget.usage = RunUsage(input_tokens=20, output_tokens=20, requests=90)
    budget.open_report_allowance()

    assert budget.report_limits.request_limit > first
    assert budget.report_limits.request_limit - 90 == config.budgets.report_request_allowance


def test_an_unset_request_allowance_leaves_the_write_up_unconstrained() -> None:
    """0 means "not configured", not "zero requests".

    A `RunBudget` built directly - as library callers and tests do - would otherwise get
    `baseline + 0`, i.e. permission to make no requests at all, recreating the pay-for-research-
    and-get-nothing failure the allowances exist to prevent. Config forbids 0 (`gt=0`), so this
    path is only reachable by direct construction.
    """
    budget = RunBudget(usage=RunUsage(requests=4), report_token_allowance=100_000)
    budget.open_report_allowance()

    assert budget.report_limits.request_limit is None


# --------------------------------------------------------- configured prices (config `prices:`)


def priced_budget(spend_limit: str = "1.00") -> RunBudget:
    """A budget for an unpriced model, with a configured rate of $1/1M in, $10/1M out."""
    return RunBudget(
        usage=RunUsage(),
        prices={"openrouter:some/new-model": ModelPrice(input_usd_per_1m=1.0, output_usd_per_1m=10.0)},
        report_spend_allowance_usd=Decimal("0.50"),
        _configured_spend_limit_usd=Decimal(spend_limit),
    )


def burn(budget: RunBudget, model: str, input_tokens: int, output_tokens: int) -> None:
    """Simulate one call: usage accrues, then the call is billed against configured prices."""
    before_in, before_out = budget.usage.input_tokens, budget.usage.output_tokens
    budget.usage = RunUsage(
        input_tokens=before_in + input_tokens,
        output_tokens=before_out + output_tokens,
        requests=budget.usage.requests + 1,
    )
    budget.charge(model, before_in, before_out)


def test_configured_prices_cost_an_otherwise_unpriced_model() -> None:
    budget = priced_budget()
    burn(budget, "openrouter:some/new-model", 1_000_000, 100_000)

    assert budget.spent_usd() == pytest.approx(2.0)  # $1 input + $1 output
    assert budget.cost_is_estimated


def test_only_the_delta_of_each_call_is_billed() -> None:
    """Every agent shares one accumulator, so billing the total each time would compound it."""
    budget = priced_budget()
    burn(budget, "openrouter:some/new-model", 1_000_000, 0)
    burn(budget, "openrouter:some/new-model", 1_000_000, 0)

    assert budget.spent_usd() == pytest.approx(2.0), "second call re-billed the first call's tokens"


def test_a_model_without_a_configured_price_is_not_billed() -> None:
    budget = priced_budget()
    burn(budget, "anthropic:some-other-model", 5_000_000, 5_000_000)

    assert budget.spent_usd() == 0.0
    assert not budget.cost_is_estimated


def test_measured_pricing_always_wins_over_a_configured_rate() -> None:
    """A hand-entered number must never override what the provider actually charged."""
    budget = priced_budget()
    budget.usage = RunUsage(input_tokens=1_000_000, output_tokens=0, cost=Decimal("7.77"))
    budget.charge("openrouter:some/new-model", 0, 0)

    assert budget.spent_usd() == pytest.approx(7.77)
    assert not budget.cost_is_estimated


def test_research_stops_when_configured_prices_reach_the_spend_limit() -> None:
    budget = priced_budget(spend_limit="1.00")
    burn(budget, "openrouter:some/new-model", 900_000, 0)
    budget.enforce_configured_spend(report=False)  # $0.90 - still inside

    burn(budget, "openrouter:some/new-model", 200_000, 0)  # now $1.10
    with pytest.raises(UsageLimitExceeded) as excinfo:
        budget.enforce_configured_spend(report=False)
    assert "budgets.spend_limit_usd" in str(excinfo.value)
    assert "genai-prices has no data" in str(excinfo.value), "must say why an estimate was used"


def test_the_write_up_allowance_is_on_top_of_estimated_research_cost() -> None:
    """The same guarantee the measured path gets: research overspend cannot eat the report."""
    budget = priced_budget(spend_limit="1.00")
    burn(budget, "openrouter:some/new-model", 3_000_000, 0)  # $3.00, well past the ceiling
    budget.open_report_allowance()

    budget.enforce_configured_spend(report=True)  # allowance is fresh, so the report may run

    burn(budget, "openrouter:some/new-model", 400_000, 0)  # $0.40 of a $0.50 allowance
    budget.enforce_configured_spend(report=True)

    burn(budget, "openrouter:some/new-model", 200_000, 0)  # now $0.60, over
    with pytest.raises(UsageLimitExceeded) as excinfo:
        budget.enforce_configured_spend(report=True)
    assert "budgets.report_spend_allowance_usd" in str(excinfo.value)


def test_no_configured_prices_means_no_estimated_enforcement() -> None:
    """Unchanged behaviour for everyone who has not configured prices."""
    budget = RunBudget(usage=RunUsage(input_tokens=9_000_000), _configured_spend_limit_usd=Decimal("0.01"))
    budget.enforce_configured_spend(report=False)
    assert budget.spent_usd() == 0.0
