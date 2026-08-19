"""The declared dependency ranges must match what this code is actually tested against.

This ships via `pipx install`, which resolves from `pyproject.toml` and never reads `uv.lock`.
So the ranges - not the lockfile - are what end users get, and a floor-only pin means they
silently receive whatever released most recently. That is not hypothetical: `rich>=14` was
resolving 15 and `logfire>=3.14.1` was resolving 4, each a major version past anything this
code had ever run against, and nothing anywhere noticed.

These tests close the loop from the other side: whatever the dev environment installs and runs
the suite against must satisfy the ranges we publish. Bumping the lockfile past a declared
bound now fails here instead of shipping an untested combination.
"""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def declared_requirements() -> list[Requirement]:
    raw = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    return [Requirement(r) for r in raw]


def declared_extras() -> list[Requirement]:
    """Requirements from `[project.optional-dependencies]` — the bundled MCP servers (PRD §4a).

    Held to the same upper-bound rule as the mandatory ones: an extra is installed by an end user
    from these ranges too, so a floor-only pin there ships the same untested-version promise.
    """
    extras = tomllib.loads(PYPROJECT.read_text())["project"].get("optional-dependencies", {})
    return [Requirement(r) for group in extras.values() for r in group]


@pytest.mark.parametrize("requirement", declared_requirements(), ids=lambda r: r.name)
def test_installed_version_satisfies_the_declared_range(requirement: Requirement) -> None:
    installed = metadata.version(requirement.name)
    assert requirement.specifier.contains(installed, prereleases=True), (
        f"{requirement.name} {installed} is installed and the suite passes against it, but "
        f"pyproject declares {requirement.specifier} - so users would get an untested version"
    )


@pytest.mark.parametrize(
    "requirement", declared_requirements() + declared_extras(), ids=lambda r: r.name
)
def test_every_dependency_has_an_upper_bound(requirement: Requirement) -> None:
    """A floor-only pin is an open-ended promise about versions that don't exist yet."""
    operators = {spec.operator for spec in requirement.specifier}
    assert operators & {"<", "<=", "==", "~="}, (
        f"{requirement.name} has no upper bound: a pipx install will take any future release, "
        "including the one that changes an API this code depends on"
    )


@pytest.mark.parametrize("requirement", declared_extras(), ids=lambda r: r.name)
def test_an_installed_extra_satisfies_its_declared_range(requirement: Requirement) -> None:
    """Same check as for mandatory deps, but an extra is legitimately absent from a dev env.

    `pydantic-zotero-mcp` is not a hard dependency and `uv sync` will not install it, so this skips rather
    than fails when it isn't there — it only has something to say once someone has injected it.
    """
    try:
        installed = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        pytest.skip(f"{requirement.name} is not installed in this environment")
    assert requirement.specifier.contains(installed, prereleases=True), (
        f"{requirement.name} {installed} is installed and the suite passes against it, but "
        f"pyproject declares {requirement.specifier}"
    )
