"""Deep research harness: chat-scoped, parallel, budgeted research into a cited report.

See ../PRD.md for the design and ../README.md for usage.

Version note: the package version lives in `pyproject.toml` and is read back here from the
installed metadata. Module docstrings deliberately do *not* stamp themselves with a milestone
("v1", "v2", ...): those stamps went stale the moment the next milestone landed — `models.py`
claimed "no critic yet" while `CriticVerdict` sat 80 lines below it — and nothing ever fails
when they drift. Modules describe what they do and cite stable PRD sections; the milestone
history lives in PRD §14, where it is actually maintained.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deep-research-harness")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
