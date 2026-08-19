"""Small helpers shared by the CLI and the research pipeline."""

from __future__ import annotations

import re
from datetime import UTC, datetime


def slug(text: str, max_len: int = 60) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value[:max_len] or "report"


def run_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
