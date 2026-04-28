from __future__ import annotations

from typing import Optional


def coerce_float(value: object, field_name: str, allow_none: bool = False) -> Optional[float]:
    """
    Convert a config value to float, optionally allowing None, and fail fast with context.
    """
    if value is None:
        if allow_none:
            return None
        raise SystemExit(f"{field_name} must be numeric (got None).")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{field_name} must be numeric (got {value!r}).") from exc


def safe_float(value: object) -> Optional[float]:
    """
    Convert a runtime value to float, returning None when parsing fails.
    Useful for per-row fields where a single bad value should be skipped, not crash the job.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
