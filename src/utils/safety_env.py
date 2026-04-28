"""Helpers for building SafetySettings from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from sampling.sample_text import SafetySettings


def _env_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    return default


def _env_int(value: Optional[str]) -> Optional[int]:
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_float(value: Optional[str]) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _env_path(value: Optional[str]) -> Optional[Path]:
    if value in (None, "", "null"):
        return None
    return Path(value).expanduser()


def safety_settings_from_env(prefix: str = "") -> SafetySettings:
    enabled = _env_bool(os.getenv(f"{prefix}SAFETY_ENABLED"), False)
    eta = _env_float(os.getenv(f"{prefix}SAFETY_ETA"))
    scale = _env_float(os.getenv(f"{prefix}SAFETY_SCALE"))
    t_start = _env_int(os.getenv(f"{prefix}SAFETY_T_START"))
    t_end = _env_int(os.getenv(f"{prefix}SAFETY_T_END"))
    auto_build_unsafe = _env_bool(
        os.getenv(f"{prefix}SAFETY_AUTO_BUILD_UNSAFE_ARTIFACTS"), True
    )
    unsafe_artifacts = _env_path(os.getenv(f"{prefix}UNSAFE_ARTIFACTS"))
    unsafe_artifact_root = _env_path(os.getenv(f"{prefix}UNSAFE_ARTIFACT_ROOT"))
    unsafe_artifact_name = os.getenv(f"{prefix}UNSAFE_ARTIFACT_NAME")
    if unsafe_artifact_name in ("", "null"):
        unsafe_artifact_name = None
    return SafetySettings(
        enabled=enabled,
        eta=eta,
        scale=scale,
        unsafe_artifacts=unsafe_artifacts,
        unsafe_artifact_root=unsafe_artifact_root,
        unsafe_artifact_name=unsafe_artifact_name,
        t_start=t_start,
        t_end=t_end,
        auto_build_unsafe_artifacts=auto_build_unsafe,
    )
