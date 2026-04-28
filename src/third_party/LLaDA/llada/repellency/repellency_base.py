"""Lightweight registry shim for LLaDA repellency methods."""

from third_party.mdlm.repellency.repellency_methods_fast import (
    RepellencyMethod,
    get_repellency_method,
    register_conditioning_method,
)

__all__ = ["RepellencyMethod", "register_conditioning_method", "get_repellency_method"]

