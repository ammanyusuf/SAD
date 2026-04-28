"""Expose semantic gating helpers for LLaDA sampling."""

from unsafe_prep.semantic_utils import masked_mean_pool  # type: ignore

__all__ = ["masked_mean_pool"]

