from __future__ import annotations

import logging
import os
from typing import Optional

from sampling.backends.base import TextGenerationBackend
from sampling.backends.mdlm_backend import MDLMBackend
from sampling.backends.dream_backend import DreamBackend
from sampling.backends.dream_diffuguard_backend import DreamDiffuGuardBackend
from sampling.backends.llada_upstream_backend import LLaDAUpstreamBackend
from sampling.backends.llada_local_backend import LLaDALocalBackend
from sampling.backends.llada_diffuguard_backend import LLaDADiffuGuardBackend
from sampling.backends.llada_dija_backend import LLaDADIJABackend

LOGGER = logging.getLogger(__name__)

_N_PER_PROMPT_DEFAULT = 8


def _n_per_prompt() -> int:
  val = os.getenv("N_PER_PROMPT")
  if val:
    try:
      return max(1, int(val))
    except ValueError:
      pass
  return _N_PER_PROMPT_DEFAULT


def _resolve_llada_variant(variant: Optional[str]) -> str:
    if variant:
        return str(variant).lower()
    env_variant = os.getenv("LLADA_VARIANT") or os.getenv("MODEL_VARIANT")
    if env_variant:
        return str(env_variant).lower()
    return "upstream"


def get_backend(family: str, variant: Optional[str] = None) -> TextGenerationBackend:
    family = str(family).lower()
    variant_name = str(variant).lower() if variant else (os.getenv("MODEL_VARIANT") or "").lower()

    # Post-hoc filter and Best-of-N: wrap the base backend for each family
    if variant_name in ("posthoc_filter", "best_of_n"):
        from sampling.backends.posthoc_filter_backend import PosthocFilterBackend, BestOfNBackend
        n = _n_per_prompt()
        # family may be null/none/empty for MDLM (which doesn't set model.family explicitly)
        effective_family = family if family not in ("none", "", "null") else "mdlm"
        if effective_family == "llada":
            inner = LLaDAUpstreamBackend()
        elif effective_family == "mdlm":
            inner = MDLMBackend()
        elif effective_family == "dream":
            inner = DreamBackend()
        else:
            raise ValueError(f"Unsupported family for {variant_name}: {family}")
        if variant_name == "posthoc_filter":
            backend = PosthocFilterBackend(inner, n=n)
        else:
            backend = BestOfNBackend(inner, n=n)
        LOGGER.warning("Selected backend: %s (family=%s, variant=%s, n=%d)", backend.__class__.__name__, family, variant_name, n)
        print("[backend] selected %s (family=%s, variant=%s, n=%d)" % (backend.__class__.__name__, family, variant_name, n), flush=True)
        return backend

    # Normalize null/empty family to mdlm (MDLM doesn't set model.family explicitly)
    if family in ("none", "", "null"):
        family = "mdlm"

    if family == "mdlm":
        if variant_name == "fk_steering":
            from sampling.backends.mdlm_fk_backend import MDLMFKBackend
            backend = MDLMFKBackend()
            LOGGER.warning("Selected backend: %s (family=%s, variant=%s)", backend.__class__.__name__, family, variant_name)
            print("[backend] selected %s (family=%s, variant=%s)" % (backend.__class__.__name__, family, variant_name), flush=True)
            return backend
        backend = MDLMBackend()
        LOGGER.warning("Selected backend: %s (family=%s, variant=%s)", backend.__class__.__name__, family, variant)
        print(
            "[backend] selected %s (family=%s, variant=%s)" % (backend.__class__.__name__, family, variant),
            flush=True,
        )
        return backend
    if family == "dream":
        if variant_name == "diffuguard":
            backend = DreamDiffuGuardBackend()
            LOGGER.warning(
                "Selected backend: %s (family=%s, variant=%s)", backend.__class__.__name__, family, variant_name
            )
            print(
                "[backend] selected %s (family=%s, variant=%s)"
                % (backend.__class__.__name__, family, variant_name),
                flush=True,
            )
            return backend
        backend = DreamBackend()
        LOGGER.warning("Selected backend: %s (family=%s, variant=%s)", backend.__class__.__name__, family, variant)
        print(
            "[backend] selected %s (family=%s, variant=%s)" % (backend.__class__.__name__, family, variant),
            flush=True,
        )
        return backend
    if family == "llada":
        variant_name = _resolve_llada_variant(variant)  # handles "upstream" default
        if variant_name == "diffuguard":
            backend = LLaDADiffuGuardBackend()
            LOGGER.warning(
                "Selected backend: %s (family=%s, variant=%s)", backend.__class__.__name__, family, variant_name
            )
            print(
                "[backend] selected %s (family=%s, variant=%s)"
                % (backend.__class__.__name__, family, variant_name),
                flush=True,
            )
            return backend
        if variant_name == "dija":
            backend = LLaDADIJABackend()
            LOGGER.warning(
                "Selected backend: %s (family=%s, variant=%s)", backend.__class__.__name__, family, variant_name
            )
            print(
                "[backend] selected %s (family=%s, variant=%s)"
                % (backend.__class__.__name__, family, variant_name),
                flush=True,
            )
            return backend
        if variant_name in ("local", "native"):
            backend = LLaDALocalBackend()
            LOGGER.warning(
                "Selected backend: %s (family=%s, variant=%s)", backend.__class__.__name__, family, variant_name
            )
            print(
                "[backend] selected %s (family=%s, variant=%s)"
                % (backend.__class__.__name__, family, variant_name),
                flush=True,
            )
            return backend
        if variant_name in ("upstream", ""):
            backend = LLaDAUpstreamBackend()
            LOGGER.warning(
                "Selected backend: %s (family=%s, variant=%s)", backend.__class__.__name__, family, variant_name
            )
            print(
                "[backend] selected %s (family=%s, variant=%s)"
                % (backend.__class__.__name__, family, variant_name),
                flush=True,
            )
            return backend
        raise ValueError(f"Unsupported LLaDA variant: {variant_name}")
    raise ValueError(f"Unsupported model family: {family}")
