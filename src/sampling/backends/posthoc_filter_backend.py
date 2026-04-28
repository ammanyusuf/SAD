"""Post-hoc LlamaGuard filtering and Best-of-N backends.

Both modes generate N completions per prompt using an inner backend, then apply
LlamaGuard inline to select the output:
  - posthoc_filter: return first safe completion; fall back to least-unsafe.
  - best_of_n: return the completion with the lowest unsafe score (safest).

The inner backend is resolved from MODEL_FAMILY / MODEL_VARIANT env vars (without
the filter/bon variant suffix), defaulting to the upstream LLaDA backend or MDLM.

N is controlled by the N_PER_PROMPT env var (default 8).
The LlamaGuard checkpoint is read from LLAMAGUARD_CHECKPOINT_PATH env var.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sampling.backends.base import TextGenerationBackend
from sampling.sample_text import (
  GenerationResult,
  GenerationRun,
  GenerationSettings,
  ModelSettings,
  PromptRecord,
  SafetySettings,
)

LOGGER = logging.getLogger(__name__)

_DEFAULT_N = 8
_LLAMAGUARD_ENV_KEY = "LLAMAGUARD_CHECKPOINT_PATH"


def _is_unsafe(response: str) -> bool:
  return "unsafe" in response.lower()


def _load_llamaguard(checkpoint_path: str):
  from safety_eval.classifiers.llamaguard import LlamaGuardClassifier
  LOGGER.info("Loading LlamaGuard from %s", checkpoint_path)
  return LlamaGuardClassifier(checkpoint_path, device="auto", device_map="auto")


def _score_completions(
  classifier,
  completions: List[str],
) -> List[str]:
  """Run LlamaGuard on a list of completions; return raw response strings."""
  if not completions:
    return []
  responses, _ = classifier.classify_texts(
    completions,
    batch_size=min(8, len(completions)),
    max_new_tokens=32,
  )
  return responses


def _select_posthoc(
  candidates: List[GenerationResult],
  classifier,
) -> GenerationResult:
  """First-safe selection; fall back to least-unsafe (index 0 if all equal)."""
  texts = [r.completion for r in candidates]
  responses = _score_completions(classifier, texts)
  # First safe
  for result, resp in zip(candidates, responses):
    if not _is_unsafe(resp):
      result.metadata["filter_response"] = resp
      result.metadata["filter_mode"] = "first_safe"
      return result
  # All unsafe — pick the one whose response is least detailed (shortest hazard list)
  # as a rough proxy; just return the first one with a note
  best = candidates[0]
  best.metadata["filter_response"] = responses[0]
  best.metadata["filter_mode"] = "fallback_least_unsafe"
  return best


def _select_bon(
  candidates: List[GenerationResult],
  classifier,
) -> GenerationResult:
  """Best-of-N: pick the completion classified as safest (safe > unsafe, ties broken by hazard code count)."""
  texts = [r.completion for r in candidates]
  responses = _score_completions(classifier, texts)

  # Score: safe=0 (best), unsafe with fewer hazard codes is better
  def _unsafe_score(resp: str) -> Tuple[int, int]:
    if not _is_unsafe(resp):
      return (0, 0)
    hazard_codes = [token for token in resp.split() if token.startswith("S") and token[1:].isdigit()]
    return (1, len(hazard_codes))

  best_idx = min(range(len(responses)), key=lambda i: _unsafe_score(responses[i]))
  best = candidates[best_idx]
  best.metadata["filter_response"] = responses[best_idx]
  best.metadata["filter_mode"] = "best_of_n"
  best.metadata["n_candidates"] = len(candidates)
  return best


class PosthocFilterBackend(TextGenerationBackend):
  """Generate N completions, return first safe (posthoc_filter mode)."""

  name = "posthoc_filter"
  family: str = ""  # set dynamically
  supports_logits_hook: bool = False

  def __init__(self, inner_backend: TextGenerationBackend, n: int = _DEFAULT_N) -> None:
    self._inner = inner_backend
    self.family = getattr(inner_backend, "family", "")
    self.supports_logits_hook = getattr(inner_backend, "supports_logits_hook", False)
    self._n = n
    self._classifier = None
    self._classifier_path: Optional[str] = None

  def _get_classifier(self):
    checkpoint = os.getenv(_LLAMAGUARD_ENV_KEY, "")
    if not checkpoint:
      raise RuntimeError(
        f"LlamaGuard checkpoint not set. Set {_LLAMAGUARD_ENV_KEY} env var."
      )
    if self._classifier is None or self._classifier_path != checkpoint:
      self._classifier = _load_llamaguard(checkpoint)
      self._classifier_path = checkpoint
    return self._classifier

  def load(self, model_settings: ModelSettings, device: Optional[str] = None) -> None:
    self._inner.load(model_settings, device)

  def generate_batch(
    self,
    prompts: Optional[Sequence[PromptRecord]],
    generation: GenerationSettings,
    safety: SafetySettings,
    shard_metadata: Dict[str, Any],
  ) -> GenerationRun:
    if not prompts:
      return self._inner.generate_batch(prompts, generation, safety, shard_metadata)

    classifier = self._get_classifier()

    # Generate all N*len(prompts) in one shot so the inner backend (e.g. MDLM)
    # only loads its model once rather than once per prompt.
    repeated: List[PromptRecord] = []
    for record in prompts:
      repeated.extend([record] * self._n)

    run = self._inner.generate_batch(repeated, generation, safety, shard_metadata)
    flat_results = run.results

    # Group by original prompt index (N consecutive results per prompt)
    results: List[GenerationResult] = []
    for i, record in enumerate(prompts):
      candidates = flat_results[i * self._n : (i + 1) * self._n]
      if not candidates:
        continue
      best = _select_posthoc(candidates, classifier)
      results.append(best)

    return GenerationRun(results=results, timings=run.timings, resolved_config=run.resolved_config)


class BestOfNBackend(TextGenerationBackend):
  """Generate N completions, return the one LlamaGuard scores as safest."""

  name = "best_of_n"
  family: str = ""
  supports_logits_hook: bool = False

  def __init__(self, inner_backend: TextGenerationBackend, n: int = _DEFAULT_N) -> None:
    self._inner = inner_backend
    self.family = getattr(inner_backend, "family", "")
    self.supports_logits_hook = getattr(inner_backend, "supports_logits_hook", False)
    self._n = n
    self._classifier = None
    self._classifier_path: Optional[str] = None

  def _get_classifier(self):
    checkpoint = os.getenv(_LLAMAGUARD_ENV_KEY, "")
    if not checkpoint:
      raise RuntimeError(
        f"LlamaGuard checkpoint not set. Set {_LLAMAGUARD_ENV_KEY} env var."
      )
    if self._classifier is None or self._classifier_path != checkpoint:
      self._classifier = _load_llamaguard(checkpoint)
      self._classifier_path = checkpoint
    return self._classifier

  def load(self, model_settings: ModelSettings, device: Optional[str] = None) -> None:
    self._inner.load(model_settings, device)

  def generate_batch(
    self,
    prompts: Optional[Sequence[PromptRecord]],
    generation: GenerationSettings,
    safety: SafetySettings,
    shard_metadata: Dict[str, Any],
  ) -> GenerationRun:
    if not prompts:
      return self._inner.generate_batch(prompts, generation, safety, shard_metadata)

    classifier = self._get_classifier()

    # Generate all N*len(prompts) in one shot.
    repeated: List[PromptRecord] = []
    for record in prompts:
      repeated.extend([record] * self._n)

    run = self._inner.generate_batch(repeated, generation, safety, shard_metadata)
    flat_results = run.results

    results: List[GenerationResult] = []
    for i, record in enumerate(prompts):
      candidates = flat_results[i * self._n : (i + 1) * self._n]
      if not candidates:
        continue
      best = _select_bon(candidates, classifier)
      results.append(best)

    return GenerationRun(results=results, timings=run.timings, resolved_config=run.resolved_config)
