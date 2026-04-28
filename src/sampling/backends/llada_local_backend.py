"""Experimental local LLaDA backend using the repo's own generation engine.

NOTE: This backend was NOT used in the paper experiments. All paper results for LLaDA
use LLaDAUpstreamBackend (model.variant=upstream), which calls the official upstream
generate() function. This backend uses llada_engine.py — a local reimplementation that
gives finer control over the denoising trajectory but is not guaranteed to be identical
to the upstream. Use scripts/debug_llada_parity.py to verify parity if needed.

Select this backend with: model.family=llada model.variant=local
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from sampling.backends.base import TextGenerationBackend
from sampling.llada_engine import LLaDAGenerationEngine
from sampling.sample_text import (
    GenerationRun,
    GenerationSettings,
    ModelSettings,
    PromptRecord,
    SafetySettings,
)


class LLaDALocalBackend:
    name = "llada_local"
    family = "llada"
    supports_logits_hook = False

    def __init__(self) -> None:
        self.model_settings: Optional[ModelSettings] = None

    def load(self, model_settings: ModelSettings, device: Optional[str] = None) -> None:
        self.model_settings = model_settings

    def generate_batch(
        self,
        prompts: Optional[Sequence[PromptRecord]],
        generation: GenerationSettings,
        safety: SafetySettings,
        shard_metadata: Dict[str, Any],
    ) -> GenerationRun:
        engine = LLaDAGenerationEngine(
            prompts=prompts,
            model=self._require_model_settings(),
            generation=generation,
            safety=safety,
            shard_metadata=shard_metadata,
        )
        return engine.run()

    def _require_model_settings(self) -> ModelSettings:
        if self.model_settings is None:
            raise RuntimeError("LLaDA local backend requires load() before generate_batch().")
        return self.model_settings
