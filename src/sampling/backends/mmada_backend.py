from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from sampling.sample_text import GenerationRun, GenerationSettings, ModelSettings, PromptRecord, SafetySettings


class MMADABackend:
    name = "mmada"
    family = "mmada"
    supports_logits_hook = False

    def load(self, model_settings: ModelSettings, device: Optional[str] = None) -> None:
        raise NotImplementedError("MMaDA backend is not implemented in this repo.")

    def generate_batch(
        self,
        prompts: Optional[Sequence[PromptRecord]],
        generation: GenerationSettings,
        safety: SafetySettings,
        shard_metadata: Dict[str, Any],
    ) -> GenerationRun:
        raise NotImplementedError("MMaDA backend is not implemented in this repo.")
