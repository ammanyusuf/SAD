from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from sampling.sample_text import GenerationRun, GenerationSettings, ModelSettings, PromptRecord, SafetySettings, run_generation


class MDLMBackend:
    name = "mdlm"
    family = "mdlm"
    supports_logits_hook = False

    def __init__(self) -> None:
        self.model_settings: Optional[ModelSettings] = None

    def load(self, model_settings: ModelSettings, device: Optional[str] = None) -> None:
        self.model_settings = model_settings
        return None

    def generate_batch(
        self,
        prompts: Optional[Sequence[PromptRecord]],
        generation: GenerationSettings,
        safety: SafetySettings,
        shard_metadata: Dict[str, Any],
    ) -> GenerationRun:
        if self.model_settings is None:
            raise RuntimeError("MDLM backend requires load() before generate_batch().")
        return run_generation(
            prompts=prompts,
            model=self.model_settings,
            generation=generation,
            safety=safety,
            shard_metadata=shard_metadata,
        )
