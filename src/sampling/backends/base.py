from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Sequence

from sampling.sample_text import GenerationRun, GenerationSettings, ModelSettings, PromptRecord, SafetySettings

class LogitsHook(Protocol):
    def __call__(
        self,
        logits: Any,
        *,
        x: Any,
        t: int,
        mask_index: Any,
        prompt_index: Any,
        attention_mask: Any,
        extra: Dict[str, Any],
    ) -> Any:
        ...


class TextGenerationBackend(Protocol):
    name: str
    family: str
    supports_logits_hook: bool

    def load(self, model_settings: ModelSettings, device: Optional[str] = None) -> None:
        ...

    def generate_batch(
        self,
        prompts: Optional[Sequence[PromptRecord]],
        generation: GenerationSettings,
        safety: SafetySettings,
        shard_metadata: Dict[str, Any],
    ) -> GenerationRun:
        ...
