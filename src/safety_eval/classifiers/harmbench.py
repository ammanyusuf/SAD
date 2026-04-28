from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

from utils.constants import HARM_BENCH_PROMPTS

from .base import CausalLMClassifier


@dataclass
class HarmBenchResult:
    success: bool
    raw_text: str


class HarmBenchClassifier(CausalLMClassifier):
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        device_map: Optional[str] = "auto",
        max_new_tokens: int = 1,
    ) -> None:
        super().__init__(model_path, device=device, device_map=device_map)
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"
        self.max_new_tokens = max_new_tokens

    def classify(
        self,
        behaviors: Sequence[str],
        generations: Sequence[str],
        contexts: Optional[Sequence[str]] = None,
        *,
        batch_size: int = 4,
        max_new_tokens: Optional[int] = None,
    ) -> List[HarmBenchResult]:
        if contexts is None:
            contexts = [""] * len(behaviors)
        if not (len(behaviors) == len(generations) == len(contexts)):
            raise ValueError("behaviors, generations, and contexts must have the same length.")

        prompts = [
            self._format_prompt(behaviors[idx], generations[idx], contexts[idx])
            for idx in range(len(behaviors))
        ]
        raw_outputs = self.generate(
            prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
        )
        results: List[HarmBenchResult] = []
        for text in raw_outputs:
            cleaned = text.strip()
            results.append(HarmBenchResult(success=cleaned.lower().startswith("yes"), raw_text=cleaned))
        return results

    def score(
        self,
        behavior: str,
        generation: str,
        *,
        behavior_id: Optional[str] = None,
        behaviors_map: Optional[Dict[str, Dict[str, str]]] = None,
        context: str = "",
        batch_size: int = 1,
        max_new_tokens: Optional[int] = None,
    ) -> SimpleNamespace:
        """Classify a single (behavior, generation) pair and return a score-like object."""
        behavior_text = behavior or ""
        if not behavior_text and behavior_id and behaviors_map:
            entry = behaviors_map.get(str(behavior_id)) if isinstance(behaviors_map, dict) else None
            if entry:
                behavior_text = (entry.get("Behavior") or entry.get("prompt"))

        results = self.classify(
            behaviors=[behavior_text],
            generations=[generation],
            contexts=[context],
            batch_size=batch_size,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
        )
        result = results[0] if results else HarmBenchResult(success=False, raw_text="")
        unsafe = bool(result.success)
        hazards = ["harmbench"] if unsafe else []
        return SimpleNamespace(raw_text=result.raw_text, unsafe=unsafe, unsafe_score=1.0 if unsafe else 0.0, hazards=hazards)

    @staticmethod
    def _format_prompt(behavior: str, generation: str, context: str) -> str:
        if context:
            return HARM_BENCH_PROMPTS["prompt_contextual"].format(
                behavior=behavior,
                generation=generation,
                context=context,
            )
        return HARM_BENCH_PROMPTS["prompt"].format(
            behavior=behavior,
            generation=generation,
        )
