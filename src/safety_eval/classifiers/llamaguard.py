from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .base import CausalLMClassifier

import torch


def load_texts(path: Path, text_key: str) -> List[str]:
    """Load a list of strings from a JSONL/JSON generations file."""
    texts: List[str] = []
    with path.open() as fp:
        for line_idx, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_idx} invalid JSON: {exc}") from exc
            if text_key not in obj:
                raise KeyError(f"Missing key '{text_key}' in {path}:{line_idx}")
            value = obj[text_key]
            if not isinstance(value, str):
                raise TypeError(f"Expected string at key '{text_key}' in {path}:{line_idx}")
            texts.append(value)
    return texts


def batched(seq: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


def count_unsafe(responses: Sequence[str]) -> int:
    return sum(1 for text in responses if "unsafe" in text.lower())


class LlamaGuardClassifier(CausalLMClassifier):
    def __init__(
        self,
        model_path: Union[str, Path],
        *,
        device: str = "auto",
        device_map: Optional[str] = "auto",
        system_prompt: Optional[str] = None,
        use_chat_template: Optional[bool] = None,
    ) -> None:
        super().__init__(model_path, device=device, device_map=device_map)
        if use_chat_template is None:
            use_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt

    def classify_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_new_tokens: int,
        log_every: int = 10,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[List[str], int]:
        if not texts:
            return [], 0

        outputs: List[str] = []
        unsafe = 0
        total_batches = math.ceil(len(texts) / batch_size)
        for batch_idx, batch in enumerate(batched(texts, batch_size), start=1):
            encoded = self._encode_batch(batch)
            responses = self._generate_from_encoded(
                encoded,
                max_new_tokens=max_new_tokens,
            )
            outputs.extend(responses)
            unsafe += count_unsafe(responses)
            if logger and log_every > 0 and (batch_idx % log_every == 0 or batch_idx == total_batches):
                logger.info(
                    "Processed batch %d/%d (cumulative unsafe=%d).",
                    batch_idx,
                    total_batches,
                    unsafe,
                )
        return outputs, unsafe

    def score(self, prompt: str, *, max_new_tokens: int = 32, batch_size: int = 1) -> str:
        """Classify a single prompt and return the raw model response."""
        outputs, _ = self.classify_texts(
            [prompt],
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        return outputs[0] if outputs else ""

    def _encode_batch(self, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
        if self.use_chat_template:
            if self.system_prompt:
                messages = [
                    [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": sample},
                    ]
                    for sample in prompts
                ]
            else:
                messages = [[{"role": "user", "content": sample}] for sample in prompts]
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
            )
            if hasattr(encoded, "input_ids"):
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
            else:
                input_ids = encoded
                attention_mask = torch.ones_like(input_ids)
            return {
                "input_ids": input_ids.to(self.device),
                "attention_mask": attention_mask.to(self.device),
            }

        plain_prompts = [
            f"{self.system_prompt}\n{sample}" if self.system_prompt else sample
            for sample in prompts
        ]
        return super()._encode_batch(plain_prompts)
