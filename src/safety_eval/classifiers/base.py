from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CausalLMClassifier:
    """Shared utilities for causal LM based safety classifiers."""

    def __init__(
        self,
        model_path: Union[str, Path],
        *,
        device: str = "auto",
        device_map: Optional[str] = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        tokenizer_kwargs: Optional[Dict] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.device = self._resolve_device(device)
        self.device_map = device_map
        dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        tokenizer_kwargs = tokenizer_kwargs or {}
        if "use_fast" not in tokenizer_kwargs:
            tokenizer_kwargs["use_fast"] = False
        tokenizer_kwargs.setdefault("trust_remote_code", True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
        pad_added = self._ensure_padding_token()
        self._configure_tokenizer()

        model_kwargs = model_kwargs or {}
        if "device_map" not in model_kwargs:
            model_kwargs["device_map"] = device_map
        if "torch_dtype" not in model_kwargs:
            model_kwargs["torch_dtype"] = dtype
        model_kwargs.setdefault("trust_remote_code", True)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        if pad_added:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompts: Sequence[str],
        *,
        batch_size: int,
        max_new_tokens: int,
        **generate_kwargs,
    ) -> Sequence[str]:
        if not prompts:
            return []
        outputs = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = self._encode_batch(batch)
            outputs.extend(
                self._generate_from_encoded(
                    encoded,
                    max_new_tokens=max_new_tokens,
                    generate_kwargs=generate_kwargs,
                )
            )
        return outputs

    def _configure_tokenizer(self) -> None:
        """Hook for subclasses to tweak tokenizer settings."""

    def _ensure_padding_token(self) -> bool:
        if self.tokenizer.pad_token is not None:
            return False
        if self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return False
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        return True

    def _encode_batch(self, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
        )
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
            attention_mask = torch.ones_like(encoded)
        else:
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

    def _generate_from_encoded(
        self,
        encoded_inputs: Dict[str, torch.Tensor],
        *,
        max_new_tokens: int,
        generate_kwargs: Optional[Dict] = None,
    ) -> Sequence[str]:
        generate_kwargs = dict(generate_kwargs or {})
        generate_kwargs.setdefault("do_sample", False)
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
        decoded = self.tokenizer.batch_decode(
            generated[:, input_ids.shape[-1] :],
            skip_special_tokens=True,
        )
        return [text.strip() for text in decoded]

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
