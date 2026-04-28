from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence, Set

import torch
from transformers import AutoModel, AutoTokenizer

from sampling.sample_text import (
    GenerationResult,
    GenerationRun,
    GenerationSettings,
    ModelSettings,
    PromptRecord,
    SafetySettings,
    _assert_no_extra_turn_tokens,
    _chunk,
    _resolve_stop_tokens,
    _strip_completion_tokens,
)
from sampling.safe_hooks import build_dream_repellency_hook
from unsafe_prep.utils import ensure_pad_token, resolve_mask_index


class DreamBackend:
    name = "dream"
    family = "dream"
    supports_logits_hook = True

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.device: Optional[torch.device] = None
        self.model_settings: Optional[ModelSettings] = None
        self.stop_tokens = None
        self.mask_token_id: Optional[int] = None
        self.effective_vocab: Optional[int] = None

    def _map_precision(self, precision: str) -> torch.dtype:
        lookup = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        return lookup.get(str(precision).lower(), torch.float32)

    def load(self, model_settings: ModelSettings, device: Optional[str] = None) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        self.model_settings = model_settings
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(target_device)
        dtype = self._map_precision(model_settings.precision)

        checkpoint_path = str(model_settings.checkpoint_path)
        tokenizer_path = str(model_settings.tokenizer_name or checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        ensure_pad_token(self.tokenizer, eos_token_id=eos_id)
        self.mask_token_id = resolve_mask_index(self.tokenizer, getattr(self.tokenizer, "mask_token", None))

        self.model = AutoModel.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device).eval()

        self.stop_tokens = _resolve_stop_tokens(self.tokenizer)
        self.effective_vocab = getattr(self.model.config, "vocab_size", None) or len(self.tokenizer)

    def _use_chat_template(self) -> bool:
        if self.tokenizer is None or self.model_settings is None:
            return False
        use_chat_template = getattr(self.tokenizer, "chat_template", None) is not None
        model_name = str(self.model_settings.model_name).lower()
        if "instruct" in model_name:
            return use_chat_template
        return False

    def _embed_fn(self):
        if self.model is None:
            return None
        model = self.model

        def _semantic_embed(tokens: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                emb_layer = None
                if hasattr(model, "get_input_embeddings"):
                    emb_layer = model.get_input_embeddings()
                if emb_layer is None and hasattr(model, "vocab_embed"):
                    emb_layer = model.vocab_embed
                if emb_layer is None and hasattr(model, "embedding"):
                    emb_layer = model.embedding
                if emb_layer is None and hasattr(model, "word_embeddings"):
                    emb_layer = model.word_embeddings
                if emb_layer is None:
                    raise RuntimeError(
                        "Semantic gating requested but no embedding layer found on Dream model."
                    )
                return emb_layer(tokens)

        return _semantic_embed

    def _capture_peak_vram(self) -> int:
        if torch.cuda.is_available():
            try:
                return torch.cuda.max_memory_allocated(torch.cuda.current_device())
            except RuntimeError:
                return torch.cuda.max_memory_allocated()
        return 0

    def generate_batch(
        self,
        prompts: Optional[Sequence[PromptRecord]],
        generation: GenerationSettings,
        safety: SafetySettings,
        shard_metadata: Dict[str, Any],
    ) -> GenerationRun:
        start_time = time.perf_counter()
        if not prompts and generation.unconditional_samples <= 0:
            timings = {
                "load_seconds": 0.0,
                "generation_seconds": 0.0,
                "total_seconds": 0.0,
                "peak_vram_bytes": 0,
                "repellency": {},
            }
            return GenerationRun(results=[], timings=timings, resolved_config={})

        if self.model_settings is None:
            raise RuntimeError("Dream backend requires load() before generate_batch().")
        load_start = time.perf_counter()
        self.load(self.model_settings)
        load_seconds = time.perf_counter() - load_start

        if self.tokenizer is None or self.model is None or self.device is None:
            raise RuntimeError("Dream backend failed to load model/tokenizer.")

        stop_tokens = self.stop_tokens
        stop_ids: Set[int] = stop_tokens.stop_ids if stop_tokens else set()

        steps = max(1, min(int(generation.sampling_steps), int(generation.max_new_tokens)))

        generation_start = time.perf_counter()
        results: list[GenerationResult] = []
        logits_hook: Optional[Any] = None

        def _emit_metadata(extra: Dict[str, Any]) -> Dict[str, Any]:
            return {
                **extra,
                **shard_metadata,
                "safe_sampling_enabled": bool(logits_hook),
            }

        use_chat_template = self._use_chat_template()

        if prompts:
            for batch in _chunk(prompts, generation.batch_size):
                prompts_text = [record.prompt for record in batch]
                if use_chat_template:
                    prompts_text = [
                        self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": text}],
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                        for text in prompts_text
                    ]
                encoded = self.tokenizer(
                    prompts_text,
                    add_special_tokens=not use_chat_template,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                    if attention_mask.dtype is not torch.bool:
                        attention_mask = attention_mask.to(torch.bool)

                prompt_width = input_ids.shape[1]
                true_prompt_lengths = (
                    attention_mask.sum(dim=1).tolist()
                    if attention_mask is not None
                    else [prompt_width] * input_ids.shape[0]
                )

                logits_hook = None
                if safety.enabled:
                    logits_hook = build_dream_repellency_hook(
                        self.tokenizer,
                        safety,
                        self.device,
                        mask_token_id=self.mask_token_id,
                        attention_mask=attention_mask,
                        prompt_width=prompt_width,
                        total_steps=steps,
                        vocab_size=self.effective_vocab,
                        embed_fn=self._embed_fn() if safety.use_semantic_gating else None,
                    )

                outputs = self.model.diffusion_generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=generation.max_new_tokens,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=steps,
                    temperature=generation.temperature,
                    generation_logits_hook_func=logits_hook,
                )
                sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs

                for row, record in enumerate(batch):
                    tokens = sequences[row].tolist()
                    prompt_len = prompt_width
                    if attention_mask is not None:
                        prompt_mask = attention_mask[row, :prompt_width].to(torch.int64).tolist()
                    else:
                        prompt_mask = [1] * int(prompt_len)
                    if len(tokens) > len(prompt_mask):
                        prompt_mask.extend([0] * (len(tokens) - len(prompt_mask)))
                    completion_tokens, _, _ = _strip_completion_tokens(
                        tokens,
                        prompt_len,
                        stop_ids,
                        self.mask_token_id,
                        stop_sequences=stop_tokens.stop_sequences if stop_tokens else (),
                    )
                    raw_completion_text = self.tokenizer.decode(
                        completion_tokens,
                        skip_special_tokens=False,
                    )
                    completion_text = raw_completion_text.strip()
                    if stop_tokens is not None:
                        _assert_no_extra_turn_tokens(
                            completion_tokens=completion_tokens,
                            decoded_completion=raw_completion_text,
                            prompt_length=prompt_len,
                            tokens=tokens,
                            stop_tokens=stop_tokens,
                            logger=logging.getLogger(__name__),
                        )
                    metadata = _emit_metadata(
                        {
                            **record.metadata,
                            "true_prompt_len": int(true_prompt_lengths[row]),
                            "prompt_width": prompt_width,
                        }
                    )
                    results.append(
                        GenerationResult(
                            prompt_id=record.prompt_id,
                            prompt=record.prompt,
                            completion=completion_text,
                            full_text=completion_text,
                            token_ids=tokens,
                            prompt_length=prompt_len,
                            prompt_mask=prompt_mask,
                            metadata=metadata,
                        )
                    )

        if generation.unconditional_samples > 0:
            remaining = generation.unconditional_samples
            counter = 0
            while remaining > 0:
                batch_size = min(generation.batch_size, remaining)
                prompt_tensor = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
                logits_hook = None
                if safety.enabled:
                    logits_hook = build_dream_repellency_hook(
                        self.tokenizer,
                        safety,
                        self.device,
                        mask_token_id=self.mask_token_id,
                        attention_mask=None,
                        prompt_width=0,
                        total_steps=steps,
                        vocab_size=self.effective_vocab,
                        embed_fn=self._embed_fn() if safety.use_semantic_gating else None,
                    )
                outputs = self.model.diffusion_generate(
                    input_ids=prompt_tensor,
                    attention_mask=None,
                    max_new_tokens=generation.max_new_tokens,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=steps,
                    temperature=generation.temperature,
                    generation_logits_hook_func=logits_hook,
                )
                sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
                for row in range(batch_size):
                    tokens = sequences[row].tolist()
                    completion_tokens, _, _ = _strip_completion_tokens(
                        tokens,
                        0,
                        stop_ids,
                        self.mask_token_id,
                        stop_sequences=stop_tokens.stop_sequences if stop_tokens else (),
                    )
                    raw_completion_text = self.tokenizer.decode(
                        completion_tokens,
                        skip_special_tokens=False,
                    )
                    completion_text = self.tokenizer.decode(
                        completion_tokens,
                        skip_special_tokens=True,
                    ).strip()
                    if stop_tokens is not None:
                        _assert_no_extra_turn_tokens(
                            completion_tokens=completion_tokens,
                            decoded_completion=raw_completion_text,
                            prompt_length=0,
                            tokens=tokens,
                            stop_tokens=stop_tokens,
                            logger=logging.getLogger(__name__),
                        )
                    metadata = _emit_metadata({"prompt_type": "unconditional"})
                    results.append(
                        GenerationResult(
                            prompt_id=f"uncond-{counter:06d}",
                            prompt="",
                            completion=completion_text,
                            full_text=completion_text,
                            token_ids=tokens,
                            prompt_length=0,
                            prompt_mask=[0] * len(tokens),
                            metadata=metadata,
                        )
                    )
                    counter += 1
                remaining -= batch_size

        generation_seconds = time.perf_counter() - generation_start
        total_seconds = time.perf_counter() - start_time
        timings = {
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "total_seconds": total_seconds,
            "peak_vram_bytes": self._capture_peak_vram(),
            "repellency": {},
        }
        resolved_config = {
            "model": asdict(self.model_settings) if self.model_settings else {},
            "generation": asdict(generation),
            "safety": asdict(safety),
        }
        return GenerationRun(results=results, timings=timings, resolved_config=resolved_config)
