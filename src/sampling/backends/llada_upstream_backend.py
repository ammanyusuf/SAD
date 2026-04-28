from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence, Set

import torch
import inspect
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
from sampling.safe_hooks import build_llada_repellency_hook
from unsafe_prep.utils import ensure_pad_token
from utils.constants import LLADA_EOS_TOKEN_ID, LLADA_MASK_TOKEN_ID, LLADA_EOT_TOKEN_ID
from third_party.LLaDA.generate import generate as llada_generate

_LLADA_GENERATE_SUPPORTS_LOGITS_HOOK = "logits_hook" in inspect.signature(llada_generate).parameters

LOGGER = logging.getLogger(__name__)


def _map_precision(precision: str) -> torch.dtype:
    lookup = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return lookup.get(str(precision).lower(), torch.float32)


class LLaDAUpstreamBackend:
    name = "llada_upstream"
    family = "llada"
    supports_logits_hook = _LLADA_GENERATE_SUPPORTS_LOGITS_HOOK

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.device: Optional[torch.device] = None
        self.model_settings: Optional[ModelSettings] = None
        self.stop_tokens = None
        self.mask_token_id: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.eot_id: Optional[int] = None
        self.effective_vocab: Optional[int] = None
        self._logged_generation_debug = False

    def load(self, model_settings: ModelSettings, device: Optional[str] = None) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        self.model_settings = model_settings
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(target_device)
        dtype = _map_precision(model_settings.precision)

        checkpoint_path = str(model_settings.checkpoint_path)
        tokenizer_path = str(model_settings.tokenizer_name or checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        ensure_pad_token(self.tokenizer, eos_token_id=eos_id)

        self.model = AutoModel.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device).eval()

        self.stop_tokens = _resolve_stop_tokens(self.tokenizer)
        self.mask_token_id = LLADA_MASK_TOKEN_ID
        self.eos_id = LLADA_EOS_TOKEN_ID
        self.eot_id = LLADA_EOT_TOKEN_ID
        self.effective_vocab = getattr(self.model.config, "vocab_size", None) or len(self.tokenizer)

    def _use_chat_template(self) -> bool:
        if self.tokenizer is None or self.model_settings is None:
            return False
        use_chat_template = getattr(self.tokenizer, "chat_template", None) is not None
        model_name = str(self.model_settings.model_name).lower()
        if "llada-8b-base" in model_name:
            return False
        if "llada-8b-instruct" in model_name:
            return True
        return use_chat_template

    def _capture_peak_vram(self) -> int:
        if torch.cuda.is_available():
            try:
                return torch.cuda.max_memory_allocated(torch.cuda.current_device())
            except RuntimeError:
                return torch.cuda.max_memory_allocated()
        return 0

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
                        "Semantic gating requested but no embedding layer found on LLaDA model."
                    )
                return emb_layer(tokens)

        return _semantic_embed

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
            raise RuntimeError("LLaDA backend requires load() before generate_batch().")
        load_start = time.perf_counter()
        self.load(self.model_settings)
        load_seconds = time.perf_counter() - load_start

        if self.tokenizer is None or self.model is None or self.device is None:
            raise RuntimeError("LLaDA backend failed to load model/tokenizer.")

        stop_tokens = self.stop_tokens
        stop_ids: Set[int] = stop_tokens.stop_ids if stop_tokens else set()

        generation_start = time.perf_counter()
        results: list[GenerationResult] = []

        logits_hook = None
        logits_hook_ctx: Dict[str, Any] = {}
        if safety.enabled:
            embed_fn = self._embed_fn() if safety.use_semantic_gating else None
            logits_hook_ctx = {
                "vocab_size": int(self.effective_vocab or len(self.tokenizer)),
                "embed_fn": embed_fn,
            }
            logits_hook = build_llada_repellency_hook(self.tokenizer, safety, self.device)

        def _emit_metadata(extra: Dict[str, Any]) -> Dict[str, Any]:
            return {
                **extra,
                **shard_metadata,
                "safe_sampling_enabled": bool(logits_hook),
            }

        use_chat_template = self._use_chat_template()
        steps = min(generation.sampling_steps, generation.max_new_tokens)
        if generation.block_length is not None:
            block_length = generation.block_length
        else:
            block_length = min(generation.max_new_tokens, generation.sampling_steps)
        if block_length > generation.max_new_tokens:
            block_length = generation.max_new_tokens
        if generation.max_new_tokens % block_length != 0:
            LOGGER.warning(
                "LLaDA upstream requires gen_length divisible by block_length; using block_length=gen_length."
            )
            block_length = generation.max_new_tokens
        num_blocks = max(generation.max_new_tokens // block_length, 1)
        steps_per_block = max(steps // num_blocks, 1)
        steps = steps_per_block * num_blocks
        total_steps = num_blocks * steps_per_block
        logits_eos_inf = False
        confidence_eos_eot_inf = False
        if self.model_settings and "llada-8b-instruct" in str(self.model_settings.model_name).lower():
            logits_eos_inf = True
            confidence_eos_eot_inf = True

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
                prompt_width = input_ids.shape[1]
                true_prompt_lengths = (
                    attention_mask.sum(dim=1).tolist() if attention_mask is not None else [prompt_width] * input_ids.shape[0]
                )
                prompt_ids = [record.prompt_id for record in batch]
                prompt_id = (
                    prompt_ids[0]
                    if len(prompt_ids) == 1
                    else f"batch:{prompt_ids[0]}..{prompt_ids[-1]}"
                )
                variants = []
                for record in batch:
                    meta = record.metadata or {}
                    variant = meta.get("prompt_variant")
                    if variant is None and "prompt_is_safe" in meta:
                        variant = "benign" if bool(meta["prompt_is_safe"]) else "unsafe"
                    if variant is not None:
                        variants.append(str(variant))
                prompt_variant = None
                if variants:
                    prompt_variant = variants[0] if all(v == variants[0] for v in variants) else "mixed"
                if prompt_variant is not None:
                    os.environ["SAFE_PROMPT_VARIANT"] = str(prompt_variant)
                if not self._logged_generation_debug:
                    LOGGER.info(
                        "LLaDA upstream debug: use_chat_template=%s mask_id=%s prompt_width=%d",
                        use_chat_template,
                        int(self.mask_token_id or LLADA_MASK_TOKEN_ID),
                        prompt_width,
                    )

                generate_kwargs = dict(
                    model=self.model,
                    prompt=input_ids,
                    attention_mask=attention_mask,
                    steps=steps,
                    gen_length=generation.max_new_tokens,
                    block_length=block_length,
                    temperature=generation.temperature,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=int(self.mask_token_id or LLADA_MASK_TOKEN_ID),
                    logits_eos_inf=logits_eos_inf,
                    confidence_eos_eot_inf=confidence_eos_eot_inf,
                )
                if _LLADA_GENERATE_SUPPORTS_LOGITS_HOOK:
                    hook_ctx = {
                        **logits_hook_ctx,
                        "prompt_width": prompt_width,
                        "total_steps": total_steps,
                        "prompt_id": prompt_id,
                        "prompt_variant": prompt_variant,
                    }
                    generate_kwargs.update(
                        logits_hook=logits_hook,
                        logits_hook_ctx=hook_ctx,
                        t_start=safety.t_start,
                        t_end=safety.t_end,
                    )
                elif logits_hook is not None:
                    LOGGER.warning(
                        "LLaDA upstream generate() does not support logits_hook; skipping safety hook."
                    )
                outputs = llada_generate(**generate_kwargs)
                if not self._logged_generation_debug and self.tokenizer is not None and outputs.numel() > 0:
                    sample_ids = outputs[0].tolist()
                    sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_ids[:32])
                    LOGGER.info(
                        "LLaDA upstream debug: sample token_ids[:32]=%s tokens[:32]=%s",
                        sample_ids[:32],
                        sample_tokens,
                    )
                    self._logged_generation_debug = True

                for row, record in enumerate(batch):
                    tokens = outputs[row].tolist()
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
                            logger=LOGGER,
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
                outputs = llada_generate(
                    model=self.model,
                    prompt=prompt_tensor,
                    attention_mask=None,
                    steps=steps,
                    gen_length=generation.max_new_tokens,
                    block_length=block_length,
                    temperature=generation.temperature,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=int(self.mask_token_id or LLADA_MASK_TOKEN_ID),
                    logits_eos_inf=logits_eos_inf,
                    confidence_eos_eot_inf=confidence_eos_eot_inf,
                    logits_hook=logits_hook,
                    logits_hook_ctx={
                        **logits_hook_ctx,
                        "prompt_width": 0,
                        "total_steps": total_steps,
                    },
                    t_start=safety.t_start,
                    t_end=safety.t_end,
                )
                for row in range(batch_size):
                    tokens = outputs[row].tolist()
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
                            logger=LOGGER,
                        )
                    metadata = _emit_metadata({"prompt_type": "unconditional"})
                    results.append(
                        GenerationResult(
                            prompt_id=f"uncond:{counter}",
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
