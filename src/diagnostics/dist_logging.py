from __future__ import annotations

import heapq
import json
import os
import shutil
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() not in ("0", "false", "off", "")


def _parse_timesteps(value: str | None) -> Union[str, Sequence[int]]:
    if not value:
        return "auto"
    if value.lower() in {"auto", "all"}:
        return value.lower()
    try:
        entries = [int(item.strip()) for item in value.split(",") if item.strip()]
        return entries if entries else "auto"
    except ValueError:
        return "auto"


@dataclass(frozen=True)
class DistributionLogConfig:
    enabled: bool
    timesteps: Union[str, Sequence[int]]
    topk: int
    positions: str
    max_positions: int
    include_full_vocab: bool
    dtype: torch.dtype
    auto_top_n: int
    path: Optional[str]
    position_sample_seed: Optional[int]
    dump_tokens: bool
    dump_prompt: bool

    @staticmethod
    def from_env() -> "DistributionLogConfig":
        enabled = _parse_bool(os.getenv("SAFE_DIST_LOG_ENABLED"))
        timesteps = _parse_timesteps(os.getenv("SAFE_DIST_LOG_TIMESTEPS"))
        topk = max(1, int(os.getenv("SAFE_DIST_LOG_TOPK", "50")))
        max_positions = max(1, int(os.getenv("SAFE_DIST_LOG_MAX_POS", "16")))
        raw_positions = (
            os.getenv("SAFE_DIST_LOG_POSITION_MODE")
            or os.getenv("SAFE_DIST_LOG_POSITIONS")
            or "masked"
        )
        positions = str(raw_positions).lower()
        if positions == "masked":
            positions = "masked_only"
        elif positions == "subset":
            positions = "sampled"
        if positions not in {"masked_only", "unmasked_only", "all", "sampled"}:
            positions = "masked_only"
        include_full_vocab = _parse_bool(os.getenv("SAFE_DIST_LOG_FULL_VOCAB"))
        dtype_choice = os.getenv("SAFE_DIST_LOG_DTYPE", "float16").lower()
        dtype = torch.float16 if dtype_choice == "float16" else torch.float32
        auto_top_n = max(1, int(os.getenv("SAFE_DIST_LOG_AUTO_TOP_N", "3")))
        path = os.getenv("SAFE_DIST_LOG_PATH")
        seed_raw = os.getenv("SAFE_DIST_LOG_POSITION_SAMPLE_SEED")
        seed = None
        if seed_raw not in (None, ""):
            try:
                seed = int(seed_raw)
            except ValueError:
                seed = None
        dump_tokens = _parse_bool(os.getenv("SAFE_DIST_LOG_DUMP_TOKENS"))
        dump_prompt = _parse_bool(os.getenv("SAFE_DIST_LOG_DUMP_PROMPT"))
        return DistributionLogConfig(
            enabled=enabled,
            timesteps=timesteps,
            topk=topk,
            positions=positions,
            max_positions=max_positions,
            include_full_vocab=include_full_vocab,
            dtype=dtype,
            auto_top_n=auto_top_n,
            path=path,
            position_sample_seed=seed,
            dump_tokens=dump_tokens,
            dump_prompt=dump_prompt,
        )


class DistributionLogger:
    _reset_done: bool = False

    def __init__(self, run_id: Optional[str], config: DistributionLogConfig, t_start: int, t_end: int) -> None:
        self._config = config
        self._enabled = config.enabled
        self._decode_fn: Optional[Callable[[Sequence[int]], str]] = None
        self._logged_steps: set[Tuple[int, Optional[str]]] = set()
        self._sampled_steps: set[Tuple[int, Optional[str]]] = set()
        if not self._enabled:
            return
        self._logger = logging.getLogger(__name__)
        base_dir = Path(os.getenv("SAFE_DIST_LOG_DIR", "diagnostics/dist_logs")).expanduser()
        if not DistributionLogger._reset_done:
            if base_dir.exists():
                shutil.rmtree(base_dir)
            DistributionLogger._reset_done = True
        base_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id or os.getenv("SLURM_JOB_ID") or "local"
        if config.path:
            path = Path(config.path).expanduser()
            if not path.is_absolute():
                path = base_dir / path
            self._file_path = path
        else:
            self._file_path = base_dir / "dist_logs.jsonl"
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._sample_dir = base_dir / "sample"
        self._sample_dir.mkdir(parents=True, exist_ok=True)
        self._auto_steps = self._build_auto_steps(t_start, t_end)
        self._top_strength_limit = config.auto_top_n
        self._strength_heap: list[Tuple[float, int]] = []
        self._strength_steps: set[int] = set()
        self._logger.info(
            "Distribution logging enabled: jsonl=%s sample_dir=%s dump_tokens=%s",
            str(self._file_path),
            str(self._sample_dir),
            bool(self._config.dump_tokens),
        )

    @classmethod
    def from_env(cls, run_id: Optional[str], t_start: int, t_end: int) -> "DistributionLogger":
        config = DistributionLogConfig.from_env()
        return cls(run_id=run_id, config=config, t_start=t_start, t_end=t_end)

    def set_decoder(self, decode_fn: Optional[Callable[[Sequence[int]], str]]) -> None:
        self._decode_fn = decode_fn
        if self._enabled and self._config.dump_tokens:
            if self._decode_fn is None:
                self._logger.warning("Distribution logging: dump_tokens enabled but no decoder set.")
            else:
                self._logger.info("Distribution logging: token decoder attached.")

    def _build_auto_steps(self, start: int, end: int) -> set[int]:
        if start is None or end is None:
            return set()
        steps = {start, end}
        if start + 1 <= end:
            steps.add(start + 1)
        if end - 1 >= start:
            steps.add(end - 1)
        steps.add((start + end) // 2)
        return steps

    def _should_log_step(self, step: int, strength_val: float | None) -> bool:
        if not self._enabled:
            return False
        step = int(step)
        target = self._config.timesteps
        if target == "all":
            return True
        if target == "auto":
            if step in self._auto_steps:
                return True
            return self._update_top_strength(step, strength_val)
        if isinstance(target, Sequence):
            return step in set(target)
        return False

    def _update_top_strength(self, step: int, strength_val: float | None) -> bool:
        if strength_val is None or strength_val <= 0.0:
            return False
        if len(self._strength_heap) < self._top_strength_limit:
            heapq.heappush(self._strength_heap, (strength_val, step))
            self._strength_steps.add(step)
            return True
        if strength_val > self._strength_heap[0][0]:
            _, removed = heapq.heapreplace(self._strength_heap, (strength_val, step))
            self._strength_steps.discard(removed)
            self._strength_steps.add(step)
            return True
        return step in self._strength_steps

    @torch.no_grad()
    def maybe_log(
        self,
        *,
        step: int,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        logits_data: torch.Tensor,
        logits_unsafe: torch.Tensor,
        logits_safe: torch.Tensor,
        prompt_id: Optional[str] = None,
        effective_strength: float | None = None,
        mask_frac: float | None = None,
        num_masked: int | None = None,
        seq_len: int | None = None,
        extra: Optional[dict] = None,
        step_header: Optional[dict] = None,
        mask_token_id: Optional[int] = None,
        prompt_len: Optional[int] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        prompt_token_ids: Optional[torch.Tensor] = None,
    ) -> None:
        if not self._should_log_step(step, effective_strength):
            return
        self._maybe_log_step_header(
            step=step,
            prompt_id=prompt_id,
            mask_frac=mask_frac,
            num_masked=num_masked,
            seq_len=seq_len,
            effective_strength=effective_strength,
            extra=step_header,
        )
        self._maybe_log_samples(
            step=step,
            prompt_id=prompt_id,
            token_ids=token_ids,
            mask_token_id=mask_token_id,
            prompt_len=prompt_len,
            prompt_mask=prompt_mask,
            prompt_token_ids=prompt_token_ids,
        )
        mask_bool = mask.to(torch.bool)
        positions = self._select_positions(mask_bool, step)
        if not positions:
            return
        log_probs_data = F.log_softmax(logits_data, dim=-1)
        log_probs_unsafe = F.log_softmax(logits_unsafe, dim=-1)
        log_probs_safe = F.log_softmax(logits_safe, dim=-1)
        probs_data = log_probs_data.exp()
        probs_unsafe = log_probs_unsafe.exp()
        probs_safe = log_probs_safe.exp()
        with self._file_path.open("a", encoding="utf-8") as handle:
            for batch_idx, token_idx in positions:
                if token_idx >= token_ids.size(1):
                    continue
                record = self._build_record(
                    step=step,
                    prompt_id=prompt_id,
                    batch_idx=int(batch_idx),
                    token_idx=int(token_idx),
                    mask_frac=mask_frac,
                    num_masked=num_masked,
                    seq_len=seq_len,
                    probs_data=probs_data[batch_idx, token_idx],
                    probs_unsafe=probs_unsafe[batch_idx, token_idx],
                    probs_safe=probs_safe[batch_idx, token_idx],
                    log_probs_data=log_probs_data[batch_idx, token_idx],
                    log_probs_unsafe=log_probs_unsafe[batch_idx, token_idx],
                    log_probs_safe=log_probs_safe[batch_idx, token_idx],
                    token_id=int(token_ids[batch_idx, token_idx].item()),
                    effective_strength=effective_strength,
                    extra=extra,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _maybe_log_step_header(
        self,
        *,
        step: int,
        prompt_id: Optional[str],
        mask_frac: float | None,
        num_masked: int | None,
        seq_len: int | None,
        effective_strength: float | None,
        extra: Optional[dict],
    ) -> None:
        key = (int(step), prompt_id)
        if key in self._logged_steps:
            return
        self._logged_steps.add(key)
        record = {
            "type": "step",
            "run_id": self._run_id,
            "prompt_id": prompt_id,
            "step": int(step),
            "mask_frac": None if mask_frac is None else float(mask_frac),
            "num_masked": num_masked,
            "seq_len": seq_len,
            "effective_strength": None if effective_strength is None else float(effective_strength),
        }
        if extra:
            record.update(extra)
        with self._file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _maybe_log_samples(
        self,
        *,
        step: int,
        prompt_id: Optional[str],
        token_ids: torch.Tensor,
        mask_token_id: Optional[int],
        prompt_len: Optional[int],
        prompt_mask: Optional[torch.Tensor],
        prompt_token_ids: Optional[torch.Tensor],
    ) -> None:
        if not self._config.dump_tokens or self._decode_fn is None:
            return
        key = (int(step), prompt_id)
        if key in self._sampled_steps:
            return
        self._sampled_steps.add(key)
        sample_path = self._sample_dir / f"step_{int(step):04d}.jsonl"
        with sample_path.open("a", encoding="utf-8") as handle:
            for batch_idx, row in enumerate(token_ids):
                token_list = row.detach().cpu().tolist()
                decoded = self._decode_fn(token_list)
                token_strings = [self._decode_fn([tok]) for tok in token_list]
                prompt_text = None
                prompt_mask_row = None
                prompt_mask_row_full = None
                if prompt_mask is not None and prompt_mask.numel() > 0:
                    try:
                        prompt_mask_row_full = prompt_mask[batch_idx].detach().cpu().to(torch.int64).tolist()
                    except Exception:
                        prompt_mask_row_full = None
                if prompt_mask_row_full:
                    prompt_mask_row = prompt_mask_row_full
                    if len(prompt_mask_row_full) != len(token_list):
                        prompt_mask_row = None
                    else:
                        prompt_token_ids_masked = [
                            tok for tok, keep in zip(token_list, prompt_mask_row_full) if keep
                        ]
                        if prompt_token_ids_masked:
                            prompt_text = self._decode_fn(prompt_token_ids_masked)
                if prompt_text is None and prompt_token_ids is not None and prompt_mask_row_full:
                    try:
                        full_tokens = prompt_token_ids[batch_idx].detach().cpu().tolist()
                        prompt_token_ids_masked = [
                            tok for tok, keep in zip(full_tokens, prompt_mask_row_full) if keep
                        ]
                        if prompt_token_ids_masked:
                            prompt_text = self._decode_fn(prompt_token_ids_masked)
                    except Exception:
                        prompt_text = None
                if prompt_text is None and prompt_len is not None and prompt_len > 0:
                    if prompt_len <= len(token_list):
                        prompt_text = self._decode_fn(token_list[: int(prompt_len)])
                    else:
                        prompt_len = None
                record = {
                    "run_id": self._run_id,
                    "prompt_id": prompt_id,
                    "step": int(step),
                    "batch_idx": int(batch_idx),
                    "token_ids": token_list,
                    "text": decoded,
                    "token_strs": token_strings,
                    "mask_token_id": mask_token_id,
                    "prompt_len": prompt_len,
                    "prompt_mask": prompt_mask_row,
                    "prompt_text": prompt_text,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _select_positions(self, mask: torch.Tensor, step: int) -> list[Tuple[int, int]]:
        if mask.numel() == 0:
            return []
        B, L = mask.shape
        if self._config.positions == "all":
            selection_mask = torch.ones_like(mask, dtype=torch.bool)
        elif self._config.positions == "unmasked_only":
            selection_mask = ~mask
        elif self._config.positions == "sampled":
            selection_mask = mask if mask.any() else torch.ones_like(mask, dtype=torch.bool)
        else:
            selection_mask = mask
        coords = torch.nonzero(selection_mask, as_tuple=False)
        if coords.numel() == 0:
            return []
        if self._config.positions == "sampled":
            max_positions = min(self._config.max_positions, coords.size(0))
            generator = None
            if self._config.position_sample_seed is not None:
                generator = torch.Generator(device=coords.device)
                generator.manual_seed(self._config.position_sample_seed + int(step))
            perm = torch.randperm(coords.size(0), generator=generator, device=coords.device)
            coords = coords[perm[:max_positions]]
        else:
            coords = coords[: self._config.max_positions]
        return [(int(coord[0].item()), int(coord[1].item())) for coord in coords]

    def _build_record(
        self,
        *,
        step: int,
        prompt_id: Optional[str],
        batch_idx: int,
        token_idx: int,
        mask_frac: float | None,
        num_masked: int | None,
        seq_len: int | None,
        probs_data: torch.Tensor,
        probs_unsafe: torch.Tensor,
        probs_safe: torch.Tensor,
        log_probs_data: torch.Tensor,
        log_probs_unsafe: torch.Tensor,
        log_probs_safe: torch.Tensor,
        token_id: int,
        effective_strength: float | None,
        extra: Optional[dict],
    ) -> dict:
        topk = min(self._config.topk, probs_data.numel())
        data_topk = torch.topk(probs_data, k=topk)
        unsafe_topk = torch.topk(probs_unsafe, k=topk)
        safe_topk = torch.topk(probs_safe, k=topk)
        margin_data = (
            float(data_topk.values[0] - data_topk.values[1])
            if data_topk.values.size(0) > 1
            else float(data_topk.values[0])
        )
        margin_safe = (
            float(safe_topk.values[0] - safe_topk.values[1])
            if safe_topk.values.size(0) > 1
            else float(safe_topk.values[0])
        )
        unsafe_mass = float(probs_unsafe[data_topk.indices].sum().item())
        kl_safe_data = float(
            torch.sum(probs_safe * (log_probs_safe - log_probs_data)).item()
        )
        kl_safe_unsafe = float(
            torch.sum(probs_safe * (log_probs_safe - log_probs_unsafe)).item()
        )
        kl_data_unsafe = float(
            torch.sum(probs_data * (log_probs_data - log_probs_unsafe)).item()
        )
        mixture = 0.5 * (probs_safe + probs_data)
        mixture_log = torch.log(mixture.clamp_min(1e-30))
        js = 0.5 * (
            torch.sum(probs_safe * (log_probs_safe - mixture_log))
            + torch.sum(probs_data * (log_probs_data - mixture_log))
        ).item()
        record = {
            "run_id": self._run_id,
            "prompt_id": prompt_id,
            "step": int(step),
            "batch_idx": batch_idx,
            "token_idx": token_idx,
            "token_id": token_id,
            "mask_frac": None if mask_frac is None else float(mask_frac),
            "num_masked": num_masked,
            "seq_len": seq_len,
            "effective_strength": None if effective_strength is None else float(effective_strength),
            "topk_tokens_data": data_topk.indices.cpu().tolist(),
            "topk_probs_data": data_topk.values.cpu().tolist(),
            "topk_tokens_unsafe": unsafe_topk.indices.cpu().tolist(),
            "topk_probs_unsafe": unsafe_topk.values.cpu().tolist(),
            "topk_tokens_safe": safe_topk.indices.cpu().tolist(),
            "topk_probs_safe": safe_topk.values.cpu().tolist(),
            "margin_data": margin_data,
            "margin_safe": margin_safe,
            "kl_safe_data": kl_safe_data,
            "kl_safe_unsafe": kl_safe_unsafe,
            "kl_data_unsafe": kl_data_unsafe,
            "js_safe_data": js,
            "unsafe_mass_under_data_topk": unsafe_mass,
        }
        if extra:
            record.update(extra)
        if self._config.include_full_vocab:
            record["full_vocab_probs_data"] = probs_data.cpu().tolist()
            record["full_vocab_probs_unsafe"] = probs_unsafe.cpu().tolist()
            record["full_vocab_probs_safe"] = probs_safe.cpu().tolist()
        return record
