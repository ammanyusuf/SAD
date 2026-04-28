from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from sampling.sample_text import SafetySettings, _prepare_unsafe_artifacts, resolve_eta_config, _resolve_stop_tokens
from sampling.transfer_schedule import compute_move_grid
from third_party.mdlm.diffusion import _load_unsafe_tensor
from third_party.mdlm.repellency.safe_denoiser import MaskKernelRepellency
from unsafe_prep.utils import ensure_pad_token
from utils.constants import LLADA_MASK_TOKEN_ID

LOGGER = logging.getLogger(__name__)

# Internal toggle: set SAFE_HOOK_QUIET=1 to suppress [safe_hook] print statements.
# Not a public config key — do not add to configs/config.yaml.
_QUIET = os.getenv("SAFE_HOOK_QUIET", "0") not in ("", "0", "false", "False")


def _compute_move_proxy(
    *,
    mask_index: torch.Tensor,
    prompt_mask: Optional[torch.Tensor],
    global_step: int,
    total_steps: int,
) -> torch.Tensor:
    """
    Proxy for MaskKernelRepellency move(t).

    Use the same linear move(t) discretization as the LLaDA sampler so
    guidance and transfer scheduling stay aligned.
    """
    if total_steps <= 1:
        move = torch.tensor(1.0, device=mask_index.device, dtype=torch.float32)
    else:
        _, move_grid, _ = compute_move_grid(
            total_steps,
            mask_schedule=None,
            device=mask_index.device,
        )
        idx = min(int(global_step), move_grid.numel() - 1)
        move = move_grid[idx].clamp_min(1e-12)
    return move.expand(mask_index.shape[0])


def build_llada_repellency_hook(tokenizer, safety: SafetySettings, device) -> Optional[Any]:
    """Build the LLaDA safe denoiser logits hook.

    Returns a closure `_logits_hook(logits, *, x, t, ...)` that is passed as
    `logits_hook=` to the upstream LLaDA generate() function. On the first call,
    it lazy-initializes MaskKernelRepellency. On subsequent calls, it applies
    repellency conditioning if t_start <= t <= t_end, then returns log(p_safe).

    Returns None if safety is disabled or no unsafe artifacts can be resolved.
    """
    LOGGER.info("Configuring LLaDA safe denoiser hook...")
    if not _QUIET:
        print("[safe_hook] build_llada_repellency_hook called", flush=True)
    if not safety.enabled:
        LOGGER.info("Safety disabled; skipping safe denoiser hook.")
        if not _QUIET:
            print("[safe_hook] safety disabled; skipping hook", flush=True)
        return None

    unsafe_path = safety.unsafe_artifacts
    if unsafe_path is None:
        try:
            unsafe_path = _prepare_unsafe_artifacts(safety, tokenizer=tokenizer)
        except Exception as exc:
            LOGGER.warning("Unsafe artifact resolution failed: %s", exc)
            if not _QUIET:
                print(f"[safe_hook] unsafe artifact resolution failed: {exc}", flush=True)
            return None
    if unsafe_path is None:
        LOGGER.warning("Safety enabled but no unsafe artifacts found; skipping safe denoiser hook.")
        if not _QUIET:
            print("[safe_hook] unsafe artifacts not found; skipping hook", flush=True)
        return None
    if os.getenv("SAFE_UNSAFE_LABEL") is None:
        unsafe_path_obj = Path(str(unsafe_path))
        label = unsafe_path_obj.parent.name or unsafe_path_obj.stem
        os.environ["SAFE_UNSAFE_LABEL"] = label

    mask_id = LLADA_MASK_TOKEN_ID
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None or (eos_id is not None and pad_id == eos_id):
        pad_id = ensure_pad_token(tokenizer, eos_token_id=eos_id)

    repellency: Optional[MaskKernelRepellency] = None
    configured_steps: Optional[int] = None
    safety_started = False
    safety_stopped = False

    def _logits_hook(
        logits: torch.Tensor,
        *,
        x: torch.Tensor,
        t: int,
        mask_index: torch.Tensor,
        prompt_index: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        extra: Dict[str, Any],
    ) -> torch.Tensor:
        nonlocal repellency, configured_steps, safety_started, safety_stopped

        prompt_width = int(extra.get("prompt_width") or (prompt_index.shape[1] if prompt_index is not None else 0))
        if prompt_width == x.shape[1] and (x == mask_id).any():
            pos = torch.arange(x.shape[1], device=x.device)[None, :].expand_as(x)
            masked_pos = torch.where(x == mask_id, pos, torch.full_like(pos, x.shape[1]))
            prompt_width = int(masked_pos.min(dim=1).values.min().item())
        total_steps = int(extra.get("total_steps") or max(prompt_width, 1))
        global_step = int(extra.get("global_step") or 0)

        if repellency is None:
            unsafe = _load_unsafe_tensor(str(unsafe_path)).to(device=device, dtype=torch.long)
            eta_value, eta_from_scale = resolve_eta_config(safety)
            weight_mode = safety.weight_mode
            beta_hat_mode = safety.beta_hat_mode
            beta_hat_clip_min = safety.beta_hat_clip_min
            beta_hat_clip_max = safety.beta_hat_clip_max
            schedule_mode = safety.schedule_mode
            stop_tokens = _resolve_stop_tokens(tokenizer)
            vocab_size = int(extra.get("vocab_size") or logits.shape[-1])
            ignore_ids = list(stop_tokens.guidance_ignore_ids) if stop_tokens else []
            ignore_ids = [idx for idx in ignore_ids if idx is not None and idx < vocab_size]
            t_end = safety.t_end if safety.t_end is not None else max(total_steps - 1, 0)
            embed_fn = extra.get("embed_fn") if safety.use_semantic_gating else None

            repellency = MaskKernelRepellency(
                ref_data=unsafe,
                embed_fn=embed_fn,
                forward_fn=None,
                num_timesteps=total_steps,
                max_idx=total_steps,
                beta_min=0.0,
                beta_max=0.0,
                vocab_size=vocab_size,
                mask_index=mask_id,
                pad_index=pad_id,
                eos_id=eos_id,
                ignore_ids=ignore_ids,
                scale=safety.scale,
                eta=eta_value,
                weight_mode=weight_mode,
                beta_hat_mode=beta_hat_mode,
                beta_hat_clip_min=beta_hat_clip_min,
                beta_hat_clip_max=beta_hat_clip_max,
                schedule_mode=schedule_mode,
                unsafe_prototypes_path=str(safety.unsafe_prototypes) if safety.unsafe_prototypes else None,
                critical_steps=safety.critical_steps,
                t_start=safety.t_start or 0,
                t_end=t_end,
                use_semantic_gating=safety.use_semantic_gating,
                semantic_weight=float(safety.semantic_weight),
                semantic_temp=float(safety.semantic_temp),
                semantic_sigma=safety.semantic_sigma,
                cache_semantic_ref=safety.cache_semantic_ref,
                semantic_ref_path=str(safety.semantic_ref_path) if safety.semantic_ref_path else None,
                alignment_strategy="left",
                tokenizer=tokenizer,
                unsafe_artifacts_name=safety.unsafe_artifact_name,
            )
            configured_steps = total_steps
            LOGGER.info(
                "Initialized LLaDA safe denoiser hook (unsafe=%s, eta=%.3f%s, t=[%s,%s], schedule=%s)",
                unsafe_path,
                eta_value,
                " (from scale)" if eta_from_scale else "",
                safety.t_start,
                t_end,
                schedule_mode,
            )

        if configured_steps != total_steps:
            repellency.num_timesteps = total_steps
            repellency.max_idx = total_steps
            if safety.t_end is None:
                repellency.t_end = max(total_steps - 1, 0)
            configured_steps = total_steps

        if safety.t_start is not None and safety.t_end is not None:
            if not safety_started and t >= safety.t_start:
                LOGGER.info("Safety applied at t_%s (end t_%s).", safety.t_start, safety.t_end)
                if not _QUIET:
                    print(f"[safe_hook] safety applied at t_{safety.t_start}", flush=True)
                safety_started = True
            if not (safety.t_start <= t <= safety.t_end):
                if not safety_stopped and t > safety.t_end:
                    LOGGER.info("Safety stopped applied at t_%s.", safety.t_end)
                    if not _QUIET:
                        print(f"[safe_hook] safety stopped at t_{safety.t_end}", flush=True)
                    safety_stopped = True
                return logits

        prompt_mask = torch.zeros_like(x, dtype=torch.bool)
        if attention_mask is not None:
            pm = attention_mask[:, :prompt_width].to(dtype=torch.bool)
            # Do NOT protect mask tokens inside the prompt span.
            pm = pm & (x[:, :prompt_width] != mask_id)
            prompt_mask[:, :prompt_width] = pm
        else:
            # No attention_mask: still don’t protect mask tokens.
            prompt_mask[:, :prompt_width] = (x[:, :prompt_width] != mask_id)

        # if LOGGER.isEnabledFor(logging.INFO):
        #     num_prompt_masks = int((x[:, :prompt_width] == mask_id).sum().item())
        #     num_protected_prompt_masks = int(
        #         (prompt_mask[:, :prompt_width] & (x[:, :prompt_width] == mask_id)).sum().item()
        #     )
        #     LOGGER.info(
        #         "hook: prompt_masks=%d protected_prompt_masks=%d",
        #         num_prompt_masks,
        #         num_protected_prompt_masks,
        #     )

        move = _compute_move_proxy(
            mask_index=mask_index,
            prompt_mask=prompt_mask,
            global_step=global_step,
            total_steps=total_steps,
        )

        probs = torch.softmax(logits, dim=-1)
        conditioned = repellency.conditioning(
            x_0_hat=probs,
            x_t=x,
            move=move,
            t_idx=t,
            prompt_mask=prompt_mask,
            prompt_width=prompt_width,
            prompt_id=extra.get("prompt_id"),
            prompt_variant=extra.get("prompt_variant"),
        )
        probs_safe = conditioned["x_0_hat"]
        probs_safe = probs_safe / probs_safe.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        logits_safe = torch.log(probs_safe.clamp_min(1e-12))
        return logits_safe

    return _logits_hook


def build_dream_repellency_hook(
    tokenizer,
    safety: SafetySettings,
    device,
    *,
    mask_token_id: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    prompt_width: Optional[int] = None,
    total_steps: Optional[int] = None,
    vocab_size: Optional[int] = None,
    embed_fn: Optional[Any] = None,
) -> Optional[Any]:
    LOGGER.info("Configuring Dream safe denoiser hook...")
    if not safety.enabled:
        LOGGER.info("Safety disabled; skipping safe denoiser hook.")
        return None

    unsafe_path = safety.unsafe_artifacts
    if unsafe_path is None:
        try:
            unsafe_path = _prepare_unsafe_artifacts(safety, tokenizer=tokenizer)
        except Exception as exc:
            LOGGER.warning("Unsafe artifact resolution failed: %s", exc)
            return None
    if unsafe_path is None:
        LOGGER.warning("Safety enabled but no unsafe artifacts found; skipping safe denoiser hook.")
        return None
    if os.getenv("SAFE_UNSAFE_LABEL") is None:
        unsafe_path_obj = Path(str(unsafe_path))
        label = unsafe_path_obj.parent.name or unsafe_path_obj.stem
        os.environ["SAFE_UNSAFE_LABEL"] = label

    mask_id = mask_token_id or getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        LOGGER.warning("Dream mask token id is missing; skipping safe denoiser hook.")
        return None

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None or (eos_id is not None and pad_id == eos_id):
        pad_id = ensure_pad_token(tokenizer, eos_token_id=eos_id)

    repellency: Optional[MaskKernelRepellency] = None
    configured_steps: Optional[int] = None

    def _logits_hook(step: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        nonlocal repellency, configured_steps

        prompt_len = int(x.shape[1] if prompt_width is None else prompt_width)
        if prompt_len == x.shape[1] and (x == mask_id).any():
            pos = torch.arange(x.shape[1], device=x.device)[None, :].expand_as(x)
            masked_pos = torch.where(x == mask_id, pos, torch.full_like(pos, x.shape[1]))
            prompt_len = int(masked_pos.min(dim=1).values.min().item())
        total = int(total_steps or max(prompt_len, 1))
        global_step = int(step)

        if repellency is None:
            unsafe = _load_unsafe_tensor(str(unsafe_path)).to(device=device, dtype=torch.long)
            eta_value, eta_from_scale = resolve_eta_config(safety)
            weight_mode = safety.weight_mode
            beta_hat_mode = safety.beta_hat_mode
            beta_hat_clip_min = safety.beta_hat_clip_min
            beta_hat_clip_max = safety.beta_hat_clip_max
            schedule_mode = safety.schedule_mode
            stop_tokens = _resolve_stop_tokens(tokenizer)
            vocab = int(vocab_size or logits.shape[-1])
            ignore_ids = list(stop_tokens.guidance_ignore_ids) if stop_tokens else []
            ignore_ids = [idx for idx in ignore_ids if idx is not None and idx < vocab]
            t_end = safety.t_end if safety.t_end is not None else max(total - 1, 0)

            repellency = MaskKernelRepellency(
                ref_data=unsafe,
                embed_fn=embed_fn if safety.use_semantic_gating else None,
                forward_fn=None,
                num_timesteps=total,
                max_idx=total,
                beta_min=0.0,
                beta_max=0.0,
                vocab_size=vocab,
                mask_index=mask_id,
                pad_index=pad_id,
                eos_id=eos_id,
                ignore_ids=ignore_ids,
                scale=safety.scale,
                eta=eta_value,
                weight_mode=weight_mode,
                beta_hat_mode=beta_hat_mode,
                beta_hat_clip_min=beta_hat_clip_min,
                beta_hat_clip_max=beta_hat_clip_max,
                schedule_mode=schedule_mode,
                unsafe_prototypes_path=str(safety.unsafe_prototypes) if safety.unsafe_prototypes else None,
                critical_steps=safety.critical_steps,
                t_start=safety.t_start or 0,
                t_end=t_end,
                use_semantic_gating=safety.use_semantic_gating,
                semantic_weight=float(safety.semantic_weight),
                semantic_temp=float(safety.semantic_temp),
                semantic_sigma=safety.semantic_sigma,
                cache_semantic_ref=safety.cache_semantic_ref,
                semantic_ref_path=str(safety.semantic_ref_path) if safety.semantic_ref_path else None,
                alignment_strategy="left",
                tokenizer=tokenizer,
                unsafe_artifacts_name=safety.unsafe_artifact_name,
            )
            configured_steps = total
            LOGGER.info(
                "Initialized Dream safe denoiser hook (unsafe=%s, eta=%.3f%s, t=[%s,%s], schedule=%s)",
                unsafe_path,
                eta_value,
                " (from scale)" if eta_from_scale else "",
                safety.t_start,
                t_end,
                schedule_mode,
            )

        if configured_steps != total:
            repellency.num_timesteps = total
            repellency.max_idx = total
            if safety.t_end is None:
                repellency.t_end = max(total - 1, 0)
            configured_steps = total

        if safety.t_start is not None and safety.t_end is not None:
            if not (safety.t_start <= global_step <= safety.t_end):
                return logits

        prompt_mask = torch.zeros_like(x, dtype=torch.bool)
        if attention_mask is not None:
            pm = attention_mask[:, :prompt_len].to(dtype=torch.bool)
            pm = pm & (x[:, :prompt_len] != mask_id)
            prompt_mask[:, :prompt_len] = pm
        else:
            prompt_mask[:, :prompt_len] = (x[:, :prompt_len] != mask_id)

        move = _compute_move_proxy(
            mask_index=(x == mask_id),
            prompt_mask=prompt_mask,
            global_step=global_step,
            total_steps=total,
        )

        probs = torch.softmax(logits, dim=-1)
        conditioned = repellency.conditioning(
            x_0_hat=probs,
            x_t=x,
            move=move,
            t_idx=global_step,
            prompt_mask=prompt_mask,
            prompt_width=prompt_len,
        )
        probs_safe = conditioned["x_0_hat"]
        probs_safe = probs_safe / probs_safe.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.log(probs_safe.clamp_min(1e-12))

    return _logits_hook
