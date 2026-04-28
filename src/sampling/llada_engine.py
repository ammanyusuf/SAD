from __future__ import annotations

import math
import os
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import lightning as L
import torch
import torch.nn.functional as F
import logging
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from sampling.sample_text import (
    GenerationResult,
    GenerationRun,
    GenerationSettings,
    ModelSettings,
    PromptRecord,
    SafetySettings,
    _chunk,
    resolve_eta_config,
    _prepare_unsafe_artifacts,
    _resolve_stop_tokens,
    _assert_no_extra_turn_tokens,
    _strip_completion_tokens,
    StopTokens,
)
from sampling.transfer_schedule import (
    compute_move_grid,
    get_num_transfer_tokens_move,
    get_num_transfer_tokens_uniform,
)
from utils.constants import LLADA_EOS_TOKEN_ID, LLADA_EOT_TOKEN_ID, LLADA_MASK_TOKEN_ID
from third_party.LLaDA.llada.repellency import MaskKernelRepellency
from third_party.mdlm.diffusion import _load_unsafe_tensor
from unsafe_prep.utils import ensure_pad_token

# =============================================================================
# LLaDA Experimental Setup (Reference)
# =============================================================================
# 1. Generation Termination:
#    - Autoregressive / Block Diffusion: Terminates on |EOS|.
#    - Pure Diffusion / Block LLaDA (semi-autoregressive):
#      - Base: Fixed length 1024.
#      - Instruct: Tuned {64, 256, 512}.
# 2. Remasking:
#    - Low-confidence remasking used for intra-block and pure diffusion.
# 3. Block Length:
#    - Base: Variable (tested multiple).
#    - Instruct: 32 (efficiency).
# 4. EOS Mitigation (Instruct Pure Diffusion):
#    - High EOS proportion observed.
#    - Mitigation: Set confidence of |EOS| to zero (or -inf logit/confidence).
#    - Implementation: logits_eos_inf=True, confidence_eos_eot_inf=True.
# =============================================================================


def _map_precision(precision: str) -> torch.dtype:
    lookup = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return lookup.get(precision.lower(), torch.float32)


def _safe_mean(items: Sequence[float]) -> Optional[float]:
    return float(sum(items) / len(items)) if items else None


_DEBUG_ENV_FLAG = "SAFE_LLADA_DEBUG"
_debug_logged_once = False


def _debug_enabled() -> bool:
    return os.getenv(_DEBUG_ENV_FLAG, "").lower() not in ("", "0", "false", "off", "no")


def _trim_to_eos(tokens: Sequence[int], eos_id: Optional[int]) -> List[int]:
    if eos_id is None:
        return list(tokens)
    for idx, tok in enumerate(tokens):
        if tok == eos_id:
            return list(tokens[: idx + 1])
    return list(tokens)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # LLaDA uses temperature as a power exponent here (not a softmax temperature).
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64).clamp_min(1e-20)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _resolve_transfer_schedule(
    transfer_schedule: Optional[str],
    safety_enabled: bool,
) -> str:
    if transfer_schedule:
        return transfer_schedule
    return "delta_move" if safety_enabled else "uniform"


@torch.no_grad()
def llada_generate(
    model,
    prompt: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    mask_id: int,
    effective_vocab: int,
    repellency: Optional[MaskKernelRepellency] = None,
    mask_schedule: Optional[Sequence[float]] = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eos_id: Optional[int] = None,
    eot_id: Optional[int] = None,
    pad_id: Optional[int] = None,
    stop_tokens: Optional[StopTokens] = None,
    extra_ban_ids: Optional[Iterable[int]] = None,
    sampling_mode: str = "pure_diffusion",
    transfer_schedule: Optional[str] = None,
    prompt_variant: Optional[str] = None,
    step_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    infill_prompt_masks: bool = False,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    LOGGER.info(
        "llada_generate: batch=%d, steps=%d, gen_length=%d, block_length=%d",
        prompt.shape[0],
        steps,
        gen_length,
        block_length,
    )

    ban_ids_initial = {mask_id}
    if pad_id is not None:
        ban_ids_initial.add(pad_id)
            
    if extra_ban_ids:
        ban_ids_initial.update(extra_ban_ids)

    ban_ids_filtered = [
        bid for bid in ban_ids_initial if bid is not None and bid < effective_vocab
    ]
    dropped_bans = [
        bid for bid in ban_ids_initial if bid is not None and bid >= effective_vocab
    ]
    if dropped_bans:
        LOGGER.warning(
            "Dropping banned token ids outside effective vocab (<=%d): %s",
            effective_vocab,
            dropped_bans,
        )
    ban_list = list(sorted(set(ban_ids_filtered)))
    LOGGER.info("Banning token IDs from logits: %s", ban_list)
    
    # Initialize sequence with mask tokens
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    # Create prompt_mask for repellency alignment
    prompt_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=model.device)
    prompt_width = prompt.shape[1]
    
    if attention_mask is not None:
        prompt_mask[:, :prompt_width] = attention_mask[:, :prompt_width].to(dtype=torch.bool)
    else:
        prompt_mask[:, :prompt_width] = True

    # if attention_mask is not None:
    #     pm = attention_mask[:, :prompt_width].to(dtype=torch.bool)
    #     # Do not protect prompt-region mask tokens in infill settings.
    #     pm = pm & (x[:, :prompt_width] != mask_id)
    #     prompt_mask[:, :prompt_width] = pm
    # else:
    #     # No attention mask available: still avoid protecting mask tokens.
    #     prompt_mask[:, :prompt_width] = (x[:, :prompt_width] != mask_id)
        
    prompt_mask[:, prompt_width:] = False

    expected_length = prompt_width + gen_length
    if x.shape[1] != expected_length:
        LOGGER.error(
            "Length mismatch: x.shape[1]=%d but prompt_width=%d + gen_length=%d = %d",
            x.shape[1],
            prompt_width,
            gen_length,
            expected_length,
        )
        raise ValueError(f"Sequence length mismatch: {x.shape[1]} != {expected_length}")

    # Expand attention mask for generation
    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=model.device,
                ),
            ],
            dim=-1,
        )

    prompt_index = prompt_mask

    target_start = 0 if infill_prompt_masks else prompt.shape[1]
    target_length = x.shape[1] - target_start
    if target_length <= 0:
        LOGGER.info("No target tokens to denoise (target_length=%d); returning input.", target_length)
        return x, {}

    if block_length > target_length:
        block_length = target_length
    if block_length <= 0:
        block_length = target_length
    num_blocks = target_length // block_length
    if target_length % block_length != 0:
        num_blocks += 1

    steps_per_block = steps // num_blocks
    if steps_per_block < 1:
        steps_per_block = 1
    total_steps = steps_per_block * num_blocks
    _, move_grid_total, delta_move_total = compute_move_grid(
        total_steps,
        mask_schedule=mask_schedule,
        device=model.device,
    )
    transfer_schedule_mode = transfer_schedule or "uniform"

    LOGGER.info(
        "llada_generate details: gen_length=%d, block_length=%d, num_blocks=%d, steps_per_block=%d",
        gen_length,
        block_length,
        num_blocks,
        steps_per_block,
    )
    global _debug_logged_once
    if _debug_enabled() and not _debug_logged_once:
        LOGGER.info(
            "SAFE_LLADA_DEBUG: prompt_len=%d steps=%d gen_length=%d block_length=%d",
            prompt.shape[1],
            steps,
            gen_length,
            block_length,
        )
        LOGGER.info("SAFE_LLADA_DEBUG: intermediate_decodes=False (final outputs only)")
        _debug_logged_once = True

    repellency_logs: Dict[str, List[float]] = {
        "mean_rho": [],
        "argmax_changed_masked": [],
        "beta_hat_mean": [],
        "beta_hat_p95": [],
        "beta_hat_max": [],
        "beta_hat_raw_mean": [],
        "beta_hat_len_mean": [],
        "guidance_strength_mean": [],
        "schedule_weight_mean": [],
        "log_beta_raw_mean": [],
        "log_beta_rel_mean": [],
        "log_beta_raw_max": [],
        "log_beta_rel_max": [],
        "mask_frac": [],
        "kl_logit_mean": [],
        "kl_prob_mean": [],
        "unsafe_shift_logit": [],
        "unsafe_shift_prob": [],
        "strength_zero_frac": [],
        "guidance_mode": [],
        "beta_hat_mode_active": [],
        "beta_hat_mode_alt": [],
        "tv_safe_data_mean": [],
        "tv_safe_unsafe_mean": [],
        "tv_data_unsafe_mean": [],
        "kl_safe_data_mean": [],
        "kl_safe_unsafe_mean": [],
        "kl_data_unsafe_mean": [],
        "js_safe_data_mean": [],
        "top1_change_rate": [],
        "top1_overlap_data_unsafe": [],
        "ess_weights": [],
        "max_weight": [],
        "entropy_weights": [],
        "effective_strength": [],
    }
    
    # Prepare prompt mask for repellency (if needed for slicing/alignment)
    # The Safe Denoiser uses it to identify the boundaries of prompt vs generation
    # if alignment strategy requires it.
    
    step_counter = 0
    progress = tqdm(
        total=steps_per_block * num_blocks,
        desc="llada_generate",
        leave=True,
        disable=False,
        mininterval=5.0,
    )

    def _verify_prompt_region(step_idx: int, transfer_index: torch.Tensor) -> None:
        if not _debug_enabled():
            return
        if infill_prompt_masks:
            # In infill mode, prompt-region masks are valid denoising targets.
            return
        prompt_tokens = x[:, :prompt_width]
        prompt_expected = prompt[:, :prompt_width]
        prompt_mismatch = (prompt_tokens != prompt_expected).any().item()
        prompt_masked = (prompt_tokens == mask_id).any().item()
        transfer_in_prompt = transfer_index[:, :prompt_width].any().item()
        if step_idx == 0 or prompt_mismatch or prompt_masked or transfer_in_prompt:
            LOGGER.info(
                "Verify initialization: step=%d prompt_len=%d mismatch=%s masked_in_prompt=%s transfer_in_prompt=%s",
                step_idx,
                prompt_width,
                bool(prompt_mismatch),
                bool(prompt_masked),
                bool(transfer_in_prompt),
            )
        if prompt_mismatch or prompt_masked or transfer_in_prompt:
            raise AssertionError(
                "Prompt region corrupted: mismatch=%s masked_in_prompt=%s transfer_in_prompt=%s"
                % (bool(prompt_mismatch), bool(prompt_masked), bool(transfer_in_prompt))
            )

    try:
        for num_block in range(num_blocks):
            block_start = target_start + num_block * block_length
            block_end = min(target_start + (num_block + 1) * block_length, x.shape[1])
            
            block_mask_index = (x[:, block_start:block_end] == mask_id)
            if not block_mask_index.any():
                progress.update(steps_per_block)
                continue
                
            block_step_start = num_block * steps_per_block
            block_move_grid = move_grid_total[
                block_step_start : block_step_start + steps_per_block + 1
            ]
            if transfer_schedule_mode == "delta_move":
                num_transfer_tokens = get_num_transfer_tokens_move(
                    block_mask_index,
                    steps_per_block,
                    block_move_grid,
                )
            else:
                num_transfer_tokens = get_num_transfer_tokens_uniform(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = (x == mask_id)
                vocab_assert = bool(os.getenv("LLADA_ASSERT_VOCAB"))
                
                # Model Forward Pass
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    attn_mask_ = None
                    if attention_mask is not None:
                        attn_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    logits = model(x_, attention_mask=attn_mask_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x, attention_mask=attention_mask).logits

                logits = logits.to(torch.float32)
                logits = logits[:, :, :effective_vocab]
                logits_vocab = logits.shape[-1]

                if logits_eos_inf and eos_id is not None and eos_id < logits_vocab:
                    logits[:, :, eos_id] = -float("inf")

                if ban_list:
                    ban_runtime = [bid for bid in ban_list if bid < logits_vocab]
                    if ban_runtime:
                        logits[:, :, ban_runtime] = -float('inf')

                # --- Repellency Logic ---
                if repellency is not None:
                    p_before = torch.softmax(logits, dim=-1)
                    
                    # Cap t_idx to schedule length if schedule is present
                    t_idx_safe = step_counter
                    if mask_schedule is not None:
                        t_idx_safe = min(step_counter, len(mask_schedule) - 1)

                    move = move_grid_total[
                        min(t_idx_safe, move_grid_total.numel() - 1)
                    ].clamp_min(1e-12)
                    move = move.expand(x.size(0))
                    if _debug_enabled():
                        assert bool((x[:, :prompt_width] != mask_id).all().item()), (
                            "Prompt region contains mask_id before repellency."
                        )

                    if _debug_enabled() and step_counter == 0:
                        LOGGER.info(
                            "Repellency first call: prompt_width=%d, x.shape=%s, prompt_mask.shape=%s",
                            prompt_width,
                            tuple(x.shape),
                            tuple(prompt_mask.shape) if prompt_mask is not None else None,
                        )
                    
                    prompt_mask_arg = None if infill_prompt_masks else prompt_mask
                    prompt_width_arg = None if infill_prompt_masks else prompt_width
                    conditioned = repellency.conditioning(
                        x_0_hat=p_before,
                        x_t=x,
                        move=move,
                        t_idx=step_counter,
                        prompt_mask=prompt_mask_arg,
                        prompt_width=prompt_width_arg,
                        prompt_variant=prompt_variant,
                    )
                    
                    p_safe = conditioned["x_0_hat"][..., :effective_vocab]
                    
                    rho_val = conditioned.get("mean_x_0_hat")
                    if rho_val is not None:
                        repellency_logs["mean_rho"].append(float(rho_val))
                    beta_hat_mean = conditioned.get("beta_hat_mean")
                    if beta_hat_mean is not None:
                        repellency_logs["beta_hat_mean"].append(float(beta_hat_mean))
                    beta_hat_p95 = conditioned.get("beta_hat_p95")
                    if beta_hat_p95 is not None:
                        repellency_logs["beta_hat_p95"].append(float(beta_hat_p95))
                    beta_hat_max = conditioned.get("beta_hat_max")
                    if beta_hat_max is not None:
                        repellency_logs["beta_hat_max"].append(float(beta_hat_max))
                    beta_hat_raw_mean = conditioned.get("beta_hat_raw_mean")
                    if beta_hat_raw_mean is not None:
                        repellency_logs["beta_hat_raw_mean"].append(float(beta_hat_raw_mean))
                    beta_hat_len_mean = conditioned.get("beta_hat_len_mean")
                    if beta_hat_len_mean is not None:
                        repellency_logs["beta_hat_len_mean"].append(float(beta_hat_len_mean))
                    strength_mean = conditioned.get("guidance_strength_mean")
                    if strength_mean is not None:
                        repellency_logs["guidance_strength_mean"].append(float(strength_mean))
                    schedule_mean = conditioned.get("schedule_weight_mean")
                    if schedule_mean is not None:
                        repellency_logs["schedule_weight_mean"].append(float(schedule_mean))
                    log_beta_raw_mean = conditioned.get("log_beta_raw_mean")
                    if log_beta_raw_mean is not None:
                        repellency_logs["log_beta_raw_mean"].append(float(log_beta_raw_mean))
                    log_beta_raw_max = conditioned.get("log_beta_raw_max")
                    if log_beta_raw_max is not None:
                        repellency_logs["log_beta_raw_max"].append(float(log_beta_raw_max))
                    log_beta_rel_mean = conditioned.get("log_beta_rel_mean")
                    if log_beta_rel_mean is not None:
                        repellency_logs["log_beta_rel_mean"].append(float(log_beta_rel_mean))
                    log_beta_rel_max = conditioned.get("log_beta_rel_max")
                    if log_beta_rel_max is not None:
                        repellency_logs["log_beta_rel_max"].append(float(log_beta_rel_max))
                    mask_frac = conditioned.get("mask_frac")
                    if mask_frac is not None:
                        repellency_logs["mask_frac"].append(float(mask_frac))
                    kl_logit_mean = conditioned.get("kl_logit_mean")
                    if kl_logit_mean is not None:
                        repellency_logs["kl_logit_mean"].append(float(kl_logit_mean))
                    kl_prob_mean = conditioned.get("kl_prob_mean")
                    if kl_prob_mean is not None:
                        repellency_logs["kl_prob_mean"].append(float(kl_prob_mean))
                    unsafe_shift_logit = conditioned.get("unsafe_shift_logit")
                    if unsafe_shift_logit is not None:
                        repellency_logs["unsafe_shift_logit"].append(float(unsafe_shift_logit))
                    unsafe_shift_prob = conditioned.get("unsafe_shift_prob")
                    if unsafe_shift_prob is not None:
                        repellency_logs["unsafe_shift_prob"].append(float(unsafe_shift_prob))
                    strength_zero_frac = conditioned.get("strength_zero_frac")
                    if strength_zero_frac is not None:
                        repellency_logs["strength_zero_frac"].append(float(strength_zero_frac))
                    guidance_mode_val = conditioned.get("guidance_mode")
                    if guidance_mode_val is not None:
                        repellency_logs.setdefault("guidance_mode", []).append(str(guidance_mode_val))
                    beta_hat_mode_active = conditioned.get("beta_hat_mode_active")
                    if beta_hat_mode_active is not None:
                        repellency_logs.setdefault("beta_hat_mode_active", []).append(str(beta_hat_mode_active))
                    beta_hat_mode_alt = conditioned.get("beta_hat_mode_alt")
                    if beta_hat_mode_alt is not None:
                        repellency_logs.setdefault("beta_hat_mode_alt", []).append(str(beta_hat_mode_alt))
                    for metric_key in (
                        "tv_safe_data_mean",
                        "tv_safe_unsafe_mean",
                        "tv_data_unsafe_mean",
                        "kl_safe_data_mean",
                        "kl_safe_unsafe_mean",
                        "kl_data_unsafe_mean",
                        "js_safe_data_mean",
                        "top1_change_rate",
                        "top1_overlap_data_unsafe",
                        "ess_weights",
                        "max_weight",
                        "entropy_weights",
                        "effective_strength",
                    ):
                        metric_val = conditioned.get(metric_key)
                        if metric_val is not None:
                            repellency_logs.setdefault(metric_key, []).append(float(metric_val))
                    
                    argmax_before = p_before.argmax(dim=-1)
                    argmax_after = p_safe.argmax(dim=-1)
                    if mask_index.any():
                        changed = (argmax_before != argmax_after) & mask_index
                        repellency_logs["argmax_changed_masked"].append(changed.float().mean().item())

                    p_safe = p_safe.clamp_min(1e-30)
                    p_safe = p_safe / p_safe.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    logits = torch.log(p_safe)
                    
                    # re-apply bans after repellency
                    if ban_list:
                        logits[:, :, ban_list] = -float('inf')
                
                # Sampling
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                
                # Confidence Calculation
                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                if confidence_eos_eot_inf:
                    if eos_id is not None and eos_id < effective_vocab:
                        x0_p = x0_p.masked_fill(x0 == eos_id, -torch.inf)
                    if eot_id is not None and eot_id < effective_vocab:
                        x0_p = x0_p.masked_fill(x0 == eot_id, -torch.inf)

                # Set confidence of future blocks to -inf
                x0_p[:, block_end:] = -torch.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -torch.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                # vectorized variable-topk per row.
                # k_per_row: [B] for this step i
                k_per_row = num_transfer_tokens[:, i].to(torch.int64)  # [B]
                max_k = int(k_per_row.max().item()) if k_per_row.numel() else 0
                if max_k > 0:
                    # topk over sequence dimension (dim=1), returns [B, max_k]
                    _, top_idx = torch.topk(confidence, k=max_k, dim=1)
                    # build a mask selecting first k_per_row indices per row
                    rank = torch.arange(max_k, device=confidence.device, dtype=torch.int64)[None, :]  # [1, max_k]
                    take = rank < k_per_row[:, None]  # [B, max_k]
                    # scatter True into transfer_index
                    transfer_index.scatter_(1, top_idx, take)

                _verify_prompt_region(step_counter, transfer_index)
                x[transfer_index] = x0[transfer_index]
                if step_callback is not None:
                    step_callback(step_counter, x)
                if os.getenv("SAFE_REPELLENCY_DEBUG"):
                    planned = int(num_transfer_tokens[0, i].item()) if num_transfer_tokens.numel() else 0
                    actual = int(transfer_index[0].sum().item()) if transfer_index.numel() else 0
                    move_val = float(
                        move_grid_total[min(step_counter, move_grid_total.numel() - 1)].item()
                    )
                    delta_val = float(
                        delta_move_total[min(step_counter, delta_move_total.numel() - 1)].item()
                    ) if delta_move_total.numel() else 0.0
                    LOGGER.info(
                        "SAFE_REPELLENCY_DEBUG: step=%d move=%.6f delta_move=%.6f planned_transfer=%d actual_transfer=%d",
                        step_counter,
                        move_val,
                        delta_val,
                        planned,
                        actual,
                    )
                
                if sampling_mode != "pure_diffusion" and eos_id is not None and eos_id < effective_vocab:
                    gen_region = x[:, prompt_width:]
                    eos_mask = gen_region == eos_id
                    if eos_mask.any():
                        for row_idx in range(eos_mask.shape[0]):
                            if eos_mask[row_idx].any():
                                first_eos = int(torch.nonzero(eos_mask[row_idx], as_tuple=False)[0].item())
                                tail_start = prompt_width + first_eos + 1
                                x[row_idx, tail_start:] = eos_id
                
                if vocab_assert:
                    assert int((x >= effective_vocab).sum().item()) == 0
                
                step_counter += 1
                progress.update(1)

    finally:
        progress.close()
        if repellency is not None and hasattr(repellency, "flush_metrics"):
            repellency.flush_metrics()

    LOGGER.info(
        "llada_generate complete: %d total steps, masked tokens remaining=%d",
        step_counter,
        int((x == mask_id).sum().item()),
    )
    return x, repellency_logs


class LLaDAGenerationEngine:
    """Sampling wrapper for LLaDA with optional repellency."""

    def __init__(
        self,
        prompts: Optional[Sequence[PromptRecord]],
        model: ModelSettings,
        generation: GenerationSettings,
        safety: SafetySettings,
        shard_metadata: Dict[str, Any],
    ) -> None:
        self.prompts: List[PromptRecord] = list(prompts) if prompts else []
        self.model_settings = model
        self.generation_settings = generation
        self.safety_settings = safety
        self.shard_metadata = shard_metadata
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.model = None
        self.mask_token_id: Optional[int] = None
        self.stop_tokens: Optional[StopTokens] = None
        self.repellency: Optional[MaskKernelRepellency] = None
        self._repellency_logs: Dict[str, List[float]] = {
            "mean_rho": [],
            "argmax_changed_masked": [],
            "beta_hat_mean": [],
            "beta_hat_p95": [],
            "beta_hat_max": [],
            "beta_hat_raw_mean": [],
            "beta_hat_len_mean": [],
            "guidance_strength_mean": [],
            "schedule_weight_mean": [],
            "log_beta_raw_mean": [],
            "log_beta_rel_mean": [],
            "log_beta_raw_max": [],
            "log_beta_rel_max": [],
            "mask_frac": [],
            "kl_logit_mean": [],
            "kl_prob_mean": [],
            "unsafe_shift_logit": [],
            "unsafe_shift_prob": [],
            "strength_zero_frac": [],
            "guidance_mode": [],
            "beta_hat_mode_active": [],
            "beta_hat_mode_alt": [],
            "tv_safe_data_mean": [],
            "tv_safe_unsafe_mean": [],
            "tv_data_unsafe_mean": [],
            "kl_safe_data_mean": [],
            "kl_safe_unsafe_mean": [],
            "kl_data_unsafe_mean": [],
            "js_safe_data_mean": [],
            "top1_change_rate": [],
            "top1_overlap_data_unsafe": [],
            "ess_weights": [],
            "max_weight": [],
            "entropy_weights": [],
            "effective_strength": [],
        }
        self.effective_vocab: Optional[int] = None
        self.model_vocab: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.eot_id: Optional[int] = None
        self._ban_generation_ids: Set[int] = set()

    def run(self) -> GenerationRun:
        start_time = time.perf_counter()
        if not self.prompts and self.generation_settings.unconditional_samples <= 0:
            timings = {
                "load_seconds": 0.0,
                "generation_seconds": 0.0,
                "total_seconds": 0.0,
                "peak_vram_bytes": 0,
                "repellency": {},
            }
            return GenerationRun(results=[], timings=timings, resolved_config={})

        load_start = time.perf_counter()
        self._prepare_model()
        load_seconds = time.perf_counter() - load_start

        generation_start = time.perf_counter()
        results = []
        results.extend(self._generate_conditioned())
        results.extend(self._generate_unconditional())
        self.logger.info("Finished LLaDA sampling for %d prompts.", len(results))
        generation_seconds = time.perf_counter() - generation_start
        total_seconds = time.perf_counter() - start_time

        repellency_stats = {
            "mean_rho": _safe_mean(self._repellency_logs.get("mean_rho", [])),
            "argmax_changed_masked": _safe_mean(
                self._repellency_logs.get("argmax_changed_masked", [])
            ),
            "beta_hat_mean": _safe_mean(self._repellency_logs.get("beta_hat_mean", [])),
            "beta_hat_p95": _safe_mean(self._repellency_logs.get("beta_hat_p95", [])),
            "beta_hat_max": _safe_mean(self._repellency_logs.get("beta_hat_max", [])),
            "beta_hat_raw_mean": _safe_mean(self._repellency_logs.get("beta_hat_raw_mean", [])),
            "beta_hat_len_mean": _safe_mean(self._repellency_logs.get("beta_hat_len_mean", [])),
            "guidance_strength_mean": _safe_mean(
                self._repellency_logs.get("guidance_strength_mean", [])
            ),
            "schedule_weight_mean": _safe_mean(
                self._repellency_logs.get("schedule_weight_mean", [])
            ),
            "log_beta_raw_mean": _safe_mean(self._repellency_logs.get("log_beta_raw_mean", [])),
            "log_beta_rel_mean": _safe_mean(self._repellency_logs.get("log_beta_rel_mean", [])),
            "log_beta_raw_max": _safe_mean(self._repellency_logs.get("log_beta_raw_max", [])),
            "log_beta_rel_max": _safe_mean(self._repellency_logs.get("log_beta_rel_max", [])),
            "mask_frac": _safe_mean(self._repellency_logs.get("mask_frac", [])),
            "kl_logit_mean": _safe_mean(self._repellency_logs.get("kl_logit_mean", [])),
            "kl_prob_mean": _safe_mean(self._repellency_logs.get("kl_prob_mean", [])),
            "unsafe_shift_logit": _safe_mean(self._repellency_logs.get("unsafe_shift_logit", [])),
            "unsafe_shift_prob": _safe_mean(self._repellency_logs.get("unsafe_shift_prob", [])),
            "strength_zero_frac": _safe_mean(self._repellency_logs.get("strength_zero_frac", [])),
        }

        timings = {
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "total_seconds": total_seconds,
            "peak_vram_bytes": self._capture_peak_vram(),
            "repellency": repellency_stats,
        }
        return GenerationRun(
            results=results,
            timings=timings,
            resolved_config=self._resolved_config(),
        )

    def _prepare_model(self) -> None:
        L.seed_everything(self.generation_settings.seed)
        _prepare_unsafe_artifacts(self.safety_settings)
        checkpoint_path = str(self.model_settings.checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
        if tokenizer.padding_side != "left":
            tokenizer.padding_side = "left"
        
        def tok_id(tok: str) -> Optional[int]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is None:
                return None
            if hasattr(tokenizer, "unk_token_id") and tid == tokenizer.unk_token_id:
                return None
            return int(tid)

        start_header_id = tok_id("<|start_header_id|>")
        end_header_id = tok_id("<|end_header_id|>")
        eot_id = tok_id("<|eot_id|>")
        bos_id = tok_id("<|startoftext|>") or tok_id("<|begin_of_text|>")
        eos_id = tokenizer.eos_token_id

        # fallback for LLaDA hardcoded IDs if not resolved by name (they hardcoded the IDs)
        if "llada" in self.model_settings.model_name.lower():
             if eot_id is None:
                 self.logger.info("Manually resolved eot_id=%d for LLaDA", LLADA_EOT_TOKEN_ID)
                 eot_id = LLADA_EOT_TOKEN_ID
             if eos_id is None:
                 self.logger.info("Manually resolved eos_id=%d for LLaDA", LLADA_EOS_TOKEN_ID)
                 eos_id = LLADA_EOS_TOKEN_ID

        pad_id = ensure_pad_token(tokenizer, eos_token_id=eos_id)
        tokenizer.pad_token_id = pad_id
        if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_id)
        if hasattr(tokenizer, "pad_token_id"):
            tokenizer.pad_token_id = pad_id

        self.logger.info(
            "Resolved control token ids: start_header=%s end_header=%s eot=%s bos=%s eos=%s",
            start_header_id, end_header_id, eot_id, bos_id, eos_id
        )

        self._ban_generation_ids = {x for x in [start_header_id, end_header_id, eot_id] if x is not None}
        
        # re-construct StopTokens with explicitly resolved IDs
        self.stop_tokens = StopTokens(
            pad_id=pad_id,
            eos_id=eos_id,
            eot_id=eot_id,
            eom_id=tok_id("<|eom_id|>"),
            start_header_id=start_header_id,
            end_header_id=end_header_id,
            bos_id=bos_id,
            stop_sequences=_resolve_stop_tokens(tokenizer).stop_sequences,
        )
        self.eos_id = eos_id
        self.eot_id = eot_id

        dtype = _map_precision(self.model_settings.precision)
        model = AutoModel.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        model.eval()
        self.logger.info(
            "Loaded LLaDA checkpoint from %s (dtype=%s, device=%s)",
            checkpoint_path,
            dtype,
            self.device,
        )

        mask_id = self._resolve_mask_id(tokenizer)
        self.effective_vocab = len(tokenizer)
        emb_layer = getattr(model, "get_input_embeddings", lambda: None)()
        self.model_vocab = (
            emb_layer.weight.shape[0]
            if emb_layer is not None and hasattr(emb_layer, "weight")
            else getattr(model.config, "vocab_size", None)
        )
        if self.model_vocab is not None and pad_id is not None and pad_id >= self.model_vocab:
            self.logger.info(
                "Resizing token embeddings to accommodate pad_token_id=%s (model_vocab=%s, tokenizer_len=%s)",
                pad_id,
                self.model_vocab,
                len(tokenizer),
            )
            model.resize_token_embeddings(len(tokenizer))
            emb_layer = getattr(model, "get_input_embeddings", lambda: None)()
            self.model_vocab = (
                emb_layer.weight.shape[0]
                if emb_layer is not None and hasattr(emb_layer, "weight")
                else getattr(model.config, "vocab_size", None)
            )
        if hasattr(model, "config"):
            try:
                model.config.pad_token_id = pad_id
            except Exception:
                pass
        self.logger.info(
            "Resolved vocab sizes: effective_vocab=%s, model_embedding_vocab=%s",
            self.effective_vocab,
            self.model_vocab,
        )
        
        def _fmt(tid):
            if tid is None: return "None"
            try:
                t = tokenizer.convert_ids_to_tokens(tid)
                return f"{tid} ('{t}')"
            except Exception:
                return f"{tid} (unknown)"

        self.logger.info(
            "Resolved control tokens:\n"
            "  mask_token_id:  %s\n"
            "  pad_token_id:   %s\n"
            "  eos_token_id:   %s\n"
            "  bos_token_id:   %s\n"
            "  eot_token_id:   %s\n"
            "  eom_token_id:   %s\n"
            "  start_header:   %s\n"
            "  end_header:     %s",
            _fmt(mask_id),
            _fmt(pad_id),
            _fmt(eos_id),
            _fmt(bos_id),
            _fmt(eot_id),
            _fmt(self.stop_tokens.eom_id if self.stop_tokens else None),
            _fmt(start_header_id),
            _fmt(end_header_id),
        )
        self.logger.info("Banning IDs for generation: %s", [(_fmt(x)) for x in sorted(self._ban_generation_ids)])
        self.tokenizer = tokenizer
        self.model = model
        self.mask_token_id = mask_id
        os.environ.setdefault("SAFE_PROMPT_VARIANT", "unknown")
        self._build_repellency()

    def _resolve_mask_id(self, tokenizer: PreTrainedTokenizerBase) -> int:
        """Find the correct mask token id without mutating the tokenizer."""
        upstream_id = LLADA_MASK_TOKEN_ID
        upstream_token = tokenizer.convert_ids_to_tokens(upstream_id)
        if upstream_token and upstream_token != tokenizer.unk_token:
            self.logger.info("Using documented LLaDA mask id=%d token=%s", upstream_id, upstream_token)
            return upstream_id

        if tokenizer.mask_token_id is not None:
            token = tokenizer.mask_token
            self.logger.info("Using tokenizer.mask_token_id=%s token=%s", tokenizer.mask_token_id, token)
            return int(tokenizer.mask_token_id)

        candidates = ["[MASK]", "<mask>", "mask", "▁mask"]
        for tok in candidates:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            if tok_id != tokenizer.unk_token_id:
                self.logger.info("Resolved mask token %s -> id=%s", tok, tok_id)
                return int(tok_id)

        raise SystemExit("Could not resolve LLaDA mask token id from tokenizer without adding new tokens.")

    def _build_repellency(self) -> None:
        if not self.safety_settings.enabled:
            return
        unsafe_path = self.safety_settings.unsafe_artifacts
        if unsafe_path is None:
            unsafe_path = _prepare_unsafe_artifacts(self.safety_settings)
        if not self.safety_settings.enabled or unsafe_path is None:
            return
        unsafe = _load_unsafe_tensor(str(unsafe_path)).to(torch.long)
        eta_value, eta_from_scale = resolve_eta_config(self.safety_settings)
        weight_mode = self.safety_settings.weight_mode
        if weight_mode != "eta_beta_hat":
            self.logger.warning(
                "safety.weight_mode=%s may decouple eta from beta_hat; set weight_mode=eta_beta_hat to ensure beta_hat controls strength.",
                weight_mode,
            )
        beta_hat_mode = self.safety_settings.beta_hat_mode
        beta_hat_clip_min = self.safety_settings.beta_hat_clip_min
        beta_hat_clip_max = self.safety_settings.beta_hat_clip_max
        schedule_mode = self.safety_settings.schedule_mode
        stop_tokens = self.stop_tokens
        pad_index = self.tokenizer.pad_token_id if self.tokenizer else None
        eos_index = self.tokenizer.eos_token_id if self.tokenizer else None
        vocab_size = int(self.effective_vocab or (len(self.tokenizer) if self.tokenizer else 0))
        if stop_tokens is not None:
            ignore_ids = list(stop_tokens.guidance_ignore_ids)
        else:
            ignore_ids = [idx for idx in (pad_index, eos_index) if idx is not None]
        ignore_ids = [idx for idx in ignore_ids if idx is not None and idx < vocab_size]
        t_end = self.safety_settings.t_end
        if t_end is None:
            t_end = self.generation_settings.sampling_steps - 1
        embed_fn = None
        if self.safety_settings.use_semantic_gating:
            def _semantic_embed(tokens: torch.Tensor, model=self.model) -> torch.Tensor:
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
            embed_fn = _semantic_embed
        vocab_size = int(self.effective_vocab or (len(self.tokenizer) if self.tokenizer else 0))
        model_vocab = self.model_vocab if self.model_vocab is not None else getattr(self.model.config, "vocab_size", None)
        if model_vocab is not None and vocab_size != model_vocab:
            self.logger.info(
                "Repellency using effective_vocab=%d (tokenizer length=%s, model embedding rows=%s)",
                vocab_size,
                len(self.tokenizer) if self.tokenizer is not None else "unknown",
                model_vocab,
            )
        if torch.is_tensor(unsafe):
            unsafe = unsafe.to(torch.long)
            vocab_oob = unsafe >= vocab_size
            if vocab_oob.any():
                replace_id_candidates = [
                    idx for idx in (self.mask_token_id, pad_index, 0) if idx is not None and idx < vocab_size
                ]
                replace_id = replace_id_candidates[0] if replace_id_candidates else 0
                if unsafe.dim() == 1:
                    unsafe = unsafe[~vocab_oob]
                else:
                    unsafe = unsafe.clone()
                    unsafe[vocab_oob] = replace_id
                self.logger.warning(
                    "Filtered/clamped %d unsafe token ids outside vocab_size=%d",
                    int(vocab_oob.sum().item()),
                    vocab_size,
                )
        self.repellency = MaskKernelRepellency(
            ref_data=unsafe,
            embed_fn=embed_fn,
            forward_fn=None,
            num_timesteps=self.generation_settings.sampling_steps,
            max_idx=self.generation_settings.sampling_steps,
            beta_min=0.0,
            beta_max=0.0,
            vocab_size=vocab_size,
            mask_index=self.mask_token_id,
            pad_index=pad_index,
            eos_id=eos_index,
            ignore_ids=ignore_ids,
            scale=self.safety_settings.scale,
            eta=eta_value,
            weight_mode=weight_mode,
            beta_hat_mode=beta_hat_mode,
            beta_hat_clip_min=beta_hat_clip_min,
            beta_hat_clip_max=beta_hat_clip_max,
            schedule_mode=schedule_mode,
            unsafe_prototypes_path=(
                str(self.safety_settings.unsafe_prototypes)
                if self.safety_settings.unsafe_prototypes
                else None
            ),
            critical_steps=self.safety_settings.critical_steps,
            t_start=self.safety_settings.t_start or 0,
            t_end=t_end,
            use_semantic_gating=self.safety_settings.use_semantic_gating,
            semantic_weight=float(self.safety_settings.semantic_weight),
            semantic_temp=float(self.safety_settings.semantic_temp),
            semantic_sigma=self.safety_settings.semantic_sigma,
            cache_semantic_ref=self.safety_settings.cache_semantic_ref,
            semantic_ref_path=(
                str(self.safety_settings.semantic_ref_path)
                if self.safety_settings.semantic_ref_path
                else None
            ),
            alignment_strategy="left",
            tokenizer=self.tokenizer,
            unsafe_artifacts_name=self.safety_settings.unsafe_artifact_name,
        )
        self.logger.info(
            "Repellency enabled for LLaDA (unsafe_tensor=%s, eta=%.3f%s, t=[%s,%s], schedule=%s, semantic=%s, prototypes=%s, beta_hat_mode=%s)",
            unsafe_path,
            eta_value,
            " (from scale)" if eta_from_scale else "",
            self.safety_settings.t_start,
            t_end,
            schedule_mode,
            self.safety_settings.use_semantic_gating,
            self.safety_settings.unsafe_prototypes,
            beta_hat_mode,
        )

    def _generate_conditioned(self) -> List[GenerationResult]:
        if not self.prompts or self.tokenizer is None or self.model is None:
            return []
        if self.mask_token_id is None:
            raise RuntimeError("Mask token id must be resolved before generation.")
        results: List[GenerationResult] = []
        stop_tokens = self.stop_tokens
        if stop_tokens is None and self.tokenizer is not None:
            stop_tokens = _resolve_stop_tokens(self.tokenizer)
        stop_ids: Set[int] = stop_tokens.stop_ids if stop_tokens else set()
        total = len(self.prompts)
        processed = 0
        
        with tqdm(total=total, desc="Prompts", unit="sample") as pbar:
            for batch in _chunk(self.prompts, self.generation_settings.batch_size):
                batch_size = len(batch)
                pbar.set_description(f"Prompts {processed}-{processed + batch_size}")
                processed += batch_size

                def _coerce_prompt_variant(record: PromptRecord) -> Optional[str]:
                    meta = record.metadata or {}
                    variant = meta.get("prompt_variant")
                    if variant is not None:
                        return str(variant)
                    if "prompt_is_safe" in meta:
                        return "benign" if bool(meta["prompt_is_safe"]) else "unsafe"
                    return None

                variants = [v for v in (_coerce_prompt_variant(r) for r in batch) if v is not None]
                prompt_variant = None
                if variants:
                    prompt_variant = variants[0] if all(v == variants[0] for v in variants) else "mixed"
                if prompt_variant is not None:
                    os.environ["SAFE_PROMPT_VARIANT"] = str(prompt_variant)
                
                prompts_text = [record.prompt for record in batch]
                # template
                use_chat_template = getattr(self.tokenizer, "chat_template", None) is not None
                if "llada-8b-base" in self.model_settings.model_name:
                    use_chat_template = False
                elif "llada-8b-instruct" in self.model_settings.model_name:
                    use_chat_template = True
                else:
                    use_chat_template = False
                    LOGGER.error("Chat template not supported for model: %s", self.model_settings.model_name)

                if use_chat_template:
                    prompts_text = [
                        self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": text}],
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                        for text in prompts_text
                    ]
                # LOGGER.info("Tokenizing %d prompts for LLaDA generation.", len(prompts_text))
                # LOGGER.info("Example prompt: %s", prompts_text[0] if prompts_text else "")
                encoded = self.tokenizer(
                    prompts_text,
                    add_special_tokens=not use_chat_template,
                    padding=True,
                    return_tensors="pt",
                )
                mask_id = int(self.mask_token_id)
                input_ids = encoded["input_ids"].to(self.device)
                attn_mask = encoded.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(self.device)
                prompt_width = input_ids.shape[1]
                if _debug_enabled() and prompts_text:
                    LOGGER.info(
                        "Prompt text (batch[0], width=%d). Prompt region is fixed and never masked.",
                        prompt_width,
                    )
                    LOGGER.info("Prompt[0]: %s", prompts_text[0])

                steps = min(
                    self.generation_settings.sampling_steps,
                    self.generation_settings.max_new_tokens,
                )
                logits_eos_inf = False
                confidence_eos_eot_inf = False
                if "llada-8b-instruct" in self.model_settings.model_name:
                    logits_eos_inf = True
                    confidence_eos_eot_inf = True
                if self.generation_settings.block_length is not None:
                    block_length = self.generation_settings.block_length
                else:
                    block_length = min(
                        self.generation_settings.max_new_tokens,
                        self.generation_settings.sampling_steps,
                    )
                transfer_schedule = _resolve_transfer_schedule(
                    getattr(self.generation_settings, "transfer_schedule", None),
                    self.safety_settings.enabled,
                )
                samples, rep_logs = llada_generate(
                    model=self.model,
                    prompt=input_ids,
                    attention_mask=attn_mask,
                    steps=steps,
                    gen_length=self.generation_settings.max_new_tokens,
                    block_length=block_length,
                    temperature=self.generation_settings.temperature,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=mask_id,
                    repellency=self.repellency,
                    effective_vocab=int(self.effective_vocab if self.effective_vocab is not None else len(self.tokenizer)),
                    logits_eos_inf=logits_eos_inf,
                    confidence_eos_eot_inf=confidence_eos_eot_inf,
                    eos_id=self.eos_id,
                    eot_id=self.eot_id,
                    pad_id=self.tokenizer.pad_token_id if self.tokenizer else None,
                    stop_tokens=self.stop_tokens,
                    extra_ban_ids=self._ban_generation_ids,
                    sampling_mode=self.generation_settings.sampling_mode,
                    transfer_schedule=transfer_schedule,
                    prompt_variant=prompt_variant,
                )

                for key, vals in rep_logs.items():
                    self._repellency_logs.setdefault(key, []).extend(vals)
    
                if attn_mask is not None:
                    true_prompt_lengths = attn_mask[:, :prompt_width].sum(dim=1).tolist()
                else:
                    true_prompt_lengths = [prompt_width] * input_ids.shape[0]
                
                for row, record in enumerate(batch):
                    tokens = samples[row].tolist()
                    true_prompt_len = int(true_prompt_lengths[row])
                    prompt_len = prompt_width  # use padded width to skip all prompt (incl. padding)
                    if attn_mask is not None:
                        prompt_mask = attn_mask[row, :prompt_width].to(torch.int64).tolist()
                    else:
                        prompt_mask = [1] * int(prompt_len)
                    if len(tokens) > len(prompt_mask):
                        prompt_mask.extend([0] * (len(tokens) - len(prompt_mask)))
                    
                    completion_tokens, _, _ = _strip_completion_tokens(
                        tokens,
                        prompt_len,
                        stop_ids,
                        mask_id,
                        stop_sequences=stop_tokens.stop_sequences if stop_tokens else (),
                    )
                    completion_tokens = _trim_to_eos(completion_tokens, self.eos_id)

                    raw_completion_text = self.tokenizer.decode(
                        completion_tokens,
                        skip_special_tokens=False,
                    )
                    completion_text = raw_completion_text.strip()
                    # LOGGER.info("Cleaned completion text: %s", completion_text)
                    
                    if stop_tokens is not None:
                        _assert_no_extra_turn_tokens(
                            completion_tokens=completion_tokens,
                            decoded_completion=raw_completion_text,
                            prompt_length=prompt_len,
                            tokens=tokens,
                            stop_tokens=stop_tokens,
                            logger=self.logger,
                        )
                    
                    metadata = {
                        **record.metadata,
                        **self.shard_metadata,
                        "safe_sampling_enabled": bool(self.repellency),
                        "repellency_mean_rho": _safe_mean(
                            self._repellency_logs.get("mean_rho", [])
                        ),
                        "true_prompt_len": true_prompt_len,
                        "prompt_width": prompt_width,
                    }
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
                pbar.update(batch_size)
        return results

    def _generate_unconditional(self) -> List[GenerationResult]:
        if self.generation_settings.unconditional_samples <= 0:
            return []
        if self.tokenizer is None or self.model is None or self.mask_token_id is None:
            return []
        results: List[GenerationResult] = []
        stop_tokens = self.stop_tokens
        if stop_tokens is None and self.tokenizer is not None:
            stop_tokens = _resolve_stop_tokens(self.tokenizer)
        stop_ids: Set[int] = stop_tokens.stop_ids if stop_tokens else set()
        remaining = self.generation_settings.unconditional_samples
        counter = 0

        while remaining > 0:
            batch_size = min(self.generation_settings.batch_size, remaining)
            prompt_tensor = torch.empty(
                (batch_size, 0),
                dtype=torch.long,
                device=self.device,
            )
            samples, rep_logs = llada_generate(
                model=self.model,
                prompt=prompt_tensor,
                attention_mask=None,
                steps=min(
                    self.generation_settings.sampling_steps,
                    self.generation_settings.max_new_tokens,
                ),
                gen_length=self.generation_settings.max_new_tokens,
                block_length=min(
                    self.generation_settings.max_new_tokens,
                    self.generation_settings.sampling_steps,
                ),
                temperature=self.generation_settings.temperature,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=int(self.mask_token_id),
                repellency=self.repellency,
                effective_vocab=int(self.effective_vocab if self.effective_vocab is not None else len(self.tokenizer)),
                logits_eos_inf=False,
                confidence_eos_eot_inf=False,
                eos_id=self.eos_id,
                eot_id=self.eot_id,
                pad_id=self.tokenizer.pad_token_id if self.tokenizer else None,
                stop_tokens=self.stop_tokens,
                extra_ban_ids=self._ban_generation_ids,
                sampling_mode=self.generation_settings.sampling_mode,
                transfer_schedule=_resolve_transfer_schedule(
                    getattr(self.generation_settings, "transfer_schedule", None),
                    self.safety_settings.enabled,
                ),
            )
            for key, vals in rep_logs.items():
                self._repellency_logs.setdefault(key, []).extend(vals)

            for row in range(batch_size):
                tokens = samples[row].tolist()
                completion_tokens, _, _ = _strip_completion_tokens(
                    tokens,
                    0,
                    stop_ids,
                    self.mask_token_id,
                    stop_sequences=stop_tokens.stop_sequences if stop_tokens else (),
                )
                completion_tokens = _trim_to_eos(completion_tokens, self.eos_id)

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
                        logger=self.logger,
                    )
                metadata = {
                    "prompt_type": "unconditional",
                    **self.shard_metadata,
                    "safe_sampling_enabled": bool(self.repellency),
                    "repellency_mean_rho": _safe_mean(
                        self._repellency_logs.get("mean_rho", [])
                    ),
                }
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
        return results

    def _capture_peak_vram(self) -> int:
        if torch.cuda.is_available():
            try:
                return torch.cuda.max_memory_allocated(torch.cuda.current_device())
            except RuntimeError:
                return torch.cuda.max_memory_allocated()
        return 0

    def _resolved_config(self) -> Dict[str, Any]:
        model = asdict(self.model_settings)
        gen = asdict(self.generation_settings)
        safety = asdict(self.safety_settings)
        return {"model": model, "generation": gen, "safety": safety}


def run_llada_generation(
    prompts: Optional[Sequence[PromptRecord]],
    model: ModelSettings,
    generation: GenerationSettings,
    safety: SafetySettings,
    shard_metadata: Dict[str, Any],
) -> GenerationRun:
    from sampling.backends.registry import get_backend

    if generation.seed is not None:
        L.seed_everything(generation.seed)

    LOGGER.info("Starting LLADA generation, model=%s, generation=%s, safety=%s, seed=%s", model, generation, safety, generation.seed)

    backend = get_backend("llada", getattr(model, "variant", None))
    backend.load(model_settings=model, device=None)
    with torch.inference_mode():
        return backend.generate_batch(
            prompts=prompts,
            generation=generation,
            safety=safety,
            shard_metadata=shard_metadata,
        )
LOGGER = logging.getLogger(__name__)
