"""
pz_estimator.py — Estimates p_hat_z, the per-sequence extraction probability,
implementing Equation 8 from "Characterizing Memorization in Diffusion Language
Models" (Luo et al., arXiv:2603.02333).

=== DLM estimator (Eq. 8) ===

Core function: recover_logprob_from_masked_batch()
  Ported directly from the paper's reference implementation
  (memorization/ldm-mem/smdm/n_eval_pii_diff.py).

Two unmasking algorithms:
  "origin" — At each timestep t→s, each masked position is independently
    transferred to the ground-truth token with probability 1 - s/t.
    Uses continuous-time schedule: timesteps = linspace(t0, eps, steps+1).
    Log-prob is accumulated under Gumbel-max sampling (softmax(logits/T)).

  "greddy" — At each step, the top-k most-confident masked positions
    (confidence = softmax(logits)[argmax]) are unmasked, where
    k = floor(n_masked * (1 - s/t)).  Confidence-ranked, greedy transfer.

Both algorithms use float64 for numerical stability throughout.

=== ARM estimator ===

For autoregressive models p_z is the product of sequential conditional
probabilities of the suffix tokens.  This is deterministic (R=1 suffices).

=== Relaxed estimator (Eq. 12) ===

p_{z,epsilon} = fraction of R trials where Hamming(z_hat_M, z_M) <= epsilon.
Estimated empirically from the R trial outputs.

=== Model interface ===

DLMWrapper.get_logits(x, ...) must return a [B, L, V] float tensor.
The paper's reference code used a custom model() callable; we adapt via
_ModelCallable, which wraps DLMWrapper to present the same interface.
"""
from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers (ported verbatim from the paper's reference code)
# ---------------------------------------------------------------------------

def _rand(shape, *, device, dtype, generator: Optional[torch.Generator] = None) -> torch.Tensor:
  return torch.rand(shape, device=device, dtype=dtype, generator=generator)


def add_gumbel_noise(
  logits: torch.Tensor,
  temperature: float,
  generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
  """Gumbel-max noise for confidence-ranked sampling (paper's add_gumbel_noise)."""
  logits = logits.to(torch.float64)
  noise = _rand(logits.shape, device=logits.device, dtype=torch.float64, generator=generator)
  gumbel_noise = (-torch.log(noise)) ** temperature
  return logits.exp() / gumbel_noise


def gt_logprob_under_gumbel_sampling(
  logits_fp64: torch.Tensor,
  gt_ids: torch.Tensor,
  temperature: float,
  already_log_probs: bool = False,
) -> torch.Tensor:
  """Return log Pr(gt | Gumbel-max sampling) = log softmax(logits/T)[gt].

  Ported verbatim from the paper's reference implementation.

  Parameters
  ----------
  already_log_probs:
      If True, logits_fp64 is already log_softmax'd output (e.g. MDLM returns
      log p_x0 directly).  Skip the log_softmax to avoid double-softmaxing.
  """
  if temperature < 0:
    raise ValueError("temperature must be >= 0")
  if already_log_probs:
    # Already log-probs; temperature scaling in log-space = subtract log(T),
    # but since this is used for sampling weight only (not normalization), we
    # just use the log-probs directly (equivalent to T=1 in probability space).
    log_probs = logits_fp64
  else:
    scaled = logits_fp64 / float(temperature)
    log_probs = F.log_softmax(scaled, dim=-1)
  return log_probs.gather(1, gt_ids.unsqueeze(1)).squeeze(1)


# ---------------------------------------------------------------------------
# Core recover function (ported from paper's recover_logprob_from_masked_batch)
# ---------------------------------------------------------------------------

def recover_logprob_from_masked_batch(
  model_fn,                       # callable: x [B,L] -> logits [B,L,V]
  x: torch.Tensor,                # [B, L] masked input
  gt: torch.Tensor,               # [B, L] ground-truth token ids
  mask_id: int,
  steps: int,
  alg: str,                       # "origin" | "greddy"
  temperature: float,
  eps: float,
  prompt_len: int = 0,
  torch_gen: Optional[torch.Generator] = None,
  already_log_probs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Estimate log p_z^(i) for each sample in batch via stochastic unmasking.

  This is the core function from Luo et al. 2025, ported to work with our
  DLMWrapper interface.  The continuous-time schedule maps mask fraction -> t0,
  then linearly interpolates to eps over `steps` steps.

  Parameters
  ----------
  model_fn:
      Callable accepting x [B, L] (long tensor with mask_id entries) and
      returning logits [B, L, V] (float, any dtype, will be cast to float64).
  x:
      [B, L] masked input; masked positions contain mask_id.
      Will be modified in-place (teacher-forcing reveals gt tokens).
  gt:
      [B, L] ground-truth ids.
  mask_id:
      Integer mask token id.
  steps:
      Number of denoising steps.
  alg:
      "origin" — random transfer at probability 1 - s/t per step.
      "greddy" — confidence-ranked top-k transfer per step.
  temperature:
      Gumbel temperature (paper uses 1.0 by default).
  eps:
      Minimum timestep (paper uses 1e-3 by default).
  prompt_len:
      Prefix length (positions 0..prompt_len-1 are context, never masked).

  Returns
  -------
  x_out:       [B, L] after teacher-forcing (gt tokens at formerly masked positions)
  total_log:   [B] float64 — accumulated log p_z^(i) per sample
               (set to nan for invalid/non-finite samples)
  invalid_mask:[B] bool — True for samples where any log-prob was non-finite
  """
  if steps <= 0:
    invalid = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
    return x, torch.zeros(x.shape[0], device=x.device, dtype=torch.float64), invalid

  B, L = x.shape
  device = x.device

  with torch.no_grad():
    num_mask_row = (x == mask_id).sum(dim=1).to(torch.int64)  # [B]
    mask_count_float = num_mask_row.to(torch.float64).mean().item()
    p0 = float(mask_count_float) / float(L) if L > 0 else 0.0
    t0 = (p0 - eps) / (1.0 - eps) if (1.0 - eps) != 0 else 1.0
    if not math.isfinite(t0):
      t0 = 1.0
    t0 = max(min(t0, 1.0), eps)

  timesteps = torch.linspace(t0, eps, steps + 1, device=device)
  total_log = torch.zeros(B, device=device, dtype=torch.float64)
  invalid_mask = torch.zeros(B, device=device, dtype=torch.bool)

  use_amp = (torch.device(device).type == "cuda")
  amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp else nullcontext()

  with torch.no_grad():
    for i in range(steps):
      mask_index = (x == mask_id)  # [B, L]
      if int(mask_index.sum().item()) == 0:
        break

      with amp_ctx:
        logits_full = model_fn(x)  # [B, L, V]

      t = timesteps[i]
      s = timesteps[i + 1]

      if alg == "origin":
        p_transfer = 1.0 - (s / t).item() if i < steps - 1 else 1.0
        r = _rand((B, L), device=device, dtype=torch.float32, generator=torch_gen)
        transfer_pos_mask = (r < p_transfer) & mask_index  # [B, L]

        b_idx, pos_idx = transfer_pos_mask.nonzero(as_tuple=True)
        if b_idx.numel() > 0:
          sel_logits = logits_full[b_idx, pos_idx, :].to(torch.float64)  # [N, V]
          sel_gt = gt[b_idx, pos_idx]                                     # [N]
          sel_lp = gt_logprob_under_gumbel_sampling(
            sel_logits, sel_gt, temperature=temperature, already_log_probs=already_log_probs
          )

          finite = torch.isfinite(sel_lp)
          if not finite.all():
            bad_b = b_idx[~finite]
            invalid_mask[bad_b] = True
            sel_lp = torch.where(finite, sel_lp, torch.zeros_like(sel_lp))

          total_log.index_add_(0, b_idx, sel_lp)

          good = finite
          if good.any():
            x[b_idx[good], pos_idx[good]] = sel_gt[good]

      elif alg == "greddy":
        logits_masked = logits_full[mask_index]  # [Nmask_total, V]
        logits_with_noise = add_gumbel_noise(logits_masked, temperature=temperature, generator=torch_gen)
        x0_masked = torch.argmax(logits_with_noise, dim=-1)  # [Nmask_total]

        logits_masked_fp64 = logits_masked.to(torch.float64)
        p = F.softmax(logits_masked_fp64, dim=-1)
        confidence_masked = torch.gather(p, dim=-1, index=x0_masked.unsqueeze(-1)).squeeze(-1)

        confidence_full = torch.full((B, L), float("-inf"), device=device, dtype=torch.float64)
        confidence_full[mask_index] = confidence_masked

        num_mask = mask_index.sum(dim=1).to(torch.int64)
        if i < steps - 1:
          frac = 1.0 - (s / t).item()
          k = torch.floor(num_mask.to(torch.float64) * float(frac)).to(torch.int64)
        else:
          k = num_mask

        max_k = int(k.max().item())
        if max_k <= 0:
          continue

        _, top_pos = torch.topk(confidence_full, k=max_k, dim=1)

        ar = torch.arange(max_k, device=device).unsqueeze(0)
        take = ar < k.unsqueeze(1)

        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_k)
        sel_b = b_idx[take]
        sel_pos = top_pos[take]

        if sel_b.numel() == 0:
          continue

        sel_logits = logits_full[sel_b, sel_pos, :].to(torch.float64)
        sel_gt = gt[sel_b, sel_pos]
        sel_lp = gt_logprob_under_gumbel_sampling(
          sel_logits, sel_gt, temperature=temperature, already_log_probs=already_log_probs
        )

        finite = torch.isfinite(sel_lp)
        if not finite.all():
          bad_b = sel_b[~finite]
          invalid_mask[bad_b] = True
          sel_lp = torch.where(finite, sel_lp, torch.zeros_like(sel_lp))

        total_log.index_add_(0, sel_b, sel_lp)

        good = finite
        if good.any():
          x[sel_b[good], sel_pos[good]] = sel_gt[good]

      else:
        raise NotImplementedError(f"Unknown alg={alg!r}; expected 'origin' or 'greddy'")

  total_log = torch.where(invalid_mask, torch.full_like(total_log, float("nan")), total_log)
  return x, total_log, invalid_mask


# ---------------------------------------------------------------------------
# p_hat aggregation (paper's estimate_p_hat_from_log_sums)
# ---------------------------------------------------------------------------

def estimate_p_hat_from_log_sums(log_sums: List[float]) -> Dict[str, object]:
  """Monte Carlo p_hat = E[exp(total_log)] via arithmetic mean (paper's Eq. 8).

  Uses arithmetic mean over all R trajectories (including non-finite mapped to 0),
  matching the paper's reference implementation (compute_np_stats in
  evaluation_diffsampletimes.py): p_hat = sum(exp(log_sum_i)) / R.
  """
  R = len(log_sums)
  if R == 0:
    return {"p_hat": float("nan"), "log_p_hat": float("nan"), "count": 0}

  probs = []
  n_finite = 0
  for v in log_sums:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
      probs.append(math.exp(float(v)) if float(v) > -745.0 else 0.0)
      n_finite += 1
    else:
      probs.append(0.0)

  p_hat = sum(probs) / R
  log_p_hat = math.log(p_hat) if p_hat > 0 else float("-inf")
  return {"p_hat": p_hat, "log_p_hat": log_p_hat, "count": n_finite}


# ---------------------------------------------------------------------------
# DLM estimator (public API)
# ---------------------------------------------------------------------------

def _safe_seed(s: int) -> int:
  return int(s) % (2**31 - 1)


def estimate_pz_dlm(
  dlm_wrapper,
  token_ids: torch.Tensor,
  mask_bool: torch.Tensor,
  mask_id: int,
  R: int = 512,
  N: int = 10,
  attention_mask: Optional[torch.Tensor] = None,
  trial_batch_size: int = 16,
  show_progress: bool = False,
  alg: str = "origin",
  temperature: float = 1.0,
  eps: float = 1e-3,
  seed: int = 0,
  random_mask_per_trial: bool = False,
  mask_ratio: float = 0.2,
) -> Dict[str, object]:
  """Estimate p_hat_z for a single sequence using a DLM (Eq. 8).

  Uses the paper's exact recover_logprob_from_masked_batch logic.

  Parameters
  ----------
  dlm_wrapper:
      A ``DLMWrapper`` instance.
  token_ids:
      Ground-truth token ids, shape ``(L,)``.
  mask_bool:
      Boolean mask ``(L,)``; True at positions to reconstruct.
      Ignored when ``random_mask_per_trial=True`` (mask is sampled per trial).
  mask_id:
      The [MASK] token id.
  R:
      Number of independent MC sampling trials.
  N:
      Number of denoising steps per trial.  When N >= |M| (recover_each_token),
      each step unmasks at most 1 token.  Paper default: N = |M|.
  attention_mask:
      Optional ``(L,)`` or ``(1, L)`` padding mask (passed to DLMWrapper).
  trial_batch_size:
      How many trials to run in parallel per forward pass.
  alg:
      "origin" (random transfer, paper default) or "greddy" (confidence-ranked).
  temperature:
      Gumbel temperature for log-prob estimation (paper default: 1.0).
  eps:
      Minimum timestep in continuous schedule (paper default: 1e-3).
  seed:
      Base random seed; each batch uses seed + batch_start.
  random_mask_per_trial:
      If True, each trajectory in the batch gets an independent random mask
      pattern (mask_ratio * L positions chosen uniformly without replacement).
      This matches the paper's evaluation_diffsampletimes.py approach for
      verbatim memorization (Section 6.2).  prompt_len = L - mask_count.
      If False (default), all trials use the fixed mask_bool (prefix→suffix
      split used for PII/Table 1 evaluation).
  mask_ratio:
      Fraction of tokens to mask when random_mask_per_trial=True (default 0.2).

  Returns
  -------
  dict with:
    "p_hat_z"       : float — MC estimate of p_z (arithmetic mean of exp(log_sum))
    "log_p_hat_z"   : float — log(p_hat_z)
    "log_sums"      : list[float] — per-trial total_log values (nan = invalid)
    "trial_outputs" : list[Tensor] — final sequences per trial (shape L)
    "n_masked"      : int — |M| (for random_mask mode: mask_count = round(mask_ratio*L))
    "N_actual"      : int — N used (clamped to >= 1)
    "n_invalid"     : int — number of invalid (non-finite) trials
  """
  device = dlm_wrapper.device
  L = token_ids.shape[0]

  if random_mask_per_trial:
    mask_count = max(1, int(round(L * mask_ratio)))
    n_masked = mask_count
    # prompt_len: unmasked complement (paper: seq_len - mask_count)
    prompt_len = L - mask_count
  else:
    mask_positions = mask_bool.nonzero(as_tuple=True)[0]
    n_masked = len(mask_positions)
    if n_masked == 0:
      return {
        "p_hat_z": 1.0,
        "log_p_hat_z": 0.0,
        "log_sums": [0.0] * R,
        "trial_outputs": [token_ids.clone()] * R,
        "n_masked": 0,
        "N_actual": 0,
        "n_invalid": 0,
      }
    # Count prefix length (positions before the first masked token)
    prompt_len = int(mask_bool.long().argmax().item()) if n_masked > 0 else L

  N_actual = max(1, N)
  token_ids = token_ids.to(device)

  if not random_mask_per_trial:
    mask_bool = mask_bool.to(device)

  if attention_mask is not None:
    att_mask = attention_mask.view(1, L).to(device)
  else:
    att_mask = None

  # MDLM returns log p_x0 (already log-softmax'd); other backends return raw logits.
  already_log_probs = (dlm_wrapper.backend == "mdlm")

  # Build a model callable that wraps DLMWrapper.get_logits
  def _model_fn(x: torch.Tensor) -> torch.Tensor:
    batt = att_mask.expand(x.shape[0], -1) if att_mask is not None else None
    return dlm_wrapper.get_logits(x, attention_mask=batt)

  log_sums_all: List[float] = []
  trial_outputs: List[torch.Tensor] = []

  total_batches = math.ceil(R / trial_batch_size)
  iter_range = range(total_batches)

  if show_progress:
    from tqdm.auto import tqdm
    iter_range = tqdm(iter_range, desc="p_z trials", leave=False)

  base_seed = _safe_seed(seed)

  for batch_idx in iter_range:
    batch_start = batch_idx * trial_batch_size
    batch_end = min(batch_start + trial_batch_size, R)
    B = batch_end - batch_start

    gt = token_ids.unsqueeze(0).expand(B, -1).clone()

    if random_mask_per_trial:
      # Each trajectory gets an independent random mask pattern (paper's approach)
      g_cpu = torch.Generator()
      g_cpu.manual_seed(_safe_seed(base_seed + batch_start))
      rand_scores = torch.rand((B, L), generator=g_cpu)
      _, idx = torch.topk(rand_scores, k=mask_count, dim=1, largest=True, sorted=False)
      mask_indices = torch.zeros((B, L), dtype=torch.bool)
      mask_indices.scatter_(1, idx, True)

      x = gt.clone()
      x[mask_indices.to(device)] = mask_id
    else:
      # Fixed contiguous mask (prefix→suffix, for PII/Table 1)
      x = token_ids.unsqueeze(0).expand(B, -1).clone()
      x[:, mask_bool] = mask_id

    g = torch.Generator(device=device)
    g.manual_seed(_safe_seed(base_seed + batch_start))

    x_out, total_log, invalid = recover_logprob_from_masked_batch(
      model_fn=_model_fn,
      x=x,
      gt=gt,
      mask_id=mask_id,
      steps=N_actual,
      alg=alg,
      temperature=temperature,
      eps=eps,
      prompt_len=prompt_len,
      torch_gen=g,
      already_log_probs=already_log_probs,
    )

    log_sums_all.extend(total_log.detach().cpu().tolist())
    trial_outputs.extend([x_out[b].cpu() for b in range(B)])

  pstats = estimate_p_hat_from_log_sums(log_sums_all)
  n_invalid = sum(1 for v in log_sums_all if not (isinstance(v, float) and math.isfinite(v)))

  return {
    "p_hat_z": pstats["p_hat"],
    "log_p_hat_z": pstats["log_p_hat"],
    "log_sums": log_sums_all,
    "trial_outputs": trial_outputs,
    "n_masked": n_masked,
    "N_actual": N_actual,
    "n_invalid": n_invalid,
  }


# ---------------------------------------------------------------------------
# ARM estimator
# ---------------------------------------------------------------------------

def estimate_pz_arm(
  arm_wrapper,
  prefix_ids: torch.Tensor,
  suffix_ids: torch.Tensor,
) -> Dict[str, object]:
  """Estimate p_z for an ARM (sequential product of conditional probs).

  This is deterministic so R=1 suffices.
  """
  log_pz = arm_wrapper.suffix_log_prob(prefix_ids, suffix_ids)
  p_z = math.exp(log_pz) if math.isfinite(log_pz) else 0.0
  return {
    "p_hat_z": p_z,
    "log_p_hat_z": log_pz,
    "log_sums": [log_pz],
    "n_masked": suffix_ids.view(-1).shape[0],
  }


# ---------------------------------------------------------------------------
# Relaxed estimator (Eq. 12)
# ---------------------------------------------------------------------------

def estimate_pz_relaxed(
  trial_outputs: List[torch.Tensor],
  z_true: torch.Tensor,
  mask_bool: torch.Tensor,
  epsilon: int,
) -> float:
  """Estimate p_{z,epsilon} = fraction of trials where Hamming(pred, true) <= epsilon.

  Parameters
  ----------
  trial_outputs:
      List of reconstructed sequences ``(L,)`` from the R trials.
  z_true:
      Ground-truth sequence ``(L,)``.
  mask_bool:
      Boolean mask ``(L,)``; True at target positions.
  epsilon:
      Maximum allowed Hamming distance.
  """
  R = len(trial_outputs)
  if R == 0:
    return 0.0
  count = 0
  z_true_m = z_true[mask_bool]
  for out in trial_outputs:
    pred_m = out[mask_bool.to(out.device)]
    hamming = int((pred_m != z_true_m.to(out.device)).sum().item())
    if hamming <= epsilon:
      count += 1
  return count / R
