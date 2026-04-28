"""
masks.py — Mask sampler for the (n,p)-discoverable extraction framework.

Supports two masking modes:
  1. prefix_conditioned  — mask only the suffix; keep the prefix as observed
                           context.  Used for PII replication (Table 1).
  2. random              — mask r * L positions chosen uniformly at random.
                           Used for arbitrary-position memorization experiments.

Each function returns a dict with:
  {
    "x_masked":     Tensor (L,)  token ids with mask_id at masked positions
    "mask_bool":    Tensor (L,)  bool, True at masked positions
    "z_true_masked": Tensor (L,) ground-truth ids (equals z at all positions;
                                 the caller reads only z_true_masked[mask_bool])
  }
"""
from __future__ import annotations

import math
import random
from typing import Dict, Optional, Union

import torch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_prefix_conditioned_mask(
  token_ids: torch.Tensor,
  prefix_len: int,
  mask_id: int,
) -> Dict[str, torch.Tensor]:
  """Mask the suffix of a sequence, keeping the prefix as fixed context.

  Parameters
  ----------
  token_ids:
      1-D int64 tensor of shape ``(L,)``.
  prefix_len:
      Number of tokens to keep as observed context.  Must be < L.
  mask_id:
      Token id to substitute at masked positions.

  Returns
  -------
  dict with keys "x_masked", "mask_bool", "z_true_masked".
  """
  L = token_ids.shape[0]
  if prefix_len >= L:
    raise ValueError(
      f"prefix_len ({prefix_len}) must be < sequence length ({L})."
    )
  mask_bool = torch.zeros(L, dtype=torch.bool)
  mask_bool[prefix_len:] = True

  x_masked = token_ids.clone()
  x_masked[mask_bool] = mask_id

  return {
    "x_masked": x_masked,
    "mask_bool": mask_bool,
    "z_true_masked": token_ids.clone(),
  }


def sample_random_mask(
  token_ids: torch.Tensor,
  mask_ratio: float,
  mask_id: int,
  rng: Optional[random.Random] = None,
) -> Dict[str, torch.Tensor]:
  """Mask a random subset of positions (uniform without replacement).

  Parameters
  ----------
  token_ids:
      1-D int64 tensor of shape ``(L,)``.
  mask_ratio:
      Fraction of positions to mask, e.g. 0.20, 0.25, or 0.30.
  mask_id:
      Token id to substitute at masked positions.
  rng:
      Optional ``random.Random`` instance for reproducibility.

  Returns
  -------
  dict with keys "x_masked", "mask_bool", "z_true_masked".
  """
  L = token_ids.shape[0]
  n_mask = max(1, int(math.ceil(mask_ratio * L)))
  n_mask = min(n_mask, L)

  all_positions = list(range(L))
  if rng is not None:
    masked_positions = rng.sample(all_positions, n_mask)
  else:
    masked_positions = random.sample(all_positions, n_mask)

  mask_bool = torch.zeros(L, dtype=torch.bool)
  mask_bool[masked_positions] = True

  x_masked = token_ids.clone()
  x_masked[mask_bool] = mask_id

  return {
    "x_masked": x_masked,
    "mask_bool": mask_bool,
    "z_true_masked": token_ids.clone(),
  }


def apply_mask_to_batch(
  token_ids: torch.Tensor,
  mask_bool: torch.Tensor,
  mask_id: int,
) -> torch.Tensor:
  """Apply a pre-computed boolean mask to a sequence tensor.

  Parameters
  ----------
  token_ids:
      ``(B, L)`` or ``(L,)`` int64 tensor.
  mask_bool:
      ``(L,)`` boolean tensor; True at positions to mask.
  mask_id:
      Token id to place at masked positions.

  Returns
  -------
  Tensor of the same shape as ``token_ids`` with mask_id at masked positions.
  """
  out = token_ids.clone()
  if out.dim() == 1:
    out[mask_bool] = mask_id
  else:
    out[:, mask_bool] = mask_id
  return out
