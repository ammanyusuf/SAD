"""
arm_wrapper.py — Wrapper around an autoregressive LM (GPT-2, LLaMA, etc.)
for suffix log-probability computation used in the p_z estimator.

For ARMs the joint probability of a suffix z_{suffix} given a prefix z_{prefix}
is simply the product of conditional next-token probabilities:

  p_z = prod_{l in suffix} Pr(z^l | z^{1:l-1})

This is deterministic (no sampling trials needed, R=1 suffices).

Usage
-----
  wrapper = ARMWrapper(model, tokenizer, device="cuda")
  log_pz = wrapper.suffix_log_prob(prefix_ids, suffix_ids)   # scalar
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class ARMWrapper:
  """Wraps a causal LM for suffix log-probability computation.

  Parameters
  ----------
  model:
      A HuggingFace causal LM (e.g. GPT2LMHeadModel, LlamaForCausalLM).
      Must return an object with a ``.logits`` attribute.
  device:
      Torch device string.
  """

  def __init__(self, model, device: Optional[str] = None) -> None:
    self.model = model
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

  # ------------------------------------------------------------------
  # Core: p_z for a single (prefix, suffix) pair
  # ------------------------------------------------------------------

  @torch.no_grad()
  def suffix_log_prob(
    self,
    prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
  ) -> float:
    """Compute log p(suffix | prefix) = sum_l log Pr(z^l | z^{1:l-1}).

    Parameters
    ----------
    prefix_ids:
        1-D int64 tensor of shape ``(prefix_len,)`` or batched ``(1, prefix_len)``.
    suffix_ids:
        1-D int64 tensor of shape ``(suffix_len,)`` or batched ``(1, suffix_len)``.
    attention_mask:
        Optional ``(1, prefix_len + suffix_len)`` mask; if None, all-ones assumed.

    Returns
    -------
    float
        Sum of log-probabilities over the suffix tokens (natural log).
    """
    prefix_ids = prefix_ids.view(-1)
    suffix_ids = suffix_ids.view(-1)

    full = torch.cat([prefix_ids, suffix_ids]).unsqueeze(0).to(self.device)  # (1, L)
    if attention_mask is None:
      attention_mask = torch.ones_like(full)
    else:
      attention_mask = attention_mask.to(self.device)

    out = self.model(full, attention_mask=attention_mask)
    logits = out.logits  # (1, L, V)
    log_probs = F.log_softmax(logits, dim=-1)  # (1, L, V)

    prefix_len = prefix_ids.shape[0]
    suffix_len = suffix_ids.shape[0]

    # The prediction at position i predicts token i+1.
    # Suffix starts at prefix_len; last suffix token is at prefix_len + suffix_len - 1.
    # We need predictions at positions [prefix_len-1 .. prefix_len+suffix_len-2]
    # to score tokens [prefix_len .. prefix_len+suffix_len-1].
    pred_positions = torch.arange(prefix_len - 1, prefix_len + suffix_len - 1, device=self.device)
    target_ids = suffix_ids.to(self.device)

    # log_probs[0, pred_positions, target_ids]
    lp = log_probs[0, pred_positions, target_ids]  # (suffix_len,)
    return float(lp.sum().item())

  # ------------------------------------------------------------------
  # Batched version for efficiency
  # ------------------------------------------------------------------

  @torch.no_grad()
  def batch_suffix_log_prob(
    self,
    prefix_ids_list: list,
    suffix_ids_list: list,
    pad_token_id: int = 0,
  ) -> list:
    """Compute suffix log-probs for a batch of (prefix, suffix) pairs.

    Uses left-padding so all sequences share the same forward pass.
    Note: left-padding is standard for decoder-only LMs when we need the
    model to "see" the prefix before predicting the suffix.

    Parameters
    ----------
    prefix_ids_list, suffix_ids_list:
        Lists of 1-D int64 tensors.
    pad_token_id:
        Token id used for padding shorter sequences on the left.

    Returns
    -------
    list of float
        Log-probability for each (prefix, suffix) pair.
    """
    B = len(prefix_ids_list)
    full_seqs = [
      torch.cat([p.view(-1), s.view(-1)])
      for p, s in zip(prefix_ids_list, suffix_ids_list)
    ]
    max_len = max(t.shape[0] for t in full_seqs)

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=self.device)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
    prefix_lengths = [p.view(-1).shape[0] for p in prefix_ids_list]
    suffix_lengths = [s.view(-1).shape[0] for s in suffix_ids_list]

    for i, seq in enumerate(full_seqs):
      pad_len = max_len - seq.shape[0]
      input_ids[i, pad_len:] = seq.to(self.device)
      attention_mask[i, pad_len:] = 1

    out = self.model(input_ids, attention_mask=attention_mask)
    logits = out.logits  # (B, L, V)
    log_probs = F.log_softmax(logits, dim=-1)

    results = []
    for i in range(B):
      seq = full_seqs[i]
      pad_len = max_len - seq.shape[0]
      p_len = prefix_lengths[i]
      s_len = suffix_lengths[i]

      # Absolute positions in the padded sequence
      prefix_start = pad_len  # first prefix token
      suffix_start = pad_len + p_len  # first suffix token

      # Predictions: position j predicts token j+1
      pred_positions = torch.arange(
        suffix_start - 1, suffix_start + s_len - 1, device=self.device
      )
      target_ids = seq[p_len:].to(self.device)  # the suffix tokens

      lp = log_probs[i, pred_positions, target_ids]
      results.append(float(lp.sum().item()))

    return results
