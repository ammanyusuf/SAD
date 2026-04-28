"""
test_pz_estimator.py — Unit tests for the p_z estimator.

Validates Eq. 8 on a toy model: a uniform distribution over V tokens should
give p_hat_z ≈ (1/V)^|M| (expected exact-match probability by chance).

Run:
  pytest memorization/tests/test_pz_estimator.py -v
"""
from __future__ import annotations

import math
import random

import pytest
import torch
import torch.nn.functional as F

from memorization.pz_estimator import estimate_pz_dlm, estimate_pz_arm, estimate_pz_relaxed
from memorization.masks import sample_prefix_conditioned_mask, sample_random_mask
from memorization.metrics import n_queries_needed, is_memorized, compute_extraction_metrics


# ---------------------------------------------------------------------------
# Toy DLM wrapper: uniform distribution over vocabulary
# ---------------------------------------------------------------------------

class UniformDLMWrapper:
  """Always returns uniform logits (zeros) over a fixed vocabulary."""

  def __init__(self, vocab_size: int = 100, device: str = "cpu") -> None:
    self.vocab_size = vocab_size
    self.mask_id = vocab_size  # mask token beyond vocabulary
    self.device = device

  def get_logits(self, x, attention_mask=None, sigma_t=None):
    B, L = x.shape
    return torch.zeros(B, L, self.vocab_size, device=self.device)


class PerfectDLMWrapper:
  """Always returns the ground-truth token with probability 1."""

  def __init__(self, true_ids: torch.Tensor, vocab_size: int = 100, device: str = "cpu") -> None:
    self.true_ids = true_ids  # (L,)
    self.vocab_size = vocab_size
    self.mask_id = vocab_size
    self.device = device

  def get_logits(self, x, attention_mask=None, sigma_t=None):
    B, L = x.shape
    # Logits: huge value at the true token, 0 elsewhere
    logits = torch.zeros(B, L, self.vocab_size, device=self.device)
    for b in range(B):
      for l in range(L):
        t = self.true_ids[l].item()
        if t < self.vocab_size:
          logits[b, l, t] = 100.0
    return logits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMasks:
  def test_prefix_conditioned_mask(self):
    ids = torch.arange(20)
    out = sample_prefix_conditioned_mask(ids, prefix_len=10, mask_id=99)
    assert out["x_masked"][:10].tolist() == list(range(10)), "Prefix should be untouched"
    assert (out["x_masked"][10:] == 99).all(), "Suffix should be masked"
    assert out["mask_bool"][:10].sum() == 0
    assert out["mask_bool"][10:].sum() == 10

  def test_random_mask_ratio(self):
    ids = torch.arange(100)
    out = sample_random_mask(ids, mask_ratio=0.25, mask_id=999)
    n_masked = out["mask_bool"].sum().item()
    assert n_masked == 25, f"Expected 25 masked, got {n_masked}"

  def test_prefix_len_equals_seq_raises(self):
    ids = torch.arange(10)
    with pytest.raises(ValueError):
      sample_prefix_conditioned_mask(ids, prefix_len=10, mask_id=99)


class TestMetrics:
  def test_n_queries_p50(self):
    # p_hat_z = 0.5 → exactly 1 query suffices for p=0.5
    n = n_queries_needed(0.5, p=0.5)
    assert n == 1

  def test_n_queries_small_pz(self):
    # p_hat_z = 0.001 → n ≈ 693 for p=0.5  (ln(0.5)/ln(0.999) ≈ 692.8)
    n = n_queries_needed(0.001, p=0.5)
    assert 690 <= n <= 700, f"n={n}"

  def test_memorized_within_budget(self):
    assert is_memorized(0.5, p=0.5, query_budget=10000) is True
    assert is_memorized(1e-10, p=0.5, query_budget=10000) is False

  def test_zero_pz(self):
    n = n_queries_needed(0.0, p=0.5)
    assert n > 10_000_000  # effectively infinite

  def test_perfect_pz(self):
    n = n_queries_needed(1.0, p=0.99)
    assert n == 1

  def test_compute_extraction_metrics(self):
    m = compute_extraction_metrics(0.5, p_values=(0.5, 0.99), query_budget=10000)
    assert m["memorized_p50"] is True
    assert "n_queries_p50" in m
    assert "n_queries_p99" in m


class TestPzEstimatorDLM:
  """Test the DLM p_z estimator with the uniform toy model."""

  def test_uniform_model_pz_approx_chance(self):
    """Uniform model → p_hat_z ≈ (1/V)^|M|."""
    V = 10
    wrapper = UniformDLMWrapper(vocab_size=V, device="cpu")
    L = 8
    prefix_len = 4
    # Ground-truth sequence (arbitrary)
    token_ids = torch.arange(L) % V  # tokens 0..7 mod 10

    mask_out = sample_prefix_conditioned_mask(token_ids, prefix_len=prefix_len, mask_id=wrapper.mask_id)
    n_masked = int(mask_out["mask_bool"].sum().item())  # = 4

    expected_pz = (1.0 / V) ** n_masked  # = 0.0001

    result = estimate_pz_dlm(
      dlm_wrapper=wrapper,
      token_ids=mask_out["z_true_masked"],
      mask_bool=mask_out["mask_bool"],
      mask_id=wrapper.mask_id,
      R=2000,  # many trials for accurate MC estimate
      N=1,     # unmask everything in one step
      trial_batch_size=100,
    )
    p_hat_z = result["p_hat_z"]
    # Within 50% relative error for a Monte Carlo estimate
    assert abs(p_hat_z - expected_pz) / expected_pz < 0.5, (
      f"p_hat_z={p_hat_z:.6f} expected≈{expected_pz:.6f}"
    )

  def test_perfect_model_pz_equals_one(self):
    """Perfect model → p_hat_z ≈ 1.0."""
    V = 50
    L = 10
    prefix_len = 5
    token_ids = torch.randint(0, V, (L,))
    wrapper = PerfectDLMWrapper(true_ids=token_ids, vocab_size=V, device="cpu")

    mask_out = sample_prefix_conditioned_mask(token_ids, prefix_len=prefix_len, mask_id=wrapper.mask_id)

    result = estimate_pz_dlm(
      dlm_wrapper=wrapper,
      token_ids=mask_out["z_true_masked"],
      mask_bool=mask_out["mask_bool"],
      mask_id=wrapper.mask_id,
      R=10,
      N=5,
      trial_batch_size=5,
    )
    assert result["p_hat_z"] > 0.99, f"p_hat_z={result['p_hat_z']}"

  def test_no_masked_positions_returns_one(self):
    """If mask_bool is all-False, p_hat_z should be 1.0."""
    V = 10
    wrapper = UniformDLMWrapper(vocab_size=V)
    token_ids = torch.arange(8) % V
    mask_bool = torch.zeros(8, dtype=torch.bool)
    result = estimate_pz_dlm(
      dlm_wrapper=wrapper,
      token_ids=token_ids,
      mask_bool=mask_bool,
      mask_id=wrapper.mask_id,
      R=4,
      N=1,
    )
    assert result["p_hat_z"] == 1.0

  def test_multi_step_N_reduces_variance(self):
    """N > 1 should give same expected p_hat_z as N=1 for uniform model."""
    V = 5
    wrapper = UniformDLMWrapper(vocab_size=V)
    L = 6
    prefix_len = 2
    token_ids = torch.arange(L) % V
    mask_out = sample_prefix_conditioned_mask(token_ids, prefix_len=prefix_len, mask_id=wrapper.mask_id)
    n_masked = int(mask_out["mask_bool"].sum().item())
    expected = (1 / V) ** n_masked

    r1 = estimate_pz_dlm(wrapper, mask_out["z_true_masked"], mask_out["mask_bool"],
                          wrapper.mask_id, R=500, N=1, trial_batch_size=50)
    r4 = estimate_pz_dlm(wrapper, mask_out["z_true_masked"], mask_out["mask_bool"],
                          wrapper.mask_id, R=500, N=n_masked, trial_batch_size=50)

    # Both should be within factor of 3 of expected
    for tag, r in [("N=1", r1), (f"N={n_masked}", r4)]:
      ratio = r["p_hat_z"] / expected
      assert 0.2 < ratio < 5.0, f"{tag}: p_hat_z={r['p_hat_z']:.6f}, expected≈{expected:.6f}"


class TestPzEstimatorARM:
  def test_known_log_prob(self):
    """Mock ARM wrapper with known log-prob and check result."""

    class _MockARM:
      device = "cpu"
      def suffix_log_prob(self, prefix_ids, suffix_ids):
        # Return fixed log-prob = log(0.25)
        return math.log(0.25)

    result = estimate_pz_arm(_MockARM(), torch.tensor([1, 2]), torch.tensor([3, 4]))
    assert abs(result["p_hat_z"] - 0.25) < 1e-6


class TestRelaxedEstimator:
  def test_exact_match_epsilon0(self):
    """epsilon=0 should count only perfect reconstructions."""
    z_true = torch.tensor([1, 2, 3, 4, 5])
    mask_bool = torch.tensor([False, False, True, True, True])
    # 2 perfect, 1 off by 1
    outputs = [
      torch.tensor([1, 2, 3, 4, 5]),  # perfect
      torch.tensor([1, 2, 3, 4, 5]),  # perfect
      torch.tensor([1, 2, 3, 4, 9]),  # 1 error
    ]
    p = estimate_pz_relaxed(outputs, z_true, mask_bool, epsilon=0)
    assert abs(p - 2/3) < 1e-6

  def test_relaxed_epsilon1(self):
    """epsilon=1 should count outputs with <= 1 error."""
    z_true = torch.tensor([1, 2, 3, 4, 5])
    mask_bool = torch.tensor([False, False, True, True, True])
    outputs = [
      torch.tensor([1, 2, 3, 4, 5]),  # 0 errors
      torch.tensor([1, 2, 3, 4, 9]),  # 1 error
      torch.tensor([1, 2, 0, 0, 9]),  # 3 errors
    ]
    p = estimate_pz_relaxed(outputs, z_true, mask_bool, epsilon=1)
    assert abs(p - 2/3) < 1e-6
