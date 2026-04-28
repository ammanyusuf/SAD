"""
metrics.py — (n,p)-discoverable extraction metric and its relaxed variant.

=== (n,p)-discoverable extraction (Eq. 6) ===

Given p_hat_z (estimated per-query extraction probability) and a target
success probability p in (0,1), the minimum number of queries n needed so
that the adversary succeeds at least once with probability >= p is:

  n = ceil( log(1-p) / log(1 - p_hat_z) )

A sequence is "memorized at (n,p)" if n <= query_budget.

=== (epsilon,n,p)-discoverable extraction (Eq. 12 relaxed) ===

Same formula but uses p_{z,epsilon} (fraction of R trials where Hamming
distance to the true sequence is at most epsilon) in place of p_hat_z.
"""
from __future__ import annotations

import math
from typing import Dict, Optional


_INF_QUERIES: int = int(1e18)  # sentinel for "never succeeds"


def n_queries_needed(
  p_hat_z: float,
  p: float,
  query_budget: Optional[int] = None,
) -> int:
  """Compute the minimum number of queries to achieve success probability >= p.

  Parameters
  ----------
  p_hat_z:
      Estimated per-query extraction probability in (0, 1].
  p:
      Target success probability, e.g. 0.5 or 0.99.
  query_budget:
      If given, return min(n, query_budget + 1) so that the caller can check
      ``n <= query_budget`` for memorization.

  Returns
  -------
  int
      n = ceil( log(1-p) / log(1-p_hat_z) ).
      Returns ``_INF_QUERIES`` if p_hat_z == 0.
      Returns 1 if p_hat_z >= p (single query already exceeds p).
  """
  if p_hat_z <= 0.0:
    return _INF_QUERIES
  if p_hat_z >= p:
    return 1

  # 1 - (1 - p_hat_z)^n >= p  =>  (1 - p_hat_z)^n <= 1-p
  # n >= log(1-p) / log(1-p_hat_z)   (both logs are negative, so direction flips)
  fail_prob = min(1.0 - p_hat_z, 1.0 - 1e-15)  # clamp to avoid log(0)
  if fail_prob <= 0.0:
    return 1
  log_fail = math.log(fail_prob)        # negative
  log_target = math.log(1.0 - p)       # negative (p < 1)
  n = math.ceil(log_target / log_fail)

  if query_budget is not None:
    return min(n, query_budget + 1)
  return n


def is_memorized(
  p_hat_z: float,
  p: float,
  query_budget: int = 10_000,
) -> bool:
  """Return True if the sequence is (n,p)-discoverable within query_budget."""
  n = n_queries_needed(p_hat_z, p, query_budget=query_budget)
  return n <= query_budget


def compute_extraction_metrics(
  p_hat_z: float,
  p_values: tuple = (0.5, 0.99),
  query_budget: int = 10_000,
  p_hat_z_relaxed: Optional[Dict[int, float]] = None,
) -> Dict[str, object]:
  """Compute all (n,p) and relaxed (eps,n,p) memorization metrics.

  Parameters
  ----------
  p_hat_z:
      Exact extraction probability estimate.
  p_values:
      Tuple of target probabilities to evaluate.
  query_budget:
      Maximum number of queries budget.
  p_hat_z_relaxed:
      Optional dict mapping epsilon -> p_{z,epsilon}.  If provided,
      relaxed metrics are also computed.

  Returns
  -------
  dict with keys like:
    "n_queries_p50":    int   — queries needed for p=0.5
    "n_queries_p99":    int   — queries needed for p=0.99
    "memorized_p50":    bool
    "memorized_p99":    bool
    "p_hat_z":          float
    "relaxed_{eps}_p50": bool  (if p_hat_z_relaxed provided)
    etc.
  """
  result: Dict[str, object] = {"p_hat_z": p_hat_z}

  for p in p_values:
    p_label = f"p{int(p * 100)}"
    n = n_queries_needed(p_hat_z, p, query_budget=query_budget)
    result[f"n_queries_{p_label}"] = n
    result[f"memorized_{p_label}"] = (n <= query_budget)

  if p_hat_z_relaxed:
    for eps, pze in p_hat_z_relaxed.items():
      result[f"p_hat_z_eps{eps}"] = pze
      for p in p_values:
        p_label = f"p{int(p * 100)}"
        n = n_queries_needed(pze, p, query_budget=query_budget)
        result[f"relaxed_eps{eps}_n_queries_{p_label}"] = n
        result[f"relaxed_eps{eps}_memorized_{p_label}"] = (n <= query_budget)

  return result
