"""
wikitext.py — WikiText-103 dataset wrapper for verbatim memorization analysis.

Measures whether an MDLM trained on WikiText-103 has memorized verbatim
training sequences: given a prefix of k tokens from the training corpus,
can the model reconstruct the next n tokens?

The (prefix, suffix) split is purely positional.

Two sampling strategies
-----------------------
  "random"    — Randomly sampled windows from the training split.
                Baseline: tests general verbatim memorization across the corpus.

  "high_freq" — Windows whose suffix n-gram appears at least `min_freq` times
                in the training data.  These are the sequences most likely to
                be memorized due to repetition.

The dataset caches a JSONL of raw documents on first use (no download needed;
wikitext-103-raw-v1 is already on Compute Canada via HuggingFace datasets).

Usage
-----
  from memorization.data.wikitext import WikitextDataset

  ds = WikitextDataset(
      data_dir="/path/to/data/memorization/wikitext",
      tokenizer=tokenizer,
      n_samples=1000,
      prefix_tokens=100,
      suffix_tokens=50,
      strategy="random",   # or "high_freq"
  )
"""
from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

HF_DATASET_NAME = "wikitext"
HF_DATASET_CONFIG = "wikitext-103-raw-v1"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(data_dir: Path, split: str) -> Path:
  return data_dir / f"wikitext103_{split}.jsonl"


def _windows_cache_path(data_dir: Path, split: str, prefix_tokens: int, suffix_tokens: int) -> Path:
  return data_dir / f"windows_{split}_p{prefix_tokens}_s{suffix_tokens}.pt"


def download_wikitext(
  data_dir: str,
  split: str = "train",
  force: bool = False,
) -> Path:
  """Load wikitext-103-raw-v1 from HuggingFace and cache as JSONL.

  On Compute Canada the dataset is available offline (HF_DATASETS_OFFLINE=1).
  The JSONL cache is a list of {"doc_id": ..., "text": ...} records, one per
  non-empty wiki article/paragraph as delivered by the HF dataset.

  Parameters
  ----------
  data_dir:
      Directory to store the cache file.
  split:
      HuggingFace split: "train", "validation", or "test".
  force:
      Re-download even if cache exists.
  """
  data_dir = Path(data_dir)
  data_dir.mkdir(parents=True, exist_ok=True)
  cache = _cache_path(data_dir, split)

  if cache.exists() and not force:
    LOGGER.info("Using cached wikitext103 (%s) at %s", split, cache)
    return cache

  LOGGER.info("Loading wikitext-103-raw-v1 (%s) from HuggingFace...", split)
  try:
    from datasets import load_dataset
  except ImportError:
    raise ImportError("pip install datasets")

  ds = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split=split, trust_remote_code=False)

  n = 0
  tmp = cache.with_suffix(".jsonl.tmp")
  with tmp.open("w", encoding="utf-8") as fout:
    for i, record in enumerate(ds):
      text = (record.get("text") or "").strip()
      # Skip empty lines and wikitext section headers (e.g. " = Heading = ")
      if not text or text.startswith(" = "):
        continue
      fout.write(json.dumps({"doc_id": str(i), "text": text}, ensure_ascii=False) + "\n")
      n += 1

  tmp.rename(cache)
  LOGGER.info("Cached %d non-empty records → %s", n, cache)
  return cache


def load_raw_wikitext(data_dir: str, split: str = "train", force: bool = False) -> List[Dict]:
  cache = download_wikitext(data_dir, split=split, force=force)
  records = []
  with cache.open(encoding="utf-8") as fh:
    for line in fh:
      line = line.strip()
      if line:
        records.append(json.loads(line))
  LOGGER.info("Loaded %d raw wikitext103 records (%s split)", len(records), split)
  return records


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def _extract_windows(
  records: List[Dict],
  tokenizer,
  prefix_tokens: int,
  suffix_tokens: int,
  cache_file: Optional[Path] = None,
) -> List[Dict]:
  """Tokenize all records and extract every non-overlapping (prefix, suffix) window.

  Results are cached to cache_file (a .pt file) so tokenization only runs once
  per (split, prefix_tokens, suffix_tokens) configuration.

  Returns a flat list of dicts:
    doc_id, window_idx, prefix_ids (Tensor), suffix_ids (Tensor), text (str)
  """
  if cache_file is not None and cache_file.exists():
    LOGGER.info("Loading tokenized windows from cache: %s", cache_file)
    windows = torch.load(cache_file, weights_only=False)
    LOGGER.info("Loaded %d cached windows", len(windows))
    return windows

  total_needed = prefix_tokens + suffix_tokens
  windows = []

  from tqdm.auto import tqdm
  for record in tqdm(records, desc="Tokenizing wikitext", unit="doc"):
    text = record["text"]
    all_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(all_ids) < total_needed:
      continue

    # Extract non-overlapping windows
    n_windows = len(all_ids) // total_needed
    for wi in range(n_windows):
      start = wi * total_needed
      p_ids = all_ids[start: start + prefix_tokens]
      s_ids = all_ids[start + prefix_tokens: start + total_needed]
      windows.append({
        "doc_id": record["doc_id"],
        "window_idx": wi,
        "prefix_ids": torch.tensor(p_ids, dtype=torch.long),
        "suffix_ids": torch.tensor(s_ids, dtype=torch.long),
        "pii_raw": None,   # no PII — matches expected interface
        "pii_type": "verbatim",
        "text": text,
      })

  LOGGER.info("Extracted %d windows (prefix=%d, suffix=%d)", len(windows), prefix_tokens, suffix_tokens)

  if cache_file is not None:
    tmp = cache_file.with_suffix(".pt.tmp")
    torch.save(windows, tmp)
    tmp.rename(cache_file)
    LOGGER.info("Cached windows → %s", cache_file)

  return windows


def _build_ngram_freq(windows: List[Dict]) -> Counter:
  """Count how many times each suffix token tuple appears across all windows."""
  freq: Counter = Counter()
  for w in windows:
    key = tuple(w["suffix_ids"].tolist())
    freq[key] += 1
  return freq


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class WikitextDataset(Dataset):
  """PyTorch Dataset of WikiText-103 (prefix, suffix) pairs for memorization analysis.

  Parameters
  ----------
  data_dir:
      Directory to cache the downloaded JSONL.
  tokenizer:
      HuggingFace tokenizer.  Must match the target model's tokenizer.
  n_samples:
      Number of (prefix, suffix) pairs to return.
  prefix_tokens:
      Number of prefix (context) tokens.  Paper uses 100.
  suffix_tokens:
      Number of suffix (target) tokens to reconstruct.
      Use a small value (10–50) for exact-match evaluation;
      larger values make recovery harder.
  strategy:
      "random"    — uniformly sampled windows (default)
      "high_freq" — windows whose suffix appears >= min_freq times
  min_freq:
      Minimum repetition count for "high_freq" strategy. Default 2.
  split:
      Which HuggingFace split to use ("train" recommended — MDLM was trained
      on the train split).
  seed:
      Random seed for window sampling.
  force_download:
      Re-download even if cache exists.
  """

  def __init__(
    self,
    data_dir: str,
    tokenizer,
    n_samples: int = 1000,
    prefix_tokens: int = 100,
    suffix_tokens: int = 50,
    strategy: str = "random",
    min_freq: int = 2,
    split: str = "train",
    seed: int = 42,
    force_download: bool = False,
  ) -> None:
    if strategy not in ("random", "high_freq"):
      raise ValueError(f"strategy must be 'random' or 'high_freq', got {strategy!r}")

    self.prefix_tokens = prefix_tokens
    self.suffix_tokens = suffix_tokens
    self.strategy = strategy

    data_dir_path = Path(data_dir)
    windows_cache = _windows_cache_path(data_dir_path, split, prefix_tokens, suffix_tokens)

    records = load_raw_wikitext(data_dir, split=split, force=force_download)
    all_windows = _extract_windows(
      records, tokenizer, prefix_tokens, suffix_tokens, cache_file=windows_cache
    )

    if strategy == "high_freq":
      freq = _build_ngram_freq(all_windows)
      all_windows = [w for w in all_windows if freq[tuple(w["suffix_ids"].tolist())] >= min_freq]
      LOGGER.info(
        "high_freq filter (min_freq=%d): %d windows remain", min_freq, len(all_windows)
      )
      if len(all_windows) == 0:
        LOGGER.warning(
          "No windows passed high_freq filter with min_freq=%d. "
          "The training corpus may not have many repeated n-grams of length %d. "
          "Try lowering min_freq or suffix_tokens.",
          min_freq, suffix_tokens,
        )

    rng = random.Random(seed)
    rng.shuffle(all_windows)
    self.samples = all_windows[:n_samples]
    LOGGER.info(
      "WikitextDataset: %d samples (strategy=%s, prefix=%d, suffix=%d, split=%s)",
      len(self.samples), strategy, prefix_tokens, suffix_tokens, split,
    )

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> Dict:
    return self.samples[idx]
