"""
enron.py — Enron email dataset loading, PII extraction, and dataset classes
for replicating Table 1 of "Characterizing Memorization in Diffusion Language
Models" (Luo et al., arXiv:2603.02333).

Two data sources are supported:

  corbt/enron-emails  (default, PII experiments)
    ~517k full Enron emails with signatures intact.  Contains real email
    addresses and phone numbers in signature blocks — required for Table 1
    PII memorization evaluation.
    HF id: "corbt/enron-emails"
    Cache:  <data_dir>/enron_raw.jsonl

  Yale-LILY/aeslc  (non-PII / pipeline testing)
    ~18k emails filtered for subject-line summarization.  Most signature
    blocks are stripped so PII is rare.  Useful for fast pipeline tests
    that do not need real PII (e.g. general memorization of email text).
    HF id: "Yale-LILY/aeslc"
    Cache:  <data_dir>/aeslc_raw.jsonl

PII types supported (corbt source only):
  - email addresses (EMAIL_RE from the paper)
  - phone numbers   (PHONE_RE from the paper)

Usage
-----
  from memorization.data.enron import EnronPIIDataset, AESLCDataset

  # PII experiments (Table 1)
  ds = EnronPIIDataset(
    data_dir="/tmp/enron",
    tokenizer=tokenizer,
    pii_type="email",
    n_samples=3000,
    prefix_max_tokens=100,
  )

  # Non-PII / pipeline testing
  ds = AESLCDataset(
    data_dir="/tmp/enron",
    tokenizer=tokenizer,
    n_samples=500,
    suffix_tokens=10,
  )
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns from the paper (Section 4.1)
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(
  r"([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})"
)
PHONE_RE = re.compile(
  r"[0-9][0-9][0-9][-.()][0-9][0-9][0-9][-.()][0-9][0-9][0-9][0-9]"
)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_enron(data_dir: str, force: bool = False) -> Path:
  """Download corbt/enron-emails (~517k emails) and cache as JSONL.

  This is the primary source for PII memorization experiments (Table 1).
  Emails retain full headers and signature blocks with real email addresses
  and phone numbers.

  Cache: <data_dir>/enron_raw.jsonl
  """
  data_dir = Path(data_dir)
  data_dir.mkdir(parents=True, exist_ok=True)
  cache_path = data_dir / "enron_raw.jsonl"

  if cache_path.exists() and not force:
    LOGGER.info("Using cached Enron dataset at %s", cache_path)
    return cache_path

  LOGGER.info("Downloading corbt/enron-emails from HuggingFace (~517k emails)...")
  try:
    from datasets import load_dataset
  except ImportError:
    raise ImportError("pip install datasets to use the Enron download helper.")

  ds = load_dataset("corbt/enron-emails", split="train")

  LOGGER.info("Loaded %d records; writing to %s", len(ds), cache_path)
  # corbt/enron-emails fields: message_id, subject, from, to, cc, bcc, date, body, file_name
  n = 0
  tmp_path = cache_path.with_suffix(".jsonl.tmp")
  with tmp_path.open("w", encoding="utf-8") as fout:
    for i, record in enumerate(ds):
      subject = (record.get("subject") or "").strip()
      body = (record.get("body") or "").strip()
      text = (subject + "\n" + body).strip() if subject else body
      if text:
        fout.write(json.dumps({"doc_id": str(i), "text": text}, ensure_ascii=False) + "\n")
        n += 1

  tmp_path.rename(cache_path)
  LOGGER.info("Done. Cached %d records → %s", n, cache_path)
  return cache_path


def download_aeslc(data_dir: str, force: bool = False) -> Path:
  """Download Yale-LILY/aeslc (~18k emails) and cache as JSONL.

  This is the non-PII source for pipeline testing.  Subject-line summarization
  subset of Enron — most signature blocks are stripped so PII is rare.

  Cache: <data_dir>/aeslc_raw.jsonl
  """
  data_dir = Path(data_dir)
  data_dir.mkdir(parents=True, exist_ok=True)
  cache_path = data_dir / "aeslc_raw.jsonl"

  if cache_path.exists() and not force:
    LOGGER.info("Using cached AESLC dataset at %s", cache_path)
    return cache_path

  LOGGER.info("Downloading Yale-LILY/aeslc from HuggingFace (~18k emails)...")
  try:
    from datasets import load_dataset
  except ImportError:
    raise ImportError("pip install datasets to use the AESLC download helper.")

  ds = load_dataset("Yale-LILY/aeslc", split="train+validation+test")

  LOGGER.info("Loaded %d records; writing to %s", len(ds), cache_path)
  n = 0
  with cache_path.open("w", encoding="utf-8") as fout:
    for i, record in enumerate(ds):
      body = (record.get("email_body") or "").strip()
      subj = (record.get("subject_line") or "").strip()
      text = (subj + "\n" + body).strip() if subj else body
      if text:
        fout.write(json.dumps({"doc_id": str(i), "text": text}, ensure_ascii=False) + "\n")
        n += 1

  LOGGER.info("Done. Cached %d records → %s", n, cache_path)
  return cache_path


def load_raw_enron(data_dir: str, force: bool = False) -> List[Dict]:
  """Load corbt/enron-emails cache. Downloads if not present."""
  cache_path = download_enron(data_dir, force=force)
  records = []
  with cache_path.open(encoding="utf-8") as fh:
    for line in fh:
      line = line.strip()
      if line:
        records.append(json.loads(line))
  LOGGER.info("Loaded %d raw Enron records", len(records))
  return records


def load_raw_aeslc(data_dir: str, force: bool = False) -> List[Dict]:
  """Load AESLC cache. Downloads if not present."""
  cache_path = download_aeslc(data_dir, force=force)
  records = []
  with cache_path.open(encoding="utf-8") as fh:
    for line in fh:
      line = line.strip()
      if line:
        records.append(json.loads(line))
  LOGGER.info("Loaded %d raw AESLC records", len(records))
  return records


# ---------------------------------------------------------------------------
# PII extraction (Enron only)
# ---------------------------------------------------------------------------

def extract_pii_occurrences(
  records: List[Dict],
  pii_type: str,
  filter_extra_emails_in_prefix: bool = True,
) -> List[Dict]:
  """Find all PII occurrences in a list of email records.

  Parameters
  ----------
  records:
      List of dicts with "doc_id" and "text" keys.
  pii_type:
      ``"email"`` or ``"phone"``.
  filter_extra_emails_in_prefix:
      For email PII: discard occurrences where the 200-character window
      preceding the email contains another email address (matching the
      paper's filtering protocol).

  Returns
  -------
  List of dicts with keys: doc_id, text, pii_raw, pii_start, pii_end, pii_type.
  """
  pattern = EMAIL_RE if pii_type == "email" else PHONE_RE
  results = []

  for record in records:
    text = record["text"]
    doc_id = record["doc_id"]
    for m in pattern.finditer(text):
      start, end = m.start(), m.end()
      pii_raw = m.group(0)

      if filter_extra_emails_in_prefix and pii_type == "email":
        prefix_window = text[max(0, start - 200): start]
        if EMAIL_RE.search(prefix_window):
          continue

      results.append({
        "doc_id": doc_id,
        "text": text,
        "pii_raw": pii_raw,
        "pii_start": start,
        "pii_end": end,
        "pii_type": pii_type,
      })

  LOGGER.info("Found %d %s occurrences", len(results), pii_type)
  return results


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _tokenize_prefix_suffix(
  record: Dict,
  tokenizer,
  prefix_max_tokens: int = 100,
  max_model_len: int = 1024,
) -> Optional[Dict]:
  """Tokenize a (prefix, PII-suffix) pair. Returns None if unusable."""
  text = record["text"]
  pii_raw = record["pii_raw"]
  pii_start = record["pii_start"]

  prefix_ids = tokenizer.encode(text[:pii_start], add_special_tokens=False, truncation=False)
  suffix_ids = tokenizer.encode(pii_raw, add_special_tokens=False, truncation=False)

  if len(suffix_ids) == 0:
    return None

  # Truncate prefix to fit within model context: keep the last N tokens
  max_prefix = min(prefix_max_tokens, max_model_len - len(suffix_ids))
  if max_prefix <= 0:
    return None  # suffix alone exceeds model context
  prefix_ids = prefix_ids[-max_prefix:]

  if len(prefix_ids) == 0:
    return None

  return {
    "doc_id": record["doc_id"],
    "text": text,
    "pii_raw": pii_raw,
    "pii_type": record["pii_type"],
    "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
    "suffix_ids": torch.tensor(suffix_ids, dtype=torch.long),
  }


def _tokenize_text_window(
  record: Dict,
  tokenizer,
  prefix_tokens: int,
  suffix_tokens: int,
) -> Optional[Dict]:
  """Tokenize a fixed-length (prefix, suffix) split of an email body.

  Used by AESLCDataset for non-PII memorization testing.
  Splits the full token sequence at position prefix_tokens; suffix is the
  next suffix_tokens tokens.  Returns None if the text is too short.
  """
  text = record["text"]
  all_ids = tokenizer.encode(text, add_special_tokens=False)
  total_needed = prefix_tokens + suffix_tokens
  if len(all_ids) < total_needed:
    return None

  p_ids = all_ids[:prefix_tokens]
  s_ids = all_ids[prefix_tokens: prefix_tokens + suffix_tokens]

  return {
    "doc_id": record["doc_id"],
    "text": text,
    "pii_raw": None,
    "pii_type": "none",
    "prefix_ids": torch.tensor(p_ids, dtype=torch.long),
    "suffix_ids": torch.tensor(s_ids, dtype=torch.long),
  }


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class EnronPIIDataset(Dataset):
  """PyTorch Dataset of Enron PII (prefix, suffix) pairs.

  Uses corbt/enron-emails (~517k emails) as the data source.
  Each sample is a (prefix context, PII string) pair where the suffix is
  a real email address or phone number found in the email body/signature.

  Parameters
  ----------
  data_dir:
      Directory to cache the downloaded dataset.
  tokenizer:
      HuggingFace tokenizer matching the target model.
  pii_type:
      ``"email"`` or ``"phone"``.
  n_samples:
      Maximum samples (randomly selected). Paper uses 3,000.
  prefix_max_tokens:
      Max prefix context length in tokens. Paper uses 100.
  seed:
      Random seed for sample selection.
  force_download:
      Re-download even if cache exists.
  """

  def __init__(
    self,
    data_dir: str,
    tokenizer,
    pii_type: str = "email",
    n_samples: int = 3000,
    prefix_max_tokens: int = 100,
    seed: int = 42,
    force_download: bool = False,
    oversample_factor: int = 5,
  ) -> None:
    if pii_type not in ("email", "phone"):
      raise ValueError(f"pii_type must be 'email' or 'phone', got {pii_type!r}")

    self.pii_type = pii_type
    self.prefix_max_tokens = prefix_max_tokens

    # Max total sequence length the model can handle (prefix + suffix)
    max_model_len = getattr(tokenizer, "model_max_length", None) or 1024

    raw_records = load_raw_enron(data_dir, force=force_download)
    pii_occurrences = extract_pii_occurrences(raw_records, pii_type)

    # Shuffle and subsample candidates before tokenizing to avoid
    # tokenizing all 270k+ occurrences when we only need n_samples.
    rng = random.Random(seed)
    rng.shuffle(pii_occurrences)
    candidates = pii_occurrences[: n_samples * oversample_factor]

    from tqdm.auto import tqdm
    tokenized = []
    for occ in tqdm(candidates, desc=f"Tokenizing {pii_type} occurrences", unit="occ"):
      tok = _tokenize_prefix_suffix(occ, tokenizer, prefix_max_tokens, max_model_len)
      if tok is not None:
        tokenized.append(tok)
      if len(tokenized) >= n_samples:
        break

    LOGGER.info("Tokenized %d %s occurrences (from %d candidates)",
                len(tokenized), pii_type, len(candidates))

    self.samples = tokenized
    LOGGER.info("EnronPIIDataset: %d samples (pii_type=%s)", len(self.samples), pii_type)

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> Dict:
    return self.samples[idx]

  def get_fine_tune_texts(self) -> List[str]:
    """Return unique document texts for fine-tuning (1 epoch per paper)."""
    seen: set = set()
    texts = []
    for s in self.samples:
      if s["doc_id"] not in seen:
        seen.add(s["doc_id"])
        texts.append(s["text"])
    return texts


class AESLCDataset(Dataset):
  """PyTorch Dataset of AESLC emails for non-PII memorization testing.

  Uses Yale-LILY/aeslc (~18k emails).  Since PII is largely absent, this
  is NOT suitable for Table 1 replication.  Use it to:
    - Quickly test the pipeline end-to-end without downloading 517k emails
    - Measure general text memorization (not PII-specific)

  Each sample is a fixed (prefix_tokens, suffix_tokens) split of an email.
  The "suffix" is the target to reconstruct; it is not a PII string.

  Parameters
  ----------
  data_dir:
      Directory to cache the downloaded dataset.
  tokenizer:
      HuggingFace tokenizer matching the target model.
  n_samples:
      Maximum samples to return.
  prefix_tokens:
      Number of prefix (context) tokens per sample. Default 100.
  suffix_tokens:
      Number of suffix (target) tokens per sample. Default 10.
  seed:
      Random seed for sample selection.
  force_download:
      Re-download even if cache exists.
  """

  def __init__(
    self,
    data_dir: str,
    tokenizer,
    n_samples: int = 500,
    prefix_tokens: int = 100,
    suffix_tokens: int = 10,
    seed: int = 42,
    force_download: bool = False,
  ) -> None:
    self.prefix_tokens = prefix_tokens
    self.suffix_tokens = suffix_tokens

    raw_records = load_raw_aeslc(data_dir, force=force_download)

    from tqdm.auto import tqdm
    tokenized = []
    for rec in tqdm(raw_records, desc="Tokenizing AESLC records", unit="rec"):
      tok = _tokenize_text_window(rec, tokenizer, prefix_tokens, suffix_tokens)
      if tok is not None:
        tokenized.append(tok)

    LOGGER.info("AESLCDataset: %d usable records (min_len=%d)", len(tokenized), prefix_tokens + suffix_tokens)

    rng = random.Random(seed)
    if len(tokenized) > n_samples:
      tokenized = rng.sample(tokenized, n_samples)

    self.samples = tokenized
    LOGGER.info("AESLCDataset: %d samples", len(self.samples))

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> Dict:
    return self.samples[idx]
