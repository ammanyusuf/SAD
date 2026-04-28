#!/usr/bin/env python3
"""
memorization/setup_assets.py — Download and verify all assets required for
the memorization experiments (Table 1 replication).

Run this on a login node (internet access) BEFORE submitting to a compute node.

Usage:
  # Check what's present, download what's missing
  python memorization/setup_assets.py

  # Only check (no downloads)
  python memorization/setup_assets.py --check-only

  # Force re-download of datasets even if cached
  python memorization/setup_assets.py --force-datasets

  # Skip model checks (e.g. you only care about data)
  python memorization/setup_assets.py --skip-models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(msg: str) -> None:
  print(f"  [OK]   {msg}")

def _warn(msg: str) -> None:
  print(f"  [WARN] {msg}")

def _info(msg: str) -> None:
  print(f"  [INFO] {msg}")

def _err(msg: str) -> None:
  print(f"  [ERR]  {msg}", file=sys.stderr)

def _header(msg: str) -> None:
  print(f"\n{'='*60}")
  print(f"  {msg}")
  print(f"{'='*60}")

def _nonempty_dir(path: Path) -> bool:
  return path.exists() and path.is_dir() and any(path.iterdir())

def _env(var: str, default: Optional[str] = None) -> Optional[str]:
  return os.environ.get(var, default)

def _expand(path_str: str) -> Path:
  return Path(os.path.expanduser(path_str))


# ---------------------------------------------------------------------------
# Model checks
# ---------------------------------------------------------------------------

MODELS = [
  {
    "name": "gpt2-large (ARM baseline)",
    "hf_id": "openai-community/gpt2-large",
    "env_var": "HF_MODELS_CACHE",
    "subdir": "gpt2-large",
    "manual": False,
    "note": None,
  },
  {
    "name": "mdlm-wiki (702-1250000.ckpt)",
    "hf_id": None,
    "env_var": "CHECKPOINT_PATH",
    "subdir": None,  # env_var points directly to the file
    "manual": True,
    "note": (
      "This checkpoint is not on HuggingFace. Copy it to the path shown above.\n"
      "    Expected location: ~/scratch/models/text-diffusion/702-1250000.ckpt\n"
      "    Ask your collaborator for the checkpoint file."
    ),
  },
]


def check_models(
  hf_models_cache: Optional[str],
  checkpoint_path: Optional[str],
  check_only: bool,
  skip_models: bool,
) -> List[str]:
  """Return list of warning/error strings for the summary."""
  issues = []

  if skip_models:
    _info("Skipping model checks (--skip-models).")
    return issues

  _header("Model checks")

  for m in MODELS:
    name = m["name"]

    if m["subdir"] is not None:
      # Path is $ENV_VAR / subdir
      base = hf_models_cache or _env(m["env_var"])
      if not base:
        msg = f"{name}: env var ${m['env_var']} not set — run: source scripts/env_profile.sh dlm-memorization"
        _err(msg)
        issues.append(msg)
        continue
      target = _expand(base) / m["subdir"]
    else:
      # Path is directly $ENV_VAR (points to a file or dir)
      target_str = checkpoint_path or _env(m["env_var"])
      if not target_str:
        msg = f"{name}: env var ${m['env_var']} not set — run: source scripts/env_profile.sh dlm-memorization"
        _err(msg)
        issues.append(msg)
        continue
      target = _expand(target_str)

    if m["manual"]:
      # Can't auto-download — just check presence
      if target.exists():
        _ok(f"{name}  →  {target}")
      else:
        msg = f"{name}: NOT FOUND at {target}"
        _warn(msg)
        if m["note"]:
          for line in m["note"].splitlines():
            print(f"    {line}")
        issues.append(msg)
    else:
      # Can auto-download via HuggingFace
      if _nonempty_dir(target):
        _ok(f"{name}  →  {target}")
      elif check_only:
        msg = f"{name}: missing at {target} (run without --check-only to download)"
        _warn(msg)
        issues.append(msg)
      else:
        _info(f"Downloading {m['hf_id']} → {target} ...")
        try:
          from huggingface_hub import snapshot_download
          snapshot_download(
            repo_id=m["hf_id"],
            local_dir=str(target),
            local_dir_use_symlinks=False,
          )
          _ok(f"{name}  →  {target}")
        except Exception as exc:
          msg = f"{name}: download failed — {exc}"
          _err(msg)
          issues.append(msg)

  return issues


# ---------------------------------------------------------------------------
# Dataset checks / downloads
# ---------------------------------------------------------------------------

def _download_dataset(cache_path: Path, hf_id: str, label: str, write_fn, force: bool) -> bool:
  """Generic HuggingFace dataset downloader. write_fn(ds, fout) writes JSONL rows."""
  if cache_path.exists() and not force:
    _ok(f"{label} cache already present: {cache_path} ({cache_path.stat().st_size // 1024} KB)")
    return True

  _info(f"Downloading {hf_id} → {cache_path} ...")
  try:
    from datasets import load_dataset
  except ImportError:
    _err("Missing 'datasets' package — pip install datasets")
    return False

  try:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(hf_id, split="train")
    n = 0
    tmp_path = cache_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as fout:
      for i, record in enumerate(ds):
        row = write_fn(i, record)
        if row:
          fout.write(json.dumps(row, ensure_ascii=False) + "\n")
          n += 1
    tmp_path.rename(cache_path)
    _ok(f"{label}: {n} records → {cache_path}")
    return True
  except Exception as exc:
    _err(f"{label} download failed: {exc}")
    return False


def check_datasets(
  memorization_data_dir: Optional[str],
  check_only: bool,
  force_datasets: bool,
) -> List[str]:
  """Return list of warning/error strings for the summary."""
  issues = []

  _header("Dataset checks")

  if not memorization_data_dir:
    msg = "$MEMORIZATION_DATA_DIR not set — run: source scripts/env_profile.sh dlm-memorization"
    _err(msg)
    return [msg]

  data_dir = _expand(memorization_data_dir)
  enron_dir = data_dir / "enron"

  # ---- corbt/enron-emails (PII experiments, Table 1) ----
  enron_cache = enron_dir / "enron_raw.jsonl"
  if enron_cache.exists() and not force_datasets:
    _ok(f"corbt/enron-emails: {enron_cache} ({enron_cache.stat().st_size // 1024} KB)")
  elif check_only:
    msg = f"corbt/enron-emails cache missing at {enron_cache} (run without --check-only to download)"
    _warn(msg)
    issues.append(msg)
  else:
    def _enron_row(i, record):
      subject = (record.get("subject") or "").strip()
      body = (record.get("body") or "").strip()
      text = (subject + "\n" + body).strip() if subject else body
      return {"doc_id": str(i), "text": text} if text else None

    ok = _download_dataset(enron_cache, "corbt/enron-emails", "corbt/enron-emails", _enron_row, force_datasets)
    if not ok:
      issues.append("corbt/enron-emails download failed — check internet access")

  # ---- Yale-LILY/aeslc (non-PII, pipeline testing) ----
  aeslc_cache = enron_dir / "aeslc_raw.jsonl"
  if aeslc_cache.exists() and not force_datasets:
    _ok(f"Yale-LILY/aeslc:       {aeslc_cache} ({aeslc_cache.stat().st_size // 1024} KB)")
  elif check_only:
    msg = f"Yale-LILY/aeslc cache missing at {aeslc_cache} (run without --check-only to download)"
    _warn(msg)
    issues.append(msg)
  else:
    def _aeslc_row(i, record):
      body = (record.get("email_body") or "").strip()
      subj = (record.get("subject_line") or "").strip()
      text = (subj + "\n" + body).strip() if subj else body
      return {"doc_id": str(i), "text": text} if text else None

    # AESLC has train/validation/test splits — download all
    try:
      from datasets import load_dataset as _ld
      enron_dir.mkdir(parents=True, exist_ok=True)
      ds = _ld("Yale-LILY/aeslc", split="train+validation+test")
      n = 0
      tmp_path = aeslc_cache.with_suffix(".jsonl.tmp")
      with tmp_path.open("w", encoding="utf-8") as fout:
        for i, record in enumerate(ds):
          row = _aeslc_row(i, record)
          if row:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
      tmp_path.rename(aeslc_cache)
      _ok(f"Yale-LILY/aeslc: {n} records → {aeslc_cache}")
    except Exception as exc:
      msg = f"Yale-LILY/aeslc download failed: {exc}"
      _err(msg)
      issues.append(msg)

  # ---- wikitext-103-raw-v1 (verbatim memorization, WikiText experiment) ----
  wiki_dir = data_dir / "wikitext"
  wiki_cache = wiki_dir / "wikitext103_train.jsonl"
  if wiki_cache.exists() and not force_datasets:
    _ok(f"wikitext-103 train:    {wiki_cache} ({wiki_cache.stat().st_size // 1024} KB)")
  elif check_only:
    msg = f"wikitext-103 cache missing at {wiki_cache} (run without --check-only to download)"
    _warn(msg)
    issues.append(msg)
  else:
    _info("Downloading wikitext-103-raw-v1 (train split) → JSONL cache ...")
    try:
      from memorization.data.wikitext import download_wikitext
      download_wikitext(str(wiki_dir), split="train", force=force_datasets)
      _ok(f"wikitext-103 train → {wiki_cache}")
    except Exception as exc:
      msg = f"wikitext-103 download failed: {exc}"
      _err(msg)
      issues.append(msg)

  print()
  print("  corbt/enron-emails  — ~517k emails with real PII (Table 1 replication)")
  print("  Yale-LILY/aeslc     — ~18k emails, no PII, for pipeline testing only")
  print("  wikitext-103        — Wikipedia text, verbatim memorization experiment")

  return issues


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(summary_path: Path, issues: List[str], all_checks: Dict) -> None:
  data = {"issues": issues, "checks": all_checks}
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
  _info(f"Summary written to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description="Download and verify assets for memorization experiments.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  p.add_argument("--check-only", action="store_true",
                 help="Only check presence; do not download anything.")
  p.add_argument("--skip-models", action="store_true",
                 help="Skip model presence checks.")
  p.add_argument("--force-datasets", action="store_true",
                 help="Re-download datasets even if cached.")
  p.add_argument("--summary-json", default=None,
                 help="Write a JSON summary to this path.")
  return p.parse_args(argv)


def main(argv=None) -> None:
  args = parse_args(argv)

  # Read env vars (set by env_profile.sh dlm-memorization)
  hf_models_cache     = _env("HF_MODELS_CACHE")
  checkpoint_path     = _env("CHECKPOINT_PATH")
  memorization_data_dir = _env("MEMORIZATION_DATA_DIR")
  memorization_results_dir = _env("MEMORIZATION_RESULTS_DIR")

  print("\nMemorization experiment asset checker")
  print(f"  HF_MODELS_CACHE:          {hf_models_cache or '(not set)'}")
  print(f"  CHECKPOINT_PATH:          {checkpoint_path or '(not set)'}")
  print(f"  MEMORIZATION_DATA_DIR:    {memorization_data_dir or '(not set)'}")
  print(f"  MEMORIZATION_RESULTS_DIR: {memorization_results_dir or '(not set)'}")

  if not any([hf_models_cache, checkpoint_path, memorization_data_dir]):
    print()
    print("  No environment variables detected.")
    print("  Run this first:  source scripts/env_profile.sh dlm-memorization")
    print()

  all_issues: List[str] = []

  model_issues = check_models(
    hf_models_cache=hf_models_cache,
    checkpoint_path=checkpoint_path,
    check_only=args.check_only,
    skip_models=args.skip_models,
  )
  all_issues.extend(model_issues)

  dataset_issues = check_datasets(
    memorization_data_dir=memorization_data_dir,
    check_only=args.check_only,
    force_datasets=args.force_datasets,
  )
  all_issues.extend(dataset_issues)

  # ---- Results dir ----
  _header("Results directory")
  if memorization_results_dir:
    results_dir = _expand(memorization_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    _ok(f"Results dir ready: {results_dir}")
  else:
    _warn("$MEMORIZATION_RESULTS_DIR not set — output will use a default path")

  # ---- Final summary ----
  _header("Summary")
  if all_issues:
    print(f"  {len(all_issues)} issue(s) found:")
    for issue in all_issues:
      print(f"    - {issue}")
  else:
    print("  All checks passed. Ready to run experiments.")

  if args.summary_json:
    write_summary(
      Path(os.path.expanduser(args.summary_json)),
      all_issues,
      {
        "hf_models_cache": hf_models_cache,
        "checkpoint_path": checkpoint_path,
        "memorization_data_dir": memorization_data_dir,
        "memorization_results_dir": memorization_results_dir,
      },
    )

  if all_issues:
    sys.exit(1)


if __name__ == "__main__":
  main()
