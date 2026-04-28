"""truncate_artifact.py — Create nested artifact subsets from a source artifact.

Reads the first K records from a pre-built artifact (shard-00000.pt, shard-00001.pt, ...)
and writes them as a new artifact directory. Because we always take records in order
(no re-sampling), N=100 is a guaranteed strict prefix of N=200, etc.

Usage:
  python -m unsafe_prep.truncate_artifact \
    --source  $UNSAFE_ARTIFACT_ROOT/real-toxicity-prompts-nested-5000-llada \
    --out-dir $UNSAFE_ARTIFACT_ROOT \
    --sizes   100 200 300 500 750 1000 2500 \
    --suffix  -llada        # appended after the size, e.g. real-toxicity-prompts-nested-0100-llada
    --name-prefix real-toxicity-prompts-nested

The source artifact itself (N=5000) is already usable as-is; this script only produces
the smaller truncated copies.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import torch


def _iter_shard_records(artifact_dir: Path):
  """Yield (input_ids_row, length, meta_entry) in shard order."""
  shard_paths = sorted(artifact_dir.glob("shard-*.pt"))
  if not shard_paths:
    raise FileNotFoundError(f"No shard files found in {artifact_dir}")
  for shard_path in shard_paths:
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    input_ids = payload["input_ids"]   # [K, L]
    lengths = payload["lengths"]       # [K]
    meta = payload["meta"]             # list[dict]
    for i in range(input_ids.shape[0]):
      yield input_ids[i], lengths[i].item(), meta[i]


def _write_artifact(
    records: list,
    out_dir: Path,
    shard_size: int,
    name: str,
    source_name: str,
) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  shards_written = 0
  for shard_start in range(0, len(records), shard_size):
    chunk = records[shard_start : shard_start + shard_size]
    ids = torch.stack([r[0] for r in chunk])
    lens = torch.tensor([r[1] for r in chunk], dtype=torch.long)
    meta = [r[2] for r in chunk]
    shard_path = out_dir / f"shard-{shards_written:05d}.pt"
    torch.save({"input_ids": ids, "lengths": lens, "meta": meta}, shard_path)
    stats_path = shard_path.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps({
            "count": len(chunk),
            "mean_length": float(lens.float().mean().item()),
            "max_length": int(lens.max().item()),
            "min_length": int(lens.min().item()),
        }, indent=2),
        encoding="utf-8",
    )
    shards_written += 1

  count = len(records)
  lengths_all = torch.tensor([r[1] for r in records], dtype=torch.float)
  artifact_entry = {
      "name": name,
      "source": source_name,
      "count": count,
      "num_shards": shards_written,
      "mean_length": float(lengths_all.mean().item()),
      "std_length": float(lengths_all.std().item()) if count > 1 else 0.0,
      "min_length": int(lengths_all.min().item()),
      "max_length": int(lengths_all.max().item()),
      "storage": {
          "layout": "single_shard" if shards_written == 1 else "sharded",
          "paths": [f"shard-{i:05d}.pt" for i in range(shards_written)],
          "materialized_path": None,
      },
      "filters": {"truncated_from": source_name},
      "sample_seed": None,
      "sample_size_requested": count,
      "category_counts": {},
      "categories": [],
  }
  return artifact_entry


def _update_root_index(root: Path, new_entry: dict) -> None:
  index_path = root / "index.json"
  if index_path.exists():
    index = json.loads(index_path.read_text(encoding="utf-8"))
  else:
    index = {"unsafe_artifacts": []}
  artifacts = {e["name"]: e for e in index.get("unsafe_artifacts", [])}
  artifacts[new_entry["name"]] = new_entry
  index["unsafe_artifacts"] = sorted(artifacts.values(), key=lambda e: e["name"])
  index["built_at"] = datetime.now(timezone.utc).isoformat()
  index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def truncate(
    source_dir: Path,
    out_root: Path,
    sizes: List[int],
    name_prefix: str,
    suffix: str,
    shard_size: int,
) -> None:
  max_size = max(sizes)
  print(f"Reading up to {max_size} records from {source_dir} ...")
  records = []
  for row in _iter_shard_records(source_dir):
    records.append(row)
    if len(records) >= max_size:
      break
  print(f"  Read {len(records)} records.")

  if len(records) < max_size:
    print(f"WARNING: source has only {len(records)} records; sizes larger than this will be capped.")

  for k in sorted(sizes):
    subset = records[:k]
    if not subset:
      print(f"  Skipping N={k}: no records available.")
      continue
    name = f"{name_prefix}-{k:04d}{suffix}"
    out_dir = out_root / name
    print(f"  Writing N={len(subset)} → {out_dir.name} ...")
    entry = _write_artifact(subset, out_dir, shard_size, name, source_dir.name)
    _update_root_index(out_root, entry)
    print(f"    Done ({entry['num_shards']} shard(s)).")

  print("All done.")


def main() -> None:
  parser = argparse.ArgumentParser(description="Create nested artifact subsets by prefix truncation.")
  parser.add_argument("--source", type=Path, required=True,
                      help="Path to the source artifact directory (e.g. .../real-toxicity-prompts-nested-1000-llada).")
  parser.add_argument("--out-dir", type=Path, required=True,
                      help="Root artifact directory where new sub-artifacts will be written (same as UNSAFE_ARTIFACT_ROOT).")
  parser.add_argument("--sizes", type=int, nargs="+", required=True,
                      help="Subset sizes to produce (must all be <= source artifact size).")
  parser.add_argument("--name-prefix", type=str, required=True,
                      help="Prefix for output artifact names, e.g. 'real-toxicity-prompts-nested'.")
  parser.add_argument("--suffix", type=str, default="", nargs="?", const="",
                      help="Suffix appended after the zero-padded size, e.g. '--suffix=-llada'.")
  parser.add_argument("--shard-size", type=int, default=1024,
                      help="Records per shard file (default: 1024).")
  args = parser.parse_args()
  truncate(
      source_dir=args.source,
      out_root=args.out_dir,
      sizes=args.sizes,
      name_prefix=args.name_prefix,
      suffix=args.suffix,
      shard_size=args.shard_size,
  )


if __name__ == "__main__":
  main()
