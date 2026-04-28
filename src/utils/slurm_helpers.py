"""Shared utilities for SLURM staging scripts.

Provides lightweight CLIs to (a) count prompts inside a dataset JSON/JSONL file
and (b) aggregate per-shard `run_metadata.json` artifacts into a compact
summary that captures throughput and memory telemetry.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable, List, Optional

from tools import generate as generate_module


def _load_metadata_files(root: Path) -> List[dict]:
    records: List[dict] = []
    for meta_path in root.rglob("run_metadata.json"):
        try:
            records.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return records


def _print_dataset_count(dataset_path: Path) -> None:
    print(f"[slurm_helpers] Counting prompts in {dataset_path}")
    records = generate_module._load_dataset(dataset_path)
    count = len(records)
    print(f"[slurm_helpers] Dataset contains {count} prompts")
    print(count)


def _aggregate_metadata(
    root: Path,
    output_path: Path,
    job_id: str,
    experiment_slug: str,
    total_prompts: int,
    gpu_shards: int,
    task_id: Optional[str],
    slice_bounds: Optional[tuple[int, int]],
) -> None:
    print(f"[slurm_helpers] Aggregating metadata under {root} (job_id={job_id})")
    records = _load_metadata_files(root)
    payload = {
        "job_id": job_id,
        "experiment_slug": experiment_slug,
        "total_prompts": total_prompts,
        "gpu_shards": gpu_shards,
        "records": len(records),
    }
    if task_id is not None:
        payload["task_id"] = task_id
    if slice_bounds is not None:
        payload["task_range"] = list(slice_bounds)

    if not records:
        print("[slurm_helpers] No run_metadata.json files discovered; writing error payload")
        payload["error"] = "no metadata discovered"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[slurm_helpers] Summary written to {output_path}")
        return

    seq_per_sec: Iterable[float] = [
        telemetry.get("sequences_per_second")
        for telemetry in (record.get("telemetry", {}) for record in records)
        if telemetry.get("sequences_per_second") is not None
    ]
    peak_vram: Iterable[float] = [
        telemetry.get("peak_vram_gb")
        for telemetry in (record.get("telemetry", {}) for record in records)
        if telemetry.get("peak_vram_gb") is not None
    ]

    seq_per_sec = list(seq_per_sec)
    peak_vram = list(peak_vram)
    if seq_per_sec:
        payload["seq_per_sec_min"] = min(seq_per_sec)
        payload["seq_per_sec_median"] = statistics.median(seq_per_sec)
    if peak_vram:
        payload["peak_vram_max"] = max(peak_vram)

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    median_seq = payload.get("seq_per_sec_median")
    peak = payload.get("peak_vram_max")
    median_str = f"{median_seq:.3f}" if isinstance(median_seq, (int, float)) else "n/a"
    peak_str = f"{peak:.3f}" if isinstance(peak, (int, float)) else "n/a"
    print(
        f"[slurm_helpers] Aggregated {len(records)} metadata files (median seq/s={median_str}, peak VRAM={peak_str} GB)"
    )
    print(f"[slurm_helpers] Summary written to {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utility helpers for SLURM staging scripts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    count_parser = subparsers.add_parser("count", help="Print the number of prompts in a dataset JSON.")
    count_parser.add_argument("--dataset", type=Path, required=True, help="Dataset JSON/JSONL path.")

    aggregate_parser = subparsers.add_parser("aggregate", help="Summarize run metadata under a directory.")
    aggregate_parser.add_argument("--root", type=Path, required=True, help="Directory containing shard outputs.")
    aggregate_parser.add_argument("--output", type=Path, required=True, help="Destination for summary JSON.")
    aggregate_parser.add_argument("--job-id", required=True)
    aggregate_parser.add_argument("--experiment-slug", required=True)
    aggregate_parser.add_argument("--total-prompts", type=int, required=True)
    aggregate_parser.add_argument("--gpu-shards", type=int, required=True)
    aggregate_parser.add_argument("--task-id", default=None)
    aggregate_parser.add_argument("--slice-start", type=int, default=None)
    aggregate_parser.add_argument("--slice-end", type=int, default=None)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "count":
        _print_dataset_count(args.dataset)
    elif args.command == "aggregate":
        bounds = None
        if args.slice_start is not None and args.slice_end is not None:
            bounds = (args.slice_start, args.slice_end)
        _aggregate_metadata(
            root=args.root,
            output_path=args.output,
            job_id=args.job_id,
            experiment_slug=args.experiment_slug,
            total_prompts=args.total_prompts,
            gpu_shards=args.gpu_shards,
            task_id=args.task_id,
            slice_bounds=bounds,
        )
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
