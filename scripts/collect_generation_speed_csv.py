#!/usr/bin/env python3
"""Collect generation speed summaries into a single CSV.

Scans for job_run_metadata.json / task_metadata.json (aggregate outputs) and
optionally falls back to per-shard run_metadata.json when needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple


AGGREGATE_FILENAMES = ("job_run_metadata.json", "task_metadata.json")


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _iter_run_metadata(root: Path) -> Iterable[Path]:
    return root.rglob("run_metadata.json")


def _summarize_shards(root: Path) -> Optional[Tuple[float, float, float, int]]:
    values = []
    for meta_path in _iter_run_metadata(root):
        payload = _load_json(meta_path)
        if not payload:
            continue
        telemetry = payload.get("telemetry", {})
        seq = telemetry.get("sequences_per_second")
        if isinstance(seq, (int, float)):
            values.append(float(seq))
    if not values:
        return None
    values.sort()
    return (min(values), statistics.median(values), max(values), len(values))


def _extract_model_id(root: Path) -> str:
    for meta_path in _iter_run_metadata(root):
        payload = _load_json(meta_path)
        if not payload:
            continue
        model = payload.get("model", {})
        checkpoint = model.get("checkpoint")
        name = model.get("name")
        if isinstance(checkpoint, str) and checkpoint:
            return checkpoint
        if isinstance(name, str) and name:
            return name
    return "unknown"


def _extract_run_id(root: Path) -> Optional[str]:
    for meta_path in _iter_run_metadata(root):
        payload = _load_json(meta_path)
        if not payload:
            continue
        run_id = payload.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
        io = payload.get("io", {})
        run_id = io.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
    return None


def _extract_experiment_slug(root: Path, fallback: Optional[str]) -> str:
    if isinstance(fallback, str) and fallback:
        return fallback
    for meta_path in _iter_run_metadata(root):
        payload = _load_json(meta_path)
        if not payload:
            continue
        io = payload.get("io", {})
        slug = io.get("experiment_slug")
        if isinstance(slug, str) and slug:
            return slug
    return "unknown"


def _extract_generation(root: Path) -> tuple[Optional[int], Optional[int], Optional[int]]:
    for meta_path in _iter_run_metadata(root):
        payload = _load_json(meta_path)
        if not payload:
            continue
        gen = payload.get("generation", {})
        steps = gen.get("sampling_steps")
        batch = gen.get("batch_size")
        max_new = gen.get("max_new_tokens")
        return (
            steps if isinstance(steps, int) else None,
            batch if isinstance(batch, int) else None,
            max_new if isinstance(max_new, int) else None,
        )
    return (None, None, None)


def _iter_aggregate_files(root: Path) -> Iterable[Path]:
    for name in AGGREGATE_FILENAMES:
        yield from root.rglob(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan results and emit CSV with generation speed summaries."
    )
    parser.add_argument("--root", type=Path, required=True, help="Results root to scan.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. If omitted, writes to stdout.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log discovered metadata files and fallbacks.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[collect_speed] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    rows = []
    seen_roots = set()

    for meta_path in _iter_aggregate_files(args.root):
        payload = _load_json(meta_path)
        if not payload:
            logger.warning("Skip unreadable JSON: %s", meta_path)
            continue
        logger.info("Found aggregate: %s", meta_path)
        seq_median = payload.get("seq_per_sec_median")
        if not isinstance(seq_median, (int, float)):
            # Fallback to shards if aggregate missing
            shard_stats = _summarize_shards(meta_path.parent)
            if shard_stats is None:
                logger.warning("No shard telemetry under %s", meta_path.parent)
                continue
            seq_min, seq_med, seq_max, records = shard_stats
        else:
            seq_min = payload.get("seq_per_sec_min")
            seq_max = payload.get("seq_per_sec_max")
            records = payload.get("records")
            seq_med = float(seq_median)
            if not isinstance(seq_min, (int, float)) or not isinstance(seq_max, (int, float)):
                shard_stats = _summarize_shards(meta_path.parent)
                if shard_stats is not None:
                    logger.info("Aggregate missing min/max; using shards under %s", meta_path.parent)
                    seq_min, _, seq_max, records = shard_stats
        model_id = _extract_model_id(meta_path.parent)
        run_id = _extract_run_id(meta_path.parent)
        experiment_slug = _extract_experiment_slug(meta_path.parent, payload.get("experiment_slug"))
        sampling_steps, batch_size, max_new_tokens = _extract_generation(meta_path.parent)

        rows.append(
            {
                "experiment_slug": experiment_slug,
                "model": model_id,
                "run_id": run_id or "",
                "seq_per_sec_median": f"{seq_med:.6f}",
                "seq_per_sec_min": "" if seq_min is None else f"{float(seq_min):.6f}",
                "seq_per_sec_max": "" if seq_max is None else f"{float(seq_max):.6f}",
                "records": "" if records is None else str(records),
                "sampling_steps": "" if sampling_steps is None else str(sampling_steps),
                "batch_size": "" if batch_size is None else str(batch_size),
                "max_new_tokens": "" if max_new_tokens is None else str(max_new_tokens),
                "source": str(meta_path),
            }
        )
        seen_roots.add(meta_path.parent)

    # Optional: include roots that only have run_metadata.json (no aggregate)
    for meta_path in args.root.rglob("run_metadata.json"):
        run_root = meta_path.parent
        if run_root in seen_roots:
            continue
        shard_stats = _summarize_shards(run_root)
        if shard_stats is None:
            logger.warning("Skip run with no telemetry under %s", run_root)
            continue
        logger.info("Found shard-only telemetry under %s", run_root)
        seq_min, seq_med, seq_max, records = shard_stats
        model_id = _extract_model_id(run_root)
        run_id = _extract_run_id(run_root)
        experiment_slug = _extract_experiment_slug(run_root, None)
        sampling_steps, batch_size, max_new_tokens = _extract_generation(run_root)
        rows.append(
            {
                "experiment_slug": experiment_slug,
                "model": model_id,
                "run_id": run_id or "",
                "seq_per_sec_median": f"{seq_med:.6f}",
                "seq_per_sec_min": f"{seq_min:.6f}",
                "seq_per_sec_max": f"{seq_max:.6f}",
                "records": str(records),
                "sampling_steps": "" if sampling_steps is None else str(sampling_steps),
                "batch_size": "" if batch_size is None else str(batch_size),
                "max_new_tokens": "" if max_new_tokens is None else str(max_new_tokens),
                "source": str(run_root),
            }
        )

    fieldnames = [
        "experiment_slug",
        "model",
        "run_id",
        "seq_per_sec_median",
        "seq_per_sec_min",
        "seq_per_sec_max",
        "records",
        "sampling_steps",
        "batch_size",
        "max_new_tokens",
        "source",
    ]

    output = args.output
    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
