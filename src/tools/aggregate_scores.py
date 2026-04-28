#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-shard safety scoring outputs into consolidated tables."
    )
    parser.add_argument(
        "--classifier",
        default="llamaguard",
        help="Classifier identifier (matches the folder under scores/).",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        dest="run_dirs",
        required=True,
        help="Run directory containing scores/<classifier>/*.parquet or .jsonl files. "
             "Specify multiple times to aggregate several runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for aggregated parquet/jsonl and summary CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def collect_score_files(scores_dir: Path) -> List[Path]:
    if not scores_dir.exists():
        raise SystemExit(f"Scores directory missing: {scores_dir}")
    files = sorted(scores_dir.glob("*.parquet"))
    if not files:
        files = sorted(scores_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No shard score files found in {scores_dir}")
    return files


def load_records(files: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for file in files:
        if file.suffix == ".parquet":
            frames.append(pd.read_parquet(file))
        else:
            with file.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> List[List]:
    if df.empty:
        return []
    required = {"category", "unsafe", "length"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing expected columns in score data: {sorted(missing)}")
    grouped = df.groupby("category", dropna=False)
    rows: List[List] = []
    for category, group in grouped:
        count = len(group)
        unsafe = int(group["unsafe"].astype(bool).sum())
        unsafe_rate = unsafe / count if count else 0.0
        avg_len = float(group["length"].mean()) if count else 0.0
        rows.append([category or "unknown", count, unsafe, round(unsafe_rate, 4), round(avg_len, 2)])
    rows.sort(key=lambda item: item[0])
    return rows


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated_frames: List[pd.DataFrame] = []
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        scores_dir = run_dir / "scores" / args.classifier
        files = collect_score_files(scores_dir)
        LOGGER.info("Loaded %d score files from %s", len(files), scores_dir)
        aggregated_frames.append(load_records(files))

    if not aggregated_frames:
        LOGGER.warning("No score data found.")
        return

    combined = pd.concat(aggregated_frames, ignore_index=True)
    combined_path = output_dir / f"{args.classifier}_scores.parquet"
    if combined_path.exists() and not args.force:
        raise SystemExit(f"{combined_path} already exists. Use --force to overwrite.")
    combined.to_parquet(combined_path, index=False)
    LOGGER.info("Wrote combined scores to %s", combined_path)

    summary_rows = summarize(combined)
    summary_path = output_dir / f"{args.classifier}_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "count", "unsafe", "unsafe_rate", "avg_length"])
        writer.writerows(summary_rows)
    LOGGER.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
