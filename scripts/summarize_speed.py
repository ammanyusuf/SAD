#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


KNOWN_SIZES = {50, 100, 200, 500, 1000, 1500, 2000, 3000, 5000}


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1:
        df = pd.read_csv(path)
    return df


def _extract_ref_size(run_id: str) -> Optional[int]:
    if not run_id:
        return None
    matches = re.findall(r"-(\d{3,5})(?=[-_])", run_id)
    for raw in reversed(matches):
        try:
            val = int(raw)
        except ValueError:
            continue
        if val in KNOWN_SIZES:
            return val
    return None


def _extract_timestep_window(run_id: str) -> str:
    if not run_id:
        return "unknown"
    match = re.search(r"ts(\d+)[_-](\d+)", run_id)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return "unknown"


def _is_baseline(run_id: str) -> bool:
    if not run_id:
        return False
    return run_id.endswith("-baseline")


def _active_steps(time_window: str) -> Optional[int]:
    if not time_window or time_window == "unknown":
        return None
    try:
        start_str, end_str = time_window.split("-", maxsplit=1)
        start = int(start_str)
        end = int(end_str)
        return max(0, end - start)
    except (ValueError, AttributeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize wall-clock speed by ref size and time window.")
    parser.add_argument("--speed-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--metric", type=str, default="seq_per_sec_median")
    args = parser.parse_args()

    df = _read_csv(args.speed_csv)
    if df.empty:
        raise SystemExit("speed_summary.csv is empty.")
    if args.metric not in df.columns:
        raise SystemExit(f"Missing '{args.metric}' column in speed csv.")
    if "run_id" not in df.columns:
        raise SystemExit("Missing 'run_id' column in speed csv.")

    df = df.copy()
    df["ref_size"] = df["run_id"].astype(str).apply(_extract_ref_size)
    df["time_window"] = df["run_id"].astype(str).apply(_extract_timestep_window)
    df["is_baseline"] = df["run_id"].astype(str).apply(_is_baseline)
    df["active_steps"] = df["time_window"].apply(_active_steps)
    if "experiment_slug" not in df.columns:
        df["experiment_slug"] = "unknown"

    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=[args.metric])

    df["sec_per_sample"] = 1.0 / df[args.metric]

    baseline_df = df[df["is_baseline"]].copy()
    main_df = df[~df["is_baseline"]].copy()
    main_df = main_df.dropna(subset=["ref_size"])

    summary = (
        main_df.groupby(["experiment_slug", "ref_size", "time_window", "active_steps"])[
            [args.metric, "sec_per_sample"]
        ]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    summary.columns = [
        "experiment_slug",
        "ref_size",
        "time_window",
        "active_steps",
        f"{args.metric}_mean",
        f"{args.metric}_median",
        "samples",
        "sec_per_sample_mean",
        "sec_per_sample_median",
        "samples_sec",
    ]
    summary = summary.drop(columns=["samples_sec"])
    summary = summary.sort_values(["experiment_slug", "ref_size", "time_window"])

    baseline_summary = None
    if not baseline_df.empty:
        baseline_keys = ["experiment_slug"]
        if (baseline_df["time_window"] != "unknown").any():
            baseline_keys.append("time_window")
        baseline_summary = (
            baseline_df.groupby(baseline_keys)[[args.metric, "sec_per_sample"]]
            .agg(["median"])
            .reset_index()
        )
        baseline_summary.columns = baseline_keys + [
            "baseline_seq_per_sec",
            "baseline_sec_per_sample",
        ]

    if baseline_summary is not None:
        summary = summary.merge(baseline_summary, on=baseline_keys, how="left")
        summary["delta_seq_per_sec"] = summary[f"{args.metric}_median"] - summary["baseline_seq_per_sec"]
        summary["slowdown_factor"] = summary["sec_per_sample_median"] / summary["baseline_sec_per_sample"]

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_csv, index=False)
        print(f"Wrote {args.output_csv}")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
