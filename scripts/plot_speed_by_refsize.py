#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1:
        df = pd.read_csv(path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot speed vs ref_size with time-window lines.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="seq_per_sec_median")
    parser.add_argument("--log-x", action="store_true")
    args = parser.parse_args()

    df = _read_csv(args.summary_csv)
    if df.empty:
        raise SystemExit("summary csv is empty.")
    metric = args.metric
    if metric not in df.columns and f"{metric}_median" in df.columns:
        metric = f"{metric}_median"
    if metric not in df.columns:
        raise SystemExit(f"Missing '{args.metric}' in summary csv.")

    df = df.copy()
    df["ref_size"] = pd.to_numeric(df["ref_size"], errors="coerce")
    df = df.dropna(subset=["ref_size", metric])

    fig, ax = plt.subplots(figsize=(8, 5))
    if "active_steps" not in df.columns:
        raise SystemExit("Missing 'active_steps' in summary csv.")

    for active_steps, group in df.groupby("active_steps"):
        group = group.sort_values("ref_size")
        if pd.isna(active_steps):
            label = "steps=unknown"
        else:
            label = f"steps={int(active_steps)}"
        ax.plot(
            group["ref_size"],
            group[metric],
            marker="o",
            linestyle="-",
            label=label,
        )

    baseline_cols = [c for c in ["baseline_seq_per_sec", "baseline_sec_per_sample"] if c in df.columns]
    if "baseline_seq_per_sec" in df.columns and df["baseline_seq_per_sec"].notna().any():
        baseline_value = df["baseline_seq_per_sec"].dropna().median()
        x_min = df["ref_size"].min()
        x_max = df["ref_size"].max()
        ax.plot(
            [x_min, x_max],
            [baseline_value, baseline_value],
            color="red",
            linestyle="--",
            label="baseline",
        )

    if args.log_x:
        ax.set_xscale("log")
    ax.set_xlabel("unsafe reference size (tensor_size)")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="active steps", loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
