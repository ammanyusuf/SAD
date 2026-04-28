#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORBLIND_SAFE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
    "#F0E442",  # yellow
    "#000000",  # black
]


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "lines.linewidth": 1.6,
            "lines.markersize": 6,
            "axes.grid": True,
            "grid.color": "0.85",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1:
        df = pd.read_csv(path)
    return df


def _round_steps(value: object, step: int) -> int | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(val):
        return None
    return int(round(val / step) * step)


def _pick_best_time_window(df: pd.DataFrame) -> pd.DataFrame:
    if "samples" not in df.columns:
        return df
    df = df.copy()
    df["samples"] = pd.to_numeric(df["samples"], errors="coerce")
    df = df.sort_values(["ref_size", "active_steps_bucket", "samples"], ascending=[True, True, False])
    return df.drop_duplicates(subset=["ref_size", "active_steps_bucket"], keep="first")


def _interactive_rename_labels(labels: list[str], prompt_prefix: str, enabled: bool) -> list[str]:
    if not enabled or not sys.stdin.isatty():
        return labels
    renamed: dict[str, str] = {}
    for label in labels:
        if label in renamed:
            continue
        response = input(f"{prompt_prefix} rename '{label}' -> ").strip()
        if response:
            renamed[label] = response
    return [renamed.get(label, label) for label in labels]


def _plot_metric(
    df: pd.DataFrame,
    metric_col: str,
    baseline_col: str | None,
    output_path: Path,
    x_label: str,
    y_label: str,
    log_x: bool,
    title: str | None,
    interactive_rename: bool,
    legend_title: str,
) -> None:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    grouped = df.groupby("active_steps_bucket")
    buckets = sorted([b for b in grouped.groups.keys() if b is not None])
    line_labels = [f"steps={bucket}" for bucket in buckets]
    line_labels = _interactive_rename_labels(line_labels, "line", interactive_rename)
    label_map = {bucket: label for bucket, label in zip(buckets, line_labels)}
    color_map = {bucket: COLORBLIND_SAFE[i % len(COLORBLIND_SAFE)] for i, bucket in enumerate(buckets)}

    for bucket in buckets:
        group = grouped.get_group(bucket).sort_values("ref_size")
        ax.plot(
            group["ref_size"],
            group[metric_col],
            marker="o",
            linestyle="-",
            color=color_map[bucket],
            label=label_map.get(bucket, f"steps={bucket}"),
        )

    baseline_value = None
    if baseline_col and baseline_col in df.columns:
        baseline_series = pd.to_numeric(df[baseline_col], errors="coerce").dropna()
        if not baseline_series.empty:
            baseline_value = float(baseline_series.median())
    if baseline_value is not None:
        x_min = df["ref_size"].min()
        x_max = df["ref_size"].max()
        ax.plot(
            [x_min, x_max],
            [baseline_value, baseline_value],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="baseline",
        )

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.legend(title=legend_title, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_path}")


def _aggregate_time_windows(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "best":
        return _pick_best_time_window(df)
    metric_cols = [
        "seq_per_sec_median_mean",
        "seq_per_sec_median_median",
        "sec_per_sample_mean",
        "sec_per_sample_median",
        "baseline_seq_per_sec",
        "baseline_sec_per_sample",
    ]
    cols = [c for c in metric_cols if c in df.columns]
    agg = df.groupby(["ref_size", "active_steps_bucket"])[cols].agg(mode).reset_index()
    return agg


def _parse_exclude_steps(raw: str | None) -> set[int]:
    if not raw:
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-ready ref-size plot with active-step lines (bucketed)."
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument("--bucket-step", type=int, default=10)
    parser.add_argument("--exclude-steps", type=str, default="600")
    parser.add_argument(
        "--time-window",
        choices=["best", "mean", "median"],
        default="best",
        help="How to combine multiple time windows for the same ref_size + steps.",
    )
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--x-label", type=str, default="unsafe reference size (|D_unsafe|)")
    parser.add_argument("--legend-title", type=str, default="active steps")
    parser.add_argument("--y-label-seq", type=str, default="seq/sec (median)")
    parser.add_argument("--y-label-sec", type=str, default="sec/sample (median)")
    parser.add_argument("--interactive-rename", action="store_true")
    args = parser.parse_args()

    df = _read_csv(args.summary_csv)
    if df.empty:
        raise SystemExit("summary csv is empty.")

    required = {"ref_size", "active_steps"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    df = df.copy()
    df["ref_size"] = pd.to_numeric(df["ref_size"], errors="coerce")
    df["active_steps_bucket"] = df["active_steps"].apply(lambda v: _round_steps(v, args.bucket_step))
    df = df.dropna(subset=["ref_size", "active_steps_bucket"])
    df["active_steps_bucket"] = pd.to_numeric(df["active_steps_bucket"], errors="coerce")
    exclude_steps = _parse_exclude_steps(args.exclude_steps)
    if exclude_steps:
        df = df[~df["active_steps_bucket"].isin(exclude_steps)]

    df = _aggregate_time_windows(df, args.time_window)

    use_log_x = args.log_x and not args.linear_x

    metric_pairs = [
        ("seq_per_sec_median_median", "baseline_seq_per_sec", args.y_label_seq),
        ("sec_per_sample_median", "baseline_sec_per_sample", args.y_label_sec),
    ]
    for metric_col, baseline_col, y_label in metric_pairs:
        if metric_col not in df.columns:
            continue
        out_path = args.output_prefix.with_name(
            f"{args.output_prefix.name}_{metric_col}{'_logx' if use_log_x else ''}.png"
        )
        _plot_metric(
            df=df,
            metric_col=metric_col,
            baseline_col=baseline_col,
            output_path=out_path,
            x_label=args.x_label,
            y_label=y_label,
            log_x=use_log_x,
            title=args.title,
            interactive_rename=args.interactive_rename,
            legend_title=args.legend_title,
        )


if __name__ == "__main__":
    main()
