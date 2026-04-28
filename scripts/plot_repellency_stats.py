#!/usr/bin/env python3
"""
Simple utility to inspect repellency CSV logs.

Example:
    python scripts/plot_repellency_stats.py --csv /mnt/data/repellency_stats.csv --outdir ./artifacts/repellency_plots --rolling 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE_METRICS = ["log_q_t_mean", "q_t_mean"]
BETA_LOG_MEANS = ["log_beta_raw_len_mean", "log_beta_raw_raw_mean"]
BETA_HAT_MEANS = ["beta_hat_len_mean", "beta_hat_raw_mean"]
BETA_CLAMP_FRACS = ["log_beta_raw_len_clamp_min_frac", "log_beta_raw_raw_clamp_min_frac"]
STRENGTH_METRICS = ["strength_mean", "g_t_mean"]
HEALTH_METRICS = [
    "rho_mean",
    "w_variance_mean",
    "p_unsafe_entropy_mean",
    "changed_frac_masked",
    "kl_logit_mean",
    "kl_prob_mean",
    "unsafe_shift_prob",
    "unsafe_shift_logit",
    "mask_frac",
    "tv_safe_data_mean",
    "kl_safe_data_mean",
    "top1_change_rate",
    "effective_strength",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot repellency debug stats from CSV.")
    parser.add_argument("--csv", default="/mnt/data/repellency_stats.csv", help="Path to the input CSV.")
    parser.add_argument(
        "--outdir",
        default="./artifacts/repellency_plots",
        help="Directory to write plot PNGs.",
    )
    parser.add_argument(
        "--rolling",
        type=int,
        default=1,
        help="Window size for moving average smoothing (min 1).",
    )
    parser.add_argument(
        "--window-start",
        type=float,
        default=None,
        help="Highlight span start (step) for axvspan overlays.",
    )
    parser.add_argument(
        "--window-end",
        type=float,
        default=None,
        help="Highlight span end (step) for axvspan overlays.",
    )
    return parser.parse_args()


def _load_frame(csv_path: Path, rolling: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "step" not in df.columns:
        raise ValueError("CSV must contain 'step' column.")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df[df["step"].notna()].copy()
    if df.empty:
        raise ValueError("No valid rows after filtering 'step'.")
    metric_cols = set(CORE_METRICS + BETA_LOG_MEANS + BETA_HAT_MEANS + BETA_CLAMP_FRACS + STRENGTH_METRICS + HEALTH_METRICS)
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("step").reset_index(drop=True)
    window = max(1, rolling)
    if window > 1:
        for col in metric_cols:
            if col in df:
                df[col] = df[col].rolling(window=window, min_periods=1).mean()
    return df


def _draw_window(ax: plt.Axes, window_span: tuple[float, float] | None) -> None:
    if window_span is None:
        return
    start, end = window_span
    if start is None or end is None or start >= end:
        return
    ax.axvspan(start, end, alpha=0.1, color="gray")


def _derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    derived = df.copy()
    if "strength_mean" in derived.columns and "mask_frac" in derived.columns:
        derived["effective_strength"] = derived["strength_mean"] * derived["mask_frac"]
    if "log_q_t_mean" in derived.columns and "log_q_len_mean" in derived.columns:
        denom = derived["log_q_len_mean"].replace(0, np.nan)
        derived["log_qt_per_token"] = derived["log_q_t_mean"] / denom
    if "strength_mean" in derived.columns and "beta_hat_len_mean" in derived.columns:
        denom = derived["beta_hat_len_mean"].replace(0, np.nan)
        derived["strength_over_beta"] = derived["strength_mean"] / denom
    return derived


def _build_window_span(df: pd.DataFrame, args: argparse.Namespace) -> tuple[float, float] | None:
    step_min = float(df["step"].min())
    step_max = float(df["step"].max())
    start = args.window_start if args.window_start is not None else step_min
    end = args.window_end if args.window_end is not None else step_max
    if start >= end:
        return None
    return start, end


def _plot_core(df: pd.DataFrame, outdir: Path, window_span: tuple[float, float] | None) -> None:
    if not any(col in df for col in CORE_METRICS):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(True)
    ax.set_title("Core kernel magnitudes")
    ax.set_xlabel("step")
    if "log_q_t_mean" in df:
        ax.plot(df["step"], df["log_q_t_mean"], label="log_q_t_mean", color="tab:blue")
    if "log_qt_per_token" in df:
        ax.plot(df["step"], df["log_qt_per_token"], label="log_qt_per_token", color="tab:green", linestyle="--")
    ax2 = ax.twinx()
    if "q_t_mean" in df:
        series = df["q_t_mean"].clip(lower=1e-20)
        ax2.plot(df["step"], series, label="q_t_mean", color="tab:orange")
        ax2.set_yscale("log")
    ax.set_ylabel("log_q_t_mean")
    ax2.set_ylabel("q_t_mean (log scale)")
    handles = []
    labels = []
    for axis in (ax, ax2):
        h, l = axis.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        ax.legend(handles, labels)
    _draw_window(ax, window_span)
    fig.tight_layout()
    fig_path = outdir / "core_qt.png"
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_beta_logs(df: pd.DataFrame, outdir: Path, window_span: tuple[float, float] | None) -> None:
    if not any(col in df for col in BETA_LOG_MEANS):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(True)
    ax.set_title("Log beta diagnostics")
    ax.set_xlabel("step")
    plotted = False
    for col in BETA_LOG_MEANS:
        if col in df:
            ax.plot(df["step"], df[col], label=col)
            plotted = True
    if plotted:
        ax.legend()
        ax.set_ylabel("log beta mean")
        _draw_window(ax, window_span)
        fig.tight_layout()
        fig_path = outdir / "beta_logs.png"
        fig.savefig(fig_path)
    plt.close(fig)


def _plot_beta_hat(df: pd.DataFrame, outdir: Path, window_span: tuple[float, float] | None) -> None:
    if not any(col in df for col in BETA_HAT_MEANS):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(True)
    ax.set_title("Beta hat summaries")
    ax.set_xlabel("step")
    for col in BETA_HAT_MEANS:
        if col in df:
            series = df[col].clip(lower=1e-20)
            ax.plot(df["step"], series, label=col)
    ax.set_yscale("log")
    ax.set_ylabel("beta_hat (log scale)")
    ax.legend()
    _draw_window(ax, window_span)
    fig.tight_layout()
    fig_path = outdir / "beta_vals.png"
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_beta_clamp(df: pd.DataFrame, outdir: Path, window_span: tuple[float, float] | None) -> None:
    if not any(col in df for col in BETA_CLAMP_FRACS):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(True)
    ax.set_title("Beta clamp min fraction")
    ax.set_xlabel("step")
    plotted = False
    for col in BETA_CLAMP_FRACS:
        if col in df:
            ax.plot(df["step"], df[col], label=col)
            plotted = True
    if plotted:
        ax.legend()
        ax.set_ylabel("clamp_min_frac")
        _draw_window(ax, window_span)
        fig.tight_layout()
        fig_path = outdir / "beta_clamp.png"
        fig.savefig(fig_path)
    plt.close(fig)


def _plot_strength(df: pd.DataFrame, outdir: Path, window_span: tuple[float, float] | None) -> None:
    if "strength_mean" not in df and "g_t_mean" not in df and "effective_strength" not in df:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(True)
    ax.set_title("Guidance strength vs g(t)")
    ax.set_xlabel("step")
    if "strength_mean" in df:
        series = df["strength_mean"].clip(lower=1e-20)
        ax.plot(df["step"], series, label="strength_mean", color="tab:blue")
    if "effective_strength" in df:
        series = df["effective_strength"].clip(lower=1e-20)
        ax.plot(df["step"], series, label="effective_strength", color="tab:green", linestyle="--")
    ax.set_ylabel("strength (log scale)")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    if "g_t_mean" in df:
        ax2.plot(df["step"], df["g_t_mean"], label="g_t_mean", color="tab:orange")
        ax2.set_ylabel("g_t_mean")
    if "mask_frac" in df:
        ax2.plot(df["step"], df["mask_frac"], label="mask_frac", color="tab:purple", linestyle=":")
    handles = []
    labels = []
    for axis in (ax, ax2):
        h, l = axis.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        ax.legend(handles, labels)
    _draw_window(ax, window_span)
    fig.tight_layout()
    fig_path = outdir / "strength.png"
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_health(df: pd.DataFrame, outdir: Path, window_span: tuple[float, float] | None) -> None:
    available = [col for col in HEALTH_METRICS if col in df]
    if not available and "strength_over_beta" not in df:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
    ax.set_title("Health / quality metrics")
    ax.set_xlabel("step")
    for col in available:
        ax.plot(df["step"], df[col], label=col)
    if "strength_over_beta" in df:
        ax.plot(df["step"], df["strength_over_beta"], label="strength_over_beta")
    ax.legend(loc="best")
    ax.set_ylabel("value")
    _draw_window(ax, window_span)
    fig.tight_layout()
    fig_path = outdir / "health.png"
    fig.savefig(fig_path)
    plt.close(fig)


def _print_summary(df: pd.DataFrame) -> None:
    step_min = df["step"].min()
    step_max = df["step"].max()
    print(f"Rows={len(df)}, step range=[{step_min:.0f},{step_max:.0f}]")
    keys = ["log_q_t_mean", "q_t_mean", "beta_hat_len_mean", "beta_hat_raw_mean", "strength_mean", "rho_mean"]
    for key in keys:
        if key in df:
            col = df[key]
            print(f"{key}: min={col.min():.4f} max={col.max():.4f}")


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = _load_frame(csv_path, args.rolling)
    df = _derive_columns(df)
    window_span = _build_window_span(df, args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _plot_core(df, outdir, window_span)
    _plot_beta_logs(df, outdir, window_span)
    _plot_beta_hat(df, outdir, window_span)
    _plot_beta_clamp(df, outdir, window_span)
    _plot_strength(df, outdir, window_span)
    _plot_health(df, outdir, window_span)
    _print_summary(df)


if __name__ == "__main__":
    main()
