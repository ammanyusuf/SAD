#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    points = sorted(points, key=lambda p: p[0])
    frontier: List[Tuple[float, float]] = []
    best_y = -math.inf
    for x, y in points:
        if y >= best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def _pick_quality_metric(df: pd.DataFrame) -> Optional[str]:
    for key in ("mauve_exp_vs_ref", "bertscore_f1_mean"):
        if key in df.columns:
            return key
    if "perplexity" in df.columns:
        return "perplexity"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot safety/quality tradeoff curves.")
    parser.add_argument("--overall-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--x-metric", type=str, default=None)
    parser.add_argument("--y-metric", type=str, default=None)
    parser.add_argument("--invert-y", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.overall_csv)
    if df.empty:
        raise SystemExit("overall_rates.csv is empty.")

    output_dir = args.output_dir / "artifacts" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for (exp, variant), group in df.groupby(["experiment_slug", "prompt_variant"]):
        x_metric = args.x_metric
        if x_metric is None:
            x_metric = "unsafe_rate" if variant == "unsafe" else "unsafe_rate_delta_vs_baseline"
            if x_metric not in group.columns:
                x_metric = "unsafe_rate"
        y_metric = args.y_metric or _pick_quality_metric(group)
        if y_metric is None or x_metric not in group.columns:
            continue

        subset = group.dropna(subset=[x_metric, y_metric])
        if subset.empty:
            continue

        invert_y = args.invert_y or y_metric == "perplexity"
        y_vals = -subset[y_metric] if invert_y else subset[y_metric]
        points = list(zip(subset[x_metric], y_vals))
        frontier = _pareto_frontier(points)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(subset[x_metric], y_vals, c=subset["safety_scale"], cmap="viridis", alpha=0.8)
        if frontier:
            frontier = sorted(frontier, key=lambda p: p[0])
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, color="red", linewidth=2, label="Pareto frontier")
            ax.legend()

        ax.set_title(f"Tradeoff: {exp} | {variant}")
        ax.set_xlabel(x_metric + " (lower is better)")
        y_label = f"{y_metric} (higher is better)"
        if invert_y:
            y_label = f"{y_metric} (lower is better)"
        ax.set_ylabel(y_label)
        plt.colorbar(scatter, ax=ax, label="safety_scale")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        safe_exp = str(exp or "experiment").replace("/", "_")
        safe_variant = str(variant or "variant").replace("/", "_")
        out_path = output_dir / f"tradeoff_{safe_exp}_{safe_variant}.png"
        fig.savefig(out_path)
        plt.close(fig)

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
