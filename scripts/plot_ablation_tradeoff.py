#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRIC_ALIASES: Dict[str, str] = {
    "unsafe_rate": "unsafe_rate",
    "harmbench_asr": "harmbench_asr",
    "advbench_asr": "advbench_asr",
    "bertscore": "bertscore_f1_mean",
    "bertscore_f1": "bertscore_f1_mean",
    "mmd": "mmd2_rbf",
    "mmd2_rbf": "mmd2_rbf",
    "mauve": "mauve_exp_vs_ref",
}


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
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.title_fontsize": 11,
            "lines.linewidth": 1.5,
            "lines.markersize": 7,
            "axes.grid": True,
            "grid.color": "0.85",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _interactive_rename_labels(labels: List[str], prompt_prefix: str, enabled: bool) -> List[str]:
    if not enabled or not sys.stdin.isatty():
        return labels
    renamed: Dict[str, str] = {}
    for label in labels:
        if label in renamed:
            continue
        response = input(f"{prompt_prefix} rename '{label}' -> ").strip()
        if response:
            renamed[label] = response
    return [renamed.get(label, label) for label in labels]


def _parse_ts_label(label: str) -> Optional[Tuple[float, float]]:
    if not label.startswith("ts:"):
        return None
    try:
        _, window = label.split("ts:", 1)
        start_text, end_text = window.split("-", 1)
        return float(start_text), float(end_text)
    except ValueError:
        return None


def _sort_time_window_labels(labels: List[str]) -> List[str]:
    def sort_key(label: str) -> Tuple[int, float, float, str]:
        parsed = _parse_ts_label(label)
        if parsed is None:
            return (1, 0.0, 0.0, label)
        start, end = parsed
        return (0, -start, -end, label)

    return sorted(labels, key=sort_key)


def _resolve_metric(name: str, df: pd.DataFrame) -> Optional[str]:
    if name in df.columns:
        return name
    alias = METRIC_ALIASES.get(name)
    if alias and alias in df.columns:
        return alias
    return None


def _parse_float(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _make_window_label(t_start: object, t_end: object, critical_steps: object) -> str:
    if isinstance(critical_steps, str) and critical_steps:
        return f"cs:{critical_steps}"
    if isinstance(critical_steps, (list, tuple)) and critical_steps:
        return "cs:" + "_".join(str(x) for x in critical_steps)
    start = "full" if pd.isna(t_start) or t_start is None else str(t_start)
    end = "full" if pd.isna(t_end) or t_end is None else str(t_end)
    return f"ts:{start}-{end}"


def _apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.experiment:
        out = out[out["experiment_slug"].isin(args.experiment)]
    if args.prompt_variant:
        out = out[out["prompt_variant"].isin(args.prompt_variant)]
    if args.artifact:
        normalized: List[str] = []
        for raw in args.artifact:
            if raw is None:
                continue
            value = str(raw).strip()
            if not value:
                continue
            normalized.append(value)
            if value.endswith(".pt"):
                normalized.append(value[:-3])
        out = out[out["artifact_name"].isin(normalized)]
    if args.dataset and "dataset_name" in out.columns:
        out = out[out["dataset_name"].isin(args.dataset)]
    if args.eta is not None:
        out = out[_maybe_float_series(out["safety_scale"]) == float(args.eta)]
    if args.tensor_size:
        out = out[out["tensor_size"].isin(args.tensor_size)]
    if args.t_start is not None:
        out = out[_maybe_float_series(out["t_start"]) == float(args.t_start)]
    if args.t_end is not None:
        out = out[_maybe_float_series(out["t_end"]) == float(args.t_end)]
    if args.gating_label:
        out = out[out["gating_label"].isin(args.gating_label)]
    if not args.include_baseline:
        out = out[~out["safety_scale"].isna()]
    return out


def _default_utility_metric(df: pd.DataFrame) -> Optional[str]:
    for key in ("bertscore_f1_mean", "mauve_exp_vs_ref", "perplexity"):
        if key in df.columns:
            return key
    return None


def _is_lower_better(metric: str) -> bool:
    return metric in {"perplexity", "mmd2_rbf"}


def _pareto_frontier_mask(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_higher_better: bool,
    y_higher_better: bool,
) -> np.ndarray:
    order = np.argsort(x_vals)
    if x_higher_better:
        order = order[::-1]
    if y_higher_better:
        best_y = -math.inf
        better = lambda y, best: y >= best
    else:
        best_y = math.inf
        better = lambda y, best: y <= best
    mask = np.zeros(len(x_vals), dtype=bool)
    for idx in order:
        y = y_vals[idx]
        if better(y, best_y):
            mask[idx] = True
            best_y = y
    return mask


def _format_label(row: pd.Series, window_label_map: Optional[Dict[str, str]] = None) -> str:
    eta = _parse_float(row.get("safety_scale"))
    if eta is None:
        eta_label = "η=baseline"
    else:
        eta_label = f"η={eta:g}"
    window = _make_window_label(row.get("t_start"), row.get("t_end"), row.get("critical_steps"))
    if window_label_map and window in window_label_map:
        window = window_label_map[window]
    return f"{eta_label} | {window}"


def _group_keys(df: pd.DataFrame) -> List[str]:
    keys = ["experiment_slug", "prompt_variant"]
    if "dataset_name" in df.columns:
        keys.append("dataset_name")
    return keys


def _channel_series(df: pd.DataFrame, channel: str) -> Tuple[Optional[pd.Series], str]:
    if channel == "none":
        return None, "none"
    if channel == "time_window":
        series = df.apply(
            lambda row: _make_window_label(row.get("t_start"), row.get("t_end"), row.get("critical_steps")),
            axis=1,
        )
        return series, "categorical"
    if channel == "tensor_size":
        series = df.get("tensor_size")
        if series is None:
            return None, "none"
        return series.astype(str), "categorical"
    if channel == "safety_scale":
        series = pd.to_numeric(df.get("safety_scale"), errors="coerce")
        return series, "continuous"
    return None, "none"


def _map_markers(values: pd.Series) -> Dict[str, str]:
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "8"]
    unique_vals = [str(v) for v in values.dropna().unique().tolist()]
    if not unique_vals:
        return {}
    mapping: Dict[str, str] = {}
    for idx, val in enumerate(sorted(unique_vals, key=lambda v: str(v))):
        mapping[val] = markers[idx % len(markers)]
    return mapping


def _map_markers_ordered(values: Sequence[str]) -> Dict[str, str]:
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "8"]
    mapping: Dict[str, str] = {}
    for idx, val in enumerate(values):
        mapping[val] = markers[idx % len(markers)]
    return mapping


def _map_sizes(values: pd.Series) -> Dict[str, float]:
    sizes = [40, 55, 70, 85, 100, 115, 130, 145]
    unique_vals = [str(v) for v in values.dropna().unique().tolist()]
    if not unique_vals:
        return {}
    mapping: Dict[str, float] = {}
    for idx, val in enumerate(sorted(unique_vals, key=lambda v: str(v))):
        mapping[val] = sizes[idx % len(sizes)]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot safety-utility tradeoff curves for ablations.")
    parser.add_argument("--overall-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--x-metric", type=str, default=None)
    parser.add_argument("--y-metric", type=str, default="unsafe_rate")
    parser.add_argument("--x-lower-better", action="store_true")
    parser.add_argument("--x-higher-better", action="store_true")
    parser.add_argument("--max-labels", type=int, default=40)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--log-safety-scale", action="store_true")
    parser.add_argument("--label-pareto", action="store_true")
    parser.add_argument("--show-pareto", action="store_true")
    parser.add_argument("--delta-plot", action="store_true")
    parser.add_argument("--hexbin", action="store_true")
    parser.add_argument("--label-top-k", type=int, default=4)
    parser.add_argument("--point-alpha", type=float, default=0.6)
    parser.add_argument("--point-size", type=float, default=90)
    parser.add_argument("--baseline-lines", action="store_true")
    parser.add_argument(
        "--color-by",
        type=str,
        default="safety_scale",
        choices=["safety_scale", "tensor_size", "time_window", "none"],
    )
    parser.add_argument(
        "--marker-by",
        type=str,
        default="none",
        choices=["safety_scale", "tensor_size", "time_window", "none"],
    )
    parser.add_argument(
        "--size-by",
        type=str,
        default="none",
        choices=["safety_scale", "tensor_size", "time_window", "none"],
    )
    parser.add_argument("--x-label", type=str, default=None)
    parser.add_argument("--y-label-left", type=str, default=None)
    parser.add_argument("--y-label-right", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--legend-name", type=str, default=None)
    parser.add_argument("--interactive-legend", action="store_true")

    parser.add_argument("--experiment", action="append")
    parser.add_argument("--prompt-variant", action="append")
    parser.add_argument("--artifact", action="append")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--eta", type=float)
    parser.add_argument("--tensor-size", action="append")
    parser.add_argument("--t-start", type=float)
    parser.add_argument("--t-end", type=float)
    parser.add_argument("--gating-label", action="append")

    args = parser.parse_args()
    _apply_paper_style()

    df = pd.read_csv(args.overall_csv)
    if df.empty:
        raise SystemExit("overall_rates.csv is empty.")

    filtered = _apply_filters(df, args)
    if filtered.empty:
        raise SystemExit("No rows left after filtering.")

    x_metric_raw = args.x_metric or _default_utility_metric(filtered)
    if x_metric_raw is None:
        raise SystemExit("No utility metric found; pass --x-metric.")

    x_metric = _resolve_metric(x_metric_raw, filtered)
    y_metric = _resolve_metric(args.y_metric, filtered)
    if x_metric is None:
        raise SystemExit(f"Unknown x metric '{x_metric_raw}'.")
    if y_metric is None:
        raise SystemExit(f"Unknown y metric '{args.y_metric}'.")

    output_dir = args.output_dir / "ablations" / "tradeoff"
    output_dir.mkdir(parents=True, exist_ok=True)

    group_keys = _group_keys(filtered)
    for key_values, group in filtered.groupby(group_keys):
        key_tuple = key_values if isinstance(key_values, tuple) else (key_values,)
        subset = group.dropna(subset=[x_metric, y_metric])
        if subset.empty:
            continue

        x_vals = subset[x_metric].to_numpy(dtype=float)
        y_vals = subset[y_metric].to_numpy(dtype=float)

        x_lower_better = args.x_lower_better or (
            not args.x_higher_better and _is_lower_better(x_metric)
        )
        x_higher_better = not x_lower_better
        y_higher_better = False

        baseline = subset[pd.to_numeric(subset.get("safety_scale"), errors="coerce").isna()]
        baseline_x = None
        baseline_y = None
        if not baseline.empty:
            baseline_x = pd.to_numeric(baseline[x_metric], errors="coerce").mean()
            baseline_y = pd.to_numeric(baseline[y_metric], errors="coerce").mean()

        if args.delta_plot:
            if baseline_x is None or baseline_y is None or np.isnan(baseline_x) or np.isnan(baseline_y):
                raise SystemExit("--delta-plot requires baseline rows; add --include-baseline.")
            x_vals = x_vals - baseline_x
            y_vals = y_vals - baseline_y
            x_lower_better = True
            x_higher_better = False
            y_higher_better = False

        pareto_mask = None
        pareto_subset = None
        if args.show_pareto or args.label_pareto:
            pareto_mask = _pareto_frontier_mask(x_vals, y_vals, x_higher_better, y_higher_better)
            pareto_subset = subset[pareto_mask]

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        if args.hexbin:
            ax.hexbin(
                x_vals,
                y_vals,
                gridsize=30,
                cmap="Greys",
                mincnt=1,
                alpha=0.25,
                linewidths=0.2,
            )
        if args.baseline_lines:
            if args.delta_plot:
                ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
                ax.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0)
            elif baseline_x is not None and baseline_y is not None:
                ax.axhline(baseline_y, color="0.35", linestyle="--", linewidth=1.0)
                ax.axvline(baseline_x, color="0.35", linestyle="--", linewidth=1.0)
        color_series, color_kind = _channel_series(subset, args.color_by)
        marker_series, marker_kind = _channel_series(subset, args.marker_by)
        size_series, size_kind = _channel_series(subset, args.size_by)

        if color_series is not None and color_kind == "continuous":
            color_values = pd.to_numeric(color_series, errors="coerce")
        if args.color_by == "safety_scale" and args.log_safety_scale:
            color_values = color_values.where(color_values > 0)
            color_values = np.log10(color_values)
            point_colors = None
        else:
            color_values = None
            if color_series is not None and color_kind == "categorical":
                color_vals = color_series.astype(str)
                raw_categories = [str(v) for v in color_vals.dropna().unique().tolist()]
                if args.color_by == "time_window":
                    categories = _sort_time_window_labels(raw_categories)
                else:
                    categories = sorted(raw_categories, key=lambda v: str(v))
                palette = np.array(COLORBLIND_SAFE)
                color_map = {cat: palette[idx % len(palette)] for idx, cat in enumerate(categories)}
                point_colors = [color_map.get(str(v), "#0072B2") for v in color_vals]
            else:
                point_colors = "#0072B2"

        marker_map = {}
        marker_values = marker_series
        if marker_series is not None and marker_kind == "continuous":
            numeric_markers = pd.to_numeric(marker_series, errors="coerce")
            if numeric_markers.notna().any():
                try:
                    marker_values = pd.qcut(numeric_markers, q=4, duplicates="drop")
                except ValueError:
                    marker_values = numeric_markers
            marker_kind = "categorical"
        if marker_values is not None and marker_kind == "categorical":
            marker_series_str = marker_values.astype(str)
            if args.marker_by == "time_window":
                ordered_vals = _sort_time_window_labels(
                    [str(v) for v in marker_series_str.dropna().unique().tolist()]
                )
                marker_map = _map_markers_ordered(ordered_vals)
            else:
                marker_map = _map_markers(marker_series_str)

        size_map = {}
        sizes = None
        if size_series is not None:
            if size_kind == "categorical":
                size_map = _map_sizes(size_series.astype(str))
                sizes = [size_map.get(str(v), 60) for v in size_series.astype(str)]
            elif size_kind == "continuous":
                numeric_sizes = pd.to_numeric(size_series, errors="coerce")
                if args.size_by == "safety_scale" and args.log_safety_scale:
                    numeric_sizes = numeric_sizes.where(numeric_sizes > 0)
                    numeric_sizes = np.log10(numeric_sizes)
                if numeric_sizes.notna().any():
                    min_v = numeric_sizes.min()
                    max_v = numeric_sizes.max()
                    if max_v > min_v:
                        sizes = 40 + 100 * (numeric_sizes - min_v) / (max_v - min_v)
                    else:
                        sizes = np.full(len(numeric_sizes), 60)

        markers = None
        if marker_map:
            markers = [marker_map.get(str(v), "o") for v in marker_values.astype(str)]

        scatter_for_colorbar = None
        if markers is None:
            if color_values is not None:
                scatter_for_colorbar = ax.scatter(
                    x_vals,
                    y_vals,
                    c=color_values,
                    cmap="YlGnBu",
                    s=sizes if sizes is not None else args.point_size,
                    alpha=args.point_alpha,
                    edgecolor="none",
                )
            else:
                ax.scatter(
                    x_vals,
                    y_vals,
                    c=point_colors,
                    s=sizes if sizes is not None else args.point_size,
                    alpha=args.point_alpha,
                    edgecolor="none",
                )
        else:
            for marker in sorted(set(markers)):
                idxs = [i for i, m in enumerate(markers) if m == marker]
                if color_values is not None:
                    scatter_for_colorbar = ax.scatter(
                        x_vals[idxs],
                        y_vals[idxs],
                        c=np.array(color_values)[idxs],
                        cmap="YlGnBu",
                        s=np.array(sizes)[idxs] if sizes is not None else args.point_size,
                        marker=marker,
                        alpha=args.point_alpha,
                        edgecolor="none",
                        label=None,
                    )
                else:
                    ax.scatter(
                        x_vals[idxs],
                        y_vals[idxs],
                        c=np.array(point_colors)[idxs],
                        s=np.array(sizes)[idxs] if sizes is not None else args.point_size,
                        marker=marker,
                        alpha=args.point_alpha,
                        edgecolor="none",
                        label=None,
                    )

        if color_values is not None and scatter_for_colorbar is not None:
            color_label = args.color_by
            if args.color_by == "safety_scale" and args.log_safety_scale:
                color_label = "log10(safety_scale)"
            plt.colorbar(scatter_for_colorbar, ax=ax, label=color_label)

        window_label_map: Dict[str, str] = {}

        if color_series is not None and color_kind == "categorical":
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=str(cat),
                    markerfacecolor=color_map[cat],
                    markersize=10,
                )
                for cat in categories
            ]
            color_labels = [h.get_label() for h in legend_handles]
            color_labels = _interactive_rename_labels(
                color_labels, "Color legend label", args.interactive_legend
            )
            for handle, label in zip(legend_handles, color_labels):
                handle.set_label(label)
            if args.color_by == "time_window":
                window_label_map.update(dict(zip(categories, color_labels)))
            legend_title = args.legend_name or args.color_by
            color_legend = ax.legend(handles=legend_handles, title=legend_title, loc="upper left")
            ax.add_artist(color_legend)

        if marker_map:
            from matplotlib.lines import Line2D

            marker_handles = [
                Line2D([0], [0], marker=mk, color="w", label=val, markerfacecolor="gray", markersize=10)
                for val, mk in marker_map.items()
            ]
            marker_labels = [h.get_label() for h in marker_handles]
            marker_labels = _interactive_rename_labels(
                marker_labels, "Marker legend label", args.interactive_legend
            )
            for handle, label in zip(marker_handles, marker_labels):
                handle.set_label(label)
            if args.marker_by == "time_window":
                marker_order = list(marker_map.keys())
                window_label_map.update(dict(zip(marker_order, marker_labels)))
            legend_title = args.legend_name or args.marker_by
            marker_legend = ax.legend(handles=marker_handles, title=legend_title, loc="upper right")
            ax.add_artist(marker_legend)

        if args.show_pareto and pareto_subset is not None and not pareto_subset.empty:
            ax.scatter(
                x_vals[pareto_mask],
                y_vals[pareto_mask],
                color="#0072B2",
                edgecolor="white",
                s=70,
                zorder=3,
            )

        if args.label_pareto:
            if pareto_subset is None or pareto_subset.empty:
                raise SystemExit("--label-pareto requires pareto points; add --show-pareto.")
            labels_added = 0
            for _, row in pareto_subset.iterrows():
                if labels_added >= args.max_labels:
                    break
                label = _format_label(row, window_label_map)
                ax.annotate(
                    label,
                    (x_vals[subset.index.get_loc(row.name)], y_vals[subset.index.get_loc(row.name)]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=11,
                )
                labels_added += 1

        if args.label_top_k > 0:
            label_candidates = subset.copy()
            label_candidates["_x_val"] = x_vals
            label_candidates["_y_val"] = y_vals
            best_x = label_candidates.nsmallest(1, "_x_val")
            best_y = label_candidates.nsmallest(1, "_y_val")
            x_norm = (label_candidates["_x_val"] - label_candidates["_x_val"].min())
            y_norm = (label_candidates["_y_val"] - label_candidates["_y_val"].min())
            denom_x = label_candidates["_x_val"].max() - label_candidates["_x_val"].min()
            denom_y = label_candidates["_y_val"].max() - label_candidates["_y_val"].min()
            if denom_x > 0:
                x_norm = x_norm / denom_x
            if denom_y > 0:
                y_norm = y_norm / denom_y
            label_candidates["_balance"] = x_norm + y_norm
            best_balance = label_candidates.nsmallest(1, "_balance")
            highlights = pd.concat([best_x, best_y, best_balance]).drop_duplicates()
            for _, row in highlights.head(args.label_top_k).iterrows():
                ax.annotate(
                    _format_label(row, window_label_map),
                    (row["_x_val"], row["_y_val"]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=11,
                )

        title_parts = [f"{k}={v}" for k, v in zip(group_keys, key_tuple)]
        if args.title:
            ax.set_title(args.title)
        else:
            ax.set_title(" | ".join(title_parts))
        x_suffix = "lower is better" if x_lower_better else "higher is better"
        x_label_metric = x_metric
        y_label_metric = y_metric
        if args.delta_plot:
            x_label_metric = f"Δ {x_metric}"
            y_label_metric = f"Δ {y_metric}"
        ax.set_xlabel(args.x_label or f"{x_label_metric} ({x_suffix})")
        ax.set_ylabel(args.y_label_left or f"{y_label_metric} (lower is better)")
        if not baseline.empty and not args.delta_plot:
            baseline_artist = ax.scatter(
                baseline[x_metric],
                baseline[y_metric],
                marker="x",
                s=130,
                color="#0072B2",
                linewidths=1.5,
                zorder=4,
                label="baseline",
            )
            baseline_labels = _interactive_rename_labels(
                [baseline_artist.get_label()], "Baseline legend label", args.interactive_legend
            )
            baseline_artist.set_label(baseline_labels[0])
            baseline_legend = ax.legend(handles=[baseline_artist], loc="lower right")
            ax.add_artist(baseline_legend)
        fig.tight_layout()

        safe_name = "_".join([str(v) for v in key_tuple]).replace("/", "_")
        out_path = output_dir / f"tradeoff_{safe_name}.png"
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
