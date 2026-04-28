#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRIC_ALIASES: Dict[str, str] = {
    "unsafe_rate_delta": "unsafe_rate_delta_vs_baseline",
    "unsafe_rate_delta_vs_baseline": "unsafe_rate_delta_vs_baseline",
    "safe_to_unsafe": "safe_to_unsafe",
    "unsafe_to_safe": "unsafe_to_safe",
    "unsafe_to_unsafe": "unsafe_to_unsafe",
    "safe_to_safe": "safe_to_safe",
    "perplexity": "perplexity",
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

_TIME_WINDOW_TOTAL_STEPS = 256.0


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


def _baseline_mask(df: pd.DataFrame) -> pd.Series:
    safety = _maybe_float_series(df["safety_scale"])
    mask = safety.isna() | (safety < 0)
    if "missing_baseline" in df.columns:
        mask = mask | df["missing_baseline"].astype(bool)
    return mask


def _extract_ref_size(text: str) -> Optional[int]:
    if not text:
        return None
    for pattern in (r"-(\d{3,5})(?=[-_])", r"_(\d{3,5})(?=[-_])"):
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def _speed_rows_for_group(speed_df: pd.DataFrame, group_info: Dict[str, object]) -> pd.DataFrame:
    subset = speed_df.copy()
    exp = str(group_info.get("experiment_slug") or "")
    if exp:
        subset = subset[subset["experiment_slug"].astype(str) == exp]
    variant = str(group_info.get("prompt_variant") or "")
    if variant:
        subset = subset[subset["run_id"].astype(str).str.contains(variant, na=False)]
    artifact = str(group_info.get("artifact_name") or "")
    if artifact:
        subset = subset[subset["run_id"].astype(str).str.contains(artifact, na=False)]
    return subset


def _sort_numeric(values: Iterable[object]) -> List[object]:
    numeric_vals = []
    non_numeric = []
    for val in values:
        num = _parse_float(val)
        if num is None:
            non_numeric.append(val)
        else:
            numeric_vals.append((num, val))
    numeric_vals.sort(key=lambda item: item[0])
    non_numeric_sorted = sorted(non_numeric, key=lambda v: str(v))
    return [val for _, val in numeric_vals] + non_numeric_sorted


def _make_window_label(t_start: object, t_end: object, critical_steps: object) -> str:
    if isinstance(critical_steps, str) and critical_steps:
        return f"cs:{critical_steps}"
    if isinstance(critical_steps, (list, tuple)) and critical_steps:
        return "cs:" + "_".join(str(x) for x in critical_steps)
    start = "full" if pd.isna(t_start) or t_start is None else str(t_start)
    end = "full" if pd.isna(t_end) or t_end is None else str(t_end)
    if start != "full" and end != "full":
        try:
            start_val = float(start)
            end_val = float(end)
            inferred_start = float(_TIME_WINDOW_TOTAL_STEPS) - start_val
            start = f"{inferred_start:g}"
            end = f"{end_val:g}"
        except ValueError:
            pass
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
    if args.dataset:
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


def _print_available_values(df: pd.DataFrame) -> None:
    def _unique(col: str, limit: int = 12) -> List[str]:
        if col not in df.columns:
            return []
        values = [v for v in df[col].dropna().unique().tolist() if str(v).strip()]
        values = sorted(values, key=lambda v: str(v))
        return [str(v) for v in values[:limit]]

    print("Available filters (sample):")
    print("  artifact_name:", _unique("artifact_name"))
    print("  t_start:", _unique("t_start"))
    print("  t_end:", _unique("t_end"))
    print("  safety_scale:", _unique("safety_scale"))
    print("  prompt_variant:", _unique("prompt_variant"))


def _infer_secondary_metrics(args: argparse.Namespace) -> List[str]:
    metrics: List[str] = []
    if args.secondary_metric:
        metrics.extend(args.secondary_metric)
    if args.add_bertscore:
        metrics.append("bertscore")
    if args.add_perplexity:
        metrics.append("perplexity")
    if args.add_mmd:
        metrics.append("mmd")
    if args.add_mauve:
        metrics.append("mauve")
    return metrics


def _collect_metrics(df: pd.DataFrame, primary: str, secondary: Sequence[str]) -> Tuple[str, List[str]]:
    resolved_primary = _resolve_metric(primary, df)
    if resolved_primary is None:
        raise SystemExit(f"Unknown primary metric '{primary}'. Available columns include: {sorted(df.columns)}")
    resolved_secondary: List[str] = []
    for metric in secondary:
        resolved = _resolve_metric(metric, df)
        if resolved is None:
            raise SystemExit(f"Unknown secondary metric '{metric}'.")
        if resolved == resolved_primary:
            continue
        if resolved not in resolved_secondary:
            resolved_secondary.append(resolved)
    return resolved_primary, resolved_secondary


def _group_keys(ablation: str) -> List[str]:
    if ablation == "eta":
        return ["experiment_slug", "prompt_variant", "artifact_name"]
    if ablation == "ref_size":
        return [
            "experiment_slug",
            "prompt_variant",
            "artifact_name",
            "safety_scale",
            "t_start",
            "t_end",
            "critical_steps",
        ]
    if ablation == "time_window":
        return ["experiment_slug", "prompt_variant", "artifact_name", "safety_scale"]
    raise SystemExit(f"Unknown ablation '{ablation}'.")


def _assign_x(group: pd.DataFrame, ablation: str) -> Tuple[pd.DataFrame, List[str]]:
    data = group.copy()
    if ablation == "eta":
        data["_x_value"] = _maybe_float_series(data["safety_scale"])
        data["_x_label"] = data["_x_value"].apply(lambda v: "" if pd.isna(v) else str(v))
        ordered = [str(v) for v in sorted(data["_x_value"].dropna().unique())]
        return data, ordered
    if ablation == "ref_size":
        data["_x_value"] = _maybe_float_series(data["tensor_size"])
        data["_x_label"] = data["tensor_size"].fillna(data["artifact_name"]).astype(str)
        ordered_vals = sorted(data["_x_value"].dropna().unique())
        if ordered_vals:
            return data, [str(v) for v in ordered_vals]
        ordered = _sort_numeric(data["_x_label"].unique().tolist())
        return data, [str(v) for v in ordered]
    if ablation == "time_window":
        data["_x_label"] = data.apply(
            lambda row: _make_window_label(row.get("t_start"), row.get("t_end"), row.get("critical_steps")),
            axis=1,
        )
        ordered = list(dict.fromkeys(data["_x_label"].tolist()))
        return data, ordered
    raise SystemExit(f"Unknown ablation '{ablation}'.")


def _aggregate(group: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    agg = group.groupby("_x_label")[list(metrics)].agg(["mean", "std"]).reset_index()
    # flatten columns
    flat_cols = ["x_label"]
    for metric in metrics:
        flat_cols.extend([f"{metric}_mean", f"{metric}_std"])
    agg.columns = flat_cols
    return agg


def _plot_group(
    group: pd.DataFrame,
    ablation: str,
    primary_metric: str,
    secondary_metrics: Sequence[str],
    output_path: Path,
    title_parts: List[str],
    x_label: str,
    show_error: bool,
    log_ref_size: bool,
    annotate_knee: bool,
    x_label_override: Optional[str],
    y_label_left: Optional[str],
    y_label_right: Optional[str],
    title_override: Optional[str],
    interactive_legend: bool,
) -> None:
    data, ordered_labels = _assign_x(group, ablation)
    metrics = [primary_metric] + list(secondary_metrics)

    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))

    if ablation == "eta":
        data = data.copy()
        data["_line_label"] = data.apply(
            lambda row: _make_window_label(row.get("t_start"), row.get("t_end"), row.get("critical_steps")),
            axis=1,
        )
        baseline = data[_baseline_mask(data)]
        baseline_y = None
        if not baseline.empty:
            baseline_y = pd.to_numeric(baseline[primary_metric], errors="coerce").mean()
        elif primary_metric == "unsafe_rate" and "baseline_unsafe_rate" in data.columns:
            baseline_y = pd.to_numeric(data["baseline_unsafe_rate"], errors="coerce").mean()
        agg = data.groupby(["_x_value", "_line_label"])[metrics].agg(["mean", "std"]).reset_index()
        agg.columns = [
            "_x_value",
            "_line_label",
        ] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
        line_labels = [str(v) for v in agg["_line_label"].unique().tolist()]
        line_color_map = {
            label: COLORBLIND_SAFE[idx % len(COLORBLIND_SAFE)] for idx, label in enumerate(line_labels)
        }
        for line_label, line_data in agg.groupby("_line_label"):
            line_data = line_data.sort_values("_x_value")
            ax1.plot(
                line_data["_x_value"],
                line_data[f"{primary_metric}_mean"],
                marker="o",
                linestyle="-",
                color=line_color_map.get(str(line_label), COLORBLIND_SAFE[0]),
                label=str(line_label),
            )
        if baseline_y is not None and not np.isnan(baseline_y):
            ax1.axhline(baseline_y, color=COLORBLIND_SAFE[0], linestyle="--", linewidth=1.5, label="baseline")
        ax1.set_xscale("log")
        ax1.set_xlabel(x_label_override or x_label)
        ax1.set_ylabel(y_label_left or primary_metric)
        if title_override:
            ax1.set_title(title_override)
        else:
            ax1.set_title(" | ".join(title_parts))
        handles, labels = ax1.get_legend_handles_labels()
        if labels:
            labels = _interactive_rename_labels(labels, "Legend label", interactive_legend)
            ax1.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1))
    elif ablation == "ref_size":
        agg = data.groupby("_x_value")[metrics].agg(["mean", "std"]).reset_index()
        agg.columns = ["_x_value"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
        agg = agg.sort_values("_x_value")
        ax1.plot(
            agg["_x_value"],
            agg[f"{primary_metric}_mean"],
            marker="o",
            linestyle="-",
            color=COLORBLIND_SAFE[0],
            label=primary_metric,
        )
        if log_ref_size:
            ax1.set_xscale("log")
        ax1.set_xlabel(x_label_override or x_label)
        ax1.set_ylabel(y_label_left or primary_metric)
        if title_override:
            ax1.set_title(title_override)
        else:
            ax1.set_title(" | ".join(title_parts))
        if annotate_knee and len(agg) >= 3:
            x_vals = np.log10(agg["_x_value"].to_numpy()) if log_ref_size else agg["_x_value"].to_numpy()
            y_vals = agg[f"{primary_metric}_mean"].to_numpy()
            slopes = np.diff(y_vals) / np.diff(x_vals)
            if len(slopes) >= 2:
                curvature = np.abs(np.diff(slopes))
                idx = int(np.argmax(curvature)) + 1
                ax1.annotate(
                    "knee",
                    (agg["_x_value"].iloc[idx], y_vals[idx]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=11,
                    color="red",
                )
        handles, labels = ax1.get_legend_handles_labels()
        if labels:
            labels = _interactive_rename_labels(labels, "Legend label", interactive_legend)
            ax1.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1))
    else:
        agg = _aggregate(data, metrics)
        agg["_order"] = agg["x_label"].apply(
            lambda v: ordered_labels.index(v) if v in ordered_labels else len(ordered_labels)
        )
        agg = agg.sort_values("_order").drop(columns=["_order"])
        x_positions = np.arange(len(agg["x_label"]))

        primary_mean = agg[f"{primary_metric}_mean"].to_numpy()
        primary_std = agg[f"{primary_metric}_std"].to_numpy() if show_error else None

        ax1.errorbar(
            x_positions,
            primary_mean,
            yerr=primary_std,
            marker="o",
            linestyle="-",
            color=COLORBLIND_SAFE[0],
            label=primary_metric,
        )
        ax1.set_ylabel(y_label_left or primary_metric)
        ax1.set_xlabel(x_label_override or x_label)

        handles = ax1.get_legend_handles_labels()[0]
        labels = ax1.get_legend_handles_labels()[1]

        if secondary_metrics:
            ax2 = ax1.twinx()
            for idx, metric in enumerate(secondary_metrics):
                mean_vals = agg[f"{metric}_mean"].to_numpy()
                std_vals = agg[f"{metric}_std"].to_numpy() if show_error else None
                ax2.errorbar(
                    x_positions,
                    mean_vals,
                    yerr=std_vals,
                    marker="s",
                    linestyle="--",
                    color=COLORBLIND_SAFE[(idx + 1) % len(COLORBLIND_SAFE)],
                    label=metric,
                )
            ax2.set_ylabel(y_label_right or "secondary metrics")
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels += l2

        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(agg["x_label"], rotation=0)
        if title_override:
            ax1.set_title(title_override)
        else:
            ax1.set_title(" | ".join(title_parts))

        if handles:
            labels = _interactive_rename_labels(labels, "Legend label", interactive_legend)
            ax1.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ablation curves from overall_rates.csv")
    parser.add_argument("--overall-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["eta", "ref_size", "time_window"],
        required=True,
        help="Ablation to plot: eta, ref_size (unsafe reference size), or time_window.",
    )
    parser.add_argument("--primary-metric", type=str, default="unsafe_rate")
    parser.add_argument("--secondary-metric", type=str, action="append")
    parser.add_argument("--add-bertscore", action="store_true")
    parser.add_argument("--add-perplexity", action="store_true")
    parser.add_argument("--add-mmd", action="store_true")
    parser.add_argument("--add-mauve", action="store_true")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--error-bars", action="store_true")
    parser.add_argument("--log-ref-size", action="store_true")
    parser.add_argument("--annotate-knee", action="store_true")
    parser.add_argument("--speed-csv", type=Path)
    parser.add_argument("--time-window-steps", type=float, default=256.0)
    parser.add_argument("--x-label", type=str, default=None)
    parser.add_argument("--y-label-left", type=str, default=None)
    parser.add_argument("--y-label-right", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
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
    global _TIME_WINDOW_TOTAL_STEPS
    _TIME_WINDOW_TOTAL_STEPS = float(args.time_window_steps)
    _apply_paper_style()

    df = pd.read_csv(args.overall_csv)
    if df.empty:
        raise SystemExit("overall_rates.csv is empty.")

    filtered = _apply_filters(df, args)
    if filtered.empty:
        _print_available_values(df)
        raise SystemExit("No rows left after filtering. Check filters or include baseline.")

    secondary_metrics_raw = _infer_secondary_metrics(args)
    primary_metric, secondary_metrics = _collect_metrics(filtered, args.primary_metric, secondary_metrics_raw)

    group_keys = _group_keys(args.ablation)

    group_df = filtered.copy()
    group_key_cols: List[str] = []
    for key in group_keys:
        group_key = f"_group_{key}"
        if key not in group_df.columns:
            group_df[group_key] = "none"
        else:
            group_df[group_key] = group_df[key].apply(
                lambda v: "none" if pd.isna(v) or str(v).strip() == "" else v
            )
        group_key_cols.append(group_key)

    speed_df = None
    if args.speed_csv and args.speed_csv.exists():
        speed_df = pd.read_csv(args.speed_csv, sep="\t")
        if "run_id" not in speed_df.columns:
            speed_df = pd.read_csv(args.speed_csv)

    for key_values, group in group_df.groupby(group_key_cols):
        if isinstance(key_values, tuple):
            key_tuple = key_values
        else:
            key_tuple = (key_values,)
        title_parts = [f"{k}={v}" for k, v in zip(group_keys, key_tuple)]
        group_orig = filtered.loc[group.index]

        if args.ablation == "eta":
            x_label = "eta (safety_scale)"
        elif args.ablation == "ref_size":
            x_label = "unsafe reference size (tensor_size)"
        else:
            x_label = "time window"

        safe_name = "_".join([str(v) for v in key_tuple]).replace("/", "_")
        output_path = (
            args.output_dir
            / "ablations"
            / args.ablation
            / f"{primary_metric}__{safe_name}.png"
        )

        _plot_group(
            group_orig,
            args.ablation,
            primary_metric,
            secondary_metrics,
            output_path,
            title_parts,
            x_label,
            args.error_bars,
            args.log_ref_size,
            args.annotate_knee,
            args.x_label,
            args.y_label_left,
            args.y_label_right,
            args.title,
            args.interactive_legend,
        )
        print(f"Wrote {output_path}")

        if args.ablation == "ref_size" and speed_df is not None:
            group_info = dict(zip(group_keys, key_tuple))
            speed_subset = _speed_rows_for_group(speed_df, group_info)
            if not speed_subset.empty:
                speed_subset = speed_subset.copy()
                speed_subset["ref_size"] = speed_subset["run_id"].astype(str).apply(_extract_ref_size)
                speed_subset = speed_subset.dropna(subset=["ref_size"])
                if not speed_subset.empty and "seq_per_sec_median" in speed_subset.columns:
                    speed_agg = speed_subset.groupby("ref_size")["seq_per_sec_median"].mean().reset_index()
                    speed_agg = speed_agg.sort_values("ref_size")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(
                        speed_agg["ref_size"],
                        speed_agg["seq_per_sec_median"],
                        marker="o",
                        linestyle="-",
                        color=COLORBLIND_SAFE[2],
                    )
                    if args.log_ref_size:
                        ax.set_xscale("log")
                    ax.set_xlabel("unsafe reference size (tensor_size)")
                    ax.set_ylabel("seq/s (median)")
                    ax.grid(True, linestyle="--", alpha=0.4)
                    ax.set_title(" | ".join(title_parts) + " | speed")
                    fig.tight_layout()
                    speed_out = (
                        args.output_dir
                        / "ablations"
                        / "ref_size"
                        / f"runtime__{safe_name}.png"
                    )
                    fig.savefig(speed_out, bbox_inches="tight", facecolor="white")
                    plt.close(fig)
                    print(f"Wrote {speed_out}")


if __name__ == "__main__":
    main()
