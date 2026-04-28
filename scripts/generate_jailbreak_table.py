#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DEFENSE_ORDER = [
    ("none", "None"),
    ("ppl", "PPL"),
    ("self-reminder", "Self-rem."),
    ("diffuguard", "DiffuGuard"),
]

ATTACK_ORDER = [
    ("zeroshot", "Zero-shot"),
    ("DIJA", "DIJA"),
    ("PAD", "PAD"),
]


@dataclass
class MetricCell:
    base: Optional[float]
    sden: Optional[float]


def _read_overall_rates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    if df.empty:
        raise SystemExit(f"{path} is empty.")
    return df


def _normalize_defense(val: Optional[str]) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "none"
    sval = str(val).strip().lower()
    if sval in ("", "none", "null"):
        return "none"
    if "ppl" in sval:
        return "ppl"
    if "self" in sval:
        return "self-reminder"
    if "diffu" in sval:
        return "diffuguard"
    return sval


def _select_best(
    df: pd.DataFrame,
    metric_col: str,
    *,
    is_sden: bool,
) -> Optional[float]:
    if metric_col not in df.columns or df.empty:
        return None
    safety_scale = pd.to_numeric(df.get("safety_scale"), errors="coerce")
    if is_sden:
        subset = df[safety_scale.notna() & (safety_scale != 0)]
    else:
        subset = df[safety_scale.isna() | (safety_scale == 0)]
    if subset.empty:
        return None
    metric_vals = pd.to_numeric(subset[metric_col], errors="coerce")
    if metric_vals.notna().sum() == 0:
        return None
    return float(metric_vals.min())


def _to_pct(value: Optional[float]) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value) * 100.0


def _format_cell(base: Optional[float], sden: Optional[float]) -> Tuple[str, str]:
    if base is None:
        base_str = "--"
    else:
        base_str = f"{base:.1f}"
    if sden is None:
        return base_str, "--"
    sden_str = f"{sden:.1f}"
    if base is not None:
        delta = sden - base
        sden_str += f"\\deltatag{{{delta:+.1f}}}"
    return base_str, sden_str


def _get_refusal_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("refusal_refusal_rate", "refusal_rate"):
        if col in df.columns:
            return col
    return None


def _build_row(
    df: pd.DataFrame,
    metric_col: str,
    refusal_col: str,
    refusal_defense: str,
) -> List[str]:
    row_cells: List[str] = []
    for defense_key, _label in DEFENSE_ORDER:
        subset = df[df["defense_key"] == defense_key]
        base = _to_pct(_select_best(subset, metric_col, is_sden=False))
        sden = _to_pct(_select_best(subset, metric_col, is_sden=True))
        base_str, sden_str = _format_cell(base, sden)
        row_cells.extend([base_str, sden_str])

    refusal_subset = df[df["defense_key"] == refusal_defense]
    refusal_base = _to_pct(_select_best(refusal_subset, refusal_col, is_sden=False))
    refusal_sden = _to_pct(_select_best(refusal_subset, refusal_col, is_sden=True))
    refusal_base_str, refusal_sden_str = _format_cell(refusal_base, refusal_sden)
    row_cells.extend([refusal_base_str, refusal_sden_str])

    return row_cells


def _render_table(
    model_rows: List[Tuple[str, List[List[str]]]],
    *,
    caption: str,
    label: str,
) -> str:
    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3.2pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.05}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}llc cc cc cc cc cc@{}}")
    lines.append("\\toprule")
    lines.append(
        "\\multirow{2}{*}{\\textbf{Model}} &"
        "\\multirow{2}{*}{\\textbf{Benchmark}} &"
        "\\multirow{2}{*}{\\textbf{Attack}} &"
        "\\multicolumn{2}{c}{\\textbf{None}} &"
        "\\multicolumn{2}{c}{\\textbf{PPL}} &"
        "\\multicolumn{2}{c}{\\textbf{Self-rem.}} &"
        "\\multicolumn{2}{c}{\\textbf{DiffuGuard}} &"
        "\\multicolumn{2}{c}{\\textbf{Refusal (\\%)}} \\\\"
    )
    lines.append("\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\\cmidrule(lr){12-13}")
    lines.append("&&&\\textbf{Base} & \\textbf{+\\sden} & \\textbf{Base} & \\textbf{+\\sden} & "
                 "\\textbf{Base} & \\textbf{+\\sden} & \\textbf{Base} & \\textbf{+\\sden} & "
                 "\\textbf{Base} & \\textbf{+\\sden} \\\\")
    lines.append("\\midrule")

    for model_name, rows in model_rows:
        for idx, row in enumerate(rows):
            model_cell = f"\\multirow{{{len(rows)}}}{{*}}{{{model_name}}}" if idx == 0 else ""
            row_line = " & ".join([model_cell] + row) + " \\\\"
            lines.append(row_line)
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines.append("\\end{tabular}}%")
    lines.append("\\vspace{2pt}")
    lines.append("\\footnotesize\\textbf{Notes.} Entries are the best-performing configuration (lowest ASR) under each defence; "
                 "\\deltatag{..} shows the change (in percentage points) from Base to +\\sden within the same defence and row.")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX jailbreak tables from overall_rates.csv files.")
    parser.add_argument(
        "--model-csv",
        action="append",
        required=True,
        help="Model name and CSV path in the form 'ModelName=/path/to/overall_rates.csv'.",
    )
    parser.add_argument("--dataset", type=str, default="wildjailbreak")
    parser.add_argument("--dataset-label", type=str, default="WildJailbreak")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["unsafe_rate"],
        help="Metric columns to render as ASR (e.g., unsafe_rate harmbench_asr).",
    )
    parser.add_argument(
        "--refusal-defense",
        type=str,
        default="none",
        choices=[k for k, _ in DEFENSE_ORDER],
        help="Which defence to use for the refusal-rate columns.",
    )
    parser.add_argument("--caption", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--output-prefix", type=str, default="jailbreak_table")
    args = parser.parse_args()

    model_inputs: List[Tuple[str, Path]] = []
    for item in args.model_csv:
        if "=" not in item:
            raise SystemExit(f"Invalid --model-csv '{item}', expected ModelName=/path/to/overall_rates.csv")
        name, path = item.split("=", 1)
        model_inputs.append((name.strip(), Path(path).expanduser()))

    model_rows_by_metric: Dict[str, List[Tuple[str, List[List[str]]]]] = {}

    for metric in args.metrics:
        model_rows: List[Tuple[str, List[List[str]]]] = []
        for model_name, csv_path in model_inputs:
            df = _read_overall_rates(csv_path)
            df = df[df["dataset_name"].astype(str).str.lower() == args.dataset.lower()]
            if df.empty:
                raise SystemExit(f"No rows for dataset '{args.dataset}' in {csv_path}")

            df = df.copy()
            df["defense_key"] = df.get("defense_method").apply(_normalize_defense)
            refusal_col = _get_refusal_col(df)
            if refusal_col is None:
                raise SystemExit(f"No refusal rate column found in {csv_path}")
            if metric not in df.columns:
                raise SystemExit(f"Metric '{metric}' not found in {csv_path}")

            model_rows_for_attacks: List[List[str]] = []
            for attack_key, attack_label in ATTACK_ORDER:
                if "attack_method" in df.columns:
                    subset = df[df["attack_method"].astype(str).str.lower() == attack_key.lower()]
                else:
                    subset = df.iloc[0:0]
                if subset.empty and "prompt_variant" in df.columns:
                    # Fallback: allow prompt_variant to drive attack if attack_method missing
                    subset = df[df["prompt_variant"].astype(str).str.lower() == attack_key.lower()]
                if subset.empty:
                    continue
                metric_cells = _build_row(
                    subset,
                    metric,
                    refusal_col,
                    args.refusal_defense,
                )
                row = [args.dataset_label, attack_label] + metric_cells
                model_rows_for_attacks.append(row)

            if not model_rows_for_attacks:
                raise SystemExit(f"No attack rows found for model '{model_name}' in {csv_path}")
            model_rows.append((model_name, model_rows_for_attacks))

        model_rows_by_metric[metric] = model_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric, model_rows in model_rows_by_metric.items():
        caption = args.caption
        if caption is None:
            caption = (
                f"Headline jailbreak robustness on {args.dataset_label} prompts. "
                f"We report {metric} (\\%, lower is better) and refusal rate (\\%). "
                "Each defence has Base / +\\sden subcolumns; \\deltatag{..} denotes change vs the corresponding Base "
                "under the same defence."
            )
        label = args.label or f"tab:jailbreak_{args.dataset}_{metric}"
        table_text = _render_table(model_rows, caption=caption, label=label)
        out_path = args.output_dir / f"{args.output_prefix}_{metric}.tex"
        out_path.write_text(table_text + "\n", encoding="utf-8")

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
