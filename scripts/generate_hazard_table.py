#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from utils.constants import LLAMAGUARD_HAZARD_CODES, LLAMAGUARD_HAZARD_MAP


DEFAULT_HAZARD_COLS = (
    "hazard_unsafe_rate_safe",
    "hazard_unsafe_rate",
    "hazard_rate",
    "unsafe_rate",
)


@dataclass
class SelectedRow:
    value: Optional[float]
    row: Optional[pd.Series]


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    if df.empty:
        raise SystemExit(f"{path} is empty.")
    return df


def _pick_hazard_col(df: pd.DataFrame, override: Optional[str]) -> str:
    if override:
        if override not in df.columns:
            raise SystemExit(f"Hazard column '{override}' not found in CSV.")
        return override
    for col in DEFAULT_HAZARD_COLS:
        if col in df.columns:
            return col
    raise SystemExit(
        "No hazard rate column found. Tried: " + ", ".join(DEFAULT_HAZARD_COLS)
    )


def _merge_overall_metrics(
    hazard_df: pd.DataFrame,
    overall_df: Optional[pd.DataFrame],
    quality_cols: Iterable[str],
) -> pd.DataFrame:
    if overall_df is None:
        return hazard_df
    if "run_dir" not in hazard_df.columns or "run_dir" not in overall_df.columns:
        return hazard_df
    keep_cols = ["run_dir"] + [col for col in quality_cols if col in overall_df.columns]
    if len(keep_cols) == 1:
        return hazard_df
    overall_trim = overall_df[keep_cols].drop_duplicates(subset=["run_dir"])
    merged = hazard_df.merge(overall_trim, on="run_dir", how="left")
    return merged


def _select_best_row(
    df: pd.DataFrame,
    metric_col: str,
    *,
    is_sden: bool,
) -> SelectedRow:
    if metric_col not in df.columns or df.empty:
        return SelectedRow(None, None)
    safety_scale = pd.to_numeric(df.get("safety_scale"), errors="coerce")
    if is_sden:
        subset = df[safety_scale.notna() & (safety_scale != 0)]
    else:
        subset = df[safety_scale.isna() | (safety_scale == 0)]
    if subset.empty:
        return SelectedRow(None, None)
    metric_vals = pd.to_numeric(subset[metric_col], errors="coerce")
    if metric_vals.notna().sum() == 0:
        return SelectedRow(None, None)
    best_idx = metric_vals.idxmin()
    best_val = float(metric_vals.loc[best_idx])
    return SelectedRow(best_val, subset.loc[best_idx])


def _to_pct(value: Optional[float]) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value) * 100.0


def _format_pair(
    base: Optional[float],
    sden: Optional[float],
    *,
    decimals: int,
    with_delta: bool,
) -> Tuple[str, str]:
    if base is None:
        base_str = "--"
    else:
        base_str = f"{base:.{decimals}f}"
    if sden is None:
        return base_str, "--"
    sden_str = f"{sden:.{decimals}f}"
    if with_delta and base is not None:
        delta = sden - base
        sden_str += f"\\deltatag{{{delta:+.{decimals}f}}}"
    return base_str, sden_str


def _quality_decimals(metric: str) -> int:
    name = metric.lower()
    if "perplex" in name:
        return 2
    if "bert" in name or "embed" in name:
        return 3
    return 3


def _quality_label(metric: str, overrides: Dict[str, str]) -> str:
    if metric in overrides:
        return overrides[metric]
    name = metric.lower()
    if name == "perplexity":
        return "PPL"
    if name in ("embedding_similarity", "bert_score", "bertscore"):
        return "BERTScore"
    if name == "mmd2_rbf":
        return "MMD2"
    return metric


def _collect_hazard_names(df: pd.DataFrame) -> Dict[str, str]:
    names: Dict[str, str] = {}
    if "hazard_code" in df.columns and "hazard_name" in df.columns:
        for _, row in df[["hazard_code", "hazard_name"]].dropna().iterrows():
            code = str(row["hazard_code"]).strip()
            name = str(row["hazard_name"]).strip()
            if code and name and code not in names:
                names[code] = name
    for code, name in LLAMAGUARD_HAZARD_MAP.items():
        names.setdefault(code, name)
    return names


def _render_table(
    hazard_rows: List[List[str]],
    model_names: List[str],
    quality_cols: List[str],
    quality_multirow_cols: List[str],
    quality_labels: Dict[str, str],
    *,
    caption: str,
    label: str,
) -> str:
    per_hazard_cols = [col for col in quality_cols if col not in quality_multirow_cols]
    metric_groups = ["Hazard (\\%)"] + [
        _escape_latex(_quality_label(metric, quality_labels)) for metric in per_hazard_cols
    ] + [
        _escape_latex(_quality_label(metric, quality_labels)) for metric in quality_multirow_cols
    ]
    cols_per_model = 2 * len(metric_groups)

    col_spec = "@{}ll" + "".join([" " + "c" * cols_per_model for _ in model_names]) + "@{}"

    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3.2pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.05}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row 1: model names
    header_cells = [
        "\\multirow{3}{*}{\\textbf{Hazard}}",
        "\\multirow{3}{*}{\\textbf{Category}}",
    ]
    header_cells += [
        f"\\multicolumn{{{cols_per_model}}}{{c}}{{\\textbf{{{_escape_latex(name)}}}}}"
        for name in model_names
    ]
    lines.append(" & ".join(header_cells) + " \\\\")

    # Header row 2: metric groups
    group_cells = ["", ""]
    for _ in model_names:
        for group in metric_groups:
            group_cells.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{group}}}}}")
    lines.append(" & ".join(group_cells) + " \\\\")

    # cmidrules for each metric group
    cmidrules: List[str] = []
    col_idx = 3
    for _ in model_names:
        for _ in metric_groups:
            cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + 1}}}")
            col_idx += 2
    lines.append("".join(cmidrules))

    # Header row 3: Base / +\sden
    subheader = ["", ""]
    for _ in model_names:
        for _ in metric_groups:
            subheader.append("\\textbf{Base}")
            subheader.append("\\textbf{+\\sden}")
    lines.append(" & ".join(subheader) + " \\\\")
    lines.append("\\midrule")

    expected_cols = 2 + cols_per_model * len(model_names)
    for row in hazard_rows:
        if len(row) != expected_cols:
            raise SystemExit(
                f"Row has {len(row)} columns, expected {expected_cols}. Row={row}"
            )
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}%")
    lines.append("\\vspace{2pt}")
    lines.append(
        "\\footnotesize\\textbf{Notes.} Entries are the best-performing configuration (lowest hazard rate) "
        "for Base and +\\sden within each model; \\deltatag{..} indicates the change (percentage points) from Base to +\\sden."
    )
    lines.append("\\end{table*}")
    return "\n".join(lines)


def _escape_latex(text: str) -> str:
    if text is None:
        return ""
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    escaped = []
    for ch in str(text):
        escaped.append(replacements.get(ch, ch))
    return "".join(escaped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX hazard tables from hazard_rates.csv files."
    )
    parser.add_argument(
        "--model-csv",
        action="append",
        required=True,
        help="Model name and hazard CSV path in the form 'ModelName=/path/to/hazard_rates.csv'.",
    )
    parser.add_argument(
        "--model-overall-csv",
        action="append",
        default=[],
        help="Optional overall_rates.csv paths keyed by model: 'ModelName=/path/to/overall_rates.csv'.",
    )
    parser.add_argument("--prompt-variant", type=str, default=None)
    parser.add_argument("--experiment-slug", type=str, default=None)
    parser.add_argument("--artifact-name", type=str, default=None)
    parser.add_argument("--gating-label", type=str, default=None)
    parser.add_argument("--hazard-col", type=str, default=None)
    parser.add_argument(
        "--quality-cols",
        nargs="+",
        default=["perplexity", "embedding_similarity"],
        help="Optional quality metrics to include (if present).",
    )
    parser.add_argument(
        "--quality-multirow-cols",
        nargs="+",
        default=["perplexity"],
        help="Quality metrics to render as multirow values per model (default: perplexity).",
    )
    parser.add_argument(
        "--quality-label",
        action="append",
        default=[],
        help="Override quality label: col=Label (e.g., embedding_similarity=BERTScore).",
    )
    parser.add_argument("--caption", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--output-prefix", type=str, default="hazard_table")
    args = parser.parse_args()

    model_inputs: List[Tuple[str, Path]] = []
    for item in args.model_csv:
        if "=" not in item:
            raise SystemExit(f"Invalid --model-csv '{item}', expected ModelName=/path/to/hazard_rates.csv")
        name, path = item.split("=", 1)
        model_inputs.append((name.strip(), Path(path).expanduser()))

    overall_inputs: Dict[str, Path] = {}
    for item in args.model_overall_csv:
        if "=" not in item:
            raise SystemExit(
                f"Invalid --model-overall-csv '{item}', expected ModelName=/path/to/overall_rates.csv"
            )
        name, path = item.split("=", 1)
        overall_inputs[name.strip()] = Path(path).expanduser()

    quality_label_overrides: Dict[str, str] = {}
    for item in args.quality_label:
        if "=" not in item:
            raise SystemExit(f"Invalid --quality-label '{item}', expected col=Label")
        col, label = item.split("=", 1)
        quality_label_overrides[col.strip()] = label.strip()

    model_names = [name for name, _ in model_inputs]
    hazard_rows: List[List[str]] = []
    for col in args.quality_multirow_cols:
        if col not in args.quality_cols:
            args.quality_cols.append(col)

    # Build per-model hazard summaries
    model_tables: List[Dict[str, Dict[str, Tuple[SelectedRow, SelectedRow]]]] = []
    model_multirow: List[Dict[str, Tuple[SelectedRow, SelectedRow]]] = []
    hazard_names: Dict[str, str] = {}

    for model_name, hazard_path in model_inputs:
        df = _read_csv(hazard_path)
        hazard_col = _pick_hazard_col(df, args.hazard_col)

        if args.prompt_variant and "prompt_variant" in df.columns:
            df = df[df["prompt_variant"].astype(str) == args.prompt_variant]
        if args.experiment_slug and "experiment_slug" in df.columns:
            df = df[df["experiment_slug"].astype(str) == args.experiment_slug]
        if args.artifact_name and "artifact_name" in df.columns:
            df = df[df["artifact_name"].astype(str) == args.artifact_name]
        if args.gating_label and "gating_label" in df.columns:
            df = df[df["gating_label"].astype(str) == args.gating_label]

        overall_path = overall_inputs.get(model_name)
        overall_df = _read_csv(overall_path) if overall_path else None
        df = _merge_overall_metrics(df, overall_df, args.quality_cols)

        hazard_names.update(_collect_hazard_names(df))

        model_summary: Dict[str, Dict[str, Tuple[SelectedRow, SelectedRow]]] = {}
        for code in df.get("hazard_code", []):
            if pd.isna(code):
                continue
            code_str = str(code).strip()
            if code_str:
                model_summary.setdefault(code_str, {})

        for code in model_summary.keys():
            subset = df[df["hazard_code"].astype(str) == code]
            base = _select_best_row(subset, hazard_col, is_sden=False)
            sden = _select_best_row(subset, hazard_col, is_sden=True)
            model_summary[code]["hazard"] = (base, sden)
            for quality_col in args.quality_cols:
                if quality_col not in subset.columns:
                    model_summary[code][quality_col] = (SelectedRow(None, None), SelectedRow(None, None))
                    continue
                base_q = _select_best_row(subset, quality_col, is_sden=False)
                sden_q = _select_best_row(subset, quality_col, is_sden=True)
                model_summary[code][quality_col] = (base_q, sden_q)

        model_tables.append(model_summary)

        multirow_metrics: Dict[str, Tuple[SelectedRow, SelectedRow]] = {}
        for quality_col in args.quality_multirow_cols:
            if quality_col not in df.columns:
                multirow_metrics[quality_col] = (SelectedRow(None, None), SelectedRow(None, None))
                continue
            base_q = _select_best_row(df, quality_col, is_sden=False)
            sden_q = _select_best_row(df, quality_col, is_sden=True)
            multirow_metrics[quality_col] = (base_q, sden_q)
        model_multirow.append(multirow_metrics)

    if not hazard_names:
        raise SystemExit("No hazard codes found in the provided CSVs.")

    hazard_order = [code for code in LLAMAGUARD_HAZARD_CODES if code in hazard_names]
    missing_codes = sorted(set(hazard_names) - set(hazard_order))
    hazard_order.extend(missing_codes)

    total_rows = len(hazard_order)
    for row_idx, code in enumerate(hazard_order):
        row: List[str] = [code, _escape_latex(hazard_names.get(code, code))]
        for model_idx, model_summary in enumerate(model_tables):
            hazard_pair = model_summary.get(code, {}).get("hazard")
            if hazard_pair is None:
                row.extend(["--", "--"])
                for _ in [c for c in args.quality_cols if c not in args.quality_multirow_cols]:
                    row.extend(["--", "--"])
                for _ in args.quality_multirow_cols:
                    row.extend(["--", "--"] if row_idx == 0 else ["", ""])
                continue
            base_sel, sden_sel = hazard_pair
            base_pct = _to_pct(base_sel.value)
            sden_pct = _to_pct(sden_sel.value)
            base_str, sden_str = _format_pair(
                base_pct, sden_pct, decimals=1, with_delta=True
            )
            row.extend([base_str, sden_str])

            for quality_col in [c for c in args.quality_cols if c not in args.quality_multirow_cols]:
                base_q, sden_q = model_summary.get(code, {}).get(
                    quality_col, (SelectedRow(None, None), SelectedRow(None, None))
                )
                decimals = _quality_decimals(quality_col)
                base_val = base_q.value
                sden_val = sden_q.value
                base_q_str, sden_q_str = _format_pair(
                    base_val, sden_val, decimals=decimals, with_delta=False
                )
                row.extend([base_q_str, sden_q_str])

            for quality_col in args.quality_multirow_cols:
                base_q, sden_q = model_multirow[model_idx].get(
                    quality_col, (SelectedRow(None, None), SelectedRow(None, None))
                )
                decimals = _quality_decimals(quality_col)
                base_val = base_q.value
                sden_val = sden_q.value
                base_q_str, sden_q_str = _format_pair(
                    base_val, sden_val, decimals=decimals, with_delta=False
                )
                if row_idx == 0:
                    row.extend(
                        [
                            f"\\multirow{{{total_rows}}}{{*}}{{{base_q_str}}}",
                            f"\\multirow{{{total_rows}}}{{*}}{{{sden_q_str}}}",
                        ]
                    )
                else:
                    row.extend(["", ""])
        hazard_rows.append(row)

    caption = args.caption
    if caption is None:
        caption = (
            "LlamaGuard hazard rates by category (\%, lower is better). "
            "Each model reports the best Base and +\\sden configuration per hazard." 
        )
        if args.quality_cols:
            caption += " Quality metrics are reported for the selected configurations when available."
    label = args.label or "tab:hazard_rates"

    table_text = _render_table(
        hazard_rows,
        model_names,
        args.quality_cols,
        args.quality_multirow_cols,
        quality_label_overrides,
        caption=caption,
        label=label,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.output_prefix}.tex"
    out_path.write_text(table_text + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
