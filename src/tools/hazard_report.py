#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils.constants import LLAMAGUARD_HAZARD_CODES, LLAMAGUARD_HAZARD_MAP
from tools.score import compute_generation_embeddings, compute_mmd_rbf_for_config

LOGGER = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise SystemExit(f"Refusing to overwrite file at {path}; choose a different output directory.")
    path.mkdir(parents=True, exist_ok=True)


def _load_perplexity(run_dir: Path) -> tuple[Optional[float], Optional[int]]:
    """Load perplexity summary if present."""
    candidates = [
        run_dir / "scores" / "perplexity.json",
    ]
    if not candidates[0].exists():
        candidates = list(run_dir.rglob("perplexity.json"))
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ppl = float(data.get("perplexity"))
            texts = int(data.get("texts")) if "texts" in data else None
            return ppl, texts
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None, None


def _load_alignment_similarity(run_dir: Path) -> Optional[float]:
    summary_path = run_dir / "scores" / "embedding_alignment_summary.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        return float(data.get("mean_similarity")) if "mean_similarity" in data else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _load_jailbreak_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "scores" / "jailbreak_metrics.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}
    return {
        "harmbench_asr": payload.get("harmbench_asr"),
        "advbench_asr": payload.get("advbench_asr"),
        "strong_reject_score": payload.get("strong_reject_score"),
        "strong_reject_count": payload.get("strong_reject_count"),
    }


def _load_alignment_pairs(run_dir: Path) -> List[Dict[str, Any]]:
    candidates = [run_dir / "scores" / "embedding_alignment_pairs.jsonl"]
    if not candidates[0].exists():
        candidates = list(run_dir.rglob("embedding_alignment_pairs.jsonl"))
    for path in candidates:
        if path.exists():
            return _load_json_records(path)
    return []


def _compute_unsafe_summary(records: List[Dict[str, Any]]) -> tuple[int, int, Optional[float]]:
    total = len(records)
    if total == 0:
        return 0, 0, None
    unsafe = sum(1 for rec in records if bool(rec.get("unsafe")))
    return total, unsafe, round(unsafe / total, 6)


def _embedding_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "pairs": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "frac_lt_0_5": None,
            "frac_lt_0_7": None,
            "frac_lt_0_9": None,
        }
    arr = np.array(values, dtype=np.float32)
    return {
        "pairs": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "frac_lt_0_5": float(np.mean(arr < 0.5)),
        "frac_lt_0_7": float(np.mean(arr < 0.7)),
        "frac_lt_0_9": float(np.mean(arr < 0.9)),
    }


def _load_hygiene_metrics(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "scores" / "hygiene_metrics.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = {
            "hygiene_stop_token_leak_rate": data.get("stop_token_leak_rate"),
            "hygiene_mask_leak_rate": data.get("mask_leak_rate"),
            "hygiene_empty_completion_rate": data.get("empty_completion_rate"),
        }
        if "early_stop" in data and isinstance(data["early_stop"], dict):
            out["hygiene_early_stop_fraction"] = data["early_stop"].get("fraction")
        return out
    except (json.JSONDecodeError, ValueError):
        return {}


def _load_lexical_metrics(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "scores" / "lexical_metrics.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = {
            "lexical_exact_match_rate": data.get("exact_match_rate"),
        }
        # overlap n-grams
        for n_key, metrics in data.get("overlap", {}).items():
            if isinstance(metrics, dict):
                for metric_name, stats in metrics.items():
                    if isinstance(stats, dict):
                        out[f"lexical_overlap_{n_key}_{metric_name}_mean"] = stats.get("mean")
        # distinct n-grams
        for n_key, stats in data.get("distinct", {}).items():
            if isinstance(stats, dict):
                out[f"lexical_distinct_{n_key}_mean"] = stats.get("mean")
        # repeat n-grams
        for n_key, stats in data.get("repeat", {}).items():
            if isinstance(stats, dict):
                out[f"lexical_repeat_{n_key}_mean"] = stats.get("mean")
        # other stats
        if "copy_4" in data and isinstance(data["copy_4"], dict):
            out["lexical_copy_4_mean"] = data["copy_4"].get("mean")
        if "fuzzy_overlap" in data and isinstance(data["fuzzy_overlap"], dict):
            summary = data["fuzzy_overlap"].get("summary")
            if isinstance(summary, dict):
                out["lexical_fuzzy_overlap_mean"] = summary.get("mean")
        if "max_repeated_span" in data and isinstance(data["max_repeated_span"], dict):
            out["lexical_max_repeated_span_mean"] = data["max_repeated_span"].get("mean")
        return out
    except (json.JSONDecodeError, ValueError):
        return {}


def _load_bertscore_metrics(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "scores" / "bertscore.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = {}
        for metric in ("precision", "recall", "f1"):
            stats = data.get(metric)
            if isinstance(stats, dict):
                out[f"bertscore_{metric}_mean"] = stats.get("mean")
        return out
    except (json.JSONDecodeError, ValueError):
        return {}


def _load_mauve_metrics(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "scores" / "mauve.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = {}
        for key in (
            "mauve_exp_vs_ref",
            "mauve_base_vs_ref",
            "mauve_num_texts",
            "mauve_max_texts_cap",
            "mauve_model_name",
        ):
            if key in data:
                out[key] = data.get(key)
        return out
    except (json.JSONDecodeError, ValueError):
        return {}


def _load_refusal_metrics(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "scores" / "refusal_metrics.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = {
            "refusal_rate": data.get("refusal_rate"),
            "non_answer_rate": data.get("non_answer_rate"),
            "num_texts": data.get("num_texts"),
        }
        if "refusal_phrases_hit_rate" in data:
            out["refusal_phrases_hit_rate"] = data.get("refusal_phrases_hit_rate")
        return {f"refusal_{k}": v for k, v in out.items()}
    except (json.JSONDecodeError, ValueError):
        return {}


def _load_degeneration_metrics(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "scores" / "degeneration_metrics.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = {
            "degeneration_rate": data.get("degeneration_rate"),
            "degeneration_num_texts": data.get("num_texts"),
        }
        components = data.get("degeneration_components")
        if isinstance(components, dict):
            for key, value in components.items():
                out[f"degeneration_{key}"] = value
        return out
    except (json.JSONDecodeError, ValueError):
        return {}
def _generate_embedding_report(
    run_reports: List[Dict[str, Any]],
    output_dir: Path,
    examples_per_run: int,
    *,
    jailbreak_split: bool = False,
) -> None:
    if not run_reports:
        LOGGER.warning("No run reports available for embedding similarity report; skipping.")
        return
    baseline_map: Dict[str, float] = {}
    for report in run_reports:
        if report.get("safety_scale") is None and report.get("unsafe_rate") is not None:
            baseline_map[_tensor_key_from_report(report, jailbreak_split=jailbreak_split)] = float(
                report.get("unsafe_rate", 0.0)
            )
    report_by_run: Dict[Path, Dict[str, Any]] = {}
    for report in run_reports:
        run_dir = Path(str(report.get("run_dir"))).resolve()
        report_by_run[run_dir] = report

    summary_rows: List[List[Any]] = []
    examples_low: List[Dict[str, Any]] = []
    examples_high: List[Dict[str, Any]] = []

    for run_dir, report in report_by_run.items():
        pairs = _load_alignment_pairs(run_dir)
        values: List[float] = []
        for rec in pairs:
            try:
                values.append(float(rec.get("similarity")))
            except (TypeError, ValueError):
                continue
        stats = _embedding_stats(values)
        tensor_key = _tensor_key_from_report(report, jailbreak_split=jailbreak_split)
        baseline_rate = baseline_map.get(tensor_key)
        if baseline_rate is None and report.get("defense_method"):
            fallback_key = _tensor_key_without_defense(report, jailbreak_split=jailbreak_split)
            baseline_rate = baseline_map.get(fallback_key)
        unsafe_rate = report.get("unsafe_rate")
        unsafe_delta = None
        if baseline_rate is not None and unsafe_rate is not None:
            unsafe_delta = float(unsafe_rate) - float(baseline_rate)
        row = [
            str(run_dir),
            report.get("experiment_slug"),
        ]
        if jailbreak_split:
            row.extend(
                [
                    report.get("dataset_name"),
                    report.get("attack_method"),
                    report.get("defense_method"),
                ]
            )
        row.extend(
            [
                report.get("prompt_source"),
                report.get("prompt_variant"),
                report.get("artifact_name"),
                report.get("tensor_size"),
                report.get("safety_scale"),
                report.get("t_start"),
                report.get("t_end"),
                report.get("critical_steps"),
                report.get("gating_label"),
                report.get("perplexity"),
                report.get("perplexity_texts"),
                unsafe_rate,
                baseline_rate,
                unsafe_delta,
                stats["pairs"],
                stats["mean"],
                report.get("mmd2_rbf"),
                stats["median"],
                stats["std"],
                stats["min"],
                stats["max"],
                stats["p10"],
                stats["p25"],
                stats["p75"],
                stats["p90"],
                stats["frac_lt_0_5"],
                stats["frac_lt_0_7"],
                stats["frac_lt_0_9"],
            ]
        )
        summary_rows.append(row)

        if pairs and examples_per_run > 0:
            sortable_pairs: List[tuple[float, Dict[str, Any]]] = []
            for rec in pairs:
                try:
                    sim_val = float(rec.get("similarity"))
                except (TypeError, ValueError):
                    continue
                sortable_pairs.append((sim_val, rec))
            sortable_pairs.sort(key=lambda item: item[0])
            sorted_pairs = [item[1] for item in sortable_pairs]
            low_pairs = sorted_pairs[:examples_per_run]
            high_pairs = sorted_pairs[-examples_per_run:]
            for rec in low_pairs:
                entry = dict(rec)
                entry.update(
                    {
                        "run_dir": str(run_dir),
                        "experiment_slug": report.get("experiment_slug"),
                        "prompt_variant": report.get("prompt_variant"),
                        "artifact_name": report.get("artifact_name"),
                        "tensor_size": report.get("tensor_size"),
                        "safety_scale": report.get("safety_scale"),
                        "gating_label": report.get("gating_label"),
                        "perplexity": report.get("perplexity"),
                        "unsafe_rate": unsafe_rate,
                        "baseline_unsafe_rate": baseline_rate,
                        "unsafe_rate_delta_vs_baseline": unsafe_delta,
                        "example_rank": "low",
                    }
                )
                examples_low.append(entry)
            for rec in high_pairs:
                entry = dict(rec)
                entry.update(
                    {
                        "run_dir": str(run_dir),
                        "experiment_slug": report.get("experiment_slug"),
                        "prompt_variant": report.get("prompt_variant"),
                        "artifact_name": report.get("artifact_name"),
                        "tensor_size": report.get("tensor_size"),
                        "safety_scale": report.get("safety_scale"),
                        "gating_label": report.get("gating_label"),
                        "perplexity": report.get("perplexity"),
                        "unsafe_rate": unsafe_rate,
                        "baseline_unsafe_rate": baseline_rate,
                        "unsafe_rate_delta_vs_baseline": unsafe_delta,
                        "example_rank": "high",
                    }
                )
                examples_high.append(entry)

    summary_header = [
        "run_dir",
        "experiment_slug",
    ]
    if jailbreak_split:
        summary_header.extend(["dataset_name", "attack_method", "defense_method"])
    summary_header.extend(
        [
            "prompt_source",
            "prompt_variant",
            "artifact_name",
            "tensor_size",
            "safety_scale",
            "t_start",
            "t_end",
            "critical_steps",
            "gating_label",
            "perplexity",
            "perplexity_texts",
            "unsafe_rate",
            "baseline_unsafe_rate",
            "unsafe_rate_delta_vs_baseline",
            "pairs",
            "mean_similarity",
            "mmd2_rbf",
            "median_similarity",
            "std_similarity",
            "min_similarity",
            "max_similarity",
            "p10_similarity",
            "p25_similarity",
            "p75_similarity",
            "p90_similarity",
            "frac_lt_0_5",
            "frac_lt_0_7",
            "frac_lt_0_9",
        ]
    )
    _write_csv(summary_rows, summary_header, output_dir / "embedding_similarity_summary.csv")
    LOGGER.info("Wrote embedding_similarity_summary.csv (%d rows)", len(summary_rows))

    if examples_low:
        low_path = output_dir / "embedding_similarity_examples_low.jsonl"
        with low_path.open("w", encoding="utf-8") as handle:
            for rec in examples_low:
                handle.write(json.dumps(rec) + "\n")
        LOGGER.info("Wrote %d low-similarity examples to %s", len(examples_low), low_path)
    if examples_high:
        high_path = output_dir / "embedding_similarity_examples_high.jsonl"
        with high_path.open("w", encoding="utf-8") as handle:
            for rec in examples_high:
                handle.write(json.dumps(rec) + "\n")
        LOGGER.info("Wrote %d high-similarity examples to %s", len(examples_high), high_path)


def _load_parquet_records(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_parquet(path)
    return df.to_dict("records")


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _tensor_key_from_report(report: Dict[str, Any], *, jailbreak_split: bool = False) -> str:
    art = report.get("artifact_name")
    prompt_variant = report.get("prompt_variant")
    dataset_name = report.get("dataset_name")
    attack_method = report.get("attack_method")
    defense_method = report.get("defense_method")
    if jailbreak_split and dataset_name:
        base = str(dataset_name)
    elif art:
        base = str(art)
    else:
        base = f"{report.get('experiment_slug')}_baseline"
    parts = [base]
    if jailbreak_split:
        parts.extend(
            [
                f"ds={dataset_name}" if dataset_name else None,
                f"attack={attack_method}" if attack_method else None,
                f"def={defense_method}" if defense_method else None,
            ]
        )
    if prompt_variant:
        parts.append(f"pv={prompt_variant}")
    return "|".join([p for p in parts if p])


def _tensor_key_without_defense(report: Dict[str, Any], *, jailbreak_split: bool = False) -> str:
    if not jailbreak_split:
        return _tensor_key_from_report(report, jailbreak_split=False)
    clone = dict(report)
    clone["defense_method"] = None
    return _tensor_key_from_report(clone, jailbreak_split=True)


def _attach_baseline_deltas(
    run_reports: Sequence[Dict[str, Any]],
    metric_keys: Sequence[str],
    *,
    only_for_prompt_variant: Optional[str] = "benign",
    jailbreak_split: bool = False,
) -> None:
    baseline_map: Dict[str, Dict[str, Any]] = {}
    for report in run_reports:
        if report.get("safety_scale") is not None:
            continue
        tensor_key = _tensor_key_from_report(report, jailbreak_split=jailbreak_split)
        baseline_map[tensor_key] = report
        fallback_key = _tensor_key_from_report(
            {
                "artifact_name": None,
                "experiment_slug": report.get("experiment_slug"),
                "dataset_name": report.get("dataset_name"),
                "attack_method": report.get("attack_method"),
                "defense_method": report.get("defense_method"),
                "prompt_variant": report.get("prompt_variant"),
            },
            jailbreak_split=jailbreak_split,
        )
        baseline_map[fallback_key] = report

    for report in run_reports:
        prompt_variant = report.get("prompt_variant")
        is_target = only_for_prompt_variant is None or prompt_variant == only_for_prompt_variant
        tensor_key = _tensor_key_from_report(report, jailbreak_split=jailbreak_split)
        baseline_report = baseline_map.get(tensor_key)
        if baseline_report is None:
            fallback_key = _tensor_key_from_report(
                {
                    "artifact_name": None,
                    "experiment_slug": report.get("experiment_slug"),
                    "dataset_name": report.get("dataset_name"),
                    "attack_method": report.get("attack_method"),
                    "defense_method": report.get("defense_method"),
                    "prompt_variant": prompt_variant,
                },
                jailbreak_split=jailbreak_split,
            )
            baseline_report = baseline_map.get(fallback_key)
        if baseline_report is None and report.get("defense_method"):
            fallback_key = _tensor_key_without_defense(
                {
                    "artifact_name": report.get("artifact_name"),
                    "experiment_slug": report.get("experiment_slug"),
                    "dataset_name": report.get("dataset_name"),
                    "attack_method": report.get("attack_method"),
                    "defense_method": report.get("defense_method"),
                    "prompt_variant": prompt_variant,
                },
                jailbreak_split=jailbreak_split,
            )
            baseline_report = baseline_map.get(fallback_key)
            if baseline_report is None:
                fallback_key = _tensor_key_without_defense(
                    {
                        "artifact_name": None,
                        "experiment_slug": report.get("experiment_slug"),
                        "dataset_name": report.get("dataset_name"),
                        "attack_method": report.get("attack_method"),
                        "defense_method": report.get("defense_method"),
                        "prompt_variant": prompt_variant,
                    },
                    jailbreak_split=jailbreak_split,
                )
                baseline_report = baseline_map.get(fallback_key)
        for metric in metric_keys:
            delta_key = f"{metric}_delta_vs_baseline"
            if not is_target:
                report[delta_key] = np.nan
                continue
            baseline_val = baseline_report.get(metric) if baseline_report else None
            current_val = report.get(metric)
            if baseline_val is None or current_val is None:
                report[delta_key] = None
                continue
            try:
                report[delta_key] = float(current_val) - float(baseline_val)
            except (TypeError, ValueError):
                report[delta_key] = None


def _score_records(run_dir: Path) -> List[Dict[str, Any]]:
    scores_dir = run_dir / "scores"
    if not scores_dir.exists():
        LOGGER.warning("No 'scores' directory found under %s; skipping.", run_dir)
        return []
    files: List[Path] = []
    files.extend(sorted(scores_dir.glob("safety_shard_*.parquet")))
    files.extend(sorted(scores_dir.glob("safety_shard_*.jsonl")))
    if not files:
        # try a recursive search (e.g., scores/llamaguard/safety_shard_*.parquet)
        files.extend(sorted(scores_dir.rglob("safety_shard_*.parquet")))
        files.extend(sorted(scores_dir.rglob("safety_shard_*.jsonl")))
    if not files:
        LOGGER.warning("No safety score shards found under %s; skipping.", scores_dir)
        return []
    LOGGER.info("Found %d safety score shard(s) under %s", len(files), scores_dir)
    records: List[Dict[str, Any]] = []
    for path in files:
        if path.suffix == ".parquet":
            LOGGER.info("Loading parquet shard %s", path)
            records.extend(_load_parquet_records(path))
        else:
            records.extend(_load_json_records(path))
    return records


def _parse_hazards(raw_value: Any) -> List[str]:
    if isinstance(raw_value, np.ndarray):
        raw_value = raw_value.tolist()
    if isinstance(raw_value, (list, tuple)):
        parsed: List[str] = []
        for item in raw_value:
            if isinstance(item, dict):
                # common keys for hazard code/name
                for key in ("hazard_code", "code", "name"):
                    if key in item and item[key]:
                        parsed.append(str(item[key]))
                        break
            elif isinstance(item, (list, tuple, np.ndarray)):
                parsed.extend(_parse_hazards(item))
            elif item:
                parsed.append(str(item))
        return parsed
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if item]
            except json.JSONDecodeError:
                pass
        return [text]
    return []


def _tensor_size(artifact_name: Optional[str]) -> Optional[str]:
    if not artifact_name:
        return None
    match = re.search(r"(\d{3,5})", artifact_name)
    if match:
        return match.group(1)
    return artifact_name


def _normalize_seq(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _gating_label(t_start: Any, t_end: Any, critical_steps: Optional[List[Any]]) -> str:
    # Gating label tag used in CSVs/figs to separate different gating choices:
    # - cs_<steps> when critical_steps is set (e.g., cs_0_1_2)
    # - ts_<t_start>_<t_end> otherwise (using "full" when a bound is None)
    if critical_steps:
        parts = "_".join(str(x) for x in critical_steps)
        return f"cs_{parts}"
    start_str = "full" if t_start is None else str(t_start)
    end_str = "full" if t_end is None else str(t_end)
    return f"ts_{start_str}_{end_str}"


def _sanitize_for_path(text: str) -> str:
    safe = re.sub(r"[\\/:]", "_", text)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "default"


def _sort_scale_labels(labels: Iterable[str]) -> List[str]:
    def _key(label: str) -> tuple[int, object]:
        if label == "baseline":
            return (0, 0.0)
        try:
            return (1, float(label))
        except (TypeError, ValueError):
            return (2, str(label))

    return sorted(set(labels), key=_key)


def _resolve_config_path(run_dir: Path) -> Optional[Path]:
    cfg_path = run_dir / "config_merged.yaml"
    if cfg_path.exists():
        return cfg_path
    candidates = list(run_dir.rglob("config_merged.yaml"))
    if candidates:
        filtered = [
            path
            for path in candidates
            if "scores" not in path.relative_to(run_dir).parts
        ]
        filtered = filtered or candidates
        return min(filtered, key=lambda path: len(path.relative_to(run_dir).parts))
    hydra_cfg = run_dir / ".hydra" / "config.yaml"
    if hydra_cfg.exists():
        return hydra_cfg
    hydra_candidates = list(run_dir.rglob(".hydra/config.yaml"))
    if hydra_candidates:
        filtered = [
            path
            for path in hydra_candidates
            if "scores" not in path.relative_to(run_dir).parts
        ]
        filtered = filtered or hydra_candidates
        return min(filtered, key=lambda path: len(path.relative_to(run_dir).parts))
    return None


def _load_run_config(run_dir: Path) -> Optional[DictConfig]:
    cfg_path = _resolve_config_path(run_dir)
    if not cfg_path:
        LOGGER.warning("No config_merged.yaml or .hydra/config.yaml found under %s; skipping embeddings.", run_dir)
        return None
    LOGGER.info("Loading run config from %s", cfg_path)
    return OmegaConf.load(cfg_path)


def _load_distribution_shift_metrics(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "scores" / "distribution_shift.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "mmd_split_half_mean": data.get("baseline_split_half_mean"),
        }
    except (json.JSONDecodeError, ValueError):
        return {}


def _infer_dataset_name(*candidates: Optional[str]) -> Optional[str]:
    dataset_order = (
        "wildjailbreak",
        "jailbreakbench",
        "advbench",
        "strongreject",
        "xstest",
        "harmbench",
    )
    for value in candidates:
        if not value:
            continue
        try:
            text = str(value).lower()
        except Exception:
            continue
        if not text:
            continue
        if "jbb" in text and "jailbreakbench" not in text:
            return "jailbreakbench"
        for name in dataset_order:
            if name in text:
                return name
    return None


def _infer_attack_method(jailbreak_cfg: Dict[str, Any], run_dir: Path) -> Optional[str]:
    attack_method = jailbreak_cfg.get("attack_method")
    if attack_method not in (None, "", "null"):
        return str(attack_method)
    match = re.search(r"-(PAD|DIJA|zeroshot|zero-shot|autodan|gcg)(?:-|$)", run_dir.name, re.IGNORECASE)
    if match:
        return match.group(1).replace("zero-shot", "zeroshot")
    return None


def _infer_defense_method(jailbreak_cfg: Dict[str, Any], run_dir: Path) -> Optional[str]:
    defense_method = jailbreak_cfg.get("defense_method")
    if defense_method not in (None, "", "null"):
        return str(defense_method)
    match = re.search(r"-def-([^-]+)", run_dir.name)
    if match:
        return match.group(1)
    return None


def _parse_run_metadata(run_dir: Path) -> Optional[Dict[str, Any]]:
    cfg_path = _resolve_config_path(run_dir)
    if not cfg_path:
        LOGGER.warning(
            "No config_merged.yaml or .hydra/config.yaml found under %s; skipping run.",
            run_dir,
        )
        return None
    cfg = OmegaConf.load(cfg_path)
    data_cfg = cfg.get("data") or {}
    prompt_source = data_cfg.get("prompt_source") or {}
    safety_cfg = cfg.get("safety") or {}
    jailbreak_cfg = cfg.get("jailbreak") or {}
    io_cfg = cfg.get("io") or {}
    model_cfg = cfg.get("model") or {}
    _mv = model_cfg.get("variant")
    model_variant = str(_mv) if _mv not in (None, "", "null", "None") else None
    prompt_variant = data_cfg.get("prompt_variant")
    if not prompt_variant and isinstance(prompt_source, dict):
        prompt_params = prompt_source.get("params")
        if isinstance(prompt_params, dict):
            prompt_variant = prompt_params.get("prompt_variant")

    artifact_name = safety_cfg.get("unsafe_artifact_name")
    proto_path = safety_cfg.get("unsafe_prototypes")
    if not artifact_name and proto_path:
        artifact_name = Path(str(proto_path)).name
    if not artifact_name and safety_cfg.get("unsafe_prototype_root") and safety_cfg.get("unsafe_artifact_name"):
        proto_candidate = Path(str(safety_cfg.get("unsafe_prototype_root"))) / f"{safety_cfg.get('unsafe_artifact_name')}_k64.pt"
        artifact_name = proto_candidate.name
    tensor_size = _tensor_size(artifact_name)
    prompt_name = prompt_source.get("name") if isinstance(prompt_source, dict) else None
    prompt_params = prompt_source.get("params") if isinstance(prompt_source, dict) else {}
    data_dir = prompt_params.get("data_dir") if isinstance(prompt_params, dict) else None
    dataset_json = data_cfg.get("dataset_json")
    dataset_name = _infer_dataset_name(
        io_cfg.get("experiment_slug"),
        prompt_name,
        dataset_json,
        data_dir,
        run_dir.name,
    )

    attack_method = _infer_attack_method(jailbreak_cfg, run_dir)
    defense_method = _infer_defense_method(jailbreak_cfg, run_dir)

    t_start = safety_cfg.get("t_start")
    t_end = safety_cfg.get("t_end")
    critical_steps = _normalize_seq(safety_cfg.get("critical_steps"))
    ppl_value, ppl_texts = _load_perplexity(run_dir)
    eta = safety_cfg.get("eta")
    scale = safety_cfg.get("scale")
    enabled = bool(safety_cfg.get("enabled"))
    effective_eta = eta if eta is not None else scale
    if not enabled:
        effective_eta = None

    _filter_variants = {"posthoc_filter", "best_of_n", "fk_steering"}

    metrics = {
        "run_dir": run_dir,
        "experiment_slug": io_cfg.get("experiment_slug"),
        "dataset_name": dataset_name,
        "prompt_source": prompt_name,
        "prompt_variant": prompt_variant,
        "data_dir": data_dir,
        "dataset_json": dataset_json,
        "artifact_name": artifact_name,
        "tensor_size": tensor_size,
        "model_variant": model_variant,
        "safety_enabled": enabled,
        "safety_eta": effective_eta,
        # For filter variants, use the variant name as safety_scale so they don't
        # merge with true baseline rows (which also have safety.enabled=false).
        "safety_scale": model_variant if model_variant in _filter_variants else effective_eta,
        "t_start": t_start,
        "t_end": t_end,
        "critical_steps": critical_steps,
        "gating_label": _gating_label(t_start, t_end, critical_steps),
        "attack_method": attack_method,
        "defense_method": defense_method,
        "perplexity": ppl_value,
        "perplexity_texts": ppl_texts,
    }
    metrics.update(_load_hygiene_metrics(run_dir))
    metrics.update(_load_lexical_metrics(run_dir))
    metrics.update(_load_bertscore_metrics(run_dir))
    metrics.update(_load_mauve_metrics(run_dir))
    metrics.update(_load_refusal_metrics(run_dir))
    metrics.update(_load_degeneration_metrics(run_dir))
    metrics.update(_load_distribution_shift_metrics(run_dir))
    return metrics


def _hazard_entry(count: int, total: int, code: str) -> Dict[str, Any]:
    return {
        "hazard_code": code,
        "hazard_name": LLAMAGUARD_HAZARD_MAP.get(code, code),
        "unsafe": count,
        "rate": round(count / total, 6) if total else 0.0,
    }


def _embedding_cache_key(run_dir: Path, cfg_root: DictConfig) -> str:
    model_cfg = cfg_root.get("model") or {}
    gen_cfg = cfg_root.get("gen") or {}
    score_cfg = cfg_root.get("score") or {}
    parts = [
        str(run_dir),
        str(model_cfg.get("checkpoint") or ""),
        str(model_cfg.get("tokenizer_name") or ""),
        str(gen_cfg.get("max_new_tokens") or ""),
        str(score_cfg.get("batch_size") or ""),
        os.environ.get("MDLM_EMBED_FN", ""),
        os.environ.get("ALIGNMENT_EMBEDDER_MODEL", ""),
        os.environ.get("MDLM_EMBED_ATTR", ""),
        os.environ.get("MODEL_CONFIG_PATH", ""),
    ]
    return "|".join(parts)


def _get_embeddings_for_run(
    run_dir: Path,
    cfg_root: DictConfig,
    cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    cache_key = _embedding_cache_key(run_dir, cfg_root)
    cached = cache.get(cache_key)
    if cached is not None:
        LOGGER.info("Using cached embeddings for %s", run_dir)
        return cached
    score_cfg = cfg_root.get("score") or {}
    text_field = str(score_cfg.get("text_field") or "completion")
    embeddings, _ = compute_generation_embeddings(cfg_root, run_dir, text_field=text_field)
    cache[cache_key] = embeddings
    return embeddings


def _populate_mmd_metrics(run_reports: List[Dict[str, Any]], *, jailbreak_split: bool = False) -> None:
    if not run_reports:
        return
    baseline_run_map: Dict[str, Path] = {}
    for report in run_reports:
        if report.get("safety_scale") is None:
            tensor_key = _tensor_key_from_report(report, jailbreak_split=jailbreak_split)
            baseline_run_map[tensor_key] = Path(str(report.get("run_dir"))).resolve()
            fallback_key = _tensor_key_from_report(
                {
                    "artifact_name": None,
                    "experiment_slug": report.get("experiment_slug"),
                    "dataset_name": report.get("dataset_name"),
                    "attack_method": report.get("attack_method"),
                    "defense_method": report.get("defense_method"),
                    "prompt_variant": report.get("prompt_variant"),
                },
                jailbreak_split=jailbreak_split,
            )
            baseline_run_map[fallback_key] = Path(str(report.get("run_dir"))).resolve()
            if report.get("defense_method"):
                no_def_key = _tensor_key_without_defense(report, jailbreak_split=jailbreak_split)
                baseline_run_map.setdefault(no_def_key, Path(str(report.get("run_dir"))).resolve())
                no_def_fallback = _tensor_key_without_defense(
                    {
                        "artifact_name": None,
                        "experiment_slug": report.get("experiment_slug"),
                        "dataset_name": report.get("dataset_name"),
                        "attack_method": report.get("attack_method"),
                        "defense_method": report.get("defense_method"),
                        "prompt_variant": report.get("prompt_variant"),
                    },
                    jailbreak_split=jailbreak_split,
                )
                baseline_run_map.setdefault(no_def_fallback, Path(str(report.get("run_dir"))).resolve())

    if not baseline_run_map:
        LOGGER.warning("No baseline runs found; skipping MMD computation.")
        return

    cfg_cache: Dict[Path, DictConfig] = {}
    embed_cache: Dict[str, torch.Tensor] = {}
    for report in run_reports:
        if report.get("safety_scale") is None:
            report["mmd2_rbf"] = None
            continue
        tensor_key = _tensor_key_from_report(report, jailbreak_split=jailbreak_split)
        baseline_dir = baseline_run_map.get(tensor_key)
        if baseline_dir is None:
            fallback_key = _tensor_key_from_report(
                {
                    "artifact_name": None,
                    "experiment_slug": report.get("experiment_slug"),
                    "dataset_name": report.get("dataset_name"),
                    "attack_method": report.get("attack_method"),
                    "defense_method": report.get("defense_method"),
                    "prompt_variant": report.get("prompt_variant"),
                },
                jailbreak_split=jailbreak_split,
            )
            baseline_dir = baseline_run_map.get(fallback_key)
        if baseline_dir is None and report.get("defense_method"):
            fallback_key = _tensor_key_without_defense(report, jailbreak_split=jailbreak_split)
            baseline_dir = baseline_run_map.get(fallback_key)
            if baseline_dir is None:
                fallback_key = _tensor_key_without_defense(
                    {
                        "artifact_name": None,
                        "experiment_slug": report.get("experiment_slug"),
                        "dataset_name": report.get("dataset_name"),
                        "attack_method": report.get("attack_method"),
                        "defense_method": report.get("defense_method"),
                        "prompt_variant": report.get("prompt_variant"),
                    },
                    jailbreak_split=jailbreak_split,
                )
                baseline_dir = baseline_run_map.get(fallback_key)
            if baseline_dir is None:
                LOGGER.warning("No baseline match for %s; skipping MMD.", report.get("run_dir"))
                report["mmd2_rbf"] = None
                continue
            LOGGER.info(
                "Using fallback baseline key %s for %s",
                fallback_key,
                report.get("run_dir"),
            )
        safe_dir = Path(str(report.get("run_dir"))).resolve()
        cfg_root = cfg_cache.get(safe_dir)
        if cfg_root is None:
            cfg_root = _load_run_config(safe_dir)
            if cfg_root is None:
                report["mmd2_rbf"] = None
                continue
            cfg_cache[safe_dir] = cfg_root
        safe_embeddings = _get_embeddings_for_run(safe_dir, cfg_root, embed_cache)
        baseline_embeddings = _get_embeddings_for_run(baseline_dir, cfg_root, embed_cache)
        if safe_embeddings.numel() == 0 or baseline_embeddings.numel() == 0:
            LOGGER.warning(
                "Missing embeddings for MMD (%s: %s, baseline: %s).",
                safe_dir,
                tuple(safe_embeddings.shape),
                tuple(baseline_embeddings.shape),
            )
            report["mmd2_rbf"] = None
            continue
        config_name = _gating_label(
            report.get("t_start"),
            report.get("t_end"),
            report.get("critical_steps"),
        )
        mmd_info = compute_mmd_rbf_for_config(
            baseline_embeddings,
            safe_embeddings,
            config_name,
        )
        report["mmd2_rbf"] = mmd_info.get("mmd2_rbf")
        LOGGER.info(
            "MMD^2 (RBF) for %s vs baseline: %.6f (near 0 => similar; larger => shift)",
            safe_dir,
            report["mmd2_rbf"],
        )


def _build_run_report(
    records: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    examples_per_hazard: int,
    examples_per_metric: int,
    examples_per_transition: int,
    baseline_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    total = len(records)
    unsafe = sum(1 for rec in records if bool(rec.get("unsafe")))
    hazard_counts: Dict[str, int] = {code: 0 for code in LLAMAGUARD_HAZARD_CODES}
    hazard_examples: Dict[str, List[Dict[str, Any]]] = {code: [] for code in LLAMAGUARD_HAZARD_CODES}

    non_empty_hazards_raw: List[Any] = []
    non_empty_hazards_count = 0
    for rec in records:
        hazards = _parse_hazards(rec.get("hazards"))
        if not hazards:
            continue
        non_empty_hazards_count += 1
        if len(non_empty_hazards_raw) < 5:
            non_empty_hazards_raw.append(rec.get("hazards"))
        for code in hazards:
            hazard_counts.setdefault(code, 0)
            hazard_counts[code] += 1
            if code in hazard_examples and len(hazard_examples[code]) >= examples_per_hazard:
                continue
            example = {
                "hazard_code": code,
                "run_dir": str(metadata["run_dir"]),
                "prompt_id": rec.get("prompt_id"),
                "sample_id": rec.get("sample_id"),
                "category": rec.get("category"),
                "prompt": rec.get("prompt", ""),
                "generation": rec.get("generation", ""),
                "unsafe_score": rec.get("unsafe_score"),
                "hazards": hazards,
                "prompt_variant": metadata.get("prompt_variant"),
                "baseline_generation": None,
            }
            baseline_key = (
                metadata.get("experiment_slug"),
                metadata.get("prompt_source"),
                metadata.get("prompt_variant"),
                str(rec.get("prompt_id")),
            )
            baseline_rec = baseline_lookup.get(baseline_key)
            if baseline_rec:
                example["baseline_generation"] = baseline_rec.get("generation", baseline_rec.get("completion", ""))
            hazard_examples.setdefault(code, [])
            if len(hazard_examples[code]) < examples_per_hazard:
                hazard_examples[code].append(example)

    if total > 0:
        top_hazards = sorted(hazard_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        LOGGER.info(
            "Top hazards for %s: %s",
            metadata.get("run_dir"),
            ", ".join(f"{code}={count}/{total}" for code, count in top_hazards if count > 0) or "none",
        )
    else:
        LOGGER.info("No records to analyze for %s", metadata.get("run_dir"))

    hazards_report: List[Dict[str, Any]] = []
    for code in LLAMAGUARD_HAZARD_CODES:
        count = hazard_counts.get(code, 0)
        entry = _hazard_entry(count, total, code)
        entry["examples"] = hazard_examples.get(code, [])
        hazards_report.append(entry)

    # Fallback: if we have hazard counts but no examples (e.g., due to schema quirks),
    # make a second pass to populate examples directly from the raw records.
    if examples_per_hazard > 0:
        for entry in hazards_report:
            if entry.get("unsafe", 0) > 0 and not entry.get("examples"):
                code = entry["hazard_code"]
                filled: List[Dict[str, Any]] = []
                for rec in records:
                    hazards = _parse_hazards(rec.get("hazards"))
                    if code not in hazards:
                        continue
                    example = {
                        "hazard_code": code,
                        "run_dir": str(metadata["run_dir"]),
                        "prompt_id": rec.get("prompt_id"),
                        "sample_id": rec.get("sample_id"),
                        "category": rec.get("category"),
                        "prompt": rec.get("prompt", ""),
                        "generation": rec.get("generation", rec.get("completion", "")),
                        "unsafe_score": rec.get("unsafe_score"),
                        "hazards": hazards,
                        "prompt_variant": metadata.get("prompt_variant"),
                        "baseline_generation": None,
                    }
                    baseline_key = (
                        metadata.get("experiment_slug"),
                        metadata.get("prompt_source"),
                        metadata.get("prompt_variant"),
                        str(rec.get("prompt_id")),
                    )
                    baseline_rec = baseline_lookup.get(baseline_key)
                    if baseline_rec:
                        example["baseline_generation"] = baseline_rec.get("generation", baseline_rec.get("completion", ""))
                    filled.append(example)
                    if len(filled) >= examples_per_hazard:
                        break
                entry["examples"] = filled

    LOGGER.info(
        "Hazard parsing for %s: %d/%d records had hazards; sample=%s",
        metadata.get("run_dir"),
        non_empty_hazards_count,
        total,
        non_empty_hazards_raw,
    )

    # Fallback: if all hazard counts are zero but hazard_summary.csv exists, use it.
    if all(item.get("unsafe", 0) == 0 for item in hazards_report):
        summary_path = Path(metadata["run_dir"]) / "hazard_summary.csv"
        if summary_path.exists():
            hazards_report = []
            fallback_unsafe = 0
            with summary_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    try:
                        unsafe_count = int(row.get("unsafe", 0))
                        unsafe_rate = float(row.get("unsafe_rate", 0.0))
                    except ValueError:
                        continue
                    fallback_unsafe += unsafe_count
                    hazards_report.append(
                        {
                            "hazard_code": row.get("hazard_code") or "",
                            "hazard_name": row.get("hazard_name") or "",
                            "unsafe": unsafe_count,
                            "rate": unsafe_rate,
                            "examples": [],
                        }
                    )
            if hazards_report:
                unsafe = fallback_unsafe

    report = {
        "run_dir": str(metadata["run_dir"]),
        "experiment_slug": metadata.get("experiment_slug"),
        "dataset_name": metadata.get("dataset_name"),
        "attack_method": metadata.get("attack_method"),
        "defense_method": metadata.get("defense_method"),
        "prompt_source": metadata.get("prompt_source"),
        "prompt_variant": metadata.get("prompt_variant"),
        "data_dir": metadata.get("data_dir"),
        "artifact_name": metadata.get("artifact_name"),
        "tensor_size": metadata.get("tensor_size"),
        "model_variant": metadata.get("model_variant"),
        # Baseline runs (safety disabled) should not be grouped by safety scale.
        # Filter variants (posthoc_filter, best_of_n, fk_steering) also have safety.enabled=false
        # but use the variant name as safety_scale so they remain distinguishable from true baseline.
        "safety_scale": metadata.get("safety_scale"),
        "t_start": metadata.get("t_start"),
        "t_end": metadata.get("t_end"),
        "critical_steps": metadata.get("critical_steps"),
        "gating_label": metadata.get("gating_label"),
        "perplexity": metadata.get("perplexity"),
        "perplexity_texts": metadata.get("perplexity_texts"),
        "total": total,
        "unsafe": unsafe,
        "unsafe_rate": round(unsafe / total, 6) if total else 0.0,
        "hazards": hazards_report,
    }
    # pass through extra metrics
    for k, v in metadata.items():
        if (
            k.startswith("hygiene_")
            or k.startswith("lexical_")
            or k.startswith("bertscore_")
            or k.startswith("mauve_")
            or k.startswith("refusal_")
            or k.startswith("degeneration_")
        ):
            report[k] = v
    # Collect top/bottom examples per numeric metric in records.
    metric_examples: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    metric_keys = ("embedding_similarity", "mmd2_rbf", "perplexity")
    for key in metric_keys:
        scored = []
        for rec in records:
            val = rec.get(key)
            if isinstance(val, (int, float)):
                entry = {
                    "prompt_id": rec.get("prompt_id"),
                    "prompt": rec.get("prompt", ""),
                    "generation": rec.get("generation", rec.get("completion", "")),
                    "metric": val,
                    "baseline_generation": None,
                }
                baseline_key = (
                    metadata.get("experiment_slug"),
                    metadata.get("prompt_source"),
                    metadata.get("prompt_variant"),
                    str(rec.get("prompt_id")),
                )
                baseline_rec = baseline_lookup.get(baseline_key)
                if baseline_rec:
                    entry["baseline_generation"] = baseline_rec.get("generation", baseline_rec.get("completion", ""))
                scored.append(entry)
        if not scored:
            continue
        scored.sort(key=lambda x: x["metric"])
        metric_examples[key] = {
            "bottom": scored[:examples_per_metric],
            "top": list(reversed(scored[-examples_per_metric:])),
        }

    report["metric_examples"] = metric_examples

    # Safety transition tracking versus baseline (if available).
    transitions = {
        "unsafe_to_safe": {"count": 0, "examples": []},
        "safe_to_unsafe": {"count": 0, "examples": []},
        "unsafe_to_unsafe": {"count": 0, "examples": []},
        "safe_to_safe": {"count": 0, "examples": []},
        "missing_baseline": 0,
    }
    if metadata.get("safety_enabled"):
        for rec in records:
            pid = str(rec.get("prompt_id"))
            baseline_key = (
                metadata.get("experiment_slug"),
                metadata.get("prompt_source"),
                metadata.get("prompt_variant"),
                pid,
            )
            baseline_rec = baseline_lookup.get(baseline_key)
            if not baseline_rec:
                transitions["missing_baseline"] += 1
                continue
            baseline_unsafe = bool(baseline_rec.get("unsafe"))
            current_unsafe = bool(rec.get("unsafe"))
            if baseline_unsafe and not current_unsafe:
                key = "unsafe_to_safe"
            elif not baseline_unsafe and current_unsafe:
                key = "safe_to_unsafe"
            elif baseline_unsafe and current_unsafe:
                key = "unsafe_to_unsafe"
            else:
                key = "safe_to_safe"
            transitions[key]["count"] += 1
            if len(transitions[key]["examples"]) < examples_per_transition:
                transitions[key]["examples"].append(
                    {
                        "prompt_id": pid,
                        "prompt": rec.get("prompt", ""),
                        "generation": rec.get("generation", rec.get("completion", "")),
                        "baseline_generation": baseline_rec.get("generation", baseline_rec.get("completion", "")),
                        "baseline_unsafe": baseline_unsafe,
                        "current_unsafe": current_unsafe,
                        "hazards": _parse_hazards(rec.get("hazards")),
                    }
                )

    report["safety_transitions"] = transitions
    return report


def _aggregate_by_tensor(
    run_reports: Sequence[Dict[str, Any]],
    *,
    jailbreak_split: bool = False,
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for run in run_reports:
        tensor_key = _tensor_key_from_report(run, jailbreak_split=jailbreak_split)
        entry = grouped.setdefault(
            tensor_key,
            {
                "artifact_name": run.get("artifact_name"),
                "tensor_size": run.get("tensor_size"),
                "dataset_name": run.get("dataset_name"),
                "attack_method": run.get("attack_method"),
                "defense_method": run.get("defense_method"),
                "prompt_source": run.get("prompt_source"),
                "prompt_variant": run.get("prompt_variant"),
                "runs": [],
            },
        )
        hazards_map = {
            hazard["hazard_code"]: hazard["rate"]
            for hazard in run.get("hazards", [])
        }
        entry["runs"].append(
            {
                "run_dir": run["run_dir"],
                "safety_scale": run.get("safety_scale"),
                "unsafe_rate": run.get("unsafe_rate"),
                "t_start": run.get("t_start"),
                "t_end": run.get("t_end"),
                "critical_steps": run.get("critical_steps"),
                "gating_label": run.get("gating_label"),
                "perplexity": run.get("perplexity"),
                "perplexity_texts": run.get("perplexity_texts"),
                "hazard_rates": hazards_map,
            }
        )
    for entry in grouped.values():
        entry["runs"].sort(
            key=lambda item: (
                item["gating_label"] or "",
                item["safety_scale"] is None,
                item["safety_scale"],
            )
        )
    return grouped


def _write_csv(rows: Iterable[Iterable[Any]], header: Sequence[str], path: Path) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def generate_report(
    run_dirs: Sequence[Path],
    output_dir: Path,
    examples_per_hazard: int,
    examples_per_metric: int,
    examples_per_transition: int,
    render_hazard_charts: bool = False,
    only_embedding: bool = False,
    *,
    jailbreak_split: bool = False,
) -> None:
    LOGGER.info("Starting hazard report for %d run(s) into %s", len(run_dirs), output_dir)
    _ensure_dir(output_dir)
    run_reports: List[Dict[str, Any]] = []
    overall_rows: List[List[Any]] = []
    hazard_rows: List[List[Any]] = []
    hazard_records: List[Dict[str, Any]] = []
    global_examples: Dict[tuple, List[Dict[str, Any]]] = {}

    run_records_map: Dict[Path, List[Dict[str, Any]]] = {}
    run_entries: List[tuple[Path, Dict[str, Any], List[Dict[str, Any]]]] = []
    for run_dir in run_dirs:
        LOGGER.info("Processing run_dir=%s", run_dir)
        metadata = _parse_run_metadata(run_dir)
        if not metadata:
            continue
        records = _score_records(run_dir)
        run_records_map[run_dir] = records
        run_entries.append((run_dir, metadata, records))

    # Build lookup for baseline generations by prompt_id to attach to examples.
    baseline_lookup: Dict[tuple, Dict[str, Any]] = {}
    for run_dir, records in run_records_map.items():
        meta = _parse_run_metadata(run_dir)
        if not meta or meta.get("safety_enabled"):
            continue
        for rec in records:
            key = (
                meta.get("experiment_slug"),
                meta.get("prompt_source"),
                meta.get("prompt_variant"),
                str(rec.get("prompt_id")),
            )
            if key not in baseline_lookup:
                baseline_lookup[key] = rec

    for run_dir, metadata, records in run_entries:
        if only_embedding:
            total, unsafe, unsafe_rate = _compute_unsafe_summary(records)
            if not records:
                LOGGER.warning("No records found for %s; embedding report will omit unsafe rates.", run_dir)
            report = dict(metadata)
            report.update(
                {
                    "run_dir": str(metadata["run_dir"]),
                    "total": total,
                    "unsafe": unsafe,
                    "unsafe_rate": unsafe_rate,
                }
            )
            report["embedding_similarity"] = _load_alignment_similarity(run_dir)
            report.update(_load_jailbreak_summary(run_dir))
            run_reports.append(report)
            continue
        if not records:
            LOGGER.warning("No records found for %s; skipping.", run_dir)
            continue
        LOGGER.info("Loaded %d safety records for %s", len(records), run_dir)
        report = _build_run_report(
            records,
            metadata,
            examples_per_hazard,
            examples_per_metric,
            examples_per_transition,
            baseline_lookup,
        )
        report["embedding_similarity"] = _load_alignment_similarity(run_dir)
        report.update(_load_jailbreak_summary(run_dir))
        run_reports.append(report)

    _populate_mmd_metrics(run_reports, jailbreak_split=jailbreak_split)
    _generate_embedding_report(
        run_reports,
        output_dir,
        examples_per_hazard,
        jailbreak_split=jailbreak_split,
    )
    if only_embedding:
        LOGGER.info("Embedding-only report complete; skipping hazard summaries.")
        return

    baseline_map: Dict[str, float] = {}
    for report in run_reports:
        if report.get("safety_scale") is None:
            tensor_key = _tensor_key_from_report(report, jailbreak_split=jailbreak_split)
            baseline_map[tensor_key] = report.get("unsafe_rate", 0.0)
            if report.get("defense_method"):
                no_def_key = _tensor_key_without_defense(report, jailbreak_split=jailbreak_split)
                baseline_map.setdefault(no_def_key, report.get("unsafe_rate", 0.0))

    metric_delta_keys = [
        "bertscore_precision_mean",
        "bertscore_recall_mean",
        "bertscore_f1_mean",
        "mauve_exp_vs_ref",
        "refusal_rate",
        "non_answer_rate",
        "degeneration_rate",
        "lexical_distinct_n2_mean",
        "lexical_repeat_n2_mean",
        "lexical_max_repeated_span_mean",
    ]
    _attach_baseline_deltas(
        run_reports,
        metric_delta_keys,
        only_for_prompt_variant="benign",
        jailbreak_split=jailbreak_split,
    )  # note: we only compute deltas for benign prompts, not harmful ones
    for report in run_reports:
        report["delta_lexical_distinct_n2_mean"] = report.get("lexical_distinct_n2_mean_delta_vs_baseline")
        report["delta_lexical_repeat_n2_mean"] = report.get("lexical_repeat_n2_mean_delta_vs_baseline")
        report["delta_lexical_max_repeated_span_mean"] = report.get(
            "lexical_max_repeated_span_mean_delta_vs_baseline"
        )

    extra_metric_keys = set()
    for report in run_reports:
        for k in report.keys():
            if (
                k.startswith("hygiene_")
                or k.startswith("lexical_")
                or k.startswith("bertscore_")
                or k.startswith("mmd_")
                or k.startswith("mauve_")
                or k.startswith("refusal_")
                or k.startswith("degeneration_")
                or k.endswith("_delta_vs_baseline")
                or k.startswith("delta_lexical_")
            ):
                extra_metric_keys.add(k)
    sorted_extra_keys = sorted(list(extra_metric_keys))

    baseline_hazard_map: Dict[tuple, float] = {}
    baseline_hazard_by_tensor: Dict[tuple, float] = {}
    baseline_hazard_values: Dict[tuple, List[float]] = {}
    for report in run_reports:
        if report.get("safety_scale") is None:
            tensor_key = _tensor_key_from_report(report, jailbreak_split=jailbreak_split)
            gating_label = report.get("gating_label")
            for hazard in report.get("hazards", []):
                rate = hazard.get("rate", 0.0)
                baseline_hazard_map[(tensor_key, gating_label, hazard.get("hazard_code"))] = rate
                if report.get("defense_method"):
                    no_def_key = _tensor_key_without_defense(report, jailbreak_split=jailbreak_split)
                    baseline_hazard_map.setdefault((no_def_key, gating_label, hazard.get("hazard_code")), rate)
                values_key = (tensor_key, hazard.get("hazard_code"))
                baseline_hazard_values.setdefault(values_key, []).append(rate)
                if report.get("defense_method"):
                    no_def_values_key = (no_def_key, hazard.get("hazard_code"))
                    baseline_hazard_values.setdefault(no_def_values_key, []).append(rate)
    for key, values in baseline_hazard_values.items():
        if not values:
            continue
        baseline_hazard_by_tensor[key] = float(sum(values) / len(values))

    for report in run_reports:
        tensor_key = _tensor_key_from_report(report, jailbreak_split=jailbreak_split)
        baseline_rate = baseline_map.get(tensor_key)
        if baseline_rate is None:
            fallback_key = _tensor_key_from_report(
                {
                    "artifact_name": None,
                    "experiment_slug": report.get("experiment_slug"),
                    "dataset_name": report.get("dataset_name"),
                    "attack_method": report.get("attack_method"),
                    "defense_method": report.get("defense_method"),
                    "prompt_variant": report.get("prompt_variant"),
                },
                jailbreak_split=jailbreak_split,
            )
            baseline_rate = baseline_map.get(fallback_key)
        if baseline_rate is None and report.get("defense_method"):
            fallback_key = _tensor_key_without_defense(report, jailbreak_split=jailbreak_split)
            baseline_rate = baseline_map.get(fallback_key)
            if baseline_rate is None:
                fallback_key = _tensor_key_without_defense(
                    {
                        "artifact_name": None,
                        "experiment_slug": report.get("experiment_slug"),
                        "dataset_name": report.get("dataset_name"),
                        "attack_method": report.get("attack_method"),
                        "defense_method": report.get("defense_method"),
                        "prompt_variant": report.get("prompt_variant"),
                    },
                    jailbreak_split=jailbreak_split,
                )
                baseline_rate = baseline_map.get(fallback_key)
        unsafe_delta = None
        if baseline_rate is not None and report.get("unsafe_rate") is not None:
            unsafe_delta = report["unsafe_rate"] - baseline_rate

        transitions = report.get("safety_transitions") or {}
        transition_counts = {
            "unsafe_to_safe": transitions.get("unsafe_to_safe", {}).get("count", 0),
            "safe_to_unsafe": transitions.get("safe_to_unsafe", {}).get("count", 0),
            "unsafe_to_unsafe": transitions.get("unsafe_to_unsafe", {}).get("count", 0),
            "safe_to_safe": transitions.get("safe_to_safe", {}).get("count", 0),
            "missing_baseline": transitions.get("missing_baseline", 0),
        }

        row = [
            report["run_dir"],
            report.get("experiment_slug"),
        ]
        if jailbreak_split:
            row.extend(
                [
                    report.get("dataset_name"),
                    report.get("attack_method"),
                    report.get("defense_method"),
                ]
            )
        row.extend(
            [
                report.get("prompt_source"),
                report.get("prompt_variant"),
                report.get("artifact_name"),
                report.get("tensor_size"),
                report.get("model_variant"),
                report.get("safety_scale"),
                report.get("t_start"),
                report.get("t_end"),
                report.get("critical_steps"),
                report.get("gating_label"),
                report.get("perplexity"),
                report.get("perplexity_texts"),
                report.get("embedding_similarity"),
                report.get("mmd2_rbf"),
                transition_counts["unsafe_to_safe"],
                transition_counts["safe_to_unsafe"],
                transition_counts["unsafe_to_unsafe"],
                transition_counts["safe_to_safe"],
                transition_counts["missing_baseline"],
                unsafe_delta,
                report["total"],
                report["unsafe"],
                report["unsafe_rate"],
                report.get("harmbench_asr"),
                report.get("advbench_asr"),
                report.get("strong_reject_score"),
                report.get("strong_reject_count"),
            ]
        )
        # append extra metrics in fixed order
        for k in sorted_extra_keys:
            row.append(report.get(k))
        overall_rows.append(row)

        for hazard in report.get("hazards", []):
            hazard_baseline = baseline_hazard_map.get(
                (tensor_key, report.get("gating_label"), hazard.get("hazard_code"))
            )
            if hazard_baseline is None:
                hazard_baseline = baseline_hazard_by_tensor.get((tensor_key, hazard.get("hazard_code")))
            if hazard_baseline is None and report.get("defense_method"):
                no_def_key = _tensor_key_without_defense(report, jailbreak_split=jailbreak_split)
                hazard_baseline = baseline_hazard_map.get(
                    (no_def_key, report.get("gating_label"), hazard.get("hazard_code"))
                )
                if hazard_baseline is None:
                    hazard_baseline = baseline_hazard_by_tensor.get((no_def_key, hazard.get("hazard_code")))
            hazard_delta = None
            if hazard_baseline is not None:
                hazard_delta = hazard["rate"] - hazard_baseline
            hazard_row = {
                "run_dir": report["run_dir"],
                "experiment_slug": report.get("experiment_slug"),
                "dataset_name": report.get("dataset_name"),
                "attack_method": report.get("attack_method"),
                "defense_method": report.get("defense_method"),
                "prompt_variant": report.get("prompt_variant"),
                "artifact_name": report.get("artifact_name"),
                "tensor_size": report.get("tensor_size"),
                "safety_scale": report.get("safety_scale"),
                "t_start": report.get("t_start"),
                "t_end": report.get("t_end"),
                "critical_steps": report.get("critical_steps"),
                "gating_label": report.get("gating_label"),
                "embedding_similarity": report.get("embedding_similarity"),
                "mmd2_rbf": report.get("mmd2_rbf"),
                "unsafe_to_safe": transition_counts["unsafe_to_safe"],
                "safe_to_unsafe": transition_counts["safe_to_unsafe"],
                "unsafe_to_unsafe": transition_counts["unsafe_to_unsafe"],
                "safe_to_safe": transition_counts["safe_to_safe"],
                "missing_baseline": transition_counts["missing_baseline"],
                "unsafe_rate_delta_vs_baseline": unsafe_delta,
                "hazard_unsafe_rate_safe": hazard["rate"],
                "hazard_unsafe_rate_delta_vs_baseline": hazard_delta,
                "hazard_code": hazard["hazard_code"],
                "hazard_name": hazard["hazard_name"],
                "unsafe": hazard["unsafe"],
                "unsafe_rate": hazard["rate"],
            }
            hazard_records.append(hazard_row)
            hazard_row_list = [
                hazard_row["run_dir"],
                hazard_row["experiment_slug"],
            ]
            if jailbreak_split:
                hazard_row_list.extend(
                    [
                        hazard_row.get("dataset_name"),
                        hazard_row.get("attack_method"),
                        hazard_row.get("defense_method"),
                    ]
                )
            hazard_row_list.extend(
                [
                    hazard_row["prompt_variant"],
                    hazard_row["artifact_name"],
                    hazard_row["tensor_size"],
                    hazard_row["safety_scale"],
                    hazard_row["t_start"],
                    hazard_row["t_end"],
                    hazard_row["critical_steps"],
                    hazard_row["gating_label"],
                    hazard_row["embedding_similarity"],
                    hazard_row["mmd2_rbf"],
                    hazard_row["unsafe_to_safe"],
                    hazard_row["safe_to_unsafe"],
                    hazard_row["unsafe_to_unsafe"],
                    hazard_row["safe_to_safe"],
                    hazard_row["missing_baseline"],
                    hazard_row["unsafe_rate_delta_vs_baseline"],
                    hazard_row["hazard_unsafe_rate_safe"],
                    hazard_row["hazard_unsafe_rate_delta_vs_baseline"],
                    hazard_row["hazard_code"],
                    hazard_row["hazard_name"],
                    hazard_row["unsafe"],
                    hazard_row["unsafe_rate"],
                ]
            )
            hazard_rows.append(hazard_row_list)
            examples = hazard.get("examples", [])
            if not examples:
                continue
            for example in examples:
                code = hazard["hazard_code"]
                key = (
                    code,
                    report.get("artifact_name"),
                    report.get("experiment_slug"),
                    report.get("dataset_name"),
                    report.get("attack_method"),
                    report.get("defense_method"),
                    report.get("safety_scale"),
                    report.get("prompt_variant"),
                )
                global_list = global_examples.setdefault(key, [])
                if len(global_list) >= examples_per_hazard:
                    continue
                example_entry = dict(example)
                example_entry["hazard_code"] = code
                example_entry["hazard_name"] = hazard["hazard_name"]
                example_entry["artifact_name"] = report.get("artifact_name")
                example_entry["experiment_slug"] = report.get("experiment_slug")
                example_entry["dataset_name"] = report.get("dataset_name")
                example_entry["attack_method"] = report.get("attack_method")
                example_entry["defense_method"] = report.get("defense_method")
                example_entry["safety_scale"] = report.get("safety_scale")
                example_entry["prompt_variant"] = report.get("prompt_variant")
                global_list.append(example_entry)

    run_reports.sort(
        key=lambda item: (
            item.get("artifact_name") or "",
            item.get("prompt_variant") or "",
            str(item.get("safety_scale") if item.get("safety_scale") is not None else -1),
        )
    )
    aggregated = _aggregate_by_tensor(run_reports, jailbreak_split=jailbreak_split)

    report_path = output_dir / "hazard_report.json"
    report_payload = {"runs": run_reports, "tensors": aggregated}
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote hazard_report.json with %d run(s)", len(run_reports))

    overall_header = [
        "run_dir",
        "experiment_slug",
    ]
    if jailbreak_split:
        overall_header.extend(["dataset_name", "attack_method", "defense_method"])
    overall_header.extend(
        [
            "prompt_source",
            "prompt_variant",
            "artifact_name",
            "tensor_size",
            "model_variant",
            "safety_scale",
            "t_start",
            "t_end",
            "critical_steps",
            "gating_label",
            "perplexity",
            "perplexity_texts",
            "embedding_similarity",
            "mmd2_rbf",
            "unsafe_to_safe",
            "safe_to_unsafe",
            "unsafe_to_unsafe",
            "safe_to_safe",
            "missing_baseline",
            "unsafe_rate_delta_vs_baseline",
            "total",
            "unsafe",
            "unsafe_rate",
            "harmbench_asr",
            "advbench_asr",
            "strong_reject_score",
            "strong_reject_count",
        ]
    )
    overall_header += sorted_extra_keys

    LOGGER.info("Writing overall_rates.csv with %d rows", len(overall_rows))
    _write_csv(overall_rows, overall_header, output_dir / "overall_rates.csv")
    LOGGER.info("Wrote overall_rates.csv (%d rows)", len(overall_rows))

    hazard_header = [
        "run_dir",
        "experiment_slug",
    ]
    if jailbreak_split:
        hazard_header.extend(["dataset_name", "attack_method", "defense_method"])
    hazard_header.extend(
        [
            "prompt_variant",
            "artifact_name",
            "tensor_size",
            "safety_scale",
            "t_start",
            "t_end",
            "critical_steps",
            "gating_label",
            "embedding_similarity",
            "mmd2_rbf",
            "unsafe_to_safe",
            "safe_to_unsafe",
            "unsafe_to_unsafe",
            "safe_to_safe",
            "missing_baseline",
            "unsafe_rate_delta_vs_baseline",
            "hazard_unsafe_rate_safe",
            "hazard_unsafe_rate_delta_vs_baseline",
            "hazard_code",
            "hazard_name",
            "unsafe",
            "unsafe_rate",
        ]
    )
    LOGGER.info("Writing hazard_rates.csv with %d rows", len(hazard_rows))
    _write_csv(hazard_rows, hazard_header, output_dir / "hazard_rates.csv")
    LOGGER.info("Wrote hazard_rates.csv (%d rows)", len(hazard_rows))

    examples_path = output_dir / "hazard_examples.json"
    examples_payload: Dict[str, Any] = {}
    for key, entries in global_examples.items():
        if not entries:
            continue
        (
            code,
            artifact_name,
            experiment_slug,
            dataset_name,
            attack_method,
            defense_method,
            safety_scale,
            prompt_variant,
        ) = key
        prompt_tag = f"|prompt_variant={prompt_variant}" if prompt_variant else ""
        split_tag = ""
        if jailbreak_split:
            split_tag = (
                f"|dataset={dataset_name or ''}|attack={attack_method or ''}|defense={defense_method or ''}"
            )
        payload_key = (
            f"{code}|{artifact_name or 'baseline'}|{experiment_slug or ''}"
            f"{split_tag}|scale={safety_scale}{prompt_tag}"
        )
        examples_payload[payload_key] = entries
    # Include safety transition examples (per run) if present.
    for report in run_reports:
        transitions = report.get("safety_transitions") or {}
        if not transitions:
            continue
        split_tag = ""
        if jailbreak_split:
            split_tag = (
                f"|dataset={report.get('dataset_name') or ''}"
                f"|attack={report.get('attack_method') or ''}"
                f"|defense={report.get('defense_method') or ''}"
            )
        base_key = (
            f"transitions|{report.get('artifact_name') or 'baseline'}|"
            f"{report.get('experiment_slug') or ''}{split_tag}|"
            f"prompt_variant={report.get('prompt_variant') or ''}|"
            f"scale={report.get('safety_scale')}"
        )
        for cat in ("unsafe_to_safe", "safe_to_unsafe", "unsafe_to_unsafe", "safe_to_safe"):
            data = transitions.get(cat) or {}
            examples = data.get("examples") or []
            if not examples:
                continue
            examples_payload[f"{base_key}|{cat}"] = examples
    LOGGER.info("Writing hazard_examples.json with grouped examples: %d groups", len(examples_payload))
    examples_path.write_text(json.dumps(examples_payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote hazard_examples.json")

    
    LOGGER.info("Rendering combined metrics charts...")
    _render_combined_metrics(run_reports, output_dir / "figs_combined", extra_metric_keys)


# Help from Gemini to create combined metrics plots
def _render_combined_metrics(
    run_reports: List[Dict[str, Any]],
    output_dir: Path,
    extra_metric_keys: Iterable[str],
) -> None:
    """
    Renders combined visualizations for multiple metrics.
    Plots bar charts where:
      - X-axis: Tensor Size (ordered numerically).
      - Bars grouped by: Safety Scale (eta).
      - Y-axis: Metric Value.
    One chart per metric per (Experiment, Prompt Variant, Gating Label).
    """
    if not run_reports:
        return
    df = pd.DataFrame(run_reports)
    if df.empty:
        return
    
    # Ensure columns exist
    df["tensor_size"] = df["tensor_size"].fillna("baseline")
    df["safety_scale"] = df["safety_scale"].fillna("baseline")
    df["gating_label"] = df["gating_label"].fillna("ts_full_full")
    df["prompt_variant"] = df["prompt_variant"].fillna("unknown")
    df["experiment_slug"] = df["experiment_slug"].fillna("unknown")

    # Metrics to plot
    core_metrics = ["unsafe_rate", "perplexity", "embedding_similarity", "mmd2_rbf"]
    metrics_to_plot = core_metrics + sorted(list(extra_metric_keys))

    # Helper to sort tensor sizes numerically if possible
    def _sort_tensor_sizes(sizes):
        def _key(s):
            s_str = str(s)
            if s_str.isdigit():
                return (0, int(s_str))
            return (1, s_str)
        return sorted(sizes, key=_key)

    for (exp, variant, gating), group in df.groupby(["experiment_slug", "prompt_variant", "gating_label"]):
        group_dir = output_dir / _sanitize_for_path(exp) / _sanitize_for_path(variant) / _sanitize_for_path(gating)
        group_dir.mkdir(parents=True, exist_ok=True)

        tensor_sizes = _sort_tensor_sizes(group["tensor_size"].unique())
        # Filter scales: ignore baseline if it's mixed with numerics for the grouping, 
        # or handle it carefully. Usually we want to see trend with increasing scale.
        # But 'baseline' is usually scale=None. 
        # We'll treat 'baseline' as a separate group or 0? 
        # Let's just use string representation for grouping.
        
        # We want to show increasing tensor size on X axis.
        # Inside each x-tick, we want bars for each scale.
        
        scales = sorted(group["safety_scale"].unique(), key=lambda x: (str(x) != "baseline", try_float(x)))

        for metric in metrics_to_plot:
            if metric not in group.columns:
                continue
            numeric_values = pd.to_numeric(group[metric], errors="coerce")
            if numeric_values.notna().sum() == 0:
                continue
            subset = group.copy()
            subset["_metric"] = numeric_values
            # Drop NaNs for this metric
            subset = subset.dropna(subset=["_metric"])
            if subset.empty:
                continue

            # Pivot: Index=TensorSize, Columns=Scale, Values=Metric
            pivot = subset.pivot_table(
                index="tensor_size", 
                columns="safety_scale", 
                values="_metric", 
                aggfunc="mean"
            )
            # Reindex to ensure sorted order
            pivot = pivot.reindex(index=tensor_sizes, columns=scales)
            if pivot.dropna(how='all').empty:
                continue

            # Plot
            fig, ax = plt.subplots(figsize=(max(8, len(tensor_sizes) * 2), 6))
            
            # Simple bar plot from pandas
            pivot.plot(kind="bar", ax=ax, width=0.8)
            
            ax.set_title(f"{metric} by Tensor Size & Scale\n({exp} | {variant} | {gating})")
            ax.set_ylabel(metric)
            ax.set_xlabel("Tensor Size")
            plt.xticks(rotation=0)
            plt.legend(title="Scale (Eta)", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            out_file = group_dir / f"{metric}_combined.png"
            plt.savefig(out_file)
            plt.close(fig)

            if metric in ("unsafe_rate", "perplexity"):
                flipped = subset.pivot_table(
                    index="safety_scale",
                    columns="tensor_size",
                    values="_metric",
                    aggfunc="mean",
                )
                flipped = flipped.reindex(index=scales, columns=tensor_sizes)
                if flipped.dropna(how="all").empty:
                    continue

                fig_flip, ax_flip = plt.subplots(figsize=(max(8, len(scales) * 2), 6))
                flipped.plot(kind="bar", ax=ax_flip, width=0.8)

                ax_flip.set_title(f"{metric} by Scale & Tensor Size\n({exp} | {variant} | {gating})")
                ax_flip.set_ylabel(metric)
                ax_flip.set_xlabel("Scale (Eta)")
                plt.xticks(rotation=0)
                plt.legend(title="Tensor Size", bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.grid(True, axis="y", linestyle="--", alpha=0.5)
                plt.tight_layout()

                flipped_out_file = group_dir / f"{metric}_combined_flipped.png"
                plt.savefig(flipped_out_file)
                plt.close(fig_flip)

def try_float(x):
    try:
        return float(x)
    except:
        return -1.0

def _render_line_charts(hazard_records: List[Dict[str, Any]], figs_root: Path) -> None:
    if not hazard_records:
        LOGGER.warning("No hazard records available for plotting; skipping figures.")
        return
    df = pd.DataFrame(hazard_records)
    # Baseline rows have safety_scale=None; keep a copy for reference lines.
    baseline_df = df[df["safety_scale"].isna()].copy()
    # Drop rows without a numeric safety_scale (cannot plot)
    df = df.dropna(subset=["safety_scale"])
    if df.empty:
        return

    def _tensor_key(row: pd.Series) -> str:
        art = row.get("artifact_name")
        prompt_variant = row.get("prompt_variant")
        if pd.isna(art) or art is None:
            base = f"{row.get('experiment_slug', 'baseline')}"
        else:
            base = str(art)
        if pd.isna(prompt_variant) or prompt_variant is None:
            return base
        return f"{base}|pv={prompt_variant}"

    df["tensor_key"] = df.apply(_tensor_key, axis=1)
    df["gating_label"] = df["gating_label"].fillna("ts_full_full")
    baseline_df["tensor_key"] = baseline_df.apply(_tensor_key, axis=1)
    baseline_df["gating_label"] = baseline_df["gating_label"].fillna("ts_full_full")

    for (tensor_key, gating_label, hazard_code, hazard_name), group in df.groupby(
        ["tensor_key", "gating_label", "hazard_code", "hazard_name"]
    ):
        # Aggregate mean unsafe_rate if multiple runs share the same scale
        agg = (
            group.groupby("safety_scale")["unsafe_rate"]
            .mean()
            .reset_index()
            .sort_values("safety_scale")
        )
        if agg.empty:
            continue

        fig_dir = figs_root / _sanitize_for_path(str(tensor_key)) / _sanitize_for_path(str(gating_label))
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(agg["safety_scale"], agg["unsafe_rate"], marker="o")
        # Add baseline reference line if present for this tensor/gating/hazard
        base_mask = (
            (baseline_df["tensor_key"] == tensor_key)
            & (baseline_df["gating_label"] == gating_label)
            & (baseline_df["hazard_code"] == hazard_code)
        )
        if not baseline_df[base_mask].empty:
            base_rate = baseline_df[base_mask]["unsafe_rate"].mean()
            plt.axhline(base_rate, color="red", linestyle="--", linewidth=1.0, label="baseline")
            plt.legend()
        plt.title(f"{tensor_key} | {gating_label} | {hazard_name}")
        plt.xlabel("safety scale")
        plt.ylabel("unsafe rate")
        plt.grid(True, linestyle="--", alpha=0.5)
        fig_path = fig_dir / f"{_sanitize_for_path(hazard_name)}.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()


def _hazard_tensor_key(row: pd.Series) -> str:
    art = row.get("artifact_name")
    prompt_variant = row.get("prompt_variant")
    if pd.isna(art) or art is None:
        base = f"{row.get('experiment_slug', 'baseline')}"
    else:
        base = str(art)
    if pd.isna(prompt_variant) or prompt_variant is None:
        return base
    return f"{base}|pv={prompt_variant}"


def _render_hazard_heatmaps(hazard_records: List[Dict[str, Any]], figs_root: Path) -> None:
    if not hazard_records:
        LOGGER.warning("No hazard records available for heatmaps; skipping.")
        return
    df = pd.DataFrame(hazard_records)
    if df.empty:
        return
    df["tensor_key"] = df.apply(_hazard_tensor_key, axis=1)
    df["gating_label"] = df["gating_label"].fillna("ts_full_full")
    df["safety_scale_label"] = df["safety_scale"].apply(lambda x: "baseline" if pd.isna(x) else str(x))
    for (tensor_key, gating_label), group in df.groupby(["tensor_key", "gating_label"]):
        pivot = group.pivot_table(
            index="hazard_name",
            columns="safety_scale_label",
            values="unsafe_rate",
            aggfunc="mean",
        ).fillna(0.0)
        if pivot.empty:
            continue
        pivot = pivot.reindex(columns=_sort_scale_labels(pivot.columns))
        baseline_rates = pivot["baseline"] if "baseline" in pivot.columns else None
        if baseline_rates is not None:
            delta = pivot.subtract(baseline_rates, axis=0)
            heatmap_data = delta
            cmap = "coolwarm"
            suffix = "delta"
        else:
            heatmap_data = pivot
            cmap = "viridis"
            suffix = "rate"
        fig_dir = figs_root / _sanitize_for_path(str(tensor_key)) / _sanitize_for_path(str(gating_label))
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, max(4, 0.3 * len(heatmap_data.index))))
        plt.imshow(heatmap_data.values, aspect="auto", cmap=cmap)
        plt.colorbar(label="unsafe rate delta" if baseline_rates is not None else "unsafe rate")
        plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
        plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=30, ha="right")
        plt.title(f"{tensor_key} | {gating_label} | hazard heatmap ({suffix})")
        plt.tight_layout()
        plt.savefig(fig_dir / f"hazard_heatmap_{suffix}.png")
        plt.close()




def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description="Summarize LlamaGuard hazard rates across runs.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        action="append",
        required=True,
        type=Path,
        help="Generation run directories that contain 'scores/' outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write hazard summaries.",
    )
    parser.add_argument(
        "--examples-per-hazard",
        type=int,
        default=10,
        help="Maximum number of prompt/completion examples to keep per hazard.",
    )
    parser.add_argument(
        "--examples-per-metric",
        type=int,
        default=5,
        help="Number of top/bottom examples to keep per metric (per run).",
    )
    parser.add_argument(
        "--examples-per-transition",
        type=int,
        default=5,
        help="Number of examples to keep per safety transition category (baseline vs safe).",
    )
    parser.add_argument(
        "--hazard-charts",
        action="store_true",
        help="Render per-hazard line/bar charts (off by default).",
    )
    parser.add_argument(
        "--only-embedding",
        action="store_true",
        help="Generate only the embedding similarity report (skip hazard summaries).",
    )
    parser.add_argument(
        "--jailbreak-split",
        action="store_true",
        help=(
            "Split baselines/aggregation by dataset + attack + defense (for jailbreak runs). "
            "Off by default to preserve legacy grouping."
        ),
    )
    args = parser.parse_args()
    run_dirs: List[Path] = []
    for group in args.run_dirs:
        for path in group:
            run_dirs.append(path.resolve())
    LOGGER.info("Generating hazard report for %d run directories", len(run_dirs))
    generate_report(
        run_dirs,
        args.output_dir.resolve(),
        examples_per_hazard=args.examples_per_hazard,
        examples_per_metric=args.examples_per_metric,
        examples_per_transition=args.examples_per_transition,
        render_hazard_charts=args.hazard_charts,
        only_embedding=args.only_embedding,
        jailbreak_split=args.jailbreak_split,
    )


if __name__ == "__main__":
    main()
