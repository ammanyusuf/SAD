#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


MODEL_CATALOG = {
    "llada-base": {
        "hf_id": "GSAI-ML/LLaDA-8B-Base",
        "dir_name": "LLaDA-8B-Base",
    },
    "llada-instruct": {
        "hf_id": "GSAI-ML/LLaDA-8B-Instruct",
        "dir_name": "LLaDA-8B-Instruct",
    },
    "dream-instruct": {
        "hf_id": "Dream-org/Dream-v0-Instruct-7B",
        "dir_name": "Dream-v0-Instruct-7B",
    },
    "mmada-mixcot": {
        "hf_id": "Gen-Verse/MMaDA-8B-MixCoT",
        "dir_name": "MMaDA-8B-MixCoT",
    },
    "dream-coder": {
        "hf_id": "Dream-org/Dream-Coder-v0-Instruct-7B",
        "dir_name": "Dream-Coder-v0-Instruct-7B",
    },
    "dreamon": {
        "hf_id": "Dream-org/DreamOn-v0-7B",
        "dir_name": "DreamOn-v0-7B",
    },
    "diffucoder-instruct": {
        "hf_id": "apple/DiffuCoder-7B-Instruct",
        "dir_name": "DiffuCoder-7B-Instruct",
    },
    "diffucoder-cpgrpo": {
        "hf_id": "apple/DiffuCoder-7B-cpGRPO",
        "dir_name": "DiffuCoder-7B-cpGRPO",
    },
    "qwen-2.5-7b-instruct": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "dir_name": "Qwen2.5-7B-Instruct",
    },
    "strongreject-15k-v1": {
        "hf_id": "qylu4156/strongreject-15k-v1",
        "dir_name": "strongreject-15k-v1",
    },
    "gemma-2b": {
        "hf_id": "google/gemma-2b",
        "dir_name": "gemma-2b",
    },
    "harmbench-llama-2-13b-cls": {
        "hf_id": "cais/HarmBench-Llama-2-13b-cls",
        "dir_name": "HarmBench-Llama-2-13b-cls",
    },
    "gpt2": {
        "hf_id": "openai-community/gpt2",
        "dir_name": "gpt2",
    },
    "gpt2-large": {
        "hf_id": "openai-community/gpt2-large",
        "dir_name": "gpt2-large",
    },
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _nonempty_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.iterdir())


def _snapshot_download(hf_id: str, target: Path, token: Optional[str]) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Missing huggingface_hub. Install with: pip install huggingface_hub"
        ) from exc

    snapshot_download(
        repo_id=hf_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        token=token,
    )


def _export_jbb(
    output_csv: Path,
    split: str,
    hf_id: str,
    token: Optional[str],
) -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise SystemExit("Missing datasets. Install with: pip install datasets") from exc

    ds = load_dataset(hf_id, "behaviors", split=split, token=token)  # type: ignore[arg-type]
    df = ds.to_pandas()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def _convert_jbb_to_dija(input_path: Path) -> List[Dict[str, str]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("JBB input must be a list of objects.")
    output: List[Dict[str, str]] = []
    for row in data:
        goal = (row.get("Goal") or row.get("goal") or "").strip()
        if not goal:
            continue
        item = {
            "behavior": str(row.get("Behavior") or row.get("behavior") or "").strip(),
            "goal": goal,
            "refined_goal": goal,
            "target": str(row.get("Target") or row.get("target") or "").strip(),
            "category": str(row.get("Category") or row.get("category") or "").strip(),
        }
        output.append(item)
    return output


def _convert_jbb_to_diffuguard(input_path: Path) -> List[Dict[str, str]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("JBB input must be a list of objects.")
    output: List[Dict[str, str]] = []
    for row in data:
        goal = (row.get("Goal") or row.get("goal") or "").strip()
        if not goal:
            continue
        item: Dict[str, str] = {
            "vanilla prompt": goal,
            "refined prompt": goal,
        }
        for key in ("Behavior", "Target", "Category", "Source"):
            if key in row:
                item[key.lower()] = str(row.get(key) or "").strip()
        output.append(item)
    return output


def _pick_prompt_text(item: Dict[str, str]) -> str:
    for key in (
        "prompt",
        "instruction",
        "goal",
        "query",
        "text",
        "behavior",
        "Behavior",
        "vanilla",
        "forbidden_prompt",
        "vanilla prompt",
        "vanilla_prompt",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _pick_refined_text(item: Dict[str, str]) -> str:
    for key in (
        "refined prompt",
        "refined_prompt",
        "refined",
        "rewrite",
        "rewritten",
        "refined_goal",
        "adversarial",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _convert_json_to_diffuguard(input_path: Path) -> List[Dict[str, str]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list of objects.")
    output: List[Dict[str, str]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        prompt = _pick_prompt_text(row)
        if not prompt:
            continue
        refined = _pick_refined_text(row) or prompt
        item: Dict[str, str] = {
            "vanilla prompt": prompt,
            "refined prompt": refined,
        }
        for key in ("Behavior", "Target", "Category", "Source"):
            if key in row:
                item[key.lower()] = str(row.get(key) or "").strip()
        output.append(item)
    return output


def _convert_json_to_dija(input_path: Path) -> List[Dict[str, str]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list of objects.")
    output: List[Dict[str, str]] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        prompt = _pick_prompt_text(row)
        if not prompt:
            continue
        refined = _pick_refined_text(row) or prompt
        output.append(
            {
                "BehaviorID": str(row.get("BehaviorID") or f"item_{idx}"),
                "Behavior": prompt,
                "Refined_behavior": refined,
            }
        )
    return output


def _convert_harmbench_csv(input_csv: Path) -> List[Dict[str, str]]:
    import csv

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("HarmBench CSV missing header.")
        output: List[Dict[str, str]] = []
        for row in reader:
            behavior = (row.get("Behavior") or "").strip()
            if not behavior:
                continue
            output.append(
                {
                    "BehaviorID": str(row.get("BehaviorID") or "").strip(),
                    "Behavior": behavior,
                    "Refined_behavior": str(row.get("Refined_behavior") or "").strip(),
                }
            )
        return output


def _convert_harmbench_csv_to_diffuguard(input_csv: Path) -> List[Dict[str, str]]:
    import csv

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("HarmBench CSV missing header.")
        output: List[Dict[str, str]] = []
        for row in reader:
            behavior = (row.get("Behavior") or "").strip()
            if not behavior:
                continue
            refined = (row.get("Refined_behavior") or "").strip() or behavior
            item: Dict[str, str] = {
                "vanilla prompt": behavior,
                "refined prompt": refined,
            }
            if "Category" in row:
                item["category"] = str(row.get("Category") or "").strip()
            if "Source" in row:
                item["source"] = str(row.get("Source") or "").strip()
            output.append(item)
        return output


def _csv_to_records(csv_path: Path) -> List[Dict[str, str]]:
    import csv

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("CSV header missing; cannot map columns.")
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def _export_hf_json(
    output_json: Path,
    hf_id: str,
    split: str,
    token: Optional[str],
    *,
    config_name: Optional[str] = None,
    streaming: bool = False,
) -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise SystemExit("Missing datasets. Install with: pip install datasets") from exc
    if config_name:
        ds = load_dataset(hf_id, config_name, split=split, token=token, streaming=streaming)  # type: ignore[arg-type]
    else:
        ds = load_dataset(hf_id, split=split, token=token, streaming=streaming)  # type: ignore[arg-type]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if streaming:
        records = [dict(row) for row in ds]
    else:
        records = ds.to_pandas().to_dict(orient="records")
    output_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_hf_dataset_cached(
    hf_id: str,
    split: str,
    cache_dir: Path,
    token: Optional[str],
    config_name: Optional[str] = None,
) -> None:
    try:
        from datasets import load_dataset, load_dataset_builder  # type: ignore
    except Exception as exc:
        raise SystemExit("Missing datasets. Install with: pip install datasets") from exc
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Some configs (e.g., MMLU auxiliary_train) only expose a train split.
    requested_split = split
    try:
        if config_name:
            builder = load_dataset_builder(
                hf_id, config_name, cache_dir=str(cache_dir), token=token
            )
        else:
            builder = load_dataset_builder(
                hf_id, cache_dir=str(cache_dir), token=token
            )
        available_splits = list(getattr(builder.info, "splits", {}).keys())
        if requested_split not in available_splits:
            if available_splits == ["train"] or set(available_splits) == {"train"}:
                requested_split = "train"
    except Exception:
        # If split detection fails, fall back to the requested split.
        requested_split = split
    if config_name:
        load_dataset(
            hf_id,
            config_name,
            split=requested_split,
            cache_dir=str(cache_dir),
            token=token,
        )  # type: ignore[arg-type]
    else:
        load_dataset(
            hf_id,
            split=requested_split,
            cache_dir=str(cache_dir),
            token=token,
        )  # type: ignore[arg-type]


def _write_subset_json(input_path: Path, output_path: Path, limit: int) -> int:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected list JSON for subset source: {input_path}")
    limit = max(int(limit), 0)
    subset = data[:limit] if limit else data
    output_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(subset)


def _run_diffuguard_refiner(
    repo_root: Path,
    *,
    model_path: str,
    attack_prompt: Path,
    output_json: Path,
    max_new_tokens: int,
) -> None:
    refine_main = repo_root / "src/third_party/DiffuGuard/refine_prompt/main.py"
    template_path = repo_root / "src/third_party/DiffuGuard/refine_prompt/redteam_prompt_template.txt"
    if not refine_main.exists():
        raise SystemExit(f"DiffuGuard refiner not found: {refine_main}")
    if not template_path.exists():
        raise SystemExit(f"DiffuGuard refine template not found: {template_path}")

    cmd = [
        sys.executable,
        str(refine_main),
        "--hf_model_path",
        str(model_path),
        "--prompt_template_path",
        str(template_path),
        "--attack_prompt",
        str(attack_prompt),
        "--output_json",
        str(output_json),
        "--max_new_tokens",
        str(max_new_tokens),
    ]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path
    print(f"[INFO] Running DiffuGuard refine_prompt on {attack_prompt.name} -> {output_json.name}")
    completed = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"DiffuGuard refine_prompt failed (exit={completed.returncode}) for input: {attack_prompt}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up jailbreak datasets + HF models.")
    parser.add_argument("--base-dir", default="~/scratch", help="Base scratch directory.")
    default_models = ",".join(MODEL_CATALOG.keys())
    parser.add_argument(
        "--models",
        default=default_models,
        help="Comma-separated model keys to download (defaults to all known models).",
    )
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads.")
    parser.add_argument("--skip-datasets", action="store_true", help="Skip dataset downloads.")
    parser.add_argument("--jbb-split", default="harmful", help="JBB split to export.")
    parser.add_argument(
        "--jbb-hf-id",
        default="JailbreakBench/JBB-Behaviors",
        help="HF dataset id for JBB.",
    )
    parser.add_argument(
        "--harmbench-csv",
        default=None,
        help="Override HarmBench CSV path (defaults to repo copy).",
    )
    parser.add_argument(
        "--strongreject-hf-id",
        default="walledai/StrongREJECT",
        help="HF dataset id for StrongREJECT.",
    )
    parser.add_argument(
        "--strongreject-config",
        default=None,
        help="Optional StrongREJECT config name (if required by the dataset).",
    )
    parser.add_argument(
        "--strongreject-split",
        default="train",
        help="StrongREJECT split to export.",
    )
    parser.add_argument(
        "--strongreject-json",
        default=None,
        help="Optional JSON path for StrongREJECT prompts (list of dicts).",
    )
    parser.add_argument(
        "--wildjailbreak-hf-id",
        default="allenai/wildjailbreak",
        help="Optional HF dataset id for WildJailbreak.",
    )
    parser.add_argument(
        "--wildjailbreak-config",
        default="train",
        help="WildJailbreak config name (dataset uses configs like 'train' or 'eval').",
    )
    parser.add_argument(
        "--wildjailbreak-streaming",
        dest="wildjailbreak_streaming",
        action="store_true",
        help="Load WildJailbreak via streaming to avoid Arrow casting issues.",
    )
    parser.add_argument(
        "--no-wildjailbreak-streaming",
        dest="wildjailbreak_streaming",
        action="store_false",
        help="Disable streaming for WildJailbreak.",
    )
    parser.set_defaults(wildjailbreak_streaming=True)
    parser.add_argument(
        "--wildjailbreak-split",
        default="train",
        help="WildJailbreak split to export if HF id provided.",
    )
    parser.add_argument(
        "--wildjailbreak-json",
        default=None,
        help="Optional JSON path for WildJailbreak prompts (list of dicts).",
    )
    parser.add_argument(
        "--advbench-hf-id",
        default="walledai/AdvBench",
        help="HF dataset id for AdvBench.",
    )
    parser.add_argument(
        "--advbench-split",
        default="train",
        help="AdvBench split to export.",
    )
    parser.add_argument(
        "--xstest-hf-id",
        default="walledai/XSTest",
        help="HF dataset id for XSTest.",
    )
    parser.add_argument(
        "--xstest-split",
        default="test",
        help="XSTest split to export.",
    )
    parser.add_argument(
        "--humaneval-hf-id",
        default="openai/openai_humaneval",
        help="HF dataset id for HumanEval.",
    )
    parser.add_argument(
        "--humaneval-split",
        default="test",
        help="HumanEval split to cache.",
    )
    parser.add_argument(
        "--gsm8k-hf-id",
        default="openai/gsm8k",
        help="HF dataset id for GSM8K.",
    )
    parser.add_argument(
        "--gsm8k-config",
        default="main",
        help="GSM8K config name.",
    )
    parser.add_argument(
        "--gsm8k-split",
        default="test",
        help="GSM8K split to cache.",
    )
    parser.add_argument(
        "--mmlu-hf-id",
        default="cais/mmlu",
        help="HF dataset id for MMLU.",
    )
    parser.add_argument(
        "--mmlu-config",
        default="all",
        help="MMLU config name.",
    )
    parser.add_argument(
        "--mmlu-all-configs",
        action="store_true",
        help="Cache every available MMLU config (required for lm_eval mmlu group).",
    )
    parser.add_argument(
        "--mmlu-split",
        default="test",
        help="MMLU split to cache.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Write summary to this path.",
    )
    parser.add_argument(
        "--setup-refined-prompts",
        action="store_true",
        help=(
            "Generate DIJA refined prompt JSONs for AdvBench, WildJailbreak, JBB, "
            "HarmBench, and StrongReject "
            "by calling DiffuGuard refine_prompt/main.py."
        ),
    )
    parser.add_argument(
        "--refine-model-path",
        default=None,
        help=(
            "Model path for DiffuGuard refiner. Defaults to downloaded "
            "qwen-2.5-7b-instruct, then ~/scratch/hf_models/Qwen2.5-7B-Instruct."
        ),
    )
    parser.add_argument(
        "--refine-max-new-tokens",
        type=int,
        default=200,
        help="max_new_tokens passed to DiffuGuard refine_prompt/main.py.",
    )
    parser.add_argument(
        "--wildjailbreak-refine-limit",
        type=int,
        default=1000,
        help="Number of WildJailbreak entries to refine (subset written to disk first).",
    )
    parser.add_argument(
        "--skip-existing-refined",
        action="store_true",
        help="Skip DiffuGuard refine_prompt generation when refined output JSON already exists.",
    )
    args = parser.parse_args()

    base_dir = Path(os.path.expanduser(args.base_dir))
    hf_models_dir = base_dir / "hf_models"
    hf_datasets_dir = base_dir / "hf_datasets"
    datasets_dir = base_dir / "jailbreak_datasets"
    _ensure_dir(hf_models_dir)
    _ensure_dir(hf_datasets_dir)
    _ensure_dir(datasets_dir)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    model_paths: Dict[str, Dict[str, str]] = {}
    if not args.skip_models:
        # Auto-include StrongREJECT base model if adapter is requested.
        if "strongreject-15k-v1" in requested_models and "gemma-2b" not in requested_models:
            requested_models.append("gemma-2b")
        for key in requested_models:
            spec = MODEL_CATALOG.get(key)
            if not spec:
                print(f"[WARN] Unknown model key: {key}")
                continue
            target = hf_models_dir / spec["dir_name"]
            if _nonempty_dir(target):
                print(f"[INFO] Model already present: {target}")
            else:
                print(f"[INFO] Downloading {spec['hf_id']} -> {target}")
                _snapshot_download(spec["hf_id"], target, token)
            model_paths[key] = {
                "model_path": str(target),
                "tokenizer_path": str(target),
            }

    datasets: Dict[str, str] = {}
    cached_hf_datasets: Dict[str, str] = {}
    repo_root = Path(__file__).resolve().parents[1]
    if not args.skip_datasets:
        jbb_csv = datasets_dir / f"jbb_behaviors_{args.jbb_split}.csv"
        if not jbb_csv.exists():
            print(f"[INFO] Downloading JBB -> {jbb_csv}")
            _export_jbb(jbb_csv, args.jbb_split, args.jbb_hf_id, token)
        else:
            print(f"[INFO] JBB CSV already present: {jbb_csv}")

        jbb_json_dija = datasets_dir / f"jbb_behaviors_{args.jbb_split}_dija.json"
        jbb_json_diffuguard = datasets_dir / f"jbb_behaviors_{args.jbb_split}_diffuguard.json"
        jbb_csv_json = datasets_dir / f"jbb_behaviors_{args.jbb_split}.json"
        jbb_data = _csv_to_records(jbb_csv)
        jbb_csv_json.write_text(json.dumps(jbb_data, indent=2, ensure_ascii=False), encoding="utf-8")

        jbb_json_dija.write_text(
            json.dumps(_convert_jbb_to_dija(jbb_csv_json), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        jbb_json_diffuguard.write_text(
            json.dumps(_convert_jbb_to_diffuguard(jbb_csv_json), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        datasets["jbb_csv"] = str(jbb_csv)
        datasets["jbb_json_dija"] = str(jbb_json_dija)
        datasets["jbb_json_diffuguard"] = str(jbb_json_diffuguard)

        if args.harmbench_csv:
            harmbench_csv = Path(os.path.expanduser(args.harmbench_csv))
        else:
            harmbench_csv = repo_root / "src/third_party/DIJA/benchmarks/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
        if harmbench_csv.exists():
            staged_harmbench = datasets_dir / "harmbench_behaviors_text_all.csv"
            if not staged_harmbench.exists():
                staged_harmbench.write_text(harmbench_csv.read_text(encoding="utf-8"), encoding="utf-8")
            harmbench_json = datasets_dir / "harmbench_behaviors_text_all.json"
            harmbench_json.write_text(
                json.dumps(_convert_harmbench_csv(staged_harmbench), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            harmbench_diffuguard = datasets_dir / "harmbench_prompts_diffuguard.json"
            harmbench_diffuguard.write_text(
                json.dumps(_convert_harmbench_csv_to_diffuguard(staged_harmbench), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            datasets["harmbench_csv"] = str(staged_harmbench)
            datasets["harmbench_json"] = str(harmbench_json)
            datasets["harmbench_diffuguard"] = str(harmbench_diffuguard)
        else:
            print(f"[WARN] HarmBench CSV not found: {harmbench_csv}")

        strongreject_raw: Optional[Path] = None
        if args.strongreject_json:
            strongreject_raw = Path(os.path.expanduser(args.strongreject_json))
            if not strongreject_raw.exists():
                print(f"[WARN] StrongReject JSON not found: {strongreject_raw}")
                strongreject_raw = None
        else:
            strongreject_raw = datasets_dir / "strongreject_raw.json"
            if not strongreject_raw.exists():
                print(f"[INFO] Downloading StrongReject -> {strongreject_raw}")
                _export_hf_json(
                    strongreject_raw,
                    args.strongreject_hf_id,
                    args.strongreject_split,
                    token,
                    config_name=args.strongreject_config,
                )

        if strongreject_raw and strongreject_raw.exists():
            strongreject_dija = datasets_dir / "strongreject_prompts.json"
            strongreject_dija.write_text(
                json.dumps(_convert_json_to_dija(strongreject_raw), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            strongreject_diffuguard = datasets_dir / "strongreject_prompts_diffuguard.json"
            strongreject_diffuguard.write_text(
                json.dumps(_convert_json_to_diffuguard(strongreject_raw), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            datasets["strongreject_raw"] = str(strongreject_raw)
            datasets["strongreject_dija"] = str(strongreject_dija)
            datasets["strongreject_diffuguard"] = str(strongreject_diffuguard)

        wildjailbreak_raw: Optional[Path] = None
        if args.wildjailbreak_json:
            wildjailbreak_raw = Path(os.path.expanduser(args.wildjailbreak_json))
            if not wildjailbreak_raw.exists():
                print(f"[WARN] WildJailbreak JSON not found: {wildjailbreak_raw}")
                wildjailbreak_raw = None
        elif args.wildjailbreak_hf_id:
            wildjailbreak_raw = datasets_dir / "wildjailbreak_raw.json"
            if not wildjailbreak_raw.exists():
                print(f"[INFO] Downloading WildJailbreak -> {wildjailbreak_raw}")
                _export_hf_json(
                    wildjailbreak_raw,
                    args.wildjailbreak_hf_id,
                    args.wildjailbreak_split,
                    token,
                    config_name=args.wildjailbreak_config,
                    streaming=args.wildjailbreak_streaming,
                )
        if wildjailbreak_raw and wildjailbreak_raw.exists():
            wildjailbreak_diffuguard = datasets_dir / "wildjailbreak_prompts_diffuguard.json"
            wildjailbreak_diffuguard.write_text(
                json.dumps(_convert_json_to_diffuguard(wildjailbreak_raw), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            datasets["wildjailbreak_raw"] = str(wildjailbreak_raw)
            datasets["wildjailbreak_diffuguard"] = str(wildjailbreak_diffuguard)

        advbench_raw = datasets_dir / "advbench_raw.json"
        if not advbench_raw.exists():
            print(f"[INFO] Downloading AdvBench -> {advbench_raw}")
            _export_hf_json(advbench_raw, args.advbench_hf_id, args.advbench_split, token)
        advbench_diffuguard = datasets_dir / "advbench_prompts_diffuguard.json"
        advbench_diffuguard.write_text(
            json.dumps(_convert_json_to_diffuguard(advbench_raw), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        advbench_dija = datasets_dir / "advbench_prompts_dija.json"
        advbench_dija.write_text(
            json.dumps(_convert_json_to_dija(advbench_raw), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        datasets["advbench_raw"] = str(advbench_raw)
        datasets["advbench_diffuguard"] = str(advbench_diffuguard)
        datasets["advbench_dija"] = str(advbench_dija)

        if args.setup_refined_prompts:
            refine_model_path = args.refine_model_path
            if not refine_model_path:
                refine_model_path = model_paths.get("qwen-2.5-7b-instruct", {}).get("model_path")
            if not refine_model_path:
                refine_model_path = str((hf_models_dir / "Qwen2.5-7B-Instruct").resolve())
            if not Path(os.path.expanduser(str(refine_model_path))).exists():
                raise SystemExit(
                    "Refine model path not found. Provide --refine-model-path or run without --skip-models "
                    "to download qwen-2.5-7b-instruct."
                )

            def _maybe_refine_dataset(
                source_path: Optional[Path],
                output_path: Path,
                dataset_label: str,
            ) -> None:
                if source_path is None or not source_path.exists():
                    print(f"[WARN] {dataset_label} source dataset missing; skipping {dataset_label} refine prompt setup.")
                    return
                if args.skip_existing_refined and output_path.exists():
                    print(f"[INFO] {dataset_label} refined output already exists; skipping: {output_path}")
                    return
                _run_diffuguard_refiner(
                    repo_root,
                    model_path=str(refine_model_path),
                    attack_prompt=source_path,
                    output_json=output_path,
                    max_new_tokens=int(args.refine_max_new_tokens),
                )

            advbench_refined = datasets_dir / "advbench_prompts_refined.json"
            _maybe_refine_dataset(advbench_diffuguard, advbench_refined, "AdvBench")
            datasets["advbench_dija_refined"] = str(advbench_refined)

            jbb_refined = datasets_dir / f"jbb_behaviors_{args.jbb_split}_refined.json"
            _maybe_refine_dataset(jbb_json_diffuguard, jbb_refined, "JBB")
            datasets["jbb_dija_refined"] = str(jbb_refined)

            harmbench_refined = datasets_dir / "harmbench_behaviors_text_all_refined.json"
            _maybe_refine_dataset(
                Path(datasets["harmbench_diffuguard"]) if "harmbench_diffuguard" in datasets else None,
                harmbench_refined,
                "HarmBench",
            )
            datasets["harmbench_dija_refined"] = str(harmbench_refined)

            strongreject_refined = datasets_dir / "strongreject_prompts_refined.json"
            _maybe_refine_dataset(
                Path(datasets["strongreject_diffuguard"]) if "strongreject_diffuguard" in datasets else None,
                strongreject_refined,
                "StrongReject",
            )
            datasets["strongreject_dija_refined"] = str(strongreject_refined)

            wild_raw_refine_source: Optional[Path] = None
            if "wildjailbreak_diffuguard" in datasets:
                wild_raw_refine_source = Path(datasets["wildjailbreak_diffuguard"])
            elif wildjailbreak_raw and wildjailbreak_raw.exists():
                wild_raw_refine_source = wildjailbreak_raw

            if wild_raw_refine_source and wild_raw_refine_source.exists():
                wild_subset = datasets_dir / f"wildjailbreak_prompts_diffuguard_subset{int(args.wildjailbreak_refine_limit)}.json"
                subset_count = _write_subset_json(
                    wild_raw_refine_source,
                    wild_subset,
                    int(args.wildjailbreak_refine_limit),
                )
                print(
                    f"[INFO] Wrote WildJailbreak subset ({subset_count} rows) for DIJA refine -> {wild_subset}"
                )
                wild_refined = (
                    datasets_dir
                    / f"wildjailbreak_prompts_refined_subset{int(args.wildjailbreak_refine_limit)}.json"
                )
                _maybe_refine_dataset(wild_subset, wild_refined, "WildJailbreak")
                datasets["wildjailbreak_dija_subset"] = str(wild_subset)
                datasets["wildjailbreak_dija_refined_subset"] = str(wild_refined)
            else:
                print(
                    "[WARN] WildJailbreak source dataset missing; skipping WildJailbreak refine prompt setup."
                )

        xstest_raw = datasets_dir / "xstest_raw.json"
        if not xstest_raw.exists():
            print(f"[INFO] Downloading XSTest -> {xstest_raw}")
            _export_hf_json(xstest_raw, args.xstest_hf_id, args.xstest_split, token)
        xstest_dija = datasets_dir / "xstest_prompts_dija.json"
        xstest_dija.write_text(
            json.dumps(_convert_json_to_dija(xstest_raw), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        datasets["xstest_raw"] = str(xstest_raw)
        datasets["xstest_dija"] = str(xstest_dija)

        _ensure_hf_dataset_cached(
            args.humaneval_hf_id,
            args.humaneval_split,
            hf_datasets_dir,
            token,
        )
        cached_hf_datasets["humaneval"] = f"{args.humaneval_hf_id}:{args.humaneval_split}"
        _ensure_hf_dataset_cached(
            args.gsm8k_hf_id,
            args.gsm8k_split,
            hf_datasets_dir,
            token,
            config_name=args.gsm8k_config,
        )
        cached_hf_datasets["gsm8k"] = f"{args.gsm8k_hf_id}:{args.gsm8k_config}:{args.gsm8k_split}"
        if args.mmlu_all_configs:
            try:
                from datasets import get_dataset_config_names  # type: ignore
            except Exception as exc:
                raise SystemExit(
                    "Missing datasets. Install with: pip install datasets"
                ) from exc
            mmlu_configs = get_dataset_config_names(args.mmlu_hf_id)
            cached_hf_datasets["mmlu"] = []
            for cfg_name in mmlu_configs:
                _ensure_hf_dataset_cached(
                    args.mmlu_hf_id,
                    args.mmlu_split,
                    hf_datasets_dir,
                    token,
                    config_name=cfg_name,
                )
                cached_hf_datasets["mmlu"].append(
                    f"{args.mmlu_hf_id}:{cfg_name}:{args.mmlu_split}"
                )
            if args.mmlu_config and args.mmlu_config not in mmlu_configs:
                _ensure_hf_dataset_cached(
                    args.mmlu_hf_id,
                    args.mmlu_split,
                    hf_datasets_dir,
                    token,
                    config_name=args.mmlu_config,
                )
                cached_hf_datasets["mmlu"].append(
                    f"{args.mmlu_hf_id}:{args.mmlu_config}:{args.mmlu_split}"
                )
        else:
            _ensure_hf_dataset_cached(
                args.mmlu_hf_id,
                args.mmlu_split,
                hf_datasets_dir,
                token,
                config_name=args.mmlu_config,
            )
            cached_hf_datasets["mmlu"] = f"{args.mmlu_hf_id}:{args.mmlu_config}:{args.mmlu_split}"

    summary = {
        "hf_models_dir": str(hf_models_dir),
        "hf_datasets_dir": str(hf_datasets_dir),
        "datasets_dir": str(datasets_dir),
        "models": model_paths,
        "datasets": datasets,
        "hf_datasets_cached": cached_hf_datasets,
    }

    summary_path = Path(os.path.expanduser(args.summary_json)) if args.summary_json else datasets_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
