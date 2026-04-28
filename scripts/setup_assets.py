#!/usr/bin/env python3
"""Download models and datasets needed to run the Safe Text Diffusion pipeline.

Usage (quickstart — downloads everything to ./models and ./data):
    python scripts/setup_assets.py --models-dir ./models --data-dir ./data

Selective download:
    python scripts/setup_assets.py --models llada-instruct --models-dir ./models
    python scripts/setup_assets.py --datasets beavertails --data-dir ./data --no-models

HuggingFace authentication:
    The main models (LLaDA, Dream) are public and do not require a token.
    Run `huggingface-cli login` once if you hit rate limits, or pass --token.

After this script finishes the paths printed at the end can be used directly
as CHECKPOINT_PATH / TOKENIZER_PATH env vars for tools/generate.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

MODEL_CATALOG = {
    "llada-base": {
        "hf_id": "GSAI-ML/LLaDA-8B-Base",
        "dir_name": "LLaDA-8B-Base",
        "description": "LLaDA-8B Base (pretrained only). Use for unconditional / base evals.",
    },
    "llada-instruct": {
        "hf_id": "GSAI-ML/LLaDA-8B-Instruct",
        "dir_name": "LLaDA-8B-Instruct",
        "description": "LLaDA-8B Instruct. Used in all paper safety evals (Table 1/2).",
    },
    "dream-instruct": {
        "hf_id": "Dream-org/Dream-v0-Instruct-7B",
        "dir_name": "Dream-v0-Instruct-7B",
        "description": "Dream-v0 Instruct (7B). Used in paper safety evals.",
    },
    "gpt2-large": {
        "hf_id": "openai-community/gpt2-large",
        "dir_name": "gpt2-large",
        "description": "GPT-2 Large. Used by score.py for perplexity (utility metric).",
    },
}

DATASET_CATALOG = {
    "beavertails": {
        "hf_id": "PKU-Alignment/BeaverTails",
        "dir_name": "BeaverTails",
        "description": "BeaverTails — primary unsafe reference corpus and prompt source.",
        "type": "dataset",
    },
    "realtoxicityprompts": {
        "hf_id": "allenai/real-toxicity-prompts",
        "dir_name": "RealToxicityPrompts",
        "description": "RealToxicityPrompts — prompt source for toxicity evals.",
        "type": "dataset",
    },
    "toxigen": {
        "hf_id": "skg/toxigen-data",
        "dir_name": "ToxiGen",
        "description": "ToxiGen — hate speech prompt source.",
        "type": "dataset",
    },
}

DEFAULT_MODELS = ["llada-instruct", "gpt2-large"]
DEFAULT_DATASETS = ["beavertails", "realtoxicityprompts", "toxigen"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _snapshot_download(hf_id: str, target: Path, token: Optional[str]) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing huggingface_hub. Install with:\n  pip install huggingface_hub"
        ) from exc

    print(f"  Downloading {hf_id} → {target} ...")
    snapshot_download(
        repo_id=hf_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"  Done: {target}")


def _download_hf_dataset(hf_id: str, target: Path, token: Optional[str]) -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing datasets. Install with:\n  pip install datasets"
        ) from exc

    print(f"  Downloading dataset {hf_id} → {target} ...")
    ds = load_dataset(hf_id, token=token)  # type: ignore[call-overload]
    target.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(target))  # type: ignore[union-attr]
    print(f"  Done: {target}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("./models"),
        help="Root directory for model downloads (default: ./models)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Root directory for dataset downloads (default: ./data)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CATALOG.keys()),
        default=None,
        help=f"Models to download. Default: {DEFAULT_MODELS}",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CATALOG.keys()),
        default=None,
        help=f"Datasets to download. Default: {DEFAULT_DATASETS}",
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Skip model downloads.",
    )
    parser.add_argument(
        "--no-datasets",
        action="store_true",
        help="Skip dataset downloads.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace API token. Defaults to $HF_TOKEN. Most assets are public.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip assets that are already downloaded (default: true).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the directory already exists.",
    )
    args = parser.parse_args(argv)

    skip_existing = args.skip_existing and not args.force

    models_to_download = args.models or DEFAULT_MODELS
    datasets_to_download = args.datasets or DEFAULT_DATASETS

    downloaded_model_paths: dict[str, Path] = {}
    downloaded_dataset_paths: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    if not args.no_models:
        print(f"\n=== Models → {args.models_dir} ===")
        args.models_dir.mkdir(parents=True, exist_ok=True)
        for key in models_to_download:
            entry = MODEL_CATALOG[key]
            target = args.models_dir / entry["dir_name"]
            if skip_existing and _nonempty_dir(target):
                print(f"  [skip] {key} already exists at {target}")
                downloaded_model_paths[key] = target
                continue
            _snapshot_download(entry["hf_id"], target, args.token)
            downloaded_model_paths[key] = target

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    if not args.no_datasets:
        print(f"\n=== Datasets → {args.data_dir} ===")
        args.data_dir.mkdir(parents=True, exist_ok=True)
        for key in datasets_to_download:
            entry = DATASET_CATALOG[key]
            target = args.data_dir / entry["dir_name"]
            if skip_existing and _nonempty_dir(target):
                print(f"  [skip] {key} already exists at {target}")
                downloaded_dataset_paths[key] = target
                continue
            _download_hf_dataset(entry["hf_id"], target, args.token)
            downloaded_dataset_paths[key] = target

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Setup complete ===")

    if downloaded_model_paths:
        print("\nModel paths (use as CHECKPOINT_PATH / TOKENIZER_PATH):")
        for key, path in downloaded_model_paths.items():
            print(f"  {key:25s}: {path.resolve()}")

    if downloaded_dataset_paths:
        print("\nDataset paths (use as DATA_DIR in configs/data/catalog.yaml):")
        for key, path in downloaded_dataset_paths.items():
            print(f"  {key:25s}: {path.resolve()}")

    # Print ready-to-copy export block
    llada = downloaded_model_paths.get("llada-instruct")
    dream = downloaded_model_paths.get("dream-instruct")
    if llada or dream:
        print("\n--- Copy-paste env vars ---")
        if llada:
            print(f"export CHECKPOINT_PATH={llada.resolve()}")
            print(f"export TOKENIZER_PATH={llada.resolve()}")
        if dream:
            print(f"# (for Dream) export CHECKPOINT_PATH={dream.resolve()}")
            print(f"# (for Dream) export TOKENIZER_PATH={dream.resolve()}")

    beavertails = downloaded_dataset_paths.get("beavertails")
    if beavertails:
        print(f"export DATA_DIR={beavertails.resolve().parent}")

    print()


if __name__ == "__main__":
    main()
