#!/usr/bin/env python3
"""
Placeholder corpus indexing utility.

The memorization track is under active design; this script simply scaffolds
the expected directory layout and writes an empty manifest so downstream jobs
can proceed without performing heavy preprocessing yet.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="(Stub) build memorization corpus indexes.")
    parser.add_argument("--corpus", required=True, help="Corpus identifier (e.g., enwiki9).")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path.")
    parser.add_argument("--dedupe", default="none")
    parser.add_argument("--min_gram", type=int, default=11)
    parser.add_argument("--max_gram", type=int, default=13)
    parser.add_argument("--outdir", type=Path, default=Path("memorization_eval/indexes"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    tokenizer_slug = args.tokenizer.replace("/", "_")
    root = args.outdir / args.corpus / tokenizer_slug
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not args.force:
        logger.info("Skipping index stub; manifest already present at %s", manifest_path)
        return
    manifest = {
        "corpus": args.corpus,
        "tokenizer": args.tokenizer,
        "dedupe": args.dedupe,
        "min_gram": args.min_gram,
        "max_gram": args.max_gram,
        "shards": [],
        "status": "pending_implementation",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote placeholder manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
