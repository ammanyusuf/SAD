from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import build_unsafe_artifacts, load_config


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Build unsafe-answer tensor artifacts.")
  parser.add_argument(
      "--config",
      type=Path,
      required=True,
      help="Path to unsafe artifact YAML configuration.",
  )
  parser.add_argument(
      "--out",
      type=Path,
      default=None,
      help="Optional override for artifact output directory.",
  )
  parser.add_argument(
      "--include",
      nargs="+",
      default=None,
      help="Optional list of dataset sources or artifact names to include.",
  )
  parser.add_argument(
      "--exclude",
      nargs="+",
      default=None,
      help="Optional list of dataset sources to exclude.",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Collect counts without writing tensors.",
  )
  parser.add_argument(
      "--force",
      action="store_true",
      help="Overwrite existing shards instead of resuming.",
  )
  parser.add_argument(
      "--set",
      dest="overrides",
      action="append",
      default=None,
      help="Override config values via dotlist syntax (e.g., --set max_length=256 --set datasets[0].sample_size=500).",
  )
  parser.add_argument(
      "--log-level",
      default="INFO",
      help="Logging level (default: INFO).",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

  config = load_config(args.config, overrides=args.overrides)
  output_dir = args.out or None
  index = build_unsafe_artifacts(
      config=config,
      output_root=output_dir,
      include=args.include,
      exclude=args.exclude,
      dry_run=args.dry_run,
      overwrite=args.force,
  )
  if args.dry_run:
    total = sum(artifact["count"] for artifact in index["unsafe_artifacts"])
    logging.getLogger(__name__).info(
        "Dry run summary: %d artifacts, %d total records.",
        len(index["unsafe_artifacts"]),
        total,
    )


if __name__ == "__main__":
  main()
