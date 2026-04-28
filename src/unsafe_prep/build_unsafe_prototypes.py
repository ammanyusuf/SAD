from __future__ import annotations

import argparse
import logging
from pathlib import Path

from unsafe_prep.prototypes import _build_arg_parser, _cli_build


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = _build_arg_parser()
    _cli_build(parser.parse_args())


if __name__ == "__main__":
    main()
