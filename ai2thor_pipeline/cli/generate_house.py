#!/usr/bin/env python3
"""Generate a deterministic multi-floor ProcTHOR house JSON for the hazard pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.procthor_house import (
    default_local_executable,
    generate_multifloor_house,
)

_DEFAULT_OUT = (
    Path(__file__).resolve().parents[1] / "assets" / "houses" / "house_multifloor_seed1234.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output house JSON path",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--local-executable",
        type=Path,
        default=default_local_executable(),
        help="Schema-2 patched AI2-THOR build",
    )
    args = parser.parse_args()

    if not args.local_executable.is_file():
        raise RuntimeError(
            f"custom executable not found: {args.local_executable}\n"
            "Run: ./ai2thor_custom/build_local.sh"
        )

    out = generate_multifloor_house(
        args.out,
        seed=args.seed,
        local_executable_path=str(args.local_executable),
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
