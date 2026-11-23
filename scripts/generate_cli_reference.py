#!/usr/bin/env python3
"""Generate or check the committed MPML CLI spec and reference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpml.spec import build_spec, render_reference, spec_json  # noqa: E402


SPEC_PATH = ROOT / "docs" / "cli_spec.json"
REFERENCE_PATH = ROOT / "docs" / "CLI_REFERENCE.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated files are stale.")
    args = parser.parse_args()

    spec = build_spec()
    outputs = {
        SPEC_PATH: spec_json(spec),
        REFERENCE_PATH: render_reference(spec),
    }
    if args.check:
        stale = [
            path.relative_to(ROOT).as_posix()
            for path, content in outputs.items()
            if not path.exists() or path.read_text(encoding="utf-8") != content
        ]
        if stale:
            print("Generated CLI reference files are stale: " + ", ".join(stale), file=sys.stderr)
            return 1
        return 0

    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
