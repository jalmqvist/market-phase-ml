#!/usr/bin/env python3
# Force reload of loader module to prevent caching
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("benchmark.loader", "benchmark/loader.py")
loader_module = importlib.util.module_from_spec(spec)
sys.modules["benchmark.loader"] = loader_module
spec.loader.exec_module(loader_module)

from benchmark.loader import load_benchmark
"""
analyze_reference_benchmark.py

Architecture-agnostic MPML benchmark analyzer.

Current workflow

results_archive/

    baseline/

    gen1_A__...

    gen1_A__...

Produces

1. Architectural uplift
2. Internal MPML improvement
3. Family specialization

Future versions may additionally produce

4. Architecture fingerprints
5. LSTM vs MLP comparisons
6. Cross-family comparisons
"""

from pathlib import Path
import argparse
import sys

from benchmark.loader import load_benchmark
from benchmark.report import print_report


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Analyze MPML benchmark archive."
    )

    parser.add_argument(
        "--archive",
        nargs="?",
        default="results_archive",
        help="Path to benchmark archive.",
    )

    parser.add_argument(
        "--sensitivity-mode",
        action="store_true",
        default=False,
        help=(
            "Recompute deltas on-the-fly using the current baseline. "
            "Use when comparing the same experiments against different baselines."
        ),
    )

    return parser.parse_args()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():

    args = parse_args()

    archive = Path(args.archive)

    if not archive.exists():

        print()

        print(
            f"Archive not found:\n\n{archive}"
        )

        return 1

    try:

        benchmark = load_benchmark(
            archive
        )

        print_report(
            benchmark,
            sensitivity_mode=args.sensitivity_mode,
        )

        return 0

    except KeyboardInterrupt:

        print()

        print("Interrupted.")

        return 130

    except Exception as exc:

        print()

        print("=" * 80)

        print("Benchmark analysis failed")

        print("=" * 80)

        print()

        print(type(exc).__name__)

        print()

        print(exc)

        print()

        return 2


# ------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    benchmark = load_benchmark(args.archive)
    print_report(benchmark, sensitivity_mode=args.sensitivity_mode)
    sys.exit(main())
