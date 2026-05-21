"""
analysis.parsers
================
CSV, JSON manifest, and log parsers for MPML pipeline outputs.

All parsers return plain dicts/lists suitable for JSON serialisation.
Missing files are handled gracefully: the corresponding section in the
summary dict is set to None (not raised).
"""

from analysis.parsers.csv_parsers import parse_run_csvs
from analysis.parsers.manifest_parser import parse_manifest
from analysis.parsers.log_parser import parse_log
from analysis.parsers.run_discovery import discover_runs

__all__ = [
    "parse_run_csvs",
    "parse_manifest",
    "parse_log",
    "discover_runs",
]
