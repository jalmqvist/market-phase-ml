#!/usr/bin/env python3
"""
Sanity-check script for the DL surface loader (src/dl_surface_loader.py).

Runs a suite of invariant checks against either:
  - A real cube parquet (when --cube-path points to an existing file), or
  - Synthetic in-memory test cubes that exercise every validation path.

Usage
-----
# Run all checks (synthetic cubes only — no real artifact needed)
python scripts/validate_dl_surface.py

# Also validate a real cube
python scripts/validate_dl_surface.py --cube-path ../market-sentiment-ml/data/output/dl_signals/dl_signals_h1_v1.parquet

Exit codes
----------
0  All checks passed.
1  One or more checks failed.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Ensure repo root is importable regardless of working directory
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dl_surface_loader import (  # noqa: E402
    REQUIRED_CUBE_COLUMNS,
    SURFACE_REQUIRED_KEYS,
    VALID_DL_REGIMES,
    empty_dl_surface_df,
    load_dl_surface,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "PASS"
_FAIL = "FAIL"

_results: list[tuple[str, str, str]] = []  # (name, status, detail)


def _check(name: str, fn) -> bool:
    try:
        fn()
        _results.append((name, _PASS, ""))
        return True
    except AssertionError as exc:
        _results.append((name, _FAIL, str(exc)))
        return False
    except Exception as exc:  # noqa: BLE001 — catch unexpected errors in test runner
        _results.append((name, _FAIL, f"{type(exc).__name__}: {exc}"))
        return False


def _make_cube(
    n_rows: int = 10,
    pair: str = "eur-usd",
    model: str = "lstm",
    target_horizon: int = 24,
    feature_set: str = "price_trend",
    dl_regime: str = "HVTF",
    start: datetime | None = None,
) -> pd.DataFrame:
    """Build a minimal valid cube DataFrame for a single surface."""
    if start is None:
        start = datetime(2024, 1, 1, 0, 0, 0)
    times = [start + timedelta(hours=i) for i in range(n_rows)]
    return pd.DataFrame(
        {
            "pair": [pair] * n_rows,
            "entry_time": pd.to_datetime(times),
            "model": [model] * n_rows,
            "target_horizon": pd.array([target_horizon] * n_rows, dtype="Int64"),
            "feature_set": [feature_set] * n_rows,
            "dl_regime": [dl_regime] * n_rows,
            "signal_strength": [float(i) / n_rows * 2 - 1 for i in range(n_rows)],
            "pred_prob_up": [0.5 + float(i) / n_rows * 0.5 for i in range(n_rows)],
        }
    )


_SURFACE = {
    "model": "lstm",
    "target_horizon": 24,
    "feature_set": "price_trend",
    "dl_regime": "HVTF",
}

# ---------------------------------------------------------------------------
# Synthetic checks
# ---------------------------------------------------------------------------


def check_empty_df_schema():
    """empty_dl_surface_df() returns consistent dtypes."""
    df = empty_dl_surface_df()
    assert df.empty, "Expected empty DataFrame"
    assert "pair" in df.columns
    assert "timestamp" in df.columns
    assert "dl_signal_strength" in df.columns
    assert df["dl_signal_strength"].dtype == "float64"


def check_missing_cube_file():
    """Missing cube file returns empty DF (strict=False)."""
    df = load_dl_surface(
        Path("/nonexistent/path/cube.parquet"), _SURFACE, strict=False
    )
    assert df.empty, "Expected empty DF for missing file"


def check_missing_cube_file_strict():
    """Missing cube file raises ValueError when strict=True."""
    raised = False
    try:
        load_dl_surface(
            Path("/nonexistent/path/cube.parquet"), _SURFACE, strict=True
        )
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for missing file with strict=True"


def check_invalid_surface_dict_missing_key():
    """Surface dict missing a required key raises / returns empty DF."""
    bad_surface = {"model": "lstm", "target_horizon": 24, "feature_set": "price_trend"}
    df = load_dl_surface(Path("/nonexistent"), bad_surface, strict=False)
    assert df.empty, "Expected empty DF for invalid surface dict"


def check_invalid_surface_regime():
    """Unknown dl_regime returns empty DF."""
    bad_surface = dict(_SURFACE, dl_regime="UNKNOWN")
    df = load_dl_surface(Path("/nonexistent"), bad_surface, strict=False)
    assert df.empty


def check_valid_cube_roundtrip(tmp_path: Path):
    """Valid cube loads cleanly and output schema is correct."""
    cube = _make_cube()
    cube_file = tmp_path / "cube.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=True)

    assert not df.empty, "Expected non-empty result"
    assert "pair" in df.columns
    assert "timestamp" in df.columns
    assert "dl_signal_strength" in df.columns
    assert "mpml_regime_equiv" in df.columns
    assert "signal_strength" not in df.columns, "signal_strength should be renamed"
    assert "entry_time" not in df.columns, "entry_time should be renamed to timestamp"

    # Check MPML mapping
    assert set(df["mpml_regime_equiv"].unique()) == {"HVTF"}


def check_signal_strength_range_violation(tmp_path: Path):
    """Out-of-range signal_strength returns empty DF (strict=False)."""
    cube = _make_cube()
    cube.loc[0, "signal_strength"] = 1.5  # out of range
    cube_file = tmp_path / "bad_range.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=False)
    assert df.empty, "Expected empty DF for out-of-range signal_strength"


def check_pred_prob_up_range_violation(tmp_path: Path):
    """Out-of-range pred_prob_up returns empty DF."""
    cube = _make_cube()
    cube.loc[0, "pred_prob_up"] = -0.1  # out of range
    cube_file = tmp_path / "bad_prob.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=False)
    assert df.empty, "Expected empty DF for out-of-range pred_prob_up"


def check_non_hourly_timestamp(tmp_path: Path):
    """Non-hourly entry_time values are rejected."""
    cube = _make_cube()
    cube.loc[0, "entry_time"] = cube.loc[0, "entry_time"] + timedelta(minutes=15)
    cube_file = tmp_path / "non_hourly.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=False)
    assert df.empty, "Expected empty DF for non-hourly timestamps"


def check_duplicate_entry_time(tmp_path: Path):
    """Duplicate (pair, entry_time) within a surface is rejected."""
    cube = _make_cube()
    cube.loc[1, "entry_time"] = cube.loc[0, "entry_time"]  # introduce duplicate
    cube_file = tmp_path / "dup.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=False)
    assert df.empty, "Expected empty DF for duplicate entry_time"


def check_non_monotone_timestamps(tmp_path: Path):
    """Non-monotone entry_time within a surface is rejected."""
    cube = _make_cube()
    # Swap first two rows to break monotonicity
    cube.loc[0, "entry_time"], cube.loc[1, "entry_time"] = (
        cube.loc[1, "entry_time"],
        cube.loc[0, "entry_time"],
    )
    cube_file = tmp_path / "non_monotone.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=False)
    assert df.empty, "Expected empty DF for non-monotone entry_time"


def check_surface_selection_filters_correctly(tmp_path: Path):
    """Only rows matching the exact surface dict are returned."""
    cube_a = _make_cube(dl_regime="HVTF", n_rows=5)
    cube_b = _make_cube(dl_regime="LVTF", n_rows=8)
    cube = pd.concat([cube_a, cube_b], ignore_index=True)
    cube_file = tmp_path / "multi_surface.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=True)
    assert len(df) == 5, f"Expected 5 rows; got {len(df)}"


def check_missing_required_column(tmp_path: Path):
    """Cube missing a required column returns empty DF."""
    cube = _make_cube()
    cube = cube.drop(columns=["signal_strength"])
    cube_file = tmp_path / "missing_col.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=False)
    assert df.empty, "Expected empty DF for missing required column"


def check_mpml_regime_mapping(tmp_path: Path):
    """HVR dl_regime maps to HVMR in mpml_regime_equiv."""
    cube = _make_cube(dl_regime="HVR")
    cube_file = tmp_path / "hvr.parquet"
    cube.to_parquet(cube_file, index=False)

    surface = dict(_SURFACE, dl_regime="HVR")
    df = load_dl_surface(cube_file, surface, strict=True)
    assert not df.empty
    assert set(df["mpml_regime_equiv"].unique()) == {"HVMR"}


def check_target_horizon_numeric(tmp_path: Path):
    """target_horizon stored as string in parquet is coerced to Int64."""
    cube = _make_cube()
    cube["target_horizon"] = cube["target_horizon"].astype(str)
    cube_file = tmp_path / "str_horizon.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=True)
    assert not df.empty, "Expected successful load with string target_horizon"


def check_no_surface_match_returns_empty(tmp_path: Path):
    """Surface dict matching nothing returns empty DF."""
    cube = _make_cube(dl_regime="LVTF")
    cube_file = tmp_path / "lvtf.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=False)  # _SURFACE expects HVTF
    assert df.empty, "Expected empty DF when surface not in cube"


# ---------------------------------------------------------------------------
# Real artifact check (optional)
# ---------------------------------------------------------------------------


def check_real_cube(cube_path: Path, surface: dict):
    """Load a real cube and validate basic invariants."""
    df = load_dl_surface(cube_path, surface, strict=True)
    assert not df.empty, "Real cube returned empty surface"
    assert df["dl_signal_strength"].between(-1, 1).all(), (
        "signal_strength out of range"
    )
    assert "timestamp" in df.columns
    assert "pair" in df.columns
    assert "mpml_regime_equiv" in df.columns
    print(f"  Real cube: {len(df):,} rows loaded for surface {surface}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_checks(cube_path: Path | None, surface: dict) -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="validate_dl_"))

    synthetic = [
        ("empty_df_schema", check_empty_df_schema, []),
        ("missing_cube_file", check_missing_cube_file, []),
        ("missing_cube_strict", check_missing_cube_file_strict, []),
        ("invalid_surface_missing_key", check_invalid_surface_dict_missing_key, []),
        ("invalid_surface_regime", check_invalid_surface_regime, []),
        ("valid_roundtrip", check_valid_cube_roundtrip, [tmp]),
        ("signal_strength_range", check_signal_strength_range_violation, [tmp]),
        ("pred_prob_up_range", check_pred_prob_up_range_violation, [tmp]),
        ("non_hourly_timestamp", check_non_hourly_timestamp, [tmp]),
        ("duplicate_entry_time", check_duplicate_entry_time, [tmp]),
        ("non_monotone_timestamps", check_non_monotone_timestamps, [tmp]),
        ("surface_selection", check_surface_selection_filters_correctly, [tmp]),
        ("missing_required_column", check_missing_required_column, [tmp]),
        ("mpml_regime_mapping", check_mpml_regime_mapping, [tmp]),
        ("target_horizon_numeric", check_target_horizon_numeric, [tmp]),
        ("no_surface_match", check_no_surface_match_returns_empty, [tmp]),
    ]

    for name, fn, args in synthetic:
        _check(name, lambda f=fn, a=args: f(*a))

    if cube_path is not None and cube_path.exists():
        _check(
            "real_cube",
            lambda: check_real_cube(cube_path, surface),
        )
    elif cube_path is not None:
        _results.append(
            ("real_cube", "SKIP", f"Cube file not found: {cube_path}")
        )

    # Print results
    max_name = max(len(r[0]) for r in _results)
    print(f"\n{'Check':<{max_name}}  Status  Detail")
    print("-" * (max_name + 30))
    n_failed = 0
    for name, status, detail in _results:
        icon = "✓" if status == _PASS else ("⚠" if status == "SKIP" else "✗")
        print(f"{name:<{max_name}}  {icon} {status:<6}  {detail}")
        if status == _FAIL:
            n_failed += 1

    print()
    if n_failed == 0:
        print(f"All {len(_results)} checks passed.")
        return 0
    else:
        print(f"{n_failed}/{len(_results)} checks FAILED.")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate DL surface loader invariants."
    )
    parser.add_argument(
        "--cube-path",
        type=Path,
        default=None,
        help="Path to a real DL signal cube parquet (optional).",
    )
    parser.add_argument("--model", default="lstm")
    parser.add_argument("--target-horizon", type=int, default=24)
    parser.add_argument("--feature-set", default="price_trend")
    parser.add_argument("--dl-regime", default="HVTF")
    args = parser.parse_args()

    surface = {
        "model": args.model,
        "target_horizon": args.target_horizon,
        "feature_set": args.feature_set,
        "dl_regime": args.dl_regime,
    }

    return run_checks(args.cube_path, surface)


if __name__ == "__main__":
    sys.exit(main())
