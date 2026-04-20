#!/usr/bin/env python3
"""
Build a versioned D1 phase label artifact for cross-repo consumption.

Output:
  data/output/regimes/phase_labels_d1.parquet

Join contract (market-sentiment-ml):
  join on (pair, floor(entry_time to UTC day)) == (pair, timestamp)

Correctness requirements:
- timestamp is UTC, aligned to 00:00:00 exactly (timezone-aware)
- one row per (pair, day), no duplicates
- phase computed via MarketPhaseDetector.detect_phases() (no reimplementation)
- deterministic config_hash based only on detector configuration
- soft gap diagnostics (warn only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.phases import MarketPhaseDetector  # authoritative detector

# --- Path resolution (do NOT depend on current working directory) ---
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]  # scripts/ -> repo root

# -----------------------------
# Constants (artifact contract)
# -----------------------------

PROCESSED_DIR_DEFAULT = REPO_ROOT / "data" / "processed"
OUTPUT_PATH_DEFAULT = REPO_ROOT / "data" / "output" / "regimes" / "phase_labels_d1.parquet"

VALID_PHASES = {"HV_Trend", "HV_Ranging", "LV_Trend", "LV_Ranging", "Unknown"}

DETECTOR_ID = "d1_native_v1"
DETECTOR_NAME = "MarketPhaseDetector"
CONFIG_VERSION = "v1"

DATA_SOURCE_DEFAULT = "Yahoo"

# Soft diagnostics
GAP_WARN_DAYS = 3
GAP_SAMPLE_ROWS = 8


# -----------------------------
# Utilities
# -----------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stable_json_dumps(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _require_cols(df: pd.DataFrame, cols: List[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AssertionError(f"{ctx}: missing required columns: {missing}")


def normalize_pair_from_processed_filename(path: Path) -> str:
    """
    data/processed filenames are like EURUSD.csv, USDJPY.csv ...
    Normalize to lowercase xxx-yyy for cross-repo joins (eur-usd, usd-jpy).
    """
    sym = path.stem.upper()
    _assert(len(sym) >= 6, f"Unexpected processed filename stem: {path.stem}")
    return f"{sym[:3].lower()}-{sym[3:6].lower()}"


def to_utc_midnight(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Processed CSV index is tz-naive. Interpret as UTC, then normalize to 00:00 UTC.
    """
    t = pd.to_datetime(idx, errors="raise")
    if t.tz is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.floor("D")


# -----------------------------
# Versioning / config hashing
# -----------------------------

@dataclass(frozen=True)
class DetectorConfig:
    detector_class: str
    adx_period: int
    adx_trend_threshold: float
    atr_period: int
    vol_rolling_window: int
    stop_multipliers: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "detector_class": self.detector_class,
            "adx_period": int(self.adx_period),
            "adx_trend_threshold": float(self.adx_trend_threshold),
            "atr_period": int(self.atr_period),
            "vol_rolling_window": int(self.vol_rolling_window),
            "stop_multipliers": {k: float(self.stop_multipliers[k]) for k in sorted(self.stop_multipliers)},
        }


def build_config_hash(detector: MarketPhaseDetector) -> str:
    cfg = DetectorConfig(
        detector_class=detector.__class__.__name__,
        adx_period=int(detector.adx_period),
        adx_trend_threshold=float(detector.adx_trend_threshold),
        atr_period=int(detector.atr_period),
        vol_rolling_window=int(detector.vol_rolling_window),
        stop_multipliers=dict(detector.STOP_MULTIPLIERS),
    )
    return sha256_hex(stable_json_dumps(cfg.to_dict()))


# -----------------------------
# Pipeline steps
# -----------------------------

def load_data(processed_dir: Path) -> pd.DataFrame:
    """
    Load all pairs from data/processed/*.csv into one dataframe.

    Output columns:
      pair, timestamp, Open, High, Low, Close, Volume
    """
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dir not found: {processed_dir.resolve()}")

    files = sorted([p for p in processed_dir.glob("*.csv") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No processed CSV files found in: {processed_dir.resolve()}")

    frames: List[pd.DataFrame] = []
    for path in files:
        pair = normalize_pair_from_processed_filename(path)

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            raise ValueError(f"{path.name}: empty CSV")

        df = df.sort_index()

        # Required OHLC; Volume may be missing in some exports
        for c in ["Open", "High", "Low", "Close"]:
            _assert(c in df.columns, f"{path.name}: missing required column {c}")
        if "Volume" not in df.columns:
            df["Volume"] = 0.0

        ts = to_utc_midnight(df.index)

        out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        out.insert(0, "timestamp", ts)
        out.insert(0, "pair", pair)

        # Hard invariants at load time
        _assert(out["timestamp"].is_monotonic_increasing, f"{pair}: timestamps not monotonic in {path.name}")
        _assert(out["timestamp"].is_unique, f"{pair}: duplicate timestamps in {path.name}")

        frames.append(out)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["pair", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Defensive: ensure timestamp is consistently UTC-aware after concat/sort.
    all_df["timestamp"] = pd.to_datetime(all_df["timestamp"], utc=True, errors="raise")

    return all_df


def compute_regimes(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regimes using existing MarketPhaseDetector.detect_phases().
    """
    _require_cols(prices, ["pair", "timestamp", "Open", "High", "Low", "Close", "Volume"], "prices")

    detector = MarketPhaseDetector()
    parts: List[pd.DataFrame] = []

    for pair, g in prices.groupby("pair", sort=True):
        g = g.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

        _assert(g["timestamp"].is_monotonic_increasing, f"{pair}: timestamps not monotonic (post-merge)")
        _assert(g["timestamp"].is_unique, f"{pair}: duplicate timestamps (post-merge)")

        detected = detector.detect_phases(g[["Open", "High", "Low", "Close", "Volume"]].copy())
        _require_cols(detected, ["phase", "trending", "high_vol"], f"{pair}: detector output")

        # IMPORTANT: do NOT use `.values` for tz-aware timestamps.
        # `.values` can silently drop the timezone and produce tz-naive datetime64[ns].
        part = pd.DataFrame(
            {
                "pair": pair,
                "timestamp": g["timestamp"],  # preserves datetime64[ns, UTC]
                "phase": detected["phase"].astype(str).values,
                "is_trending": detected["trending"].astype(bool).values,
                "is_high_vol": detected["high_vol"].astype(bool).values,
            }
        )
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["pair", "timestamp"], kind="mergesort").reset_index(drop=True)
    return out


def attach_metadata(df: pd.DataFrame, *, data_source: str) -> pd.DataFrame:
    detector = MarketPhaseDetector()
    cfg_hash = build_config_hash(detector)
    build_ts = utc_now()

    out = df.copy()
    out["detector_id"] = DETECTOR_ID
    out["detector_name"] = DETECTOR_NAME
    out["config_version"] = CONFIG_VERSION
    out["config_hash"] = cfg_hash
    out["build_timestamp"] = pd.Timestamp(build_ts)
    out["data_source"] = data_source

    # Explicit dtypes for artifact stability
    out["pair"] = out["pair"].astype("string")
    out["phase"] = out["phase"].astype("string")
    out["detector_id"] = out["detector_id"].astype("string")
    out["detector_name"] = out["detector_name"].astype("string")
    out["config_version"] = out["config_version"].astype("string")
    out["config_hash"] = out["config_hash"].astype("string")
    out["data_source"] = out["data_source"].astype("string")

    return out


def validate_output(df: pd.DataFrame) -> None:
    """
    Hard constraints only. Fail only if core invariants are violated.
    """
    required = [
        "pair",
        "timestamp",
        "phase",
        "is_trending",
        "is_high_vol",
        "detector_id",
        "detector_name",
        "config_version",
        "config_hash",
        "build_timestamp",
        "data_source",
    ]
    _require_cols(df, required, "output")

    # Parse timestamps as UTC. If timestamps are tz-naive (a bug upstream),
    # this will interpret them as UTC; we still validate alignment to 00:00:00 UTC.
    ts = pd.to_datetime(df["timestamp"], errors="raise", utc=True)

    _assert(str(ts.dt.tz) == "UTC", f"timestamp must be UTC; got {ts.dt.tz}")

    aligned = (
        (ts.dt.hour == 0)
        & (ts.dt.minute == 0)
        & (ts.dt.second == 0)
        & (ts.dt.microsecond == 0)
    )
    _assert(bool(aligned.all()), "timestamp must align to 00:00:00 UTC boundary")

    _assert(
        not df.duplicated(subset=["pair", "timestamp"]).any(),
        "duplicate (pair, timestamp) rows found",
    )

    bad = sorted(set(df["phase"].astype(str)) - VALID_PHASES)
    _assert(not bad, f"invalid phase values: {bad}")

    for pair, g in df.groupby("pair", sort=True):
        t = pd.to_datetime(g["timestamp"], utc=True)
        _assert(t.is_monotonic_increasing, f"{pair}: timestamps not monotonic increasing")
        _assert(t.is_unique, f"{pair}: duplicate timestamps within pair")


def print_coverage_summary(df: pd.DataFrame) -> None:
    rows = []
    for pair, g in df.groupby("pair", sort=True):
        t = pd.to_datetime(g["timestamp"], utc=True)
        rows.append(
            {
                "pair": pair,
                "n_days": int(len(g)),
                "start": t.min().date().isoformat(),
                "end": t.max().date().isoformat(),
            }
        )
    summary = pd.DataFrame(rows).sort_values("pair")
    print("\nCoverage summary per pair:")
    print(summary.to_string(index=False))

def build_gap_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a full gap report per pair with (prev_timestamp, timestamp, gap_days).

    Notes:
    - Uses observed timestamps only (does not enforce continuity).
    - Intended for transparency/auditing, not validation.
    """
    rows = []
    for pair, g in df.groupby("pair", sort=True):
        t = pd.to_datetime(g["timestamp"], utc=True).sort_values().reset_index(drop=True)
        gaps = t.diff().dt.total_seconds().div(86400.0)

        for i in range(1, len(t)):
            gd = gaps.iloc[i]
            if pd.notna(gd):
                rows.append({
                    "pair": pair,
                    "prev_timestamp": t.iloc[i - 1],
                    "timestamp": t.iloc[i],
                    "gap_days": float(gd),
                })

    rep = pd.DataFrame(rows)
    if not rep.empty:
        rep = rep.sort_values(["gap_days", "pair", "timestamp"], ascending=[False, True, True]).reset_index(drop=True)
    return rep

def warn_on_large_gaps(df: pd.DataFrame) -> None:
    """
    Soft diagnostics only (warnings; never fail build):
    - detect gaps > GAP_WARN_DAYS per pair
    - print summary + sample rows
    """
    warnings = []
    samples = []

    for pair, g in df.groupby("pair", sort=True):
        # Ensure sorted UTC timestamps
        t = pd.to_datetime(g["timestamp"], utc=True).sort_values().reset_index(drop=True)

        # gaps[i] is the gap between t[i-1] -> t[i], for i>=1
        gaps = t.diff().dt.total_seconds().div(86400.0)

        # Indices (positions) where gap is "big"
        big_pos = gaps[gaps > GAP_WARN_DAYS].index.tolist()
        if not big_pos:
            continue

        big_vals = gaps.loc[big_pos]

        warnings.append(
            {
                "pair": pair,
                "n_gaps_gt_3d": int(len(big_pos)),
                "max_gap_days": float(big_vals.max()),
            }
        )

        # Sample a few gap positions safely
        for pos in big_pos[:GAP_SAMPLE_ROWS]:
            # pos is the "current" timestamp position in t
            prev_ts = t.iloc[pos - 1] if pos - 1 >= 0 else pd.NaT
            curr_ts = t.iloc[pos]
            gap_days = float(gaps.iloc[pos])
            samples.append(
                {
                    "pair": pair,
                    "prev_timestamp": str(prev_ts),
                    "timestamp": str(curr_ts),
                    "gap_days": gap_days,
                }
            )

    if warnings:
        warn_df = pd.DataFrame(warnings).sort_values(
            ["n_gaps_gt_3d", "max_gap_days"], ascending=False
        )
        print(f"\n[WARN] Detected gaps > {GAP_WARN_DAYS} days (diagnostics only; not failing):")
        print(warn_df.to_string(index=False))

        if samples:
            print("\nSample gap rows:")
            print(pd.DataFrame(samples).head(GAP_SAMPLE_ROWS).to_string(index=False))
    else:
        print(f"\nGap diagnostics: no gaps > {GAP_WARN_DAYS} days detected.")


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    ensure_parent_dir(output_path)
    df.to_parquet(output_path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Build D1 phase labels parquet artifact.")
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR_DEFAULT)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH_DEFAULT)
    p.add_argument("--data-source", type=str, default=DATA_SOURCE_DEFAULT)
    args = p.parse_args()

    # 1) Load input prices (D1)
    prices = load_data(args.processed_dir)

    # 2) Optional: gap transparency on INPUT (non-failing)
    gap_report_prices = build_gap_report(prices)
    big_prices = gap_report_prices[gap_report_prices["gap_days"] > 10]
    if not big_prices.empty:
        print("\nTop gaps > 10 days in input prices:")
        print(big_prices.head(20).to_string(index=False))

    # 3) Compute regimes (authoritative detector)
    regimes = compute_regimes(prices)

    # 4) Attach metadata -> final artifact DF
    out = attach_metadata(regimes, data_source=args.data_source)

    # 5) Validate + diagnostics (HARD then SOFT)
    print("timestamp dtype:", out["timestamp"].dtype)
    validate_output(out)        # HARD checks
    print_coverage_summary(out) # INFO
    warn_on_large_gaps(out)     # WARN only

    # 6) Save
    out = out.sort_values(["pair", "timestamp"], kind="mergesort").reset_index(drop=True)
    save_output(out, args.output)

    print(f"\nWrote: {args.output}  rows={len(out):,}")
    print(f"detector_id={DETECTOR_ID} detector_name={DETECTOR_NAME} config_version={CONFIG_VERSION}")
    print(f"config_hash={out['config_hash'].iloc[0]} build_timestamp={out['build_timestamp'].iloc[0]}")

if __name__ == "__main__":
    main()
