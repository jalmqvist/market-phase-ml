# main.py
import argparse
import sys
import importlib.metadata as importlib_metadata
import traceback
import matplotlib
import platform
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
from pathlib import Path
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from src.data import (
    MarketDataPipeline,
    ALL_PAIRS, MAJORS, MINORS,
    PAIR_NAMES, PIP_VALUES,
    summarize_dataset,
    resolve_market_data_source,
)
from src.phases import MarketPhaseDetector
from src.strategies import run_backtests
from src.strategies import (
    TF1Strategy, TF2Strategy, TF3Strategy, TF4Strategy, TF5Strategy,
    MR1Strategy, MR2Strategy, MR32Strategy, MR42Strategy, MR5Strategy,
    TradeResult  # if needed for reporting
)
from src import visualization as viz
from src.visualization import PhaseVisualizer
from src.cache import (
    save_cache, load_cache, clear_cache,
    _hash_dict_of_dataframes, _hash_params
)
from src.models import (
    PhaseMLExperiment, PhaseMLPredictor,
    StrategyPerformanceTracker, StrategySelector,
    smooth_phase_labels, safe_existing_columns,
)
from src.strategies import Backtester as BT, PhaseAwareStrategy, StrategySelector_Dynamic
from src.repro import (
    DEFAULT_EXPERIMENT_SEED,
    build_run_config,
    resolve_experiment_seed,
    set_global_seed,
    write_manifest,
)
from src.dl_config import (
    infer_dl_regime_from_artifact_path,
    resolve_dl_prediction_artifact_path,
)
from src.dl_surface_loader import VALID_DL_REGIMES
from src.dl_daily_features import load_and_aggregate_d1, D1_FEATURE_COLS
from src.experiment_surface_runtime import build_runtime_experiment_surface
from experiment_semantics import (
    VALID_EXPERIMENT_VARIANTS,
    build_experiment_metadata_from_variant,
    infer_imputation_awareness_from_name,
    normalize_variant,
)


# ─────────────────────────────────────────
# DL FEATURE REGISTRY HELPERS
# ─────────────────────────────────────────

def get_dl_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the canonical DL feature columns that are present in df.

    These are the columns produced by attach_dl_features() via D1 aggregation.
    Used everywhere instead of hardcoded assumptions about DL column presence.
    Returns an empty list when DL is disabled or no DL features were attached.
    """
    return sorted(c for c in D1_FEATURE_COLS if c in df.columns)


def has_dl_features(df: pd.DataFrame) -> bool:
    """Return True if df contains at least one DL feature column."""
    return bool(get_dl_feature_columns(df))


def _stable_feature_columns(cols: list[str] | tuple[str, ...]) -> list[str]:
    """Return a deterministically sorted, de-duplicated feature column list."""
    return sorted(dict.fromkeys(cols))


# ── Uncomment to force cache refresh ─────
# clear_cache('processed_data')
# clear_cache('backtest_results')
# clear_cache('ml_results')
# clear_cache('ml_predicted_phases')
# clear_cache('ml_backtest_results')
clear_cache()   # clears everything
# ─────────────────────────────────────────
WF_TRAIN_YEARS = 7
WF_TEST_MONTHS = 6
WF_STEP_MONTHS = 6
LABEL_HORIZON_BARS = 20  # must match StrategyPerformanceTracker(window_days=...)
# ────────────────────────────────────────
# RUN FLAGS (toggle expensive experiments)
# ─────────────────────────────────────────
RUN_IN_SAMPLE_ABLATION = True
RUN_WALKFORWARD = True

# DL surface integration (optional feature layer, v1 single-surface)
# DL_SIGNALS_ENABLED = True
DL_SIGNALS_ENABLED = os.environ.get("DL_SIGNALS_ENABLED", "false").lower() == "true"
DEFAULT_DL_REGIME = "LVTF"
DL_SIGNAL_SURFACE = {
    "model": os.getenv("DL_MODEL", "mlp"),
    "target_horizon": 24,
    "feature_set": os.getenv("DL_FEATURE_SET", "price_trend"),
    "dl_regime": os.getenv("DL_REGIME", ""),
}

# DL debug verbosity (controls noisy per-pair diagnostics)
DL_DEBUG_VERBOSE = False
MIN_DL_TRAIN_COVERAGE_PCT = float(os.environ.get("MIN_DL_TRAIN_COVERAGE_PCT", "5.0"))

# Expensive sweeps (disable by default)
RUN_TAU_SWEEP = False
RUN_POLICY_SWEEP = False

# Debug
DEBUG_BASELINE_KEYS = False
DEBUG_FEATURE_COLUMNS = False
DEBUG_SIGNAL_TYPES = False
DEBUG_VOL_GUARD = False   # gate verbose per-fold vol-guard train/test prints
# ─────────────────────────────────────────
# OPTIONAL PAIR UNIVERSE FILTER
# ─────────────────────────────────────────
# Backwards compatible:
# - unset ACTIVE_PAIRS => use full universe (existing behavior)
# - set ACTIVE_PAIRS => restrict entire MPML pipeline to those pairs
#
# Example:
# export ACTIVE_PAIRS=USDJPY,EURJPY,GBPJPY,EURCHF,USDCHF
#
ACTIVE_PAIRS_ENV = os.environ.get("ACTIVE_PAIRS", "").strip()

if ACTIVE_PAIRS_ENV:
    ACTIVE_PAIRS = {
        p.strip().upper()
        for p in ACTIVE_PAIRS_ENV.split(",")
        if p.strip()
    }
else:
    ACTIVE_PAIRS = None


def filter_pair_universe(raw_data: dict) -> dict:
    """
    Restrict pipeline universe to ACTIVE_PAIRS if configured.

    Preserves metadata keys beginning with "_".
    """
    if ACTIVE_PAIRS is None:
        print("[PAIR FILTER] inactive (full universe)")
        return raw_data

    filtered = {}

    for key, value in raw_data.items():
        # Preserve metadata entries
        if str(key).startswith("_"):
            filtered[key] = value
            continue

        if str(key).upper() in ACTIVE_PAIRS:
            filtered[key] = value

    kept_pairs = sorted([
        k for k in filtered.keys()
        if not str(k).startswith("_")
    ])

    print(f"[PAIR FILTER] active")
    print(f"[PAIR FILTER] requested={sorted(ACTIVE_PAIRS)}")
    print(f"[PAIR FILTER] kept={kept_pairs}")

    return filtered

# ─────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────
RUN_ID = None  # set to a string to force a specific id; otherwise auto-generated
_CURRENT_RUN_OUTPUT_DIR: Path | None = None

_VALID_EXPERIMENT_GENERATIONS = {"gen1", "gen2"}
_LEGACY_RESULTS_DIR = "results"
_DEFAULT_RUNS_ROOT = "results_archive"


def _validate_experiment_generation(value: str) -> str:
    gen = str(value).strip().lower()
    if gen not in _VALID_EXPERIMENT_GENERATIONS:
        raise ValueError(
            f"Invalid experiment generation: {value!r}. "
            f"Allowed values={sorted(_VALID_EXPERIMENT_GENERATIONS)}"
        )
    return gen


def _validate_experiment_variant(value: str) -> str:
    variant = normalize_variant(value)
    if variant is None:
        raise ValueError(
            f"Invalid experiment variant: {value!r}. "
            f"Allowed values={sorted(VALID_EXPERIMENT_VARIANTS)}"
        )
    return variant


def _build_experiment_metadata(
    *,
    variant: str,
    factor_overrides: dict | None = None,
) -> dict:
    return build_experiment_metadata_from_variant(
        _validate_experiment_variant(variant),
        factor_overrides=factor_overrides,
    )


def _set_run_output_dir(path: Path) -> None:
    global _CURRENT_RUN_OUTPUT_DIR
    _CURRENT_RUN_OUTPUT_DIR = Path(path).resolve()
    try:
        _CURRENT_RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(
            f"Run output directory already exists: {_CURRENT_RUN_OUTPUT_DIR}. "
            "Use a different --output-dir (or MPML_OUTPUT_DIR) to keep runs immutable."
        ) from exc


def _run_output_dir() -> Path:
    if _CURRENT_RUN_OUTPUT_DIR is None:
        raise RuntimeError("Run output directory has not been initialized.")
    return _CURRENT_RUN_OUTPUT_DIR


def _resolve_output_path(path: str) -> Path:
    raw = Path(path)
    if raw.parts and raw.parts[0] == _LEGACY_RESULTS_DIR:
        raw = Path(*raw.parts[1:])
    return _run_output_dir() / raw

# ─────────────────────────────────────────
# TRANSACTION COSTS (configurable)
# ─────────────────────────────────────────
SPREAD_PIPS = 1.0       # default 1.0
SLIPPAGE_PIPS = 0.5     # default 0.5
COMMISSION_PER_TRADE = 0.0  # default 0, in account currency per trade (round-trip handled by backtester if supported)

DYNAMIC_POLICY_KWARGS = dict(
    p_margin=0.20,
    use_prob_margin=True,
    min_hold_bars=10,
    use_hysteresis=True,
    use_min_hold=True,
    use_max_hold=True,
    max_hold_bars=60, # D1: ~3 trading months
)
WF_TAU = 0.62

# ─────────────────────────────────────────────────────
# VOLATILITY GUARD (global knobs)
# ─────────────────────────────────────────────────────
USE_VOL_GUARD = True

VOL_GUARD_Q = 0.80
VOL_GUARD_MODE = "no_mr"          # default action for non-USD-quote pairs on spike
VOL_FEATURE = "atr_pct"

# Group-aware override (implemented inside StrategySelector_Dynamic):
# - USD-quote pairs (e.g., EURUSD, GBPUSD, AUDUSD, NZDUSD): force TF on volatility spikes
# - All other pairs: apply VOL_GUARD_MODE on spikes
USD_QUOTE_VOL_SPIKE_OVERRIDE = "force_tf"

# ─────────────────────────────────────────────────────
# Diagnostics / debug artifacts
# ─────────────────────────────────────────────────────
DEBUG_SAVE_SELECTED_SERIES = True
DEBUG_SAVE_EQUITY_SERIES = True
# DEBUG_SELECTED_PAIRS = {"EURUSD", "AUDUSD", "USDCAD", "USDJPY", "GBPJPY"}
# DEBUG_SELECTED_MAX_FOLDS_PER_PAIR = 1  # keep outputs small
DEBUG_SELECTED_PAIRS = {"GBPJPY"}
DEBUG_SELECTED_MAX_FOLDS_PER_PAIR = 9
# Diagnostics: "near-spike" threshold for analysis/plots (does not affect trading logic)
VOL_GUARD_NEAR_MULT = 0.90  # near_thr = VOL_GUARD_NEAR_MULT * vol_thr
# Optional per-bar selector state timeline export (enables DL conditional timeline analysis).
# Produces selector_state_timeline.csv in the run results directory.
# Does not alter any runtime behaviour; pure observation export.
EXPORT_SELECTOR_STATE_TIMELINE = (
    os.environ.get("EXPORT_SELECTOR_STATE_TIMELINE", "false").lower() in {"1", "true", "yes", "on"}
)
# ─────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────

START_DATE = os.environ.get("MPML_START_DATE", "2005-01-01")
END_DATE = os.environ.get("MPML_END_DATE", "2026-04-01")
print(
    f"[CONFIG] START_DATE={START_DATE} "
    f"END_DATE={END_DATE}"
)
INITIAL_CAPITAL     = 10000.0
MIN_PHASE_SAMPLES   = 100       # Minimum samples per phase for ML
USE_ATR_SIZING      = False     # Set True to compare ATR-based sizing


# ─────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────

def usd_role(pair: str) -> str:
    if pair.startswith("USD"):
        return "USD-base"
    if pair.endswith("USD"):
        return "USD-quote"
    if "USD" in pair:
        return "USD-in-cross"
    return "No-USD"


def is_jpy_pair(pair: str) -> bool:
    return "JPY" in pair


def major_minor_group(pair: str, majors: list[str]) -> str:
    return "Major" if pair in majors else "Minor"


def _switch_count_from_selected(selected_s: pd.Series) -> int:
    """Count how many times the selected strategy type changes bar-to-bar."""
    if selected_s is None or len(selected_s) < 2:
        return 0
    # boolean series of changes; ignore first NaN from shift
    return int((selected_s != selected_s.shift(1)).iloc[1:].sum())


def compute_vol_guard_diagnostics(
    *,
    pair_name: str,
    fold_id: int,
    df_test: pd.DataFrame,
    selected_s: pd.Series,
    vol_feature: str,
    vol_thr: float | None,
    near_mult: float,
    majors: list[str],
    tau: float | None = None,
    tag: str = "wf",
) -> dict:
    """
    Compute per-fold diagnostics for volatility spike frequency and strategy-type switching.
    Does not require instrumenting StrategySelector_Dynamic internals.
    """
    n_bars = int(len(df_test))
    out = {
        "tag": tag,  # e.g. "wf" or "tau_sweep"
        "Tau": tau,
        "Pair": pair_name,
        "Fold": int(fold_id),
        "Bars": n_bars,
        "USD_role": usd_role(pair_name),
        "JPY": "JPY" if is_jpy_pair(pair_name) else "non-JPY",
        "MajorMinor": major_minor_group(pair_name, majors),
        "vol_feature": vol_feature,
        "vol_thr": float(vol_thr) if vol_thr is not None else np.nan,
        "near_mult": float(near_mult),
    }

    if (vol_thr is None) or (vol_feature not in df_test.columns) or (n_bars == 0):
        # Fill with NaNs/zeros so groupby works cleanly
        out.update({
            "spike_bars": 0,
            "spike_pct": np.nan,
            "near_spike_bars": 0,
            "near_spike_pct": np.nan,
            "switches_total": _switch_count_from_selected(selected_s),
            "switches_per_1000_bars": np.nan,
            "switches_on_spike": np.nan,
            "switches_on_near_spike": np.nan,
            "confident_pct": np.nan,
            "confident_pct_on_spike": np.nan,
            "confident_pct_off_spike": np.nan,
            "tf_on_spike_pct": np.nan,
            "mr_on_spike_pct": np.nan,
            "phaseaware_on_spike_pct": np.nan,
        })
        return out

    v = df_test[vol_feature].astype(float)
    spike_mask = (v >= float(vol_thr)) & v.notna()

    near_thr = float(vol_thr) * float(near_mult)
    near_spike_mask = (v >= near_thr) & v.notna()

    # switching masks (define switch at bar i when selected changes from i-1 -> i)
    switch_mask = (selected_s != selected_s.shift(1)).fillna(False)
    switch_mask.iloc[0] = False

    switches_total = int(switch_mask.sum())

    # confident means not PhaseAware
    confident_mask = (selected_s != "PhaseAware")

    # selection distributions on spike bars
    # (these are proxies for "guard effectiveness": MR-on-spike should be low)
    spike_sel = selected_s.loc[spike_mask.reindex(selected_s.index, fill_value=False)]
    if len(spike_sel) > 0:
        tf_on_spike = float((spike_sel == "TrendFollowing").mean() * 100.0)
        mr_on_spike = float((spike_sel == "MeanReversion").mean() * 100.0)
        pa_on_spike = float((spike_sel == "PhaseAware").mean() * 100.0)
    else:
        tf_on_spike = np.nan
        mr_on_spike = np.nan
        pa_on_spike = np.nan

    # confident % on/off spikes
    spike_idx = spike_mask.reindex(selected_s.index, fill_value=False)
    if int(spike_idx.sum()) > 0:
        conf_on_spike = float(confident_mask.loc[spike_idx].mean() * 100.0)
    else:
        conf_on_spike = np.nan

    off_spike_idx = (~spike_idx)
    if int(off_spike_idx.sum()) > 0:
        conf_off_spike = float(confident_mask.loc[off_spike_idx].mean() * 100.0)
    else:
        conf_off_spike = np.nan

    out.update({
        "spike_bars": int(spike_mask.sum()),
        "spike_pct": float(spike_mask.mean() * 100.0),
        "near_spike_bars": int(near_spike_mask.sum()),
        "near_spike_pct": float(near_spike_mask.mean() * 100.0),

        "switches_total": switches_total,
        "switches_per_1000_bars": float(switches_total / max(1, (n_bars - 1)) * 1000.0),

        "switches_on_spike": int((switch_mask & spike_idx).sum()),
        "switches_on_near_spike": int((switch_mask & near_spike_mask.reindex(selected_s.index, fill_value=False)).sum()),

        "confident_pct": float(confident_mask.mean() * 100.0),
        "confident_pct_on_spike": conf_on_spike,
        "confident_pct_off_spike": conf_off_spike,

        "tf_on_spike_pct": tf_on_spike,
        "mr_on_spike_pct": mr_on_spike,
        "phaseaware_on_spike_pct": pa_on_spike,
    })
    return out

def _pkg_version(name: str):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def _versions_block(run_cfg):
    return {
        "python": sys.version,
        # these are already captured in RunConfig, but keeping them here makes the manifest self-contained
        "python_version": getattr(run_cfg, "python_version", None),
        "platform": getattr(run_cfg, "platform", None),
        "git_sha": getattr(run_cfg, "git_sha", None),
        "packages": {
            "numpy": _pkg_version("numpy"),
            "pandas": _pkg_version("pandas"),
            "scikit-learn": _pkg_version("scikit-learn"),
            "xgboost": _pkg_version("xgboost"),
            "yfinance": _pkg_version("yfinance"),
            "ta": _pkg_version("ta"),
            "matplotlib": _pkg_version("matplotlib"),
            "seaborn": _pkg_version("seaborn"),
            "jupyter": _pkg_version("jupyter"),
            "notebook": _pkg_version("notebook"),
            "ipykernel": _pkg_version("ipykernel"),
        },
    }

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features and rolling statistics for ML experiments.

    These features are used by PhaseMLExperiment only —
    they are not required by the backtester or phase detector.

    Adds:
        return_lag_{1,2,3,5,10}    - lagged returns
        adx_lag_{1,2,3,5,10}       - lagged ADX values
        return_mean_{5,10,20}      - rolling mean of returns
        return_std_{5,10,20}       - rolling std of returns
        return_skew_{5,10,20}      - rolling skew of returns
        di_spread                  - +DI minus -DI
        di_ratio                   - +DI / -DI
    """
    df = df.copy()

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        df[f'adx_lag_{lag}']    = df['adx'].shift(lag)

    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'return_mean_{window}'] = (
            df['returns'].rolling(window).mean()
        )
        df[f'return_std_{window}'] = (
            df['returns'].rolling(window).std()
        )
        df[f'return_skew_{window}'] = (
            df['returns'].rolling(window).skew()
        )

    # DI spread and ratio
    df['di_spread'] = df['plus_di'] - df['minus_di']
    df['di_ratio']  = df['plus_di'] / (
        df['minus_di'].replace(0, np.nan)
    )

    df['returns_recent'] = df['returns'].rolling(window=10).mean()
    df['volatility_recent'] = df['returns'].rolling(window=10).std()
    return df


def _pair_to_dl_key(pair_name: str) -> str:
    """Convert pair name to DL artifact pair key format (xxx-yyy).

    For unexpected non-6-char symbols, return a best-effort lowercase fallback.
    """
    normalized_pair = str(pair_name).strip().lower()
    clean_pair = normalized_pair.replace("/", "").replace("-", "")
    if len(clean_pair) == 6 and clean_pair.isalpha():
        return f"{clean_pair[:3]}-{clean_pair[3:]}"
    return normalized_pair.replace("/", "-")


def _dl_surface_string(surface: dict) -> str:
    return (
        f"{surface.get('model', '?')}/"
        f"{surface.get('dl_regime', '?')}/"
        f"h{surface.get('target_horizon', '?')}/"
        f"{surface.get('feature_set', '?')}"
    )


def _with_mode_tag(path: str, mode_tag: str) -> str:
    target = _resolve_output_path(path)
    stem = target.stem
    suffix = target.suffix
    tagged = target.with_name(f"{stem}{mode_tag}{suffix}") if suffix else target.with_name(f"{stem}{mode_tag}")
    tagged.parent.mkdir(parents=True, exist_ok=True)
    return str(tagged)

def attach_dl_features(
    processed_df: pd.DataFrame,
    pair_name: str,
    surface: dict,
    artifact_path: Path | None,
) -> pd.DataFrame:
    """
    Attach D1-aggregated DL features to one processed pair frame via left join.

    Join key: (pair, timestamp_start_of_day), where pair is normalized to
    'xxx-yyy' and timestamp_start_of_day is daily midnight UTC.

    Robustness:
    - If ALL D1 DL feature columns are fully NaN after join for this pair/surface,
      drop them and proceed with baseline features (prevents downstream row collapse).

    Logging:
    - When DL_DEBUG_VERBOSE is False, keep only compact/high-signal logs.
    - When True, emit detailed per-column coverage and retention diagnostics.
    """
    # ------------------------------------------------------------------
    # Early exits / safety assertions
    # ------------------------------------------------------------------
    if artifact_path is None:
        print(f"  [DL] {pair_name}: no DL artifact resolved; skipping attachment.")
        return processed_df

    if not processed_df.index.is_unique:
        raise AssertionError(
            f"[DL] {pair_name}: processed_df index must be unique before DL join"
        )

    existing_d1_cols = [col for col in D1_FEATURE_COLS if col in processed_df.columns]
    if existing_d1_cols:
        raise AssertionError(
            f"[DL] {pair_name}: D1 DL feature columns already present before join: {existing_d1_cols}"
        )

    # ------------------------------------------------------------------
    # Load + validate surface and aggregate to D1 daily features
    # ------------------------------------------------------------------
    daily_df = load_and_aggregate_d1(
        artifact_path=artifact_path,
        surface=surface,
        strict=True,
    )
    if daily_df.empty:
        print(f"  [DL] {pair_name}: D1 aggregation produced no rows; skipping attachment.")
        return processed_df

    pair_key = _pair_to_dl_key(pair_name)
    daily_pair = daily_df[daily_df["pair"] == pair_key].copy()
    if daily_pair.empty:
        print(f"  [DL] {pair_name}: no D1 DL rows for pair key '{pair_key}'; skipping attachment.")
        return processed_df

    # Integrity assertions on daily features
    if not daily_pair["trading_day"].is_monotonic_increasing:
        raise AssertionError(f"[DL] {pair_name}: trading_day not monotonic increasing")
    if daily_pair.duplicated(subset=["pair", "trading_day"]).any():
        raise AssertionError(f"[DL] {pair_name}: duplicate (pair, trading_day) rows in D1 features")

    # ------------------------------------------------------------------
    # Build join frame (processed_df is D1 indexed by timestamp)
    # ------------------------------------------------------------------
    base = processed_df.copy().reset_index(names="_timestamp_original")
    original_timestamps = pd.to_datetime(base["_timestamp_original"])
    base["timestamp"] = original_timestamps.dt.normalize()
    base["pair"] = pair_key

    if not original_timestamps.is_monotonic_increasing:
        raise AssertionError(f"[DL] {pair_name}: processed timestamps are not monotonic increasing")
    if base.duplicated(subset=["pair", "_timestamp_original"]).any():
        raise AssertionError(f"[DL] {pair_name}: duplicate (pair, timestamp) rows before DL join")

    # Compact overlap/range diagnostic (always shown; high signal)
    try:
        proc_min = pd.to_datetime(base["_timestamp_original"]).min()
        proc_max = pd.to_datetime(base["_timestamp_original"]).max()
        dl_min = pd.to_datetime(daily_pair["trading_day"]).min()
        dl_max = pd.to_datetime(daily_pair["trading_day"]).max()
        print(
            f"  [DL] {pair_name}: processed range: {proc_min.date()} -> {proc_max.date()} | "
            f"dl range: {dl_min.date()} -> {dl_max.date()}"
        )
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Left join daily features
    # ------------------------------------------------------------------
    rows_before_join = len(base)

    # ----------------------------------------------------------
    # Normalize DL timestamps to daily midnight before merge
    #
    # This prevents silent merge failures caused by:
    # - intraday timestamps
    # - timezone serialization differences
    # - parquet datetime normalization inconsistencies
    # ----------------------------------------------------------

    # CONTRACT:
    # D1 features attached on trading day D must come from H1 observations on
    # D-1, never from D itself or from any future day.
    daily_pair["dl_feature_source_day"] = (
        pd.to_datetime(daily_pair["trading_day"]).dt.normalize() - pd.Timedelta(days=1)
    )
    daily_pair = daily_pair.rename(columns={"trading_day": "timestamp"})

    daily_pair["timestamp"] = (
        pd.to_datetime(daily_pair["timestamp"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    base["timestamp"] = (
        pd.to_datetime(base["timestamp"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    join_cols = ["pair", "timestamp", "dl_feature_source_day", *D1_FEATURE_COLS]
    # ----------------------------------------------------------
    # Pre-merge overlap diagnostics
    # ----------------------------------------------------------

    base_days = set(base["timestamp"].unique())
    dl_days = set(daily_pair["timestamp"].unique())

    overlap_days = len(base_days & dl_days)

    print(
        f"  [DL] {pair_name}: "
        f"base_days={len(base_days)} "
        f"dl_days={len(dl_days)} "
        f"overlap_days={overlap_days}"
    )

    if overlap_days == 0:
        print(
            f"  [DL] {pair_name}: zero timestamp overlap between MPML bars and "
            "DL daily features; skipping attachment."
        )
        return processed_df
    merged = base.merge(
        daily_pair[join_cols],
        on=["pair", "timestamp"],
        how="left",
        validate="many_to_one",
    )
    dl_non_null_rows = int(
        merged[list(D1_FEATURE_COLS)]
        .notna()
        .any(axis=1)
        .sum()
    )

    print(
        f"  [DL] {pair_name}: "
        f"rows_with_any_dl={dl_non_null_rows}/{len(merged)}"
    )
    print(
        f"  [DL] {pair_name}: dl_merge_hit_rate_pct="
        f"{(float(dl_non_null_rows) / float(len(merged)) * 100.0) if len(merged) else 0.0:.2f}"
    )

    if dl_non_null_rows == 0:
        print(
            f"  [DL] {pair_name}: DL merge produced zero non-null rows; "
            "skipping attachment."
        )
        return processed_df

    rows_after_join = len(merged)
    if rows_after_join != rows_before_join:
        raise AssertionError(
            f"[DL] {pair_name}: DL join multiplied rows ({rows_before_join} -> {rows_after_join})"
        )
    if merged.duplicated(subset=["pair", "_timestamp_original"]).any():
        raise AssertionError(f"[DL] {pair_name}: duplicate (pair, timestamp) rows after DL join")

    matched_dl_mask = merged[list(D1_FEATURE_COLS)].notna().any(axis=1)
    if matched_dl_mask.any():
        matched_rows = merged.loc[matched_dl_mask]
        merge_lag_days = (
            pd.to_datetime(matched_rows["timestamp"]).dt.normalize()
            - pd.to_datetime(matched_rows["dl_feature_source_day"]).dt.normalize()
        ).dt.days
        if (merge_lag_days < 1).any():
            bad_lag = matched_rows.loc[
                merge_lag_days < 1,
                ["_timestamp_original", "timestamp", "dl_feature_source_day"],
            ].head(3)
            raise AssertionError(
                f"[DL] {pair_name}: non-causal DL merge lag detected (must be >= 1 day). "
                f"sample={bad_lag.to_dict('records')}"
            )
        print(
            f"  [DL] {pair_name}: dl_merge_lag_days "
            f"min={float(merge_lag_days.min()):.2f} "
            f"median={float(merge_lag_days.median()):.2f} "
            f"max={float(merge_lag_days.max()):.2f}"
        )

    # ------------------------------------------------------------------
    # Coverage detection + optional verbose diagnostics
    # ------------------------------------------------------------------
    # Determine if DL coverage is zero (all DL cols fully NaN)
    zero_dl_coverage = all(merged[col].notna().sum() == 0 for col in D1_FEATURE_COLS)

    # Keep a compact success summary even when not verbose
    if zero_dl_coverage:
        print(
            f"  [DL] {pair_name}: zero DL coverage after join; dropping DL feature columns "
            f"and reverting to baseline features"
        )
        merged = merged.drop(columns=list(D1_FEATURE_COLS))
    else:
        # Non-zero coverage summary (compact)
        if not DL_DEBUG_VERBOSE:
            # Print only the best single-line signal: % rows with any DL present
            any_dl = merged[list(D1_FEATURE_COLS)].notna().any(axis=1)
            any_dl_pct = float(any_dl.mean() * 100.0) if len(any_dl) else 0.0
            print(f"  [DL] {pair_name}: DL coverage (any col)={any_dl_pct:.2f}%")
        else:
            print(f"  [DL] {pair_name}: attached DL columns={list(D1_FEATURE_COLS)}")
            for col in D1_FEATURE_COLS:
                nn = merged[col].notna()
                coverage = float(nn.mean() * 100.0) if len(nn) else 0.0
                if nn.any():
                    ts_non_null = merged.loc[nn, "_timestamp_original"]
                    first_ts = ts_non_null.min()
                    last_ts = ts_non_null.max()
                    print(
                        f"    [DL] {col}: coverage={coverage:.2f}% "
                        f"first_non_null={first_ts} last_non_null={last_ts}"
                    )
                else:
                    print(f"    [DL] {col}: coverage=0.00% first_non_null=None last_non_null=None")

    # ------------------------------------------------------------------
    # Optional verbose retention diagnostics (can be very noisy)
    # ------------------------------------------------------------------
    if DL_DEBUG_VERBOSE:
        feature_cols = [
            col for col in merged.columns
            if col not in PhaseMLExperiment.EXCLUDE_COLS
            and is_numeric_dtype(merged[col])
            and not is_bool_dtype(merged[col])
        ]
        feature_cols = _stable_feature_columns(feature_cols)
        optional_dl_cols = [c for c in D1_FEATURE_COLS if c in feature_cols]
        required_feature_cols = [c for c in feature_cols if c not in optional_dl_cols]

        required_mask = (
            merged[required_feature_cols].notna().all(axis=1)
            if required_feature_cols
            else pd.Series(True, index=merged.index)
        )
        rows_after_required_mask = int(required_mask.sum())
        rows_after_optional_imputation = rows_after_required_mask
        dl_coverage_pct = (
            float(merged[optional_dl_cols].notna().any(axis=1).mean() * 100.0)
            if optional_dl_cols and len(merged)
            else 0.0
        )
        retention_ratio = (
            rows_after_optional_imputation / rows_after_join
            if rows_after_join
            else np.nan
        )

        print(f"  [DL] {pair_name}: rows before DL join={rows_before_join}")
        print(f"  [DL] {pair_name}: rows after DL join={rows_after_join}")
        print(f"  [DL] {pair_name}: rows_after_required_mask={rows_after_required_mask}")
        print(f"  [DL] {pair_name}: rows_after_optional_imputation={rows_after_optional_imputation}")
        print(f"  [DL] {pair_name}: dl_coverage_pct={dl_coverage_pct:.2f}")
        print(f"  [DL] {pair_name}: effective_training_samples={rows_after_optional_imputation}")
        print(f"  [DL] {pair_name}: retention ratio={retention_ratio:.4f}")

    # ------------------------------------------------------------------
    # Rebuild output with original index
    # ------------------------------------------------------------------
    out = merged.drop(columns=["pair", "timestamp", "dl_feature_source_day"]).set_index("_timestamp_original")
    out.index.name = processed_df.index.name
    return out

def process_pair(pair_name: str,
                 df: pd.DataFrame,
                 detector: MarketPhaseDetector) -> pd.DataFrame | None:
    """
    Run phase detection and feature engineering for a single pair.

    Args:
        pair_name: Short name e.g. 'EURUSD'
        df:        Prepared DataFrame from data pipeline
        detector:  Shared MarketPhaseDetector instance

    Returns:
        Fully processed DataFrame, or None if processing fails.
    """
    try:
        # Phase detection
        df = detector.detect_phases(df)

        # ML feature engineering
        df = engineer_features(df)
        returns_recent = df["returns"].shift(1).rolling(5).mean()
        volatility_recent = df["atr_pct"].shift(1).rolling(5).mean()
        df = df.dropna()

        if len(df) < 300:
            print(f'  ✗ {pair_name}: too few rows after processing '
                  f'({len(df)}), skipping')
            return None

        print(f'  ✓ {pair_name}: {len(df)} rows, '
              f'{df["phase"].nunique()} phases detected')
        return df

    except Exception as e:
        print(f'  ✗ {pair_name}: processing failed — {e}')
        return None


def aggregate_backtest_results(pair_results: dict,
                                group_pairs: list,
                                group_name: str) -> pd.DataFrame:
    """
    Aggregate backtest results across a group of pairs using
    trade-count weighted averages.

    Weighting by number of trades is fairer than simple averaging
    because pairs with more trades contribute more information.

    Args:
        pair_results: Dict of {pair_name: backtest_results_dict}
        group_pairs:  List of pair names in this group
        group_name:   Label for the group ('Majors' or 'Minors')

    Returns:
        DataFrame with per-strategy weighted average metrics.
    """
    strategy_rows = {}

    for pair_name in group_pairs:
        if pair_name not in pair_results:
            continue

        results = pair_results[pair_name]

        for strategy_name, metrics in results.items():
            # Skip metadata keys
            if strategy_name.startswith('_'):
                continue

            if strategy_name not in strategy_rows:
                strategy_rows[strategy_name] = []

            strategy_rows[strategy_name].append({
                'pair':           pair_name,
                'total_return':   metrics['total_return'],
                'sharpe_ratio':   metrics['sharpe_ratio'],
                'max_drawdown':   metrics['max_drawdown'],
                'win_rate':       metrics['win_rate'],
                'profit_factor':  metrics['profit_factor'],
                'n_trades':       metrics['n_trades'],
            })

    if not strategy_rows:
        print(f'  ✗ No results to aggregate for {group_name}')
        return pd.DataFrame()

    # Compute weighted averages
    summary_rows = []
    for strategy_name, rows in strategy_rows.items():
        rows_df = pd.DataFrame(rows)
        weights = rows_df['n_trades']
        total_weight = weights.sum()

        if total_weight == 0:
            continue

        def wavg(col):
            return (rows_df[col] * weights).sum() / total_weight

        summary_rows.append({
            'Group':            group_name,
            'Strategy':         strategy_name,
            'Total Return (%)': round(wavg('total_return'), 2),
            'Sharpe Ratio':     round(wavg('sharpe_ratio'), 4),
            'Max Drawdown (%)': round(wavg('max_drawdown'), 2),
            'Win Rate (%)':     round(wavg('win_rate'), 2),
            'Profit Factor':    round(wavg('profit_factor'), 4),
            'Total Trades':     int(weights.sum()),
            'Pairs':            len(rows_df),
        })

    return pd.DataFrame(summary_rows)


def print_phase_distribution(df: pd.DataFrame,
                              pair_name: str) -> None:
    """
    Print phase distribution and duration statistics for a single pair.

    Helps calibrate ML training window size and retraining frequency.
    """
    phase_counts = df['phase'].value_counts()
    total        = len(df)

    print(f'\n  {pair_name} phase distribution:')
    for phase, count in phase_counts.items():
        pct = count / total * 100
        print(f'    {phase:<20} {count:>5} ({pct:.1f}%)')

    # ── Phase duration statistics ─────────────────────────────────────────
    # Identify consecutive runs of the same phase
    phase_series  = df['phase']
    run_lengths   = {}
    transitions   = 0

    current_phase = phase_series.iloc[0]
    current_len   = 1

    for i in range(1, len(phase_series)):
        if phase_series.iloc[i] == current_phase:
            current_len += 1
        else:
            # Phase changed — record the completed run
            if current_phase not in run_lengths:
                run_lengths[current_phase] = []
            run_lengths[current_phase].append(current_len)
            transitions  += 1
            current_phase = phase_series.iloc[i]
            current_len   = 1

    # Don't forget the last run
    if current_phase not in run_lengths:
        run_lengths[current_phase] = []
    run_lengths[current_phase].append(current_len)

    # Calculate years in data
    n_years = (df.index[-1] - df.index[0]).days / 365.25

    print(f'\n  {pair_name} phase duration statistics (bars):')
    print(f'    {"Phase":<20} {"Mean":>6} {"Median":>8} '
          f'{"Min":>6} {"Max":>6} {"N runs":>8}')
    print(f'    {"-" * 58}')

    all_durations = []
    for phase in sorted(run_lengths.keys()):
        durations = run_lengths[phase]
        all_durations.extend(durations)
        print(
            f'    {phase:<20} '
            f'{np.mean(durations):>6.1f} '
            f'{np.median(durations):>8.1f} '
            f'{np.min(durations):>6} '
            f'{np.max(durations):>6} '
            f'{len(durations):>8}'
        )

    print(f'    {"-" * 58}')
    print(
        f'    {"ALL PHASES":<20} '
        f'{np.mean(all_durations):>6.1f} '
        f'{np.median(all_durations):>8.1f} '
        f'{np.min(all_durations):>6} '
        f'{np.max(all_durations):>6} '
        f'{len(all_durations):>8}'
    )
    print(
        f'\n    Phase transitions: {transitions} '
        f'({transitions / n_years:.1f} per year)'
    )
    print(
        f'    Avg phase duration: {np.mean(all_durations):.1f} bars '
        f'({np.mean(all_durations) / 21:.1f} months)'
    )

def save_results(all_pair_results: dict,
                 majors_summary: pd.DataFrame,
                 minors_summary: pd.DataFrame,
                 mode_tag: str) -> None:
    """Save all results to CSV files."""
    _run_output_dir().mkdir(parents=True, exist_ok=True)

    # Per-pair results
    per_pair_rows = []
    for pair_name, results in all_pair_results.items():
        for strategy_name, metrics in results.items():
            if strategy_name.startswith('_'):
                continue
            per_pair_rows.append({
                'Pair':             pair_name,
                'Group':            (
                    'Major' if pair_name in [
                        PAIR_NAMES[t] for t in MAJORS
                    ] else 'Minor'
                ),
                'Strategy':         strategy_name,
                'Total Return (%)': metrics['total_return'],
                'Sharpe Ratio':     metrics['sharpe_ratio'],
                'Max Drawdown (%)': metrics['max_drawdown'],
                'Win Rate (%)':     metrics['win_rate'],
                'Profit Factor':    metrics['profit_factor'],
                'N Trades':         metrics['n_trades'],
            })

    per_pair_df = pd.DataFrame(per_pair_rows)
    per_pair_path = _with_mode_tag('results/results_per_pair.csv', mode_tag)
    per_pair_df.to_csv(per_pair_path, index=False)
    print(f'  ✓ Saved {per_pair_path}')

    # Group summaries
    if not majors_summary.empty:
        majors_path = _with_mode_tag('results/results_majors.csv', mode_tag)
        majors_summary.to_csv(majors_path, index=False)
        print(f'  ✓ Saved {majors_path}')

    if not minors_summary.empty:
        minors_path = _with_mode_tag('results/results_minors.csv', mode_tag)
        minors_summary.to_csv(minors_path, index=False)
        print(f'  ✓ Saved {minors_path}')

    # Combined summary
    combined_summary = pd.concat(
        [majors_summary, minors_summary],
        ignore_index=True
    )
    summary_path = _with_mode_tag('results/results_summary.csv', mode_tag)
    combined_summary.to_csv(summary_path, index=False)
    print(f'  ✓ Saved {summary_path}')

#======
# fold generator helper functions:
#=====
def _find_index_pos(dt_index: pd.DatetimeIndex, dt: pd.Timestamp) -> int:
    """
    Return integer position of the last index value <= dt.
    Raises if dt is earlier than the first index value.
    """
    pos = dt_index.searchsorted(dt, side="right") - 1
    if pos < 0:
        raise ValueError(f"Date {dt} is before start of series {dt_index[0]}")
    return int(pos)


def _window_diagnostics(
    *,
    train_start_ts: pd.Timestamp,
    train_end_ts: pd.Timestamp,
    test_start_ts: pd.Timestamp,
    test_end_ts: pd.Timestamp,
) -> dict:
    """Return causal diagnostics for one train/test window."""
    train_start_ts = pd.Timestamp(train_start_ts)
    train_end_ts = pd.Timestamp(train_end_ts)
    test_start_ts = pd.Timestamp(test_start_ts)
    test_end_ts = pd.Timestamp(test_end_ts)

    assert train_start_ts <= train_end_ts, (
        "train_start_ts must be <= train_end_ts "
        f"({train_start_ts} !<= {train_end_ts})"
    )
    assert test_start_ts <= test_end_ts, (
        "test_start_ts must be <= test_end_ts "
        f"({test_start_ts} !<= {test_end_ts})"
    )
    assert train_end_ts < test_start_ts, (
        "train_end_ts must be < test_start_ts "
        f"({train_end_ts} !< {test_start_ts})"
    )

    # Normalize to calendar days because fold diagnostics are reported in day
    # units and some callers may provide non-midnight timestamps.
    train_start_day = train_start_ts.normalize()
    train_end_day = train_end_ts.normalize()
    test_start_day = test_start_ts.normalize()
    test_end_day = test_end_ts.normalize()

    overlap_start = max(train_start_day, test_start_day)
    overlap_end = min(train_end_day, test_end_day)
    overlap_days = (
        int((overlap_end - overlap_start).days + 1)
        if overlap_start <= overlap_end
        else 0
    )
    gap_days = int((test_start_day - train_end_day).days)
    return {
        "train_start_ts": train_start_ts,
        "train_end_ts": train_end_ts,
        "test_start_ts": test_start_ts,
        "test_end_ts": test_end_ts,
        "gap_days": gap_days,
        "overlap_days": overlap_days,
    }


def _print_window_diagnostics(prefix: str, **diag) -> None:
    print(
        f"{prefix} "
        f"train={diag['train_start_ts']} -> {diag['train_end_ts']} "
        f"test={diag['test_start_ts']} -> {diag['test_end_ts']} "
        f"gap_days={diag['gap_days']} overlap_days={diag['overlap_days']}"
    )


def generate_walkforward_folds_by_pos(
    dates: pd.DatetimeIndex,
    train_years: int = 7,
    test_months: int = 6,
    step_months: int = 6,
) -> list[dict]:
    """
    Walk-forward folds using date boundaries but converted to integer positions.

    Expanding window:
      - train_start fixed at first date
      - train_end advances by step_months
      - test window length = test_months
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if not dates.is_monotonic_increasing:
        dates = dates.sort_values()
    start = pd.Timestamp(dates.min())
    end = pd.Timestamp(dates.max())

    # We'll walk train_end forward in time using DateOffset,
    # then map boundaries to integer positions.
    train_start_dt = start
    train_end_dt = train_start_dt + pd.DateOffset(years=train_years)

    folds = []
    fold_id = 0

    while True:
        test_start_dt = train_end_dt + pd.Timedelta(days=1)
        test_end_dt = test_start_dt + pd.DateOffset(months=test_months)

        if test_start_dt >= end:
            break
        if test_end_dt > end:
            test_end_dt = end

        # Convert to positions (snap to nearest available bar <= boundary)
        train_start_pos = 0
        train_end_pos = _find_index_pos(dates, train_end_dt)

        # ----------------------------------------------------------
        # Causal fold boundary:
        # test must begin strictly AFTER the final train bar.
        #
        # Using calendar-based lookup for both train_end_dt and
        # test_start_dt can collapse onto the same trading bar when
        # the index is sparse/non-uniform.
        #
        # Therefore test_start_pos is defined positionally.
        # ----------------------------------------------------------
        test_start_pos = train_end_pos + 1

        if test_start_pos >= len(dates):
            break

        test_end_pos = _find_index_pos(dates, test_end_dt)

        if test_end_pos <= test_start_pos:
            break

        train_start_ts = dates[train_start_pos]
        train_end_ts = dates[train_end_pos]
        test_start_ts = dates[test_start_pos]
        test_end_ts = dates[test_end_pos]
        window_diag = _window_diagnostics(
            train_start_ts=train_start_ts,
            train_end_ts=train_end_ts,
            test_start_ts=test_start_ts,
            test_end_ts=test_end_ts,
        )

        folds.append({
            "fold": fold_id,
            "train_start_pos": train_start_pos,
            "train_end_pos": train_end_pos,
            "test_start_pos": test_start_pos,
            "test_end_pos": test_end_pos,
            "train_start_dt": str(train_start_ts.date()),
            "train_end_dt": str(train_end_ts.date()),
            "test_start_dt": str(test_start_ts.date()),
            "test_end_dt": str(test_end_ts.date()),
            "gap_days": window_diag["gap_days"],
            "overlap_days": window_diag["overlap_days"],
        })
        fold_id += 1

        # expanding window: move train_end forward
        train_end_dt = train_end_dt + pd.DateOffset(months=step_months)

        if train_end_dt >= end:
            break

    return folds
# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def _make_strategy_dicts() -> tuple[dict, dict]:
    """Return fresh TF and MR strategy instances (call per fold to avoid state sharing)."""
    tf_strategies = {
        'TF1': TF1Strategy(), 'TF2': TF2Strategy(), 'TF3': TF3Strategy(),
        'TF4': TF4Strategy(), 'TF5': TF5Strategy(),
    }
    mr_strategies = {
        'MR1': MR1Strategy(), 'MR2': MR2Strategy(), 'MR32': MR32Strategy(),
        'MR42': MR42Strategy(), 'MR5': MR5Strategy(),
    }
    return tf_strategies, mr_strategies


def _compute_vol_threshold(df_train_bars: pd.DataFrame) -> float | None:
    """Compute per-fold vol-guard threshold from training bars (no leakage).

    Uses module globals VOL_FEATURE and VOL_GUARD_Q.
    """
    if VOL_FEATURE not in df_train_bars.columns:
        return None
    s = df_train_bars[VOL_FEATURE].dropna()
    return float(s.quantile(VOL_GUARD_Q)) if len(s) > 0 else None


def _run_baseline_bt(df_test: pd.DataFrame, pip_value: float) -> dict:
    """Run PhaseAware(TF4/MR42) baseline backtest on a test slice.

    Uses module globals INITIAL_CAPITAL, SPREAD_PIPS, SLIPPAGE_PIPS,
    COMMISSION_PER_TRADE.
    """
    backtester = BT(
        initial_capital=INITIAL_CAPITAL,
        spread_pips=SPREAD_PIPS,
        slippage_pips=SLIPPAGE_PIPS,
        commission_per_trade=COMMISSION_PER_TRADE,
        pip_value=pip_value,
        use_atr_sizing=False,
    )
    pa = PhaseAwareStrategy("TF4", "MR42")
    pa_signals, pa_sl, pa_tp = pa.generate_signals(df_test)
    return backtester.run(df_test, pa_signals, "PhaseAware_TF4_MR42_WF", pa_sl, pa_tp)


def _build_causal_selector_training_data(
    *,
    pair_name: str,
    fold_id: int,
    df_full: pd.DataFrame,
    pair_results_full: dict,
    train_start_pos: int,
    train_end_pos: int,
    test_start_pos: int,
    test_end_pos: int,
    label_horizon_bars: int,
    context_label: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Build selector training data using only bars fully contained in the train fold.

    CONTRACT:
    Selector labels must be computed from future strategy performance that is
    still fully contained inside the train fold. No label horizon may cross the
    fold boundary into the test window.
    """
    if train_end_pos <= train_start_pos:
        raise AssertionError(
            f"[SELECTOR WINDOW] {pair_name} fold={fold_id}: invalid train positions "
            f"{train_start_pos}->{train_end_pos}"
        )
    if test_start_pos >= len(df_full):
        raise AssertionError(
            f"[SELECTOR WINDOW] {pair_name} fold={fold_id}: invalid test_start_pos={test_start_pos} "
            f"len(df_full)={len(df_full)}"
        )

    selector_train_end_pos = train_end_pos - label_horizon_bars
    if selector_train_end_pos < train_start_pos:
        return pd.DataFrame(), {}

    train_start_ts = df_full.index[train_start_pos]
    train_end_ts = df_full.index[train_end_pos]
    selector_train_end_ts = df_full.index[selector_train_end_pos]
    test_start_ts = df_full.index[test_start_pos]
    test_end_ts = df_full.index[test_end_pos]

    window_diag = _window_diagnostics(
        train_start_ts=train_start_ts,
        train_end_ts=train_end_ts,
        test_start_ts=test_start_ts,
        test_end_ts=test_end_ts,
    )

    assert selector_train_end_ts < test_start_ts, (
        f"[SELECTOR WINDOW] {pair_name} fold={fold_id}: "
        f"selector_train_end_ts={selector_train_end_ts} must be < test_start_ts={test_start_ts}"
    )

    # +1 is intentional: iloc end is exclusive, and we need the train_end_pos
    # bar included so the latest admissible selector label can be computed from
    # train-fold-contained history only.
    df_for_labels = df_full.iloc[train_start_pos:train_end_pos + 1].copy()

    strat_results_for_labels = {}
    for sname, res in pair_results_full.items():
        eq = res.get("equity_curve", None)
        if eq is None:
            continue
        strat_results_for_labels[sname] = dict(res)
        strat_results_for_labels[sname]["equity_curve"] = eq.reindex(df_for_labels.index).ffill()

    tracker = StrategyPerformanceTracker(window_days=label_horizon_bars)
    training_data_all = tracker.compute_strategy_returns(
        df_for_labels,
        strat_results_for_labels,
    )
    if training_data_all.empty:
        return training_data_all, {}

    training_data = training_data_all.loc[
        (training_data_all["date"] >= train_start_ts)
        & (training_data_all["date"] <= selector_train_end_ts)
    ].copy()

    if not training_data.empty:
        ranking_start_ts = pd.Timestamp(training_data["date"].min())
        ranking_end_ts = pd.Timestamp(training_data["date"].max())
        assert ranking_end_ts <= selector_train_end_ts, (
            f"[SELECTOR WINDOW] {pair_name} fold={fold_id}: "
            f"ranking_end_ts={ranking_end_ts} exceeded selector_train_end_ts={selector_train_end_ts}"
        )
    else:
        ranking_start_ts = pd.NaT
        ranking_end_ts = pd.NaT

    print(
        f"    [SELECTOR WINDOW] {context_label} pair={pair_name} fold={fold_id} "
        f"train={train_start_ts} -> {train_end_ts} "
        f"selector_train={train_start_ts} -> {selector_train_end_ts} "
        f"eval={test_start_ts} -> {test_end_ts} "
        f"label_horizon_bars={label_horizon_bars} "
        f"fold_gap_days={window_diag['gap_days']} "
        f"ranking_ts_range={ranking_start_ts} -> {ranking_end_ts}"
    )

    return training_data, {
        "train_start_ts": train_start_ts,
        "train_end_ts": train_end_ts,
        "selector_train_end_ts": selector_train_end_ts,
        "test_start_ts": test_start_ts,
        "test_end_ts": test_end_ts,
        "ranking_start_ts": ranking_start_ts,
        "ranking_end_ts": ranking_end_ts,
    }


def _assert_backtest_index_matches(pair_name: str, df_ref, results: dict, tag: str) -> None:
    for sname, res in results.items():
        eq = res.get("equity_curve")
        if eq is None:
            continue
        if not eq.index.equals(df_ref.index):
            print(f"[FATAL] {pair_name} {tag} {sname}: eq.index != df.index right after backtest")
            print("  df len:", len(df_ref), "eq len:", len(eq))
            print("  missing in eq:", list(df_ref.index.difference(eq.index)[:10]))
            print("  extra in eq:", list(eq.index.difference(df_ref.index)[:10]))
            raise RuntimeError("Equity curve index mismatch at creation time")


def main(
    *,
    output_dir: Path | None = None,
    experiment_generation: str | None = None,
    experiment_variant: str | None = None,
    experiment_seed: int | None = None,
):
    resolved_seed = resolve_experiment_seed(
        cli_seed=experiment_seed,
        default_seed=DEFAULT_EXPERIMENT_SEED,
    )
    run_cfg = build_run_config(seed=resolved_seed, run_id=RUN_ID)
    reproducibility_block = set_global_seed(run_cfg.seed)
    dl_surface = dict(DL_SIGNAL_SURFACE)
    dl_regime = str(dl_surface.get("dl_regime", "")).strip().upper()
    dl_runtime_enabled = bool(DL_SIGNALS_ENABLED)
    # ----------------------------------------------------------
    # DL artifact resolution
    #
    # Precedence:
    #   1. Explicit env override
    #   2. Automatic resolver
    # ----------------------------------------------------------

    explicit_dl_artifact = os.getenv("DL_PREDICTION_ARTIFACT_PATH", "").strip()
    artifact_regime = None

    if dl_runtime_enabled:
        if explicit_dl_artifact:
            dl_artifact_path = Path(explicit_dl_artifact)
            print("[DL] using explicit artifact override from environment")
        else:
            dl_artifact_path = resolve_dl_prediction_artifact_path()

        if dl_artifact_path is not None:
            dl_artifact_path = Path(dl_artifact_path)

    else:
        dl_artifact_path = None

    if dl_runtime_enabled and dl_artifact_path is None:
        print("[WARN] DL enabled but no artifact resolved; DL features will be skipped.")

    if dl_runtime_enabled and dl_artifact_path is not None:
        artifact_regime = infer_dl_regime_from_artifact_path(dl_artifact_path)
    if dl_runtime_enabled and not dl_regime and artifact_regime:
        dl_regime = artifact_regime
        print(f"[DL] inferred dl_regime={dl_regime} from artifact path")

    if not dl_regime:
        dl_regime = DEFAULT_DL_REGIME
        print(
            "[WARN] DL_REGIME not set and artifact did not encode regime; "
            f"defaulting dl_regime={DEFAULT_DL_REGIME}."
        )

    dl_surface["dl_regime"] = dl_regime

    if dl_runtime_enabled and dl_regime not in VALID_DL_REGIMES:
        print(
            f"[WARN] DL features will not be attached (baseline mode): "
            f"invalid dl_regime={dl_regime!r}. Valid values={sorted(VALID_DL_REGIMES)} "
            f"(no 'all' support in v1)."
        )
        dl_runtime_enabled = False
    dl_mode_tag = "__dl_enabled" if dl_runtime_enabled else "__baseline"
    dl_surface_str = _dl_surface_string(dl_surface)
    selected_variant = (
        experiment_variant
        if experiment_variant is not None
        else os.getenv("EXPERIMENT_VARIANT", "A")
    )
    requested_msml_regime = os.getenv("MSML_REGIME", dl_regime).strip().upper() or dl_regime
    overlap_only = os.getenv("OVERLAP_ONLY", "false").strip().lower() in {"1", "true", "yes", "on"}
    base_experiment_meta = _build_experiment_metadata(variant=selected_variant)
    base_factors = dict(base_experiment_meta.get("factors") or {})
    run_name_hint = str(
        output_dir
        or os.getenv("MPML_OUTPUT_DIR")
        or ""
    ).strip()
    awareness_hint = infer_imputation_awareness_from_name(
        Path(run_name_hint).name if run_name_hint else None
    )
    factor_overrides = {
        # Baseline no-DL runs are first-class factor cohorts.
        "dl_enabled": dl_runtime_enabled,
        "sentiment_enabled": bool(base_factors.get("sentiment_enabled")) and dl_runtime_enabled,
        "missing_indicators_enabled": (
            awareness_hint
            if isinstance(awareness_hint, bool)
            else bool(base_factors.get("missing_indicators_enabled"))
        ),
        "msml_regime": requested_msml_regime,
        "overlap_only": overlap_only,
        "selector_enabled": bool(RUN_WALKFORWARD),
    }
    experiment_meta = _build_experiment_metadata(
        variant=selected_variant,
        factor_overrides=factor_overrides,
    )
    experiment_surface = build_runtime_experiment_surface(
        dl_runtime_enabled=dl_runtime_enabled,
        dl_surface=dl_surface,
        dl_artifact_path=dl_artifact_path,
        experiment_factors=experiment_meta.get("factors") or {},
    )
    generation_hint = (
        experiment_generation
        if experiment_generation is not None
        else os.getenv("EXPERIMENT_GENERATION")
    )
    if generation_hint is not None:
        normalized_generation_hint = _validate_experiment_generation(generation_hint)
        if normalized_generation_hint != experiment_meta["generation"]:
            raise ValueError(
                "Conflicting experiment semantic inputs: "
                f"variant={experiment_meta['variant']} implies generation={experiment_meta['generation']}, "
                f"but generation hint was {normalized_generation_hint}."
            )
    run_ts = run_cfg.run_id.replace("run_", "")
    computed_default_output_dir = Path(
        os.getenv("MPML_RUNS_ROOT", _DEFAULT_RUNS_ROOT)
    ) / f"{experiment_meta['generation']}_{experiment_meta['variant']}__{run_ts}"
    selected_output_dir = Path(
        output_dir or os.getenv("MPML_OUTPUT_DIR", computed_default_output_dir)
    )
    market_data_source = resolve_market_data_source()
    pipeline = MarketDataPipeline(
        start=START_DATE,
        end=END_DATE,
        source=market_data_source,
        use_cache=True
    )
    if market_data_source == "broker_csv":
        loader = getattr(pipeline, "loader", None)
        if loader is None or not hasattr(loader, "data_root") or not hasattr(loader, "timezone_name"):
            raise ValueError(
                "broker_csv source requires loader.data_root and loader.timezone_name for provenance"
            )
        market_data_root = str(loader.data_root)
        market_data_timezone = str(loader.timezone_name)
    else:
        market_data_root = "yfinance"
        market_data_timezone = "unknown"
    _set_run_output_dir(selected_output_dir)

    print('=' * 60)
    print("=== RUN MODE: DL ENABLED ===" if dl_runtime_enabled else "=== RUN MODE: BASELINE ===")
    print(f"ACTIVE_PAIRS={sorted(ACTIVE_PAIRS) if ACTIVE_PAIRS else 'ALL'}")
    print(f"DL enabled: {dl_runtime_enabled}")
    print(f"DL selected surface: {dl_surface}")
    print(f"surface={dl_surface_str}")
    print(f"DL resolved artifact path: {dl_artifact_path}")
    print("[DL REGIME]")
    print(f"artifact_path={dl_artifact_path}")
    print(f"artifact_regime={artifact_regime if artifact_regime else 'unknown'}")
    print(f"runtime_regime={dl_regime}")
    print(f"Market data source: {market_data_source}")
    print(f"Output dir: {_run_output_dir()}")
    print(f"Experiment: {experiment_meta}")
    print('=' * 60)

    # Convert ticker-keyed pip values into short-name-keyed pip values
    PIP_VALUES_BY_PAIRNAME = {
        PAIR_NAMES.get(ticker, ticker.replace("=X", "")): pip
        for ticker, pip in PIP_VALUES.items()
    }

    manifest = {
        "run": {
            **run_cfg.__dict__,
            "timestamp_utc": run_cfg.run_id.replace("run_", ""),
            # Legacy compatibility for downstream tooling still reading run.*.
            # New integrations should migrate to manifest.experiment.
            "experiment_gen": experiment_meta["generation"],
            "run_variant": experiment_meta["variant"],
        },
        "experiment": experiment_meta,
        "experiment_surface": experiment_surface,
        "market_data_source": market_data_source,
        "market_data_root": market_data_root,
        "market_data_timezone": market_data_timezone,
        "reproducibility": reproducibility_block,
        "feature_ordering": {
            "dl_feature_columns": [],
            "phase_predictor_by_pair": {},
            "strategy_selector_by_pair": {},
        },
        "flags": {
            "RUN_IN_SAMPLE_ABLATION": RUN_IN_SAMPLE_ABLATION,
            "RUN_WALKFORWARD": RUN_WALKFORWARD,
            "RUN_TAU_SWEEP": RUN_TAU_SWEEP,
            "RUN_POLICY_SWEEP": RUN_POLICY_SWEEP,
            "DEBUG_FEATURE_COLUMNS": DEBUG_FEATURE_COLUMNS,
            "DEBUG_SIGNAL_TYPES": DEBUG_SIGNAL_TYPES,
            "DL_SIGNALS_ENABLED": dl_runtime_enabled,
        },
        "dl": {
            "dl_enabled": dl_runtime_enabled,
            "dl_mode_tag": dl_mode_tag,
            "dl_surface": dl_surface,
            "dl_surface_string": dl_surface_str,
            "dl_artifact_path": str(dl_artifact_path) if dl_artifact_path is not None else None,
            # Populated after pair processing (see below)
            "dl_feature_columns": [],
            "dl_feature_count": 0,
        },
        "walkforward": {
            "train_years": WF_TRAIN_YEARS,
            "test_months": WF_TEST_MONTHS,
            "step_months": WF_STEP_MONTHS,
            "label_horizon_bars": LABEL_HORIZON_BARS,
            "wf_tau": WF_TAU,
            "wf_tau_exit": max(0.0, WF_TAU - 0.05),
        },
        "dynamic_policy": {
            **DYNAMIC_POLICY_KWARGS,
            "tau_enter": WF_TAU,
            "tau_exit": max(0.0, WF_TAU - 0.05),
        },
        "vol_guard": {
            "feature": VOL_FEATURE,
            "quantile": VOL_GUARD_Q,
            "mode_default": VOL_GUARD_MODE,  # e.g. "no_mr"
            "threshold_source": "per-fold train slice quantile (no leakage)",
            "comparison": f"{VOL_FEATURE} >= vol_thr",
            "overrides": {
                # run31: USD-quote pairs force TF on volatility spikes
                "USD-quote": "force_tf",
            },
        },
        "versions": {
            "python_full": sys.version,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                "numpy": _pkg_version("numpy"),
                "pandas": _pkg_version("pandas"),
                "scikit-learn": _pkg_version("scikit-learn"),
                "xgboost": _pkg_version("xgboost"),
                "yfinance": _pkg_version("yfinance"),
                "ta": _pkg_version("ta"),
                "matplotlib": _pkg_version("matplotlib"),
                "seaborn": _pkg_version("seaborn"),
                "jupyter": _pkg_version("jupyter"),
                "notebook": _pkg_version("notebook"),
                "ipykernel": _pkg_version("ipykernel"),
            },
        },
        "costs": {
            "spread_pips": SPREAD_PIPS,
            "slippage_pips": SLIPPAGE_PIPS,
            "commission_per_trade": COMMISSION_PER_TRADE,
            "pip_value": {
                "source": "PIP_VALUES from src/data.py",
                "mapping": "converted to short pair names in main.py",
                "jpy_pairs_pip": 0.01,
                "non_jpy_pairs_pip": 0.0001,
            },
        },
    }

    manifest_path = str(_run_output_dir() / "run_manifest.json")
    write_manifest(manifest_path, manifest)
    print(f"Saved: {manifest_path}")

    print('=' * 60)
    print('MARKET PHASE ML - MULTI-PAIR ANALYSIS')
    print('=' * 60)

    # Create output directories
    os.makedirs('src/figures', exist_ok=True)
    _run_output_dir().mkdir(parents=True, exist_ok=True)
    viz.FIGURES_DIR = _run_output_dir() / "figures"

    # ─────────────────────────────────────────
    # 1. DOWNLOAD AND PREPARE ALL PAIRS
    # ─────────────────────────────────────────
    print('\n[1/5] Downloading and preparing market data...')

    raw_data = pipeline.run(pairs=ALL_PAIRS)

    # Optional pair-universe restriction
    raw_data = filter_pair_universe(raw_data)

    loaded_majors = raw_data.pop('_majors')
    loaded_minors = raw_data.pop('_minors')

    summarize_dataset(raw_data)

    if not raw_data:
        print('✗ No data loaded. Exiting.')
        return

    # ─────────────────────────────────────────
    # 2. DETECT PHASES + ENGINEER FEATURES
    # ─────────────────────────────────────────
    print('\n[2/5] Detecting phases and engineering features...')

    detector_params = dict(
        adx_period=14,
        adx_trend_threshold=25.0,
        atr_period=14,
        vol_rolling_window=252,
        risk_pct=0.01
    )
    detector = MarketPhaseDetector(**detector_params)

    # Cache key: hash of raw data + detector parameters + DL runtime attachment state
    raw_data_hash  = _hash_dict_of_dataframes(raw_data)
    detector_hash  = _hash_params(**detector_params)
    dl_cache_surface = dl_surface if dl_runtime_enabled else None
    dl_cache_artifact = str(dl_artifact_path) if dl_runtime_enabled and dl_artifact_path is not None else None
    dl_cache_hash = _hash_params(
        dl_enabled=dl_runtime_enabled,
        dl_surface=dl_cache_surface,
        dl_artifact=dl_cache_artifact,
    )
    processed_param_hash = _hash_params(detector_hash=detector_hash, dl_cache_hash=dl_cache_hash)
    processed_data_key = f"processed_data__{raw_data_hash}__{processed_param_hash}"

    processed_data = load_cache(
        'processed_data', raw_data_hash, processed_param_hash
    )
    print("[CACHE]")
    print(f"processed_data_key={processed_data_key}")
    print(f"dl_surface={dl_surface_str if dl_runtime_enabled else 'disabled'}")
    print(f"artifact={dl_cache_artifact}")
    print(f"cache_hit={processed_data is not None}")

    if processed_data is None:
        print('  No cache found — running phase detection...')
        processed_data = {}
        for pair_name, df in raw_data.items():
            processed_df = process_pair(pair_name, df, detector)
            if processed_df is not None:
                if dl_runtime_enabled:
                    processed_df = attach_dl_features(
                        processed_df=processed_df,
                        pair_name=pair_name,
                        surface=dl_surface,
                        artifact_path=dl_artifact_path,
                    )
                    dl_cols_attached = get_dl_feature_columns(processed_df)
                    print(
                        f"  [DL FEATURE SURFACE] {pair_name}: "
                        f"columns={dl_cols_attached} count={len(dl_cols_attached)}"
                    )
                processed_data[pair_name] = processed_df
                processed_df.to_csv(
                    f'data/processed/{pair_name}.csv'
                )

        save_cache(
            'processed_data', processed_data,
            raw_data_hash, processed_param_hash
        )
        print("[DL CACHE] processed_data=save")

        print(f"\n✓ Processed {len(processed_data)} pairs")

        if DEBUG_FEATURE_COLUMNS:
            print("\n[debug] processed_data columns check (first pair):")
            first_pair = next(iter(processed_data))
            print("  first_pair:", first_pair)
            print("  columns contains returns_recent:", "returns_recent" in processed_data[first_pair].columns)
            print("  columns contains volatility_recent:", "volatility_recent" in processed_data[first_pair].columns)
            print("  missing:", [c for c in ["returns_recent", "volatility_recent"]
                                 if c not in processed_data[first_pair].columns])

        # then the script continues into backtests / ML / etc.
    else:
        print('  Loaded processed data from cache.')

    if not processed_data:
        print('✗ No pairs processed successfully. Exiting.')
        return

    # ── Update manifest with DL feature surface metadata (per run) ─────────
    sample_pair_name = sorted(processed_data.keys())[0]
    sample_pair_df = processed_data[sample_pair_name]
    _run_dl_feature_cols = get_dl_feature_columns(sample_pair_df) if dl_runtime_enabled else []
    manifest["dl"]["dl_feature_columns"] = _stable_feature_columns(_run_dl_feature_cols)
    manifest["dl"]["dl_feature_count"] = len(_run_dl_feature_cols)
    manifest["feature_ordering"]["dl_feature_columns"] = _stable_feature_columns(_run_dl_feature_cols)
    write_manifest(manifest_path, manifest)

    for pair_name in sorted(processed_data.keys()):
        df = processed_data[pair_name]
        print_phase_distribution(df, pair_name)

    # ── Temporary diagnostic — phase label smoothing effect ──────────────
    print('\n  Phase smoothing diagnostic:')
    for pair_name in sorted(processed_data.keys())[:2]:
        df = processed_data[pair_name]
        raw      = df['phase']
        smoothed = smooth_phase_labels(raw, confirmation_bars=5)
        changed  = (raw != smoothed).sum()
        print(f'  {pair_name}: {changed} bars relabeled '
              f'({changed / len(raw) * 100:.1f}% of total)')

    loaded_majors = [p for p in loaded_majors if p in processed_data]
    loaded_minors = [p for p in loaded_minors if p in processed_data]

    print(f'\n✓ {len(processed_data)} pairs ready for analysis')
    print(f'  Majors: {loaded_majors}')
    print(f'  Minors: {loaded_minors}')

    # ─────────────────────────────────────────
    # 3. RUN ML EXPERIMENTS
    # ─────────────────────────────────────────
    print('\n[3/5] Running ML experiments...')

    ml_params     = dict(n_splits=5, random_state=run_cfg.seed)
    ml_data_hash  = _hash_dict_of_dataframes(processed_data)
    ml_param_hash = _hash_params(**ml_params)

    ml_results_all = load_cache(
        'ml_results', ml_data_hash, ml_param_hash
    )

    if ml_results_all is None:
        print('  No cache found — running ML experiments...')
        ml_results_all = {}

        for pair_name in sorted(processed_data.keys()):
            df = processed_data[pair_name]
            df.attrs["pair_name"] = pair_name
            print(f'\n  --- {pair_name} ---')
            try:
                experiment = PhaseMLExperiment(
                    n_splits=ml_params['n_splits'],
                    random_state=ml_params['random_state'],
                    smooth_labels=True,  # False to disable
                    confirmation_bars=5  # tune this value
                )
                experiment.run_baseline(df)
                experiment.run_phase_features(df)
                experiment.run_phase_models(
                    df, min_samples=MIN_PHASE_SAMPLES
                )
                ml_results_all[pair_name] = experiment.compare_results()
            except Exception as e:
                print(f'  ✗ {pair_name}: ML experiment failed — {e}')

        save_cache(
            'ml_results', ml_results_all,
            ml_data_hash, ml_param_hash
        )
    else:
        print('  Loaded ML results from cache.')

    if ml_results_all:
        ml_combined = pd.concat(
            [df.assign(Pair=pair)
             for pair, df in ml_results_all.items()],
            ignore_index=True
        )
        ml_results_path = _with_mode_tag('results/results_ml.csv', dl_mode_tag)
        ml_combined.to_csv(ml_results_path, index=False)
        print(f'\n✓ ML results saved to {ml_results_path}')

    # ─────────────────────────────────────────
    # 3b. ML PHASE PREDICTION
    # ─────────────────────────────────────────
    print('\n[3b/5] Running ML phase prediction...')

    predictor_params = dict(
        train_window=504,
        retrain_freq=21,
        confirmation_bars=5,
        smooth_labels=True,
        random_state=run_cfg.seed,
        seed=run_cfg.seed,
        min_dl_coverage_pct=MIN_DL_TRAIN_COVERAGE_PCT,
    )

    # Cache key: hash of processed data + predictor parameters
    pred_data_hash  = _hash_dict_of_dataframes(processed_data)
    pred_param_hash = _hash_params(**predictor_params)

    ml_predicted_data = load_cache(
        'ml_predicted_phases', pred_data_hash, pred_param_hash
    )

    if ml_predicted_data is None:
        print('  No cache found — running ML phase prediction...')
        ml_predicted_data = {}
        predictor_feature_ordering: dict[str, list[str]] = {}

        predictor = PhaseMLPredictor(**predictor_params)

        for pair_name in sorted(processed_data.keys()):
            df = processed_data[pair_name]
            df.attrs["pair_name"] = pair_name
            print(f'\n  --- {pair_name} ---')
            try:
                predictions = predictor.fit_predict(df)
                eval_scores = predictor.evaluate(df, predictions)

                # Add predicted phase column to DataFrame copy
                df_ml = df.copy()
                df_ml['predicted_phase'] = predictions

                ml_predicted_data[pair_name] = {
                    'df':         df_ml,
                    'eval':       eval_scores,
                    'predictions': predictions
                }
                predictor_feature_ordering[pair_name] = _stable_feature_columns(
                    list(getattr(predictor, "feature_cols", []) or [])
                )
            except Exception as e:
                traceback.print_exc()
                print(f'  ✗ {pair_name}: ML prediction failed — {e}')

        save_cache(
            'ml_predicted_phases', ml_predicted_data,
            pred_data_hash, pred_param_hash
        )
    else:
        print('  Loaded ML predicted phases from cache.')
        predictor_feature_ordering = {
            pair_name: _stable_feature_columns(
                list(
                    col
                    for col in processed_data[pair_name].columns
                    if col not in PhaseMLExperiment.EXCLUDE_COLS
                    and is_numeric_dtype(processed_data[pair_name][col])
                    and not is_bool_dtype(processed_data[pair_name][col])
                    and (DL_SIGNALS_ENABLED or (col not in D1_FEATURE_COLS and not str(col).startswith("dl_")))
                )
            )
            for pair_name in sorted(processed_data.keys())
        }
    manifest["feature_ordering"]["phase_predictor_by_pair"] = {
        pair_name: predictor_feature_ordering[pair_name]
        for pair_name in sorted(predictor_feature_ordering)
    }
    write_manifest(manifest_path, manifest)

    # ── Print accuracy scores regardless of cache hit ─────────────────────
    print('\n  ML Phase Prediction Accuracy Summary:')
    predictor = PhaseMLPredictor(**predictor_params)
    for pair_name, pred_data in ml_predicted_data.items():
        print(f'\n  --- {pair_name} ---')
        eval_scores = predictor.evaluate(
            processed_data[pair_name],
            pred_data['predictions']
        )
        ml_predicted_data[pair_name] ['eval'] = eval_scores

    # ─────────────────────────────────────────
    # 3c. BACKTEST WITH ML-PREDICTED PHASES
    # ─────────────────────────────────────────
    print('\n[3c/5] Running backtests with ML-predicted phases...')

    ml_bt_param_hash = _hash_params(
        **predictor_params,
        initial_capital=INITIAL_CAPITAL,
        use_atr_sizing=False,
        tf_strategy_name='TF4',
        mr_strategy_name='MR42',
        spread_pips=SPREAD_PIPS,
        slippage_pips=SLIPPAGE_PIPS,
        commission_per_trade=COMMISSION_PER_TRADE,
    )

    ml_bt_data_hash = pred_data_hash

    ml_backtest_results = load_cache(
        'ml_backtest_results', ml_bt_data_hash, ml_bt_param_hash
    )

    if ml_backtest_results is None:
        print('  No cache found — running ML backtests...')
        ml_backtest_results = {}

        for pair_name, pred_data in ml_predicted_data.items():
            print(f'\n  --- {pair_name} ---')
            df_ml = pred_data['df']

            try:
                # Temporarily swap phase column for backtesting
                df_ml_swap = df_ml.copy()
                df_ml_swap['phase'] = df_ml_swap['predicted_phase']

                # Run backtest with ML-predicted phases using run_backtests
                # Only run the best PhaseAware combo
                pip_value = PIP_VALUES_BY_PAIRNAME.get(pair_name, 0.0001)

                result = run_backtests(
                    df=df_ml_swap,
                    initial_capital=INITIAL_CAPITAL,
                    use_atr_sizing=False,
                    tf_strategy_name='TF4',
                    mr_strategy_name='MR42',
                    spread_pips=SPREAD_PIPS,
                    slippage_pips=SLIPPAGE_PIPS,
                    commission_per_trade=COMMISSION_PER_TRADE,
                    pip_value=pip_value,
                )

                # Extract just the PhaseAware_TF4_MR42 result
                if 'PhaseAware_TF4_MR42' in result:
                    ml_backtest_results[pair_name] = result['PhaseAware_TF4_MR42']
                    print(f'  ✓ {pair_name}: ML backtest complete')
                else:
                    print(f'  ✗ {pair_name}: PhaseAware_TF4_MR42 not in results')

            except Exception as e:
                traceback.print_exc()
                print(f'  ✗ {pair_name}: ML backtest failed — {e}')

        save_cache(
            'ml_backtest_results', ml_backtest_results,
            ml_bt_data_hash, ml_bt_param_hash
        )
    else:
        print('  Loaded ML backtest results from cache.')

    # ── Print and save ML backtest results ────────────────────────────────
    print('\n  ML Backtest Results Summary (PhaseAware_TF4_MR42_ML):')
    print(f'  {"Pair":<12} {"Return %":>10} {"Sharpe":>8} '
          f'{"MaxDD %":>10} {"WinRate %":>10} {"Trades":>8}')
    print(f'  {"-" * 62}')

    ml_rows = []
    for pair_name, result in ml_backtest_results.items():
        print(f'  {pair_name:<12} '
              f'{result["total_return"]:>10.2f} '
              f'{result["sharpe_ratio"]:>8.4f} '
              f'{result["max_drawdown"]:>10.2f} '
              f'{result["win_rate"]:>10.2f} '
              f'{result["n_trades"]:>8}')
        ml_rows.append({
            'Pair': pair_name,
            'Strategy': 'PhaseAware_TF4_MR42_ML',
            'Total Return (%)': result['total_return'],
            'Sharpe Ratio': result['sharpe_ratio'],
            'Max Drawdown (%)': result['max_drawdown'],
            'Win Rate (%)': result['win_rate'],
            'Profit Factor': result['profit_factor'],
            'Total Trades': result['n_trades'],
        })

    ml_df = pd.DataFrame(ml_rows)
    ml_backtest_path = _with_mode_tag('results/results_ml_backtest.csv', dl_mode_tag)
    ml_df.to_csv(ml_backtest_path, index=False)
    print(f'\n  ✓ Saved to {ml_backtest_path}')
    # ─────────────────────────────────────────
    # 4. RUN BACKTESTS
    # ─────────────────────────────────────────
    print('\n[4/5] Running strategy backtests...')

    backtest_params = dict(
        initial_capital=INITIAL_CAPITAL,
        use_atr_sizing=False,
        spread_pips=SPREAD_PIPS,
        slippage_pips=SLIPPAGE_PIPS,
        commission_per_trade=COMMISSION_PER_TRADE,
    )
    bt_param_hash = _hash_params(**backtest_params)

    bt_data_hash  = _hash_dict_of_dataframes(processed_data)

    all_pair_results = load_cache(
        'backtest_results', bt_data_hash, bt_param_hash
    )

    if all_pair_results is None:
        print('  No cache found — running backtests...')
        all_pair_results = {}

        for pair_name in sorted(processed_data.keys()):
            df = processed_data[pair_name]
            pip_value = PIP_VALUES_BY_PAIRNAME.get(pair_name, 0.0001)
            print(f'\n  --- {pair_name} (pip={pip_value}) ---')

            results_hardcoded = {}
            results_atr       = {}

            try:
                results_hardcoded = run_backtests(
                    df=df,
                    initial_capital=INITIAL_CAPITAL,
                    use_atr_sizing=False,
                    tf_strategy_name='TF4',
                    mr_strategy_name='MR42',
                    spread_pips=SPREAD_PIPS,
                    slippage_pips=SLIPPAGE_PIPS,
                    commission_per_trade=COMMISSION_PER_TRADE,
                    pip_value=pip_value,
                )
                _assert_backtest_index_matches(pair_name, df, results_hardcoded, "hardcoded")
            except Exception as e:
                traceback.print_exc()
                print(f'  ✗ {pair_name}: hardcoded backtest failed — {e}')

            try:
                results_atr = run_backtests(
                    df=df,
                    initial_capital=INITIAL_CAPITAL,
                    use_atr_sizing=True,
                    tf_strategy_name='TF4',
                    mr_strategy_name='MR42',
                    spread_pips=SPREAD_PIPS,
                    slippage_pips=SLIPPAGE_PIPS,
                    commission_per_trade=COMMISSION_PER_TRADE,
                    pip_value=pip_value,
                )
                _assert_backtest_index_matches(pair_name, df, results_atr, "atr")
            except Exception as e:
                traceback.print_exc()
                print(f'  ✗ {pair_name}: ATR backtest failed — {e}')

            if results_hardcoded or results_atr:
                all_pair_results[pair_name] = {
                    **{f'{k}_hardcoded': v
                       for k, v in results_hardcoded.items()},
                    **{f'{k}_atr': v
                       for k, v in results_atr.items()},
                }
                print(f'  ✓ {pair_name}: results stored ({len(results_hardcoded)} hardcoded + {len(results_atr)} atr)')
            else:
                print(f'  ✗ {pair_name}: NO RESULTS STORED — both backtests returned empty')

        save_cache(
            'backtest_results', all_pair_results,
            bt_data_hash, bt_param_hash
        )

    else:
        print('  Loaded backtest results from cache.')


    # ─────────────────────────────────────────
    # 5. AGGREGATE AND REPORT RESULTS
    # ─────────────────────────────────────────
    print('\n[5/5] Aggregating results and creating visualizations...')

    # Separate hardcoded and ATR results for aggregation
    # Build clean dicts with just strategy_name -> metrics
    def extract_sizing(all_results: dict,
                       suffix: str) -> dict:
        """Extract results for one sizing method."""
        extracted = {}
        for pair_name, results in all_results.items():
            extracted[pair_name] = {
                k.replace(f'_{suffix}', ''): v
                for k, v in results.items()
                if k.endswith(f'_{suffix}')
            }
        return extracted

    hardcoded_results = extract_sizing(all_pair_results, 'hardcoded')
    atr_results = extract_sizing(all_pair_results, 'atr')

    # ─────────────────────────────────────────
    # 4b. TRAIN STRATEGY SELECTOR (ML)
    # ─────────────────────────────────────────
    print('\n[4b/5] Training strategy selector ML...')
    print('  (Predicting strategy TYPE: TrendFollowing vs MeanReversion vs PhaseAware)')

    # DL pipeline diagnostics before strategy selector training
    _sample_df_4b = (
        processed_data[sorted(processed_data.keys())[0]]
        if processed_data else None
    )
    _dl_cols_4b = get_dl_feature_columns(_sample_df_4b) if _sample_df_4b is not None else []
    print(
        f"[DL PIPELINE] strategy-selector training: "
        f"dl_enabled={dl_runtime_enabled} "
        f"dl_cols={sorted(_dl_cols_4b)} "
        f"pairs={sorted(processed_data.keys())}"
    )

    selector_trained = {}
    selector_feature_ordering: dict[str, list[str]] = {}

    for pair_name in sorted(processed_data.keys()):
        df = processed_data[pair_name]
        print(f'\n  --- {pair_name} ---')

        try:
            # Get backtest results for this pair
            pair_backtest = hardcoded_results.get(pair_name, {})
            if not pair_backtest:
                print(f'    ✗ No backtest results available')
                continue

            # Track which strategy won in rolling windows
            tracker = StrategyPerformanceTracker(window_days=20)
            training_data = tracker.compute_strategy_returns(df, pair_backtest)

            # Train selector model (3-class: TF vs MR vs PhaseAware)
            selector = StrategySelector(seed=run_cfg.seed)
            metrics = selector.train(
                training_data,
                diagnostics_label=f"dynamic selector training pair={pair_name}",
            )

            if metrics:
                selector_trained[pair_name] = selector
                selector_feature_ordering[pair_name] = _stable_feature_columns(
                    list(selector.feature_cols or [])
                )
                print(f'    ✓ Model trained: CV accuracy {metrics["cv_accuracy"]:.4f}')
            else:
                print(f'    ✗ Training failed')

        except Exception as e:
            traceback.print_exc()
            print(f'    ✗ {pair_name}: selector training failed — {e}')

    if selector_trained:
        print(f'\n✓ Strategy selectors trained for {len(selector_trained)} pairs')
    else:
        print(f'✗ No strategy selectors trained')
    manifest["feature_ordering"]["strategy_selector_by_pair"] = {
        pair_name: selector_feature_ordering[pair_name]
        for pair_name in sorted(selector_feature_ordering)
    }
    write_manifest(manifest_path, manifest)

    # Aggregate by group for both sizing methods
    # DL pipeline diagnostics before aggregation
    print(
        f"[DL PIPELINE] aggregation: "
        f"dl_enabled={dl_runtime_enabled} "
        f"pairs={sorted(processed_data.keys())} "
        f"hardcoded_pairs={sorted(hardcoded_results.keys())}"
    )
    print('\n--- Hardcoded Size Multipliers ---')
    majors_hardcoded = aggregate_backtest_results(
        hardcoded_results, loaded_majors, 'Majors'
    )
    minors_hardcoded = aggregate_backtest_results(
        hardcoded_results, loaded_minors, 'Minors'
    )

    print('\n--- ATR Constant Risk Sizing ---')
    majors_atr = aggregate_backtest_results(
        atr_results, loaded_majors, 'Majors'
    )
    minors_atr = aggregate_backtest_results(
        atr_results, loaded_minors, 'Minors'
    )

    # Print summaries
    for label, df in [
        ('MAJORS — Hardcoded Sizing', majors_hardcoded),
        ('MINORS — Hardcoded Sizing', minors_hardcoded),
        ('MAJORS — ATR Constant Risk', majors_atr),
        ('MINORS — ATR Constant Risk', minors_atr),
    ]:
        if not df.empty:
            print(f'\n{"=" * 60}')
            print(label)
            print('=' * 60)
            print(df.to_string(index=False))

    # Save results
    print('\nSaving results...')
    # DL pipeline diagnostics before final export
    print(
        f"[DL PIPELINE] final export: "
        f"dl_enabled={dl_runtime_enabled} "
        f"dl_cols={sorted(_dl_cols_4b)} "
        f"mode_tag={dl_mode_tag}"
    )
    save_results(
        hardcoded_results,
        majors_hardcoded,
        minors_hardcoded,
        mode_tag=dl_mode_tag,
    )

    # Save ATR results separately
    _run_output_dir().mkdir(parents=True, exist_ok=True)
    if not majors_atr.empty:
        majors_atr_path = _with_mode_tag('results/results_majors_atr.csv', dl_mode_tag)
        majors_atr.to_csv(majors_atr_path, index=False)
    if not minors_atr.empty:
        minors_atr_path = _with_mode_tag('results/results_minors_atr.csv', dl_mode_tag)
        minors_atr.to_csv(minors_atr_path, index=False)
    print('  ✓ ATR sizing results saved')


    # ─────────────────────────────────────────
    # 4c. TEST STRATEGY SELECTOR IN BACKTESTER
    # ─────────────────────────────────────────
    print('\n[4c/5] Testing StrategySelector_Dynamic in backtester...')

    if not selector_trained:
        print('  ✗ No selectors trained; skipping dynamic backtest')
        dynamic_results = {}
    else:
        dynamic_results = {}
        # DL pipeline diagnostics before dynamic backtests
        for _pn in sorted(selector_trained.keys()):
            _sel = selector_trained[_pn]
            _sel_dl = [c for c in (_sel.feature_cols or []) if c.startswith("dl_")]
            print(
                f"[DL PIPELINE] dynamic-bt selector: pair={_pn} "
                f"feature_count={len(_sel.feature_cols or [])} "
                f"dl_cols_in_selector={sorted(_sel_dl)}"
            )

        for pair_name in sorted(processed_data.keys()):
            df = processed_data[pair_name]
            print(f'\n  --- {pair_name} ---')
            _pair_dl_cols = get_dl_feature_columns(df)
            print(
                f"  [DL PIPELINE] {pair_name}: "
                f"dl_cols_in_df={sorted(_pair_dl_cols)}"
            )

            if pair_name not in selector_trained:
                print(f'    ✗ No selector for this pair')
                continue

            try:
                tf_strats, mr_strats = _make_strategy_dicts()
                dynamic_strategy = StrategySelector_Dynamic(
                    selector_trained=selector_trained,
                    tf_strategies=tf_strats,
                    mr_strategies=mr_strats,
                    default_tf="TF4",
                    default_mr="MR42",
                    tau_enter=WF_TAU,
                    tau_exit=max(0.0, WF_TAU - 0.05),
                    dl_debug_verbose=DL_DEBUG_VERBOSE,
                    **DYNAMIC_POLICY_KWARGS,
                )
                pip_value = PIP_VALUES_BY_PAIRNAME.get(pair_name, 0.0001)

                backtester = BT(
                    initial_capital=INITIAL_CAPITAL,
                    spread_pips=SPREAD_PIPS,
                    slippage_pips=SLIPPAGE_PIPS,
                    commission_per_trade=COMMISSION_PER_TRADE,
                    pip_value=pip_value,
                    use_atr_sizing=False,
                )
                signals, sl_pcts, tp_pcts = dynamic_strategy.generate_signals(df, pair_name)

                if DEBUG_SIGNAL_TYPES:
                    print("[debug] signal types:", type(signals), type(sl_pcts), type(tp_pcts))
                    print("[debug] has .iloc:",
                          hasattr(signals, "iloc"), hasattr(sl_pcts, "iloc"), hasattr(tp_pcts, "iloc"))

                dyn_name = (
                    f"StrategySelector_Dynamic_tau{WF_TAU:.2f}"
                    f"_exit{max(0.0, WF_TAU - 0.05):.2f}"
                    f"_hold{int(DYNAMIC_POLICY_KWARGS['min_hold_bars'])}"
                    f"_max{int(DYNAMIC_POLICY_KWARGS['max_hold_bars'])}"
                )
                result = backtester.run(df, signals, dyn_name, sl_pcts, tp_pcts)

                dynamic_results[pair_name] = result

                print(f'    ✓ Return: {result["total_return"]:+7.2f}% | '
                      f'Sharpe: {result["sharpe_ratio"]:+.3f} | '
                      f'Max DD: {result["max_drawdown"]:+.2f}%')

            except Exception as e:
                traceback.print_exc()
                print(f'    ✗ Backtest failed: {e}')

        if dynamic_results:
            dyn_rows = []
            for pair_name, res in dynamic_results.items():
                dyn_rows.append({
                    "Pair": pair_name,
                    "Strategy": "StrategySelector_Dynamic",
                    "Total Return (%)": res.get("total_return", np.nan),
                    "Sharpe": res.get("sharpe_ratio", np.nan),
                    "Max DD (%)": res.get("max_drawdown", np.nan),
                    "Num Trades": len(res.get("trades", [])),
                })

            dyn_df = pd.DataFrame(dyn_rows).sort_values(["Pair"])
            dyn_path = _with_mode_tag("results/dynamic_selector_results_per_pair.csv", dl_mode_tag)
            dyn_df.to_csv(dyn_path, index=False)
            print(f"Saved: {dyn_path}")
            print(f'\n✓ StrategySelector_Dynamic tested on {len(dynamic_results)} pairs')
        else:
            print(f'✗ No dynamic backtest results')

    # ─────────────────────────────────────────
    # 4d. COMPARE BASELINE VS DYNAMIC SELECTOR
    # ─────────────────────────────────────────
    if dynamic_results:
        print('\n[4d/5] Comparing Baseline (PhaseAware_TF4_MR42) vs Dynamic Selector...\n')

        comparison = []

        for pair_name in sorted(dynamic_results.keys()):
            if DEBUG_BASELINE_KEYS:
                print(f"{pair_name}: available baseline keys: {list(hardcoded_results.get(pair_name, {}).keys())}")
            baseline_key = 'PhaseAware_TF4_MR42'

            if pair_name not in hardcoded_results or baseline_key not in hardcoded_results[pair_name]:
                print(f'  ⚠️  {pair_name}: No baseline PhaseAware_TF4_MR42')
                continue

            baseline = hardcoded_results[pair_name][baseline_key]
            dynamic = dynamic_results[pair_name]

            comparison.append({
                'Pair': pair_name,
                'Baseline Return': baseline['total_return'],
                'Dynamic Return': dynamic['total_return'],
                'Return Δ': dynamic['total_return'] - baseline['total_return'],
                'Baseline Sharpe': baseline['sharpe_ratio'],
                'Dynamic Sharpe': dynamic['sharpe_ratio'],
                'Sharpe Δ': dynamic['sharpe_ratio'] - baseline['sharpe_ratio'],
                'Baseline Max DD': baseline['max_drawdown'],
                'Dynamic Max DD': dynamic['max_drawdown'],
                'DD Δ': dynamic['max_drawdown'] - baseline['max_drawdown'],
            })

        comp_df = pd.DataFrame(comparison)

        print(comp_df.to_string(index=False))

        if comp_df.empty:
            print(
                "\nNo baseline comparison could be made. Check if run_backtests produced 'PhaseAware_TF4_MR42_hardcoded' for each pair.")
        else:
            print(f'\n--- Summary ---')
            print(f'Avg Return Δ:     {comp_df["Return Δ"].mean():+.2f}%')
            print(f'Avg Sharpe Δ:     {comp_df["Sharpe Δ"].mean():+.4f}')
            print(f'Avg Max DD Δ:     {comp_df["DD Δ"].mean():+.2f}%')
            print(f'Pairs where Sharpe improved: {(comp_df["Sharpe Δ"] > 0).sum()} / {len(comp_df)}')
            comp_path = _with_mode_tag("results/baseline_vs_dynamic_comparison.csv", dl_mode_tag)
            comp_df.to_csv(comp_path, index=False)
            print(f"Saved: {comp_path}")

    # ─────────────────────────────────────────
    # 4e. ABLATION TABLE (TF-only vs MR-only vs PhaseAware vs Dynamic)
    # ─────────────────────────────────────────
    if RUN_IN_SAMPLE_ABLATION and dynamic_results:
        print("\n[4e/5] Building ablation summary (A0-A3)...")

        A3_LABEL = (
            f"A3_DynamicSelector_tau{WF_TAU:.2f}"
            f"_exit{max(0.0, WF_TAU - 0.05):.2f}"
            f"_hold{int(DYNAMIC_POLICY_KWARGS['min_hold_bars'])}"
            f"_max{int(DYNAMIC_POLICY_KWARGS['max_hold_bars'])}"
        )

        variants = {
            "A0_TF4": ("hardcoded", "TF4"),
            "A1_MR42": ("hardcoded", "MR42"),
            "A2_PhaseAware_TF4_MR42": ("hardcoded", "PhaseAware_TF4_MR42"),
            A3_LABEL: ("dynamic", "StrategySelector_Dynamic"),  # key unused for dynamic source
        }

        ablation_rows = []

        for pair_name in sorted(dynamic_results.keys()):
            # ensure we have hardcoded results for this pair
            pair_hc = hardcoded_results.get(pair_name, {})

            for variant_name, (source, key) in variants.items():
                if source == "dynamic":
                    res = dynamic_results.get(pair_name)
                    if res is None:
                        continue
                else:
                    res = pair_hc.get(key)
                    if res is None:
                        continue

                ablation_rows.append({
                    "Pair": pair_name,
                    "Variant": variant_name,
                    "Total Return (%)": res.get("total_return", np.nan),
                    "Sharpe": res.get("sharpe_ratio", np.nan),
                    "Max DD (%)": res.get("max_drawdown", np.nan),
                    "Num Trades": len(res.get("trades", [])),
                })

        ablation_df = pd.DataFrame(ablation_rows)

        if ablation_df.empty:
            print("  ✗ Ablation table empty (missing expected keys in hardcoded_results?)")
        else:
            # Save per-pair per-variant
            ablation_df = ablation_df.sort_values(["Pair", "Variant"])
            ablation_pair_path = _with_mode_tag("results/ablation_summary_per_pair.csv", dl_mode_tag)
            ablation_df.to_csv(ablation_pair_path, index=False)
            print(f"Saved: {ablation_pair_path}")

            # Aggregate (mean metrics across pairs per variant)
            agg = (ablation_df
                   .groupby("Variant", as_index=False)
                   .agg({
                "Total Return (%)": "mean",
                "Sharpe": "mean",
                "Max DD (%)": "mean",
                "Num Trades": "mean",
            })
                   .sort_values("Variant"))

            # Add pair coverage counts (important for trust)
            coverage = ablation_df.groupby("Variant")["Pair"].nunique().reset_index(name="Pairs")
            agg = agg.merge(coverage, on="Variant", how="left")

            ablation_agg_path = _with_mode_tag("results/ablation_summary_aggregate.csv", dl_mode_tag)
            agg.to_csv(ablation_agg_path, index=False)
            print(f"Saved: {ablation_agg_path}")

            print("\nAblation aggregate (mean across pairs):")
            print(agg.to_string(index=False))

    # ─────────────────────────────────────────
    # 4f. WALK-FORWARD EVALUATION (FULL OOS)
    # ─────────────────────────────────────────
    if RUN_WALKFORWARD:
        print("\n[4f/5] Walk-forward evaluation (out-of-sample)...")

        walkforward_rows = []
        vol_diag_rows = []
        _timeline_rows: list[dict] = []  # accumulates per-bar selector state (if enabled)

        for pair_name in sorted(processed_data.keys()):
            df_full = processed_data[pair_name]
            print(f"\n  --- {pair_name} ---")

            pair_results_full = hardcoded_results.get(pair_name, {})
            if not pair_results_full:
                print("    ✗ Missing hardcoded_results for pair; skipping")
                continue

            folds = generate_walkforward_folds_by_pos(
                df_full.index,
                train_years=WF_TRAIN_YEARS,
                test_months=WF_TEST_MONTHS,
                step_months=WF_STEP_MONTHS,
            )

            if not folds:
                print("    ✗ No folds generated; skipping")
                continue

            for f in folds:
                fold_id = f["fold"]
                train_start_pos = f["train_start_pos"]
                train_end_pos = f["train_end_pos"]
                test_start_pos = f["test_start_pos"]
                test_end_pos = f["test_end_pos"]
                fold_window_diag = _window_diagnostics(
                    train_start_ts=df_full.index[train_start_pos],
                    train_end_ts=df_full.index[train_end_pos],
                    test_start_ts=df_full.index[test_start_pos],
                    test_end_ts=df_full.index[test_end_pos],
                )
                _print_window_diagnostics(
                    f"    [WF FOLD] pair={pair_name} fold={fold_id}",
                    **fold_window_diag,
                )
                training_data, selector_window_diag = _build_causal_selector_training_data(
                    pair_name=pair_name,
                    fold_id=fold_id,
                    df_full=df_full,
                    pair_results_full=pair_results_full,
                    train_start_pos=train_start_pos,
                    train_end_pos=train_end_pos,
                    test_start_pos=test_start_pos,
                    test_end_pos=test_end_pos,
                    label_horizon_bars=LABEL_HORIZON_BARS,
                    context_label="walkforward",
                )

                if len(training_data) < 200:
                    print(f"    fold {fold_id}: ✗ too few training rows ({len(training_data)}); skipping")
                    continue

                # --- Train selector on this fold ---
                selector = StrategySelector(seed=run_cfg.seed)
                selector.train(
                    training_data,
                    do_cv=False,
                    diagnostics_label=f"walkforward fold={fold_id} pair={pair_name}",
                )  # outer WF is evaluation

                # --- Test slice ---
                df_test = df_full.iloc[test_start_pos:test_end_pos + 1].copy()
                if len(df_test) < 50:
                    continue

                # ---- Volatility guard (compute ONCE per fold; no leakage; bar-level scale) ----
                df_train_bars = df_full.iloc[train_start_pos:train_end_pos + 1].copy()
                vol_thr = _compute_vol_threshold(df_train_bars)
                vol_threshold_by_pair = {pair_name: vol_thr} if vol_thr is not None else {}

                if DEBUG_VOL_GUARD and vol_thr is not None and VOL_FEATURE in df_test.columns:
                    s_train = df_train_bars[VOL_FEATURE].dropna()
                    s_test = df_test[VOL_FEATURE].dropna()
                    if len(s_test) and len(s_train):
                        print(
                            f"    [vol-guard] {pair_name} fold={fold_id} "
                            f"train min/med/max="
                            f"{float(s_train.min()):.6f}/{float(s_train.median()):.6f}/{float(s_train.max()):.6f} | "
                            f"test min/med/max="
                            f"{float(s_test.min()):.6f}/{float(s_test.median()):.6f}/{float(s_test.max()):.6f} | "
                            f"thr(q={VOL_GUARD_Q:.2f})={vol_thr:.6f} frac>thr={float((s_test >= vol_thr).mean()):.3f}"
                        )

                # Dynamic selector strategy (per-fold)
                tf_strats, mr_strats = _make_strategy_dicts()
                dynamic_strategy = StrategySelector_Dynamic(
                    selector_trained={pair_name: selector},
                    tf_strategies=tf_strats,
                    mr_strategies=mr_strats,
                    default_tf='TF4',
                    default_mr='MR42',
                    tau_enter=WF_TAU,
                    tau_exit=max(0.0, WF_TAU - 0.05),
                    dl_debug_verbose=DL_DEBUG_VERBOSE,
                    **DYNAMIC_POLICY_KWARGS,
                    use_vol_guard=USE_VOL_GUARD,
                    vol_feature=VOL_FEATURE,
                    vol_threshold_by_pair=vol_threshold_by_pair,
                    vol_guard_mode=VOL_GUARD_MODE,
                )
                pip_value = PIP_VALUES_BY_PAIRNAME.get(pair_name, 0.0001)
                backtester = BT(
                    initial_capital=INITIAL_CAPITAL,
                    spread_pips=SPREAD_PIPS,
                    slippage_pips=SLIPPAGE_PIPS,
                    commission_per_trade=COMMISSION_PER_TRADE,
                    pip_value=pip_value,
                    use_atr_sizing=False,
                )
                dyn_signals, dyn_sl, dyn_tp, selected_s = dynamic_strategy.generate_signals(
                    df_test, pair_name, return_selected=True
                )
                dyn_res = backtester.run(df_test, dyn_signals, 'StrategySelector_Dynamic_WF', dyn_sl, dyn_tp)

                # --- Optional: save selected strategy series for plotting (small whitelist) ---
                if DEBUG_SAVE_SELECTED_SERIES and (pair_name in DEBUG_SELECTED_PAIRS):
                    # count folds saved for this pair
                    if "saved_selected_folds" not in locals():
                        saved_selected_folds = {}  # type: ignore[var-annotated]
                    saved_selected_folds.setdefault(pair_name, 0)

                    if saved_selected_folds[pair_name] < DEBUG_SELECTED_MAX_FOLDS_PER_PAIR:
                        if (vol_thr is not None) and (VOL_FEATURE in df_test.columns):
                            v = df_test[VOL_FEATURE].astype(float)
                            spike = (v >= float(vol_thr)) & v.notna()
                            near_thr = float(vol_thr) * float(VOL_GUARD_NEAR_MULT)
                            near_spike = (v >= near_thr) & v.notna()
                        else:
                            spike = pd.Series(False, index=df_test.index)
                            near_spike = pd.Series(False, index=df_test.index)

                        sel_df = pd.DataFrame({
                            "date": df_test.index,
                            "selected": selected_s.values,
                            VOL_FEATURE: df_test[VOL_FEATURE].values if VOL_FEATURE in df_test.columns else np.nan,
                            "vol_thr": float(vol_thr) if vol_thr is not None else np.nan,
                            "near_thr": (
                                        float(vol_thr) * float(VOL_GUARD_NEAR_MULT)) if vol_thr is not None else np.nan,
                            "spike": spike.values,
                            "near_spike": near_spike.values,
                        })

                        out_path = _with_mode_tag(
                            f"results/selected_series_{pair_name}_fold{fold_id}.csv",
                            dl_mode_tag,
                        )
                        sel_df.to_csv(out_path, index=False)
                        print(f"Saved: {out_path}")

                        saved_selected_folds[pair_name] += 1

                # Baseline on same test slice
                pa = PhaseAwareStrategy('TF4', 'MR42')
                pa_signals, pa_sl, pa_tp = pa.generate_signals(df_test)
                base_res = backtester.run(df_test, pa_signals, 'PhaseAware_TF4_MR42_WF', pa_sl, pa_tp)

                # --- Optional: save equity curves + spike masks for plotting (small whitelist) ---
                if DEBUG_SAVE_EQUITY_SERIES and (pair_name in DEBUG_SELECTED_PAIRS):
                    # count folds saved for this pair
                    if "saved_equity_folds" not in locals():
                        saved_equity_folds = {}  # type: ignore[var-annotated]
                    saved_equity_folds.setdefault(pair_name, 0)

                    if saved_equity_folds[pair_name] < DEBUG_SELECTED_MAX_FOLDS_PER_PAIR:
                        # equity curves should align to df_test.index (Backtester returns index=df passed in)
                        eq_dyn = dyn_res.get("equity_curve", None)
                        eq_base = base_res.get("equity_curve", None)

                        if (eq_dyn is not None) and (eq_base is not None):
                            # Align safely to df_test.index (avoid surprises if something changed)
                            eq_dyn = pd.Series(eq_dyn).reindex(df_test.index).astype(float)
                            eq_base = pd.Series(eq_base).reindex(df_test.index).astype(float)

                            if (vol_thr is not None) and (VOL_FEATURE in df_test.columns):
                                v = df_test[VOL_FEATURE].astype(float)
                                spike = (v >= float(vol_thr)) & v.notna()
                                near_thr = float(vol_thr) * float(VOL_GUARD_NEAR_MULT)
                                near_spike = (v >= near_thr) & v.notna()
                                vol_thr_out = float(vol_thr)
                                near_thr_out = float(near_thr)
                                vol_vals = v
                            else:
                                spike = pd.Series(False, index=df_test.index)
                                near_spike = pd.Series(False, index=df_test.index)
                                vol_thr_out = np.nan
                                near_thr_out = np.nan
                                vol_vals = pd.Series(np.nan, index=df_test.index)

                            # Align selection + executed (previous-bar) signal to df_test.index
                            if "selected_s" in locals() and selected_s is not None:
                                selected_aligned = pd.Series(selected_s).reindex(df_test.index).astype("object")
                            else:
                                selected_aligned = pd.Series("UNKNOWN", index=df_test.index, dtype="object")

                            signal_prev = (
                                pd.Series(dyn_signals)
                                .reindex(df_test.index)
                                .shift(1)
                                .fillna(0)
                                .astype(int)
                            )

                            eq_df = pd.DataFrame({
                                "date": df_test.index,
                                "equity_baseline": eq_base.values,
                                "equity_dynamic": eq_dyn.values,
                                VOL_FEATURE: vol_vals.values,
                                "vol_thr": vol_thr_out,
                                "near_thr": near_thr_out,
                                "spike": spike.values,
                                "near_spike": near_spike.values,
                                # debugging columns
                                "selected_type": selected_aligned.values,
                                "signal_prev": signal_prev.values,
                            })

                            out_path = _with_mode_tag(
                                f"results/equity_debug_{pair_name}_fold{fold_id}.csv",
                                dl_mode_tag,
                            )
                            eq_df.to_csv(out_path, index=False)
                            print(f"Saved: {out_path}")

                            saved_equity_folds[pair_name] += 1


                # --- per-fold DL overlap attribution (from actual bar timestamps) ---
                _dl_cols_fold = get_dl_feature_columns(df_test)
                if _dl_cols_fold:
                    _dl_active_mask = df_test[_dl_cols_fold].notna().any(axis=1)
                    _dl_overlap_pct = float(_dl_active_mask.mean() * 100.0)
                else:
                    _dl_overlap_pct = 0.0

                if not _dl_cols_fold or _dl_overlap_pct <= 5.0:
                    _dl_overlap_state = "missing"
                elif _dl_overlap_pct >= 95.0:
                    _dl_overlap_state = "active"
                else:
                    _dl_overlap_state = "partial"

                _dl_overlap_active = (_dl_overlap_state == "active")
                _test_start_date = df_full.index[test_start_pos]
                _test_end_date = df_full.index[test_end_pos]
                _dl_overlap_window = (
                    f"{_test_start_date.date().isoformat()}/{_test_end_date.date().isoformat()}"
                )

                # --- diagnostics computed BEFORE appending rows ---
                vol_diag = compute_vol_guard_diagnostics(
                    pair_name=pair_name,
                    fold_id=fold_id,
                    df_test=df_test,
                    selected_s=selected_s,
                    vol_feature=VOL_FEATURE,
                    vol_thr=vol_thr,
                    near_mult=VOL_GUARD_NEAR_MULT,
                    majors=loaded_majors,
                    tau=WF_TAU,
                    tag="wf",
                )
                vol_diag_rows.append(vol_diag)

                # --- optional per-bar selector state timeline accumulation ---
                if EXPORT_SELECTOR_STATE_TIMELINE:
                    _prev_strategy: str | None = None
                    for _ti, _ts in enumerate(df_test.index):
                        _strat = selected_s.iloc[_ti]
                        _bar_dl_active = (
                            bool(df_test[_dl_cols_fold].iloc[_ti].notna().any())
                            if _dl_cols_fold else False
                        )
                        _switch_event = (_prev_strategy is not None and _strat != _prev_strategy)
                        _timeline_rows.append({
                            "timestamp": _ts.isoformat(),
                            "pair": pair_name,
                            "fold": fold_id,
                            "selected_strategy": _strat,
                            "dl_available": _bar_dl_active,
                            "dl_overlap_pct": round(_dl_overlap_pct, 4),
                            "dl_overlap_state": _dl_overlap_state,
                            "switch_event": _switch_event,
                            "previous_strategy": _prev_strategy if _prev_strategy is not None else "",
                        })
                        _prev_strategy = _strat

                selector_train_end_ts = selector_window_diag.get("selector_train_end_ts")
                walkforward_rows.append({
                    "Pair": pair_name,
                    "Fold": fold_id,
                    "Train Start": f["train_start_dt"],
                    "Train End": f["train_end_dt"],
                    "Selector Train End": (
                        selector_train_end_ts.date().isoformat()
                        if selector_train_end_ts is not None
                        and not pd.isna(selector_train_end_ts)
                        else None
                    ),
                    "Test Start": f["test_start_dt"],
                    "Test End": f["test_end_dt"],
                    "Fold Gap (days)": f.get("gap_days", np.nan),
                    "Fold Overlap (days)": f.get("overlap_days", np.nan),
                    "Label Horizon Bars": LABEL_HORIZON_BARS,
                    "Train Rows": int(len(training_data)),
                    "Test Bars": int(len(df_test)),

                    "Baseline Return (%)": base_res.get("total_return", np.nan),
                    "Dynamic Return (%)": dyn_res.get("total_return", np.nan),
                    "Return Δ": dyn_res.get("total_return", np.nan) - base_res.get("total_return", np.nan),

                    "Baseline Sharpe": base_res.get("sharpe_ratio", np.nan),
                    "Dynamic Sharpe": dyn_res.get("sharpe_ratio", np.nan),
                    "Sharpe Δ": dyn_res.get("sharpe_ratio", np.nan) - base_res.get("sharpe_ratio", np.nan),

                    "Baseline Max DD (%)": base_res.get("max_drawdown", np.nan),
                    "Dynamic Max DD (%)": dyn_res.get("max_drawdown", np.nan),
                    "DD Δ": dyn_res.get("max_drawdown", np.nan) - base_res.get("max_drawdown", np.nan),

                    "Baseline Trades": base_res.get("n_trades", np.nan),
                    "Dynamic Trades": dyn_res.get("n_trades", np.nan),
                    "Trades Δ": dyn_res.get("n_trades", np.nan) - base_res.get("n_trades", np.nan),

                    # diagnostics
                    "vol_thr": vol_thr,
                    "Spike Bars (%)": vol_diag.get("spike_pct", np.nan),
                    "Near-Spike Bars (%)": vol_diag.get("near_spike_pct", np.nan),
                    "Switches / 1000 bars": vol_diag.get("switches_per_1000_bars", np.nan),
                    "Switches Total": vol_diag.get("switches_total", np.nan),
                    "Switches on spike": vol_diag.get("switches_on_spike", np.nan),
                    "MR on spike (%)": vol_diag.get("mr_on_spike_pct", np.nan),
                    "TF on spike (%)": vol_diag.get("tf_on_spike_pct", np.nan),
                    "Confident Bars (%)": vol_diag.get("confident_pct", np.nan),

                    # fold test-window boundaries (structured; redundant with dl_overlap_window)
                    "fold_test_start": _test_start_date.date().isoformat(),
                    "fold_test_end": _test_end_date.date().isoformat(),

                    # DL overlap attribution (computed from actual bar timestamps)
                    "dl_overlap_pct": round(_dl_overlap_pct, 4),
                    "dl_overlap_active": _dl_overlap_active,
                    "dl_overlap_state": _dl_overlap_state,
                    "dl_overlap_window": _dl_overlap_window,
                })
                print(
                    f"    fold {fold_id}: Sharpe base={base_res['sharpe_ratio']:+.3f} "
                    f"dyn={dyn_res['sharpe_ratio']:+.3f} (Δ {dyn_res['sharpe_ratio'] - base_res['sharpe_ratio']:+.3f})"
                )

        wf_df = pd.DataFrame(walkforward_rows)
        if wf_df.empty:
            print("✗ Walk-forward produced no results.")
        else:
            _run_output_dir().mkdir(parents=True, exist_ok=True)
            wf_fold_path = _with_mode_tag("results/walkforward_results_per_fold.csv", dl_mode_tag)
            wf_df.to_csv(wf_fold_path, index=False)
            print(f"Saved: {wf_fold_path}")

            # Per-pair aggregation (mean across folds)
            wf_pair = (wf_df.groupby("Pair", as_index=False)
                       .agg({
                "Return Δ": "mean",
                "Sharpe Δ": "mean",
                "DD Δ": "mean",
                "Fold": "count",
            })
                       .rename(columns={"Fold": "Folds"}))
            wf_pair_path = _with_mode_tag("results/walkforward_results_per_pair.csv", dl_mode_tag)
            wf_pair.to_csv(wf_pair_path, index=False)
            print(f"Saved: {wf_pair_path}")

            # Overall summary
            overall = {
                "Pairs": int(wf_df["Pair"].nunique()),
                "Folds": int(len(wf_df)),
                "Avg Return Δ": float(wf_df["Return Δ"].mean()),
                "Avg Sharpe Δ": float(wf_df["Sharpe Δ"].mean()),
                "Avg Max DD Δ": float(wf_df["DD Δ"].mean()),
                "Folds Sharpe Improved": int((wf_df["Sharpe Δ"] > 0).sum()),
            }
            wf_summary_path = _with_mode_tag("results/walkforward_results_summary.csv", dl_mode_tag)
            pd.DataFrame([overall]).to_csv(wf_summary_path, index=False)
            print(f"Saved: {wf_summary_path}")

            # --- New diagnostics outputs ---
            if vol_diag_rows:
                diag_df = pd.DataFrame(vol_diag_rows)
                vol_diag_path = _with_mode_tag("results/vol_guard_diagnostics_per_fold.csv", dl_mode_tag)
                diag_df.to_csv(vol_diag_path, index=False)
                print(f"Saved: {vol_diag_path}")

                metric_aggs = {
                    "spike_pct": "mean",
                    "near_spike_pct": "mean",
                    "switches_per_1000_bars": "mean",
                    "mr_on_spike_pct": "mean",
                    "tf_on_spike_pct": "mean",
                    "phaseaware_on_spike_pct": "mean",
                    "confident_pct": "mean",
                    "confident_pct_on_spike": "mean",
                    "confident_pct_off_spike": "mean",
                    "Pair": "nunique",
                    "Fold": "count",
                }

                def _agg_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
                    g = (df.groupby(group_col, as_index=False)
                         .agg(metric_aggs)
                         .rename(columns={"Pair": "Pairs", "Fold": "Rows", group_col: "Group"}))
                    g.insert(0, "GroupBy", group_col)
                    # keep column order stable and avoid stray columns
                    keep = ["GroupBy", "Group"] + [c for c in g.columns if c not in ("GroupBy", "Group")]
                    return g[keep]

                diag_summary = pd.concat(
                    [_agg_group(diag_df, "USD_role"),
                     _agg_group(diag_df, "JPY"),
                     _agg_group(diag_df, "MajorMinor")],
                    ignore_index=True
                )
                vol_diag_summary_path = _with_mode_tag("results/vol_guard_diagnostics_summary.csv", dl_mode_tag)
                diag_summary.to_csv(vol_diag_summary_path, index=False)
                print(f"Saved: {vol_diag_summary_path}")

            print("\nWalk-forward summary:")
            print(pd.DataFrame([overall]).to_string(index=False))

            # --- Optional selector state timeline export ---
            if EXPORT_SELECTOR_STATE_TIMELINE and _timeline_rows:
                timeline_path = _with_mode_tag("results/selector_state_timeline.csv", dl_mode_tag)
                pd.DataFrame(_timeline_rows).to_csv(timeline_path, index=False)
                print(f"Saved: {timeline_path} ({len(_timeline_rows)} bars)")
            elif EXPORT_SELECTOR_STATE_TIMELINE:
                print("  ⚠  selector_state_timeline: no bars collected (no folds completed).")

    # ─────────────────────────────────────────
    if RUN_TAU_SWEEP:
        print("\n[4g/5] Walk-forward global tau sweep (out-of-sample)...")

        TAUS = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]

        tau_rows = []

        for pair_name in sorted(processed_data.keys()):
            df_full = processed_data[pair_name]
            print(f"\n  --- {pair_name} ---")

            pair_results_full = hardcoded_results.get(pair_name, {})
            if not pair_results_full:
                print("    ✗ Missing hardcoded_results for pair; skipping")
                continue

            folds = generate_walkforward_folds_by_pos(
                df_full.index,
                train_years=WF_TRAIN_YEARS,
                test_months=WF_TEST_MONTHS,
                step_months=WF_STEP_MONTHS,
            )
            if not folds:
                continue

            for f in folds:
                fold_id = f["fold"]
                train_start_pos = f["train_start_pos"]
                train_end_pos = f["train_end_pos"]
                test_start_pos = f["test_start_pos"]
                test_end_pos = f["test_end_pos"]
                _print_window_diagnostics(
                    f"    [TAU FOLD] pair={pair_name} fold={fold_id}",
                    **_window_diagnostics(
                        train_start_ts=df_full.index[train_start_pos],
                        train_end_ts=df_full.index[train_end_pos],
                        test_start_ts=df_full.index[test_start_pos],
                        test_end_ts=df_full.index[test_end_pos],
                    ),
                )
                training_data, _ = _build_causal_selector_training_data(
                    pair_name=pair_name,
                    fold_id=fold_id,
                    df_full=df_full,
                    pair_results_full=pair_results_full,
                    train_start_pos=train_start_pos,
                    train_end_pos=train_end_pos,
                    test_start_pos=test_start_pos,
                    test_end_pos=test_end_pos,
                    label_horizon_bars=LABEL_HORIZON_BARS,
                    context_label="tau_sweep",
                )
                if len(training_data) < 200:
                    continue

                selector = StrategySelector(seed=run_cfg.seed)
                selector.train(
                    training_data,
                    do_cv=False,
                    diagnostics_label=f"walkforward tau-sweep fold={fold_id} pair={pair_name}",
                )

                # Test slice
                df_test = df_full.iloc[test_start_pos:test_end_pos + 1].copy()
                if len(df_test) < 50:
                    continue

                # ---- Volatility guard (compute ONCE per fold; tau-independent) ----
                df_train_bars = df_full.iloc[train_start_pos:train_end_pos + 1].copy()
                vol_thr = _compute_vol_threshold(df_train_bars)
                vol_threshold_by_pair = {pair_name: vol_thr} if vol_thr is not None else {}

                if DEBUG_VOL_GUARD and vol_thr is not None and VOL_FEATURE in df_test.columns:
                    s_train = df_train_bars[VOL_FEATURE].dropna()
                    s_test = df_test[VOL_FEATURE].dropna()
                    if len(s_test) and len(s_train):
                        print(
                            f"    [vol-guard] {pair_name} fold={fold_id} "
                            f"train min/med/max="
                            f"{float(s_train.min()):.6f}/{float(s_train.median()):.6f}/{float(s_train.max()):.6f} | "
                            f"test min/med/max="
                            f"{float(s_test.min()):.6f}/{float(s_test.median()):.6f}/{float(s_test.max()):.6f} | "
                            f"thr(q={VOL_GUARD_Q:.2f})={vol_thr:.6f} frac>thr={float((s_test >= vol_thr).mean()):.3f}"
                        )

                # Baseline PhaseAware (tau-independent)
                pip_value = PIP_VALUES_BY_PAIRNAME.get(pair_name, 0.0001)
                base_res = _run_baseline_bt(df_test, pip_value)
                backtester = BT(
                    initial_capital=INITIAL_CAPITAL,
                    spread_pips=SPREAD_PIPS,
                    slippage_pips=SLIPPAGE_PIPS,
                    commission_per_trade=COMMISSION_PER_TRADE,
                    pip_value=pip_value,
                    use_atr_sizing=False,
                )

                for tau in TAUS:
                    tf_strats, mr_strats = _make_strategy_dicts()
                    dyn_strategy = StrategySelector_Dynamic(
                        selector_trained={pair_name: selector},
                        tf_strategies=tf_strats,
                        mr_strategies=mr_strats,
                        default_tf="TF4",
                        default_mr="MR42",
                        tau_enter=tau,
                        tau_exit=max(0.0, tau - 0.05),
                        dl_debug_verbose=DL_DEBUG_VERBOSE,
                        **DYNAMIC_POLICY_KWARGS,
                        use_vol_guard=USE_VOL_GUARD,
                        vol_feature=VOL_FEATURE,
                        vol_threshold_by_pair=vol_threshold_by_pair,
                        vol_guard_mode=VOL_GUARD_MODE,
                    )

                    dyn_signals, dyn_sl, dyn_tp, selected_s = dyn_strategy.generate_signals(
                        df_test, pair_name, return_selected=True
                    )
                    conf_pct = float((selected_s != "PhaseAware").mean() * 100.0)
                    dyn_res = backtester.run(df_test, dyn_signals, f"Dynamic_tau_{tau}", dyn_sl, dyn_tp)

                    # confident bars %: either compute from dyn_strategy internals (not currently returned)
                    # or approximate with your previous valid_mask/pmax>=tau metric (but it won't match hysteresis/min-hold anyway)

                    tau_rows.append({
                        "Tau": tau,
                        "Pair": pair_name,
                        "Fold": fold_id,
                        "Train Start": f["train_start_dt"],
                        "Train End": f["train_end_dt"],
                        "Test Start": f["test_start_dt"],
                        "Test End": f["test_end_dt"],
                        "Baseline Sharpe": base_res.get("sharpe_ratio", np.nan),
                        "Dynamic Sharpe": dyn_res.get("sharpe_ratio", np.nan),
                        "Sharpe Δ": dyn_res.get("sharpe_ratio", np.nan) - base_res.get("sharpe_ratio", np.nan),
                        "Baseline Return (%)": base_res.get("total_return", np.nan),
                        "Dynamic Return (%)": dyn_res.get("total_return", np.nan),
                        "Return Δ": dyn_res.get("total_return", np.nan) - base_res.get("total_return", np.nan),
                        "Baseline Max DD (%)": base_res.get("max_drawdown", np.nan),
                        "Dynamic Max DD (%)": dyn_res.get("max_drawdown", np.nan),
                        "DD Δ": dyn_res.get("max_drawdown", np.nan) - base_res.get("max_drawdown", np.nan),
                        "Baseline Trades": base_res.get("n_trades", np.nan),
                        "Dynamic Trades": dyn_res.get("n_trades", np.nan),
                        "Trades Δ": dyn_res.get("n_trades", np.nan) - base_res.get("n_trades", np.nan),
                        "Confident Bars (%)": conf_pct,
                    })
                    # --- Vol-guard + switching diagnostics (per fold, per tau) ---
                    vol_diag_tau = compute_vol_guard_diagnostics(
                        pair_name=pair_name,
                        fold_id=fold_id,
                        df_test=df_test,
                        selected_s=selected_s,
                        vol_feature=VOL_FEATURE,
                        vol_thr=vol_thr,
                        near_mult=VOL_GUARD_NEAR_MULT,
                        majors=loaded_majors,
                        tau=tau,
                        tag="tau_sweep",
                    )
                    tau_rows[-1].update({
                        "Spike Bars (%)": vol_diag_tau.get("spike_pct", np.nan),
                        "Near-Spike Bars (%)": vol_diag_tau.get("near_spike_pct", np.nan),
                        "Switches / 1000 bars": vol_diag_tau.get("switches_per_1000_bars", np.nan),
                        "MR on spike (%)": vol_diag_tau.get("mr_on_spike_pct", np.nan),
                        "TF on spike (%)": vol_diag_tau.get("tf_on_spike_pct", np.nan),
                    })

        tau_df = pd.DataFrame(tau_rows)
        if tau_df.empty:
            print("✗ Tau sweep produced no results.")
        else:
            _run_output_dir().mkdir(parents=True, exist_ok=True)
            tau_fold_path = _with_mode_tag("results/walkforward_tau_sweep_per_fold.csv", dl_mode_tag)
            tau_df.to_csv(tau_fold_path, index=False)
            print(f"Saved: {tau_fold_path}")

            # Global summary per tau

            summary = (
                tau_df.groupby("Tau", as_index=False)
                .agg(**{
                    "Sharpe Δ": ("Sharpe Δ", "mean"),
                    "Return Δ": ("Return Δ", "mean"),
                    "DD Δ": ("DD Δ", "mean"),
                    "DD Δ median": ("DD Δ", "median"),
                    "Confident Bars (%)": ("Confident Bars (%)", "mean"),
                    "Spike Bars (%)": ("Spike Bars (%)", "mean"),
                    "Near-Spike Bars (%)": ("Near-Spike Bars (%)", "mean"),
                    "Switches / 1000 bars": ("Switches / 1000 bars", "mean"),
                    "MR on spike (%)": ("MR on spike (%)", "mean"),
                    "TF on spike (%)": ("TF on spike (%)", "mean"),
                    "Rows": ("Fold", "count"),
                })
            )
            tau_summary_path = _with_mode_tag("results/walkforward_tau_sweep_summary.csv", dl_mode_tag)
            summary.to_csv(tau_summary_path, index=False)
            print(f"Saved: {tau_summary_path}")
            print("\nTau sweep summary:")
            print(summary.to_string(index=False))

    # ─────────────────────────────────────────
    if RUN_POLICY_SWEEP:
        print("\n[4h/5] Walk-forward policy sweep (tau=0.55) ...")

        POLICIES = [
            {
                "name": "tau0.55_only",
                "tau_enter": 0.55,
                "tau_exit": 0.50,  # unused when hysteresis=False
                "min_hold_bars": 0,  # unused when use_min_hold=False
                "use_hysteresis": False,
                "use_min_hold": False,
            },
            {
                "name": "tau0.55_hold5",
                "tau_enter": 0.55,
                "tau_exit": 0.50,  # unused when hysteresis=False
                "min_hold_bars": 5,
                "use_hysteresis": False,
                "use_min_hold": True,
            },
            {
                "name": "tau0.55_exit0.50_hold5",
                "tau_enter": 0.55,
                "tau_exit": 0.50,
                "min_hold_bars": 5,
                "use_hysteresis": True,
                "use_min_hold": True,
            },
        ]

        policy_rows = []

        for pair_name in sorted(processed_data.keys()):
            df_full = processed_data[pair_name]
            print(f"\n  --- {pair_name} ---")

            pair_results_full = hardcoded_results.get(pair_name, {})
            if not pair_results_full:
                print("    ✗ Missing hardcoded_results for pair; skipping")
                continue

            folds = generate_walkforward_folds_by_pos(
                df_full.index,
                train_years=WF_TRAIN_YEARS,
                test_months=WF_TEST_MONTHS,
                step_months=WF_STEP_MONTHS,
            )
            if not folds:
                continue

            for f in folds:
                fold_id = f["fold"]
                train_start_pos = f["train_start_pos"]
                train_end_pos = f["train_end_pos"]
                test_start_pos = f["test_start_pos"]
                test_end_pos = f["test_end_pos"]
                _print_window_diagnostics(
                    f"    [POLICY FOLD] pair={pair_name} fold={fold_id}",
                    **_window_diagnostics(
                        train_start_ts=df_full.index[train_start_pos],
                        train_end_ts=df_full.index[train_end_pos],
                        test_start_ts=df_full.index[test_start_pos],
                        test_end_ts=df_full.index[test_end_pos],
                    ),
                )
                training_data, _ = _build_causal_selector_training_data(
                    pair_name=pair_name,
                    fold_id=fold_id,
                    df_full=df_full,
                    pair_results_full=pair_results_full,
                    train_start_pos=train_start_pos,
                    train_end_pos=train_end_pos,
                    test_start_pos=test_start_pos,
                    test_end_pos=test_end_pos,
                    label_horizon_bars=LABEL_HORIZON_BARS,
                    context_label="policy_sweep",
                )
                if len(training_data) < 200:
                    continue

                selector = StrategySelector(seed=run_cfg.seed)
                selector.train(
                    training_data,
                    do_cv=False,
                    diagnostics_label=f"walkforward policy-sweep fold={fold_id} pair={pair_name}",
                )

                # ---- test slice ----
                df_test = df_full.iloc[test_start_pos:test_end_pos + 1].copy()
                if len(df_test) < 50:
                    continue

                # ---- vol guard per fold (no leakage; shared across policies) ----
                df_train_bars = df_full.iloc[train_start_pos:train_end_pos + 1].copy()
                vol_thr = _compute_vol_threshold(df_train_bars)
                vol_threshold_by_pair = {pair_name: vol_thr} if vol_thr is not None else {}

                # baseline PhaseAware on test slice (shared across policies)
                pip_value = PIP_VALUES_BY_PAIRNAME.get(pair_name, 0.0001)
                base_res = _run_baseline_bt(df_test, pip_value)
                backtester = BT(
                    initial_capital=INITIAL_CAPITAL,
                    spread_pips=SPREAD_PIPS,
                    slippage_pips=SLIPPAGE_PIPS,
                    commission_per_trade=COMMISSION_PER_TRADE,
                    pip_value=pip_value,
                    use_atr_sizing=False,
                )

                # run each policy (dynamic selector backtest)
                for pol in POLICIES:
                    dyn_name = f"Dynamic_{pol['name']}"
                    tf_strats, mr_strats = _make_strategy_dicts()
                    dynamic_strategy = StrategySelector_Dynamic(
                        selector_trained={pair_name: selector},
                        tf_strategies=tf_strats,
                        mr_strategies=mr_strats,
                        default_tf="TF4",
                        default_mr="MR42",
                        # policy-specific gating params (override defaults)
                        tau_enter=pol["tau_enter"],
                        tau_exit=pol["tau_exit"],
                        min_hold_bars=pol["min_hold_bars"],
                        use_hysteresis=pol["use_hysteresis"],
                        use_min_hold=pol["use_min_hold"],
                        # prob margin settings
                        p_margin=DYNAMIC_POLICY_KWARGS.get("p_margin", 0.20),
                        use_prob_margin=DYNAMIC_POLICY_KWARGS.get("use_prob_margin", True),
                        dl_debug_verbose=DL_DEBUG_VERBOSE,
                        # vol guard
                        use_vol_guard=USE_VOL_GUARD,
                        vol_feature=VOL_FEATURE,
                        vol_threshold_by_pair=vol_threshold_by_pair,
                        vol_guard_mode=VOL_GUARD_MODE,
                    )

                    dyn_signals, dyn_sl, dyn_tp, selected_s = dynamic_strategy.generate_signals(
                        df_test, pair_name, return_selected=True
                    )
                    dyn_res = backtester.run(df_test, dyn_signals, dyn_name, dyn_sl, dyn_tp)

                    conf_pct = float((selected_s != "PhaseAware").mean() * 100.0)

                    policy_rows.append({
                        "Policy": pol["name"],
                        "Pair": pair_name,
                        "Fold": fold_id,
                        "Train Start": f["train_start_dt"],
                        "Train End": f["train_end_dt"],
                        "Test Start": f["test_start_dt"],
                        "Test End": f["test_end_dt"],
                        "Baseline Sharpe": base_res.get("sharpe_ratio", np.nan),
                        "Dynamic Sharpe": dyn_res.get("sharpe_ratio", np.nan),
                        "Sharpe Δ": dyn_res.get("sharpe_ratio", np.nan) - base_res.get("sharpe_ratio", np.nan),
                        "Baseline Return (%)": base_res.get("total_return", np.nan),
                        "Dynamic Return (%)": dyn_res.get("total_return", np.nan),
                        "Return Δ": dyn_res.get("total_return", np.nan) - base_res.get("total_return", np.nan),
                        "Baseline Max DD (%)": base_res.get("max_drawdown", np.nan),
                        "Dynamic Max DD (%)": dyn_res.get("max_drawdown", np.nan),
                        "DD Δ": dyn_res.get("max_drawdown", np.nan) - base_res.get("max_drawdown", np.nan),
                        "Confident Bars (%)": conf_pct,
                    })

        pol_df = pd.DataFrame(policy_rows)
        if pol_df.empty:
            print("✗ Policy sweep produced no results.")
        else:
            _run_output_dir().mkdir(parents=True, exist_ok=True)
            policy_fold_path = _with_mode_tag("results/walkforward_policy_sweep_per_fold.csv", dl_mode_tag)
            pol_df.to_csv(policy_fold_path, index=False)
            print(f"Saved: {policy_fold_path}")

            summary = (pol_df.groupby("Policy", as_index=False)
                       .agg({
                "Sharpe Δ": "mean",
                "Return Δ": "mean",
                "DD Δ": "mean",
                "Fold": "count",
            })
                       .rename(columns={"Fold": "Rows"}))
            policy_summary_path = _with_mode_tag("results/walkforward_policy_sweep_summary.csv", dl_mode_tag)
            summary.to_csv(policy_summary_path, index=False)
            print(f"Saved: {policy_summary_path}")
            print("\nPolicy sweep summary:")
            print(summary.to_string(index=False))
    # ─────────────────────────────────────────
    # VISUALIZATIONS
    # ─────────────────────────────────────────
    print('\nCreating visualizations...')
    os.makedirs('src/figures', exist_ok=True)

    visualizer = PhaseVisualizer()

    # ── 1. Key results figure (most important — shown first) ──────────────
    viz.plot_key_results(
        hardcoded_results=hardcoded_results,
        loaded_majors=loaded_majors,
        loaded_minors=loaded_minors
    )

    # ── 2. Group summary comparison (majors vs minors) ────────────────────
    viz.plot_group_comparison(
        majors_hardcoded,
        minors_hardcoded,
        majors_atr,
        minors_atr
    )

    # ── 3. Phase distribution heatmap (cross-pair) ────────────────────────
    viz.plot_phase_distribution_heatmap(processed_data)

    # ── 4. Phase overview — EURUSD representative example ─────────────────
    first_pair = next(iter(processed_data))
    first_df = processed_data[first_pair]

    visualizer.plot_phases_overview(
        first_df, ticker=first_pair
    )
    visualizer.plot_phase_statistics(first_df, ticker=first_pair)

    # ── 5. Backtest results — EURUSD detailed view ────────────────────────
    first_pair_results = {
        k: v for k, v in hardcoded_results.get(
            first_pair, {}
        ).items()
    }
    if first_pair_results:
        viz.plot_backtest_results(
            first_pair_results, first_df,
            title=f'Backtest Results — {first_pair}'
        )
        viz.plot_phase_performance(first_pair_results)

    # ── 6. Equity curves — all pairs, one chart per strategy ──────────────
    viz.plot_equity_curves_by_strategy(
        hardcoded_results,
        processed_data,
        loaded_majors,
        loaded_minors
    )
    # --- Vol-guard diagnostics plots (optional) ---
    try:
        from src.visualization import plot_vol_guard_diagnostics_all
        plots = plot_vol_guard_diagnostics_all(
            results_dir=str(_run_output_dir()),
            figures_dir=str(_run_output_dir() / "figures"),
        )
        print("[diag-plots] wrote:", plots)
    except Exception as e:
        print("[diag-plots] skipped:", e)

    logs_dir = _run_output_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = logs_dir / "run_summary.log"
    run_log_lines = [
        f"run_id={run_cfg.run_id}",
        f"timestamp_utc={run_cfg.run_id.replace('run_', '')}",
        f"output_dir={_run_output_dir()}",
        f"dl_enabled={dl_runtime_enabled}",
        f"msml_regime={(experiment_meta.get('factors') or {}).get('msml_regime')}",
        f"variant={experiment_meta.get('variant')}",
        f"generation={experiment_meta.get('generation')}",
    ]
    run_log_path.write_text("\n".join(run_log_lines) + "\n")
    print(f"[log] wrote: {run_log_path}")

    auto_analyze = os.getenv("MPML_AUTO_ANALYZE", "true").strip().lower() in {"1", "true", "yes", "on"}
    if auto_analyze:
        try:
            from analysis.pipeline import run_pipeline as run_analysis_pipeline
            analysis_output_dir = _run_output_dir() / "analysis"
            run_analysis_pipeline(
                archive_root=_run_output_dir(),
                output_dir=analysis_output_dir,
                verbose=False,
            )
            print(f"[analysis] wrote: {analysis_output_dir}")
        except Exception as e:
            print(f"[analysis] skipped: {e}")
    else:
        print("[analysis] auto analysis disabled via MPML_AUTO_ANALYZE.")
    # ─────────────────────────────────────────
    # DONE
    # ─────────────────────────────────────────
    print('\n' + '=' * 60)
    print('✓ ANALYSIS COMPLETE!')
    print('=' * 60)
    print('\nOutput files:')
    print(f"  {_with_mode_tag('results/results_per_pair.csv', dl_mode_tag)}")
    print(f"  {_with_mode_tag('results/results_majors.csv', dl_mode_tag)}")
    print(f"  {_with_mode_tag('results/results_minors.csv', dl_mode_tag)}")
    print(f"  {_with_mode_tag('results/results_summary.csv', dl_mode_tag)}")
    print(f"  {_with_mode_tag('results/results_majors_atr.csv', dl_mode_tag)}")
    print(f"  {_with_mode_tag('results/results_minors_atr.csv', dl_mode_tag)}")
    print(f"  {_with_mode_tag('results/results_ml.csv', dl_mode_tag)}")
    print('  figures/phases_overview.png')
    print('  figures/phase_statistics.png')
    print('  figures/phase_distribution_heatmap.png')
    print('  figures/backtest_results.png')
    print('  figures/phase_performance.png')
    print('  figures/group_comparison.png')
    print('  figures/equity_curves_*.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MPML pipeline with run-owned output directories.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Explicit run output directory (writes all run artifacts here). "
            "If omitted, MPML creates results_archive/<gen>_<variant>__<timestamp>/."
        ),
    )
    parser.add_argument(
        "--experiment-generation",
        type=str,
        default=None,
        choices=sorted(_VALID_EXPERIMENT_GENERATIONS),
        help="Optional generation consistency hint (gen1|gen2). Must match selected variant semantics if provided.",
    )
    parser.add_argument(
        "--experiment-variant",
        type=str,
        default=None,
        choices=sorted(VALID_EXPERIMENT_VARIANTS),
        help=(
            "Canonical experiment variant (A|B|C|D|E|F). "
            "Precedence: --experiment-variant > EXPERIMENT_VARIANT env > default A."
        ),
    )
    parser.add_argument(
        "--experiment-seed",
        type=int,
        default=None,
        help=(
            "Experiment RNG seed. Precedence: --experiment-seed > EXPERIMENT_SEED env "
            f"> default ({DEFAULT_EXPERIMENT_SEED})."
        ),
    )
    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        experiment_generation=args.experiment_generation,
        experiment_variant=args.experiment_variant,
        experiment_seed=args.experiment_seed,
    )
