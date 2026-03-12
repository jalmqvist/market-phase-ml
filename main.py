# main.py
import sys
import importlib.metadata as importlib_metadata
import matplotlib
import platform
#matplotlib.use('TkAgg')
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os

from src.data import (
    MarketDataPipeline,
    ALL_PAIRS, MAJORS, MINORS,
    PAIR_NAMES, PIP_VALUES,
    summarize_dataset
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
from src.models import PhaseMLExperiment, PhaseMLPredictor
from src.strategies import Backtester as BT, PhaseAwareStrategy

from src.repro import set_global_seed, build_run_config, write_manifest


# ── Uncomment to force cache refresh ─────
# clear_cache('processed_data')
# clear_cache('backtest_results')
# clear_cache('ml_results')
# clear_cache('ml_predicted_phases')
# clear_cache('ml_backtest_results')
# clear_cache()   # clears everything
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

# Expensive sweeps (disable by default)
RUN_TAU_SWEEP = False
RUN_POLICY_SWEEP = False

# Debug
DEBUG_BASELINE_KEYS = False
DEBUG_FEATURE_COLUMNS = False
DEBUG_SIGNAL_TYPES = False

os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────
SEED = 42
RUN_ID = None  # set to a string to force a specific id; otherwise auto-generated

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
)
WF_TAU = 0.62

# vol guard settings
VOL_GUARD_Q = 0.80
# VOL_GUARD_MODE = "force_phaseaware"
VOL_GUARD_MODE = "no_mr"
VOL_FEATURE = "atr_pct"
# ─────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────

START_DATE          = '2005-01-01'
END_DATE            = '2024-12-31'
INITIAL_CAPITAL     = 10000.0
MIN_PHASE_SAMPLES   = 100       # Minimum samples per phase for ML
USE_ATR_SIZING      = False     # Set True to compare ATR-based sizing


# ─────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────
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
                 minors_summary: pd.DataFrame) -> None:
    """Save all results to CSV files."""
    os.makedirs('results', exist_ok=True)

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
    per_pair_df.to_csv('results/results_per_pair.csv', index=False)
    print('  ✓ Saved results/results_per_pair.csv')

    # Group summaries
    if not majors_summary.empty:
        majors_summary.to_csv(
            'results/results_majors.csv', index=False
        )
        print('  ✓ Saved results/results_majors.csv')

    if not minors_summary.empty:
        minors_summary.to_csv(
            'results/results_minors.csv', index=False
        )
        print('  ✓ Saved results/results_minors.csv')

    # Combined summary
    combined_summary = pd.concat(
        [majors_summary, minors_summary],
        ignore_index=True
    )
    combined_summary.to_csv(
        'results/results_summary.csv', index=False
    )
    print('  ✓ Saved results/results_summary.csv')

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
        test_start_pos = _find_index_pos(dates, test_start_dt)
        test_end_pos = _find_index_pos(dates, test_end_dt)

        if test_end_pos <= test_start_pos or train_end_pos <= train_start_pos:
            break

        folds.append({
            "fold": fold_id,
            "train_start_pos": train_start_pos,
            "train_end_pos": train_end_pos,
            "test_start_pos": test_start_pos,
            "test_end_pos": test_end_pos,
            "train_start_dt": str(dates[train_start_pos].date()),
            "train_end_dt": str(dates[train_end_pos].date()),
            "test_start_dt": str(dates[test_start_pos].date()),
            "test_end_dt": str(dates[test_end_pos].date()),
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

def main():
    run_cfg = build_run_config(seed=SEED, run_id=RUN_ID)
    set_global_seed(run_cfg.seed)

    # Convert ticker-keyed pip values into short-name-keyed pip values
    PIP_VALUES_BY_PAIRNAME = {
        PAIR_NAMES.get(ticker, ticker.replace("=X", "")): pip
        for ticker, pip in PIP_VALUES.items()
    }

    manifest = {
        "run": {
            **run_cfg.__dict__,
            "timestamp_utc": run_cfg.run_id.replace("run_", ""),
        },
        "flags": {
            "RUN_IN_SAMPLE_ABLATION": RUN_IN_SAMPLE_ABLATION,
            "RUN_WALKFORWARD": RUN_WALKFORWARD,
            "RUN_TAU_SWEEP": RUN_TAU_SWEEP,
            "RUN_POLICY_SWEEP": RUN_POLICY_SWEEP,
            "DEBUG_FEATURE_COLUMNS": DEBUG_FEATURE_COLUMNS,
            "DEBUG_SIGNAL_TYPES": DEBUG_SIGNAL_TYPES,
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
            "mode": VOL_GUARD_MODE,
            "threshold_source": "per-fold train slice quantile (no leakage)",
            "comparison": f"{VOL_FEATURE} >= vol_thr",
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

    manifest_path = os.path.join("results", f"run_manifest_{run_cfg.run_id}.json")
    write_manifest(manifest_path, manifest)
    print(f"Saved: {manifest_path}")

    print('=' * 60)
    print('MARKET PHASE ML - MULTI-PAIR ANALYSIS')
    print('=' * 60)

    # Create output directories
    os.makedirs('src/figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # ─────────────────────────────────────────
    # 1. DOWNLOAD AND PREPARE ALL PAIRS
    # ─────────────────────────────────────────
    print('\n[1/5] Downloading and preparing market data...')

    pipeline = MarketDataPipeline(
        start=START_DATE,
        end=END_DATE,
        use_cache=True
    )
    raw_data = pipeline.run(pairs=ALL_PAIRS)

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

    # Cache key: hash of raw data + detector parameters
    raw_data_hash  = _hash_dict_of_dataframes(raw_data)
    detector_hash  = _hash_params(**detector_params)

    processed_data = load_cache(
        'processed_data', raw_data_hash, detector_hash
    )

    if processed_data is None:
        print('  No cache found — running phase detection...')
        processed_data = {}
        for pair_name, df in raw_data.items():
            processed_df = process_pair(pair_name, df, detector)
            if processed_df is not None:
                processed_data[pair_name] = processed_df
                processed_df.to_csv(
                    f'data/processed/{pair_name}.csv'
                )

        save_cache(
            'processed_data', processed_data,
            raw_data_hash, detector_hash
        )

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

    for pair_name, df in processed_data.items():
        print_phase_distribution(df, pair_name)

    # ── Temporary diagnostic — phase label smoothing effect ──────────────
    from src.models import smooth_phase_labels
    print('\n  Phase smoothing diagnostic:')
    for pair_name, df in list(processed_data.items())[:2]:
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

    ml_params     = dict(n_splits=5, random_state=42)
    ml_data_hash  = _hash_dict_of_dataframes(processed_data)
    ml_param_hash = _hash_params(**ml_params)

    ml_results_all = load_cache(
        'ml_results', ml_data_hash, ml_param_hash
    )

    if ml_results_all is None:
        print('  No cache found — running ML experiments...')
        ml_results_all = {}

        for pair_name, df in processed_data.items():
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
        ml_combined.to_csv('results/results_ml.csv', index=False)
        print('\n✓ ML results saved to results/results_ml.csv')

    # ─────────────────────────────────────────
    # 3b. ML PHASE PREDICTION
    # ─────────────────────────────────────────
    print('\n[3b/5] Running ML phase prediction...')

    predictor_params = dict(
        train_window=504,
        retrain_freq=21,
        confirmation_bars=5,
        smooth_labels=True,
        random_state=42
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

        predictor = PhaseMLPredictor(**predictor_params)

        for pair_name, df in processed_data.items():
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
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'  ✗ {pair_name}: ML prediction failed — {e}')

        save_cache(
            'ml_predicted_phases', ml_predicted_data,
            pred_data_hash, pred_param_hash
        )
    else:
        print('  Loaded ML predicted phases from cache.')

    # ✅ ADD THIS DEBUG BLOCK:
    print(f'\n  [DEBUG] ml_predicted_data contents:')
    print(f'    Keys: {list(ml_predicted_data.keys())}')
    print(f'    Length: {len(ml_predicted_data)}')
    if ml_predicted_data:
        for pair_name in list(ml_predicted_data.keys())[:2]:  # Show first 2
            pred_data = ml_predicted_data[pair_name]
            print(f'    {pair_name}: has keys {list(pred_data.keys())}')
    else:
        print('    ⚠️  ml_predicted_data is EMPTY!')

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

        print(f'  DEBUG: ml_predicted_data has {len(ml_predicted_data)} pairs')
        for pair_name in ml_predicted_data.keys():
            print(f'    - {pair_name}')

        for pair_name, pred_data in ml_predicted_data.items():
            print(f'\n  DEBUG: Processing {pair_name}')
            print(f'  DEBUG: pred_data keys = {pred_data.keys()}')

            df_ml = pred_data['df']
            print(f'  DEBUG: df_ml shape = {df_ml.shape}')
            print(f'  DEBUG: df_ml columns = {df_ml.columns.tolist()}')
            print(f'\n  --- {pair_name} ---')

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
                import traceback
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
    ml_df.to_csv('results/results_ml_backtest.csv', index=False)
    print(f'\n  ✓ Saved to results_ml_backtest.csv')
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

        for pair_name, df in processed_data.items():
            ticker = next(
                (t for t, n in PAIR_NAMES.items() if n == pair_name),
                None
            )
            # pip_value = PIP_VALUES.get(ticker, 0.0001)
            pip_value = PIP_VALUES_BY_PAIRNAME.get(pair_name, 0.0001)
            print(f'\n  --- {pair_name} (pip={pip_value}) ---')

            results_hardcoded = {}
            results_atr       = {}

            try:
                print(f'    [DEBUG] Starting hardcoded backtest for {pair_name}')
                print(f'    [DEBUG] df shape: {df.shape}, columns: {df.columns.tolist()}')
                pip_value = PIP_VALUES_BY_PAIRNAME.get(pair_name, 0.0001)

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

                print(f'    [DEBUG] Hardcoded backtest complete: {len(results_hardcoded)} strategies')
                _assert_backtest_index_matches(pair_name, df, results_hardcoded, "hardcoded")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'  ✗ {pair_name}: hardcoded backtest failed — {e}')

            try:
                print(f'    [DEBUG] Starting ATR backtest for {pair_name}')
                pip_value = PIP_VALUES_BY_PAIRNAME.get(pair_name, 0.0001)

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

                print(f'    [DEBUG] ATR backtest complete: {len(results_atr)} strategies')
                _assert_backtest_index_matches(pair_name, df, results_atr, "atr")
            except Exception as e:
                import traceback
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

    from src.models import StrategyPerformanceTracker, StrategySelector

    selector_trained = {}

    for pair_name, df in processed_data.items():
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
            selector = StrategySelector()
            metrics = selector.train(training_data)

            if metrics:
                selector_trained[pair_name] = selector
                print(f'    ✓ Model trained: CV accuracy {metrics["cv_accuracy"]:.4f}')
            else:
                print(f'    ✗ Training failed')

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'    ✗ {pair_name}: selector training failed — {e}')

    if selector_trained:
        print(f'\n✓ Strategy selectors trained for {len(selector_trained)} pairs')
    else:
        print(f'✗ No strategy selectors trained')

    # Aggregate by group for both sizing methods
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
    save_results(
        hardcoded_results,
        majors_hardcoded,
        minors_hardcoded
    )

    # Save ATR results separately
    os.makedirs('results', exist_ok=True)
    if not majors_atr.empty:
        majors_atr.to_csv(
            'results/results_majors_atr.csv', index=False
        )
    if not minors_atr.empty:
        minors_atr.to_csv(
            'results/results_minors_atr.csv', index=False
        )
    print('  ✓ ATR sizing results saved')


    # ─────────────────────────────────────────
    # 4c. TEST STRATEGY SELECTOR IN BACKTESTER
    # ─────────────────────────────────────────
    print('\n[4c/5] Testing StrategySelector_Dynamic in backtester...')

    if not selector_trained:
        print('  ✗ No selectors trained; skipping dynamic backtest')
        dynamic_results = {}
    else:
        from src.strategies import StrategySelector_Dynamic

        dynamic_results = {}

        for pair_name, df in processed_data.items():
            print(f'\n  --- {pair_name} ---')

            if pair_name not in selector_trained:
                print(f'    ✗ No selector for this pair')
                continue

            try:
                # Create dynamic selector strategy
                dynamic_strategy = StrategySelector_Dynamic(
                    selector_trained=selector_trained,
                    tf_strategies={
                        'TF1': TF1Strategy(),
                        'TF2': TF2Strategy(),
                        'TF3': TF3Strategy(),
                        'TF4': TF4Strategy(),
                        'TF5': TF5Strategy(),
                    },
                    mr_strategies={
                        'MR1': MR1Strategy(),
                        'MR2': MR2Strategy(),
                        'MR32': MR32Strategy(),
                        'MR42': MR42Strategy(),
                        'MR5': MR5Strategy(),
                    },
                    default_tf="TF4",
                    default_mr="MR42",
                    tau_enter=WF_TAU,
                    tau_exit=max(0.0, WF_TAU - 0.05),
                    **DYNAMIC_POLICY_KWARGS,
                )
                # Run backtest
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

                dyn_name = "StrategySelector_Dynamic_tau0.55_exit0.50_hold5"
                result = backtester.run(df, signals, dyn_name, sl_pcts, tp_pcts)

                dynamic_results[pair_name] = result

                print(f'    ✓ Return: {result["total_return"]:+7.2f}% | '
                      f'Sharpe: {result["sharpe_ratio"]:+.3f} | '
                      f'Max DD: {result["max_drawdown"]:+.2f}%')

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'    ✗ Backtest failed: {e}')
        # After dynamic_results computed
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
            dyn_df.to_csv("results/dynamic_selector_results_per_pair.csv", index=False)
            print("Saved: results/dynamic_selector_results_per_pair.csv")
        if dynamic_results:
            print(f'\n✓ StrategySelector_Dynamic tested on {len(dynamic_results)} pairs')
        else:
            print(f'✗ No dynamic backtest results')

    # ─────────────────────────────────────────
    # 4d. COMPARE BASELINE VS DYNAMIC SELECTOR
    # ─────────────────────────────────────────
    if dynamic_results:
        print('\n[4d/5] Comparing Baseline (PhaseAware_TF4_MR42) vs Dynamic Selector...\n')

        comparison = []

        for pair_name in dynamic_results.keys():
            DEBUG_BASELINE_KEYS = False
            ...
            if DEBUG_BASELINE_KEYS:
                print(f"{pair_name}: available baseline keys: {list(hardcoded_results.get(pair_name, {}).keys())}")
            # baseline_key = 'PhaseAware_TF4_MR42_hardcoded'
            baseline_key = 'PhaseAware_TF4_MR42'    # TEST

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
            comp_df.to_csv("results/baseline_vs_dynamic_comparison.csv", index=False)
            print("Saved: results/baseline_vs_dynamic_comparison.csv")

    # ─────────────────────────────────────────
    # 4e. ABLATION TABLE (TF-only vs MR-only vs PhaseAware vs Dynamic)
    # ─────────────────────────────────────────
    if RUN_IN_SAMPLE_ABLATION and dynamic_results:
        print("\n[4e/5] Building ablation summary (A0-A3)...")

        variants = {
            "A0_TF4": ("hardcoded", "TF4"),
            "A1_MR42": ("hardcoded", "MR42"),
            "A2_PhaseAware_TF4_MR42": ("hardcoded", "PhaseAware_TF4_MR42"),
            "A3_DynamicSelector_tau0.55_exit0.50_hold5": (
                "dynamic",
                "StrategySelector_Dynamic_tau0.55_exit0.50_hold5"
            ),
        }

        ablation_rows = []

        for pair_name in dynamic_results.keys():
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
            ablation_df.to_csv("results/ablation_summary_per_pair.csv", index=False)
            print("Saved: results/ablation_summary_per_pair.csv")

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

            agg.to_csv("results/ablation_summary_aggregate.csv", index=False)
            print("Saved: results/ablation_summary_aggregate.csv")

            print("\nAblation aggregate (mean across pairs):")
            print(agg.to_string(index=False))

    # ─────────────────────────────────────────
    # 4f. WALK-FORWARD EVALUATION (FULL OOS)
    # ─────────────────────────────────────────
    if RUN_WALKFORWARD:
        print("\n[4f/5] Walk-forward evaluation (out-of-sample)...")

        from src.models import StrategyPerformanceTracker, StrategySelector
        from src.strategies import StrategySelector_Dynamic



        walkforward_rows = []

        for pair_name, df_full in processed_data.items():
            print(f"\n  --- {pair_name} ---")

            pair_results_full = hardcoded_results.get(pair_name, {})
            # Sanity: equity curve index must match df_full.index
            for sname, res in pair_results_full.items():
                eq = res.get("equity_curve")
                if eq is None:
                    continue

                if not eq.index.equals(df_full.index):
                    print(f"[WARN] {pair_name} {sname}: eq.index != df_full.index")
                    print("  eq len:", len(eq), "df len:", len(df_full))
                    print("  missing in eq:", len(df_full.index.difference(eq.index)))
                    print("  extra in eq:", len(eq.index.difference(df_full.index)))
                    # show a few concrete timestamps
                    print("  missing examples:", list(df_full.index.difference(eq.index)[:5]))
                    print("  extra examples:", list(eq.index.difference(df_full.index)[:5]))
                    break
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

                # Need enough room to label training rows with lookahead horizon
                label_end_pos = train_end_pos + LABEL_HORIZON_BARS
                if label_end_pos >= len(df_full):
                    continue

                # --- Build label DF (up to label_end_pos so labels don't touch test) ---
                df_for_labels = df_full.iloc[train_start_pos:label_end_pos + 1].copy()

                # Slice strategy equity curves to match df_for_labels.index
                strat_results_for_labels = {}
                for sname, res in pair_results_full.items():
                    eq = res.get("equity_curve", None)
                    if eq is None:
                        continue
                    missing = df_for_labels.index.difference(eq.index)
                    if len(missing) > 0:
                        print("Missing in eq:", list(missing[:10]), "count:", len(missing))
                        print("eq index range:", eq.index.min(), eq.index.max(), "len:", len(eq))
                        print("df index range:", df_for_labels.index.min(), df_for_labels.index.max(), "len:",
                          len(df_for_labels))
                    common_idx = df_for_labels.index.intersection(eq.index)
                    if len(common_idx) < len(df_for_labels.index):
                        # shrink df_for_labels to what the equity curve can actually support
                        df_for_labels = df_for_labels.loc[common_idx]
                    sliced_eq = eq.reindex(common_idx).ffill()
                    strat_results_for_labels[sname] = dict(res)
                    strat_results_for_labels[sname]["equity_curve"] = sliced_eq

                tracker = StrategyPerformanceTracker(window_days=LABEL_HORIZON_BARS)
                training_data_all = tracker.compute_strategy_returns(
                    df_for_labels,
                    strat_results_for_labels
                )

                # Train rows must be within the training end (avoid using labels whose "date" is beyond train_end_pos)
                train_end_dt = df_full.index[train_end_pos]
                train_start_dt = df_full.index[train_start_pos]
                train_mask = (
                        (training_data_all["date"] >= train_start_dt) &
                        (training_data_all["date"] <= train_end_dt)
                )
                training_data = training_data_all.loc[train_mask].copy()

                if len(training_data) < 200:
                    print(f"    fold {fold_id}: ✗ too few training rows ({len(training_data)}); skipping")
                    continue

                # --- Train selector on this fold ---
                selector = StrategySelector()
                selector.train(training_data, do_cv=False)  # outer WF is evaluation

                # --- Test slice ---
                df_test = df_full.iloc[test_start_pos:test_end_pos + 1].copy()
                if len(df_test) < 50:
                    continue

                # ---- Volatility guard (compute ONCE per fold; no leakage; bar-level scale) ----
                df_train_bars = df_full.iloc[train_start_pos:train_end_pos + 1].copy()

                vol_thr = None
                if "atr_pct" in df_train_bars.columns:
                    s_train = df_train_bars["atr_pct"].dropna()
                    if len(s_train) > 0:
                        vol_thr = float(s_train.quantile(VOL_GUARD_Q))

                vol_threshold_by_pair = {pair_name: vol_thr} if vol_thr is not None else {}

                # (optional) DEBUG once per fold
                print(f"[vol-guard] {pair_name=} {fold_id=} vol_thr is None? {vol_thr is None}")
                if vol_thr is not None and "atr_pct" in df_test.columns:
                    s_test = df_test["atr_pct"].dropna()
                    if len(s_test) and len(s_train):
                        print(
                            f"    [vol-guard] train atr_pct min/med/max="
                            f"{float(s_train.min()):.6f}/{float(s_train.median()):.6f}/{float(s_train.max()):.6f} | "
                            f"test min/med/max="
                            f"{float(s_test.min()):.6f}/{float(s_test.median()):.6f}/{float(s_test.max()):.6f} | "
                            f"thr(q={VOL_GUARD_Q:.2f})={vol_thr:.6f} frac>thr={float((s_test >= vol_thr).mean()):.3f}"
                        )
                # Dynamic selector strategy (per-fold)
                selector_trained_fold = {pair_name: selector}
                dynamic_strategy = StrategySelector_Dynamic(
                    selector_trained=selector_trained_fold,
                    tf_strategies={
                        'TF1': TF1Strategy(),
                        'TF2': TF2Strategy(),
                        'TF3': TF3Strategy(),
                        'TF4': TF4Strategy(),
                        'TF5': TF5Strategy(),
                    },
                    mr_strategies={
                        'MR1': MR1Strategy(),
                        'MR2': MR2Strategy(),
                        'MR32': MR32Strategy(),
                        'MR42': MR42Strategy(),
                        'MR5': MR5Strategy(),
                    },
                    default_tf='TF4',
                    default_mr='MR42',
                    tau_enter=WF_TAU,
                    tau_exit=max(0.0, WF_TAU - 0.05),
                    **DYNAMIC_POLICY_KWARGS,
                    use_vol_guard=True,
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
                dyn_signals, dyn_sl, dyn_tp = dynamic_strategy.generate_signals(df_test, pair_name)
                dyn_res = backtester.run(df_test, dyn_signals, 'StrategySelector_Dynamic_WF', dyn_sl, dyn_tp)

                # Baseline on same test slice
                pa = PhaseAwareStrategy('TF4', 'MR42')
                pa_signals, pa_sl, pa_tp = pa.generate_signals(df_test)
                base_res = backtester.run(df_test, pa_signals, 'PhaseAware_TF4_MR42_WF', pa_sl, pa_tp)

                walkforward_rows.append({
                    "Pair": pair_name,
                    "Fold": fold_id,
                    "Train Start": f["train_start_dt"],
                    "Train End": f["train_end_dt"],
                    "Test Start": f["test_start_dt"],
                    "Test End": f["test_end_dt"],
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
                })

                print(
                    f"    fold {fold_id}: Sharpe base={base_res['sharpe_ratio']:+.3f} "
                    f"dyn={dyn_res['sharpe_ratio']:+.3f} (Δ {dyn_res['sharpe_ratio'] - base_res['sharpe_ratio']:+.3f})"
                )

        wf_df = pd.DataFrame(walkforward_rows)
        if wf_df.empty:
            print("✗ Walk-forward produced no results.")
        else:
            os.makedirs("results", exist_ok=True)
            wf_df.to_csv("results/walkforward_results_per_fold.csv", index=False)
            print("Saved: results/walkforward_results_per_fold.csv")

            # Per-pair aggregation (mean across folds)
            wf_pair = (wf_df.groupby("Pair", as_index=False)
                       .agg({
                "Return Δ": "mean",
                "Sharpe Δ": "mean",
                "DD Δ": "mean",
                "Fold": "count",
            })
                       .rename(columns={"Fold": "Folds"}))
            wf_pair.to_csv("results/walkforward_results_per_pair.csv", index=False)
            print("Saved: results/walkforward_results_per_pair.csv")

            # Overall summary
            overall = {
                "Pairs": int(wf_df["Pair"].nunique()),
                "Folds": int(len(wf_df)),
                "Avg Return Δ": float(wf_df["Return Δ"].mean()),
                "Avg Sharpe Δ": float(wf_df["Sharpe Δ"].mean()),
                "Avg Max DD Δ": float(wf_df["DD Δ"].mean()),
                "Folds Sharpe Improved": int((wf_df["Sharpe Δ"] > 0).sum()),
            }
            pd.DataFrame([overall]).to_csv("results/walkforward_results_summary.csv", index=False)
            print("Saved: results/walkforward_results_summary.csv")

            print("\nWalk-forward summary:")
            print(pd.DataFrame([overall]).to_string(index=False))

    # ─────────────────────────────────────────
    if RUN_TAU_SWEEP:
        print("\n[4g/5] Walk-forward global tau sweep (out-of-sample)...")

        # TAUS = [0.45, 0.50, 0.55, 0.60, 0.65]
        # TAUS = [0.50, 0.55, 0.60]
        # TAUS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        TAUS = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]

        tau_rows = []

        for pair_name, df_full in processed_data.items():
            print(f"\n  --- {pair_name} ---")

            pair_results_full = hardcoded_results.get(pair_name, {})
            # Sanity: equity curve index must match df_full.index
            for sname, res in pair_results_full.items():
                eq = res.get("equity_curve")
                if eq is None:
                    continue

                if not eq.index.equals(df_full.index):
                    print(f"[WARN] {pair_name} {sname}: eq.index != df_full.index")
                    print("  eq len:", len(eq), "df len:", len(df_full))
                    print("  missing in eq:", len(df_full.index.difference(eq.index)))
                    print("  extra in eq:", len(eq.index.difference(df_full.index)))
                    # show a few concrete timestamps
                    print("  missing examples:", list(df_full.index.difference(eq.index)[:5]))
                    print("  extra examples:", list(eq.index.difference(df_full.index)[:5]))
                    break
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

                label_end_pos = train_end_pos + LABEL_HORIZON_BARS
                if label_end_pos >= len(df_full):
                    continue

                # Label slice
                df_for_labels = df_full.iloc[train_start_pos:label_end_pos + 1].copy()

                strat_results_for_labels = {}
                for sname, res in pair_results_full.items():
                    eq = res.get("equity_curve", None)
                    if eq is None:
                        continue
                    strat_results_for_labels[sname] = dict(res)
                    strat_results_for_labels[sname]["equity_curve"] = eq.reindex(df_for_labels.index).ffill()

                tracker = StrategyPerformanceTracker(window_days=LABEL_HORIZON_BARS)
                training_data_all = tracker.compute_strategy_returns(df_for_labels, strat_results_for_labels)

                train_end_dt = df_full.index[train_end_pos]
                train_start_dt = df_full.index[train_start_pos]
                train_mask = (
                        (training_data_all["date"] >= train_start_dt) &
                        (training_data_all["date"] <= train_end_dt)
                )
                training_data = training_data_all.loc[train_mask].copy()
                if len(training_data) < 200:
                    continue

                selector = StrategySelector()
                selector.train(training_data, do_cv=False)

                # Test slice
                df_test = df_full.iloc[test_start_pos:test_end_pos + 1].copy()
                if len(df_test) < 50:
                    continue

                # ---- Volatility guard (compute ONCE per fold; tau-independent) ----
                df_train_bars = df_full.iloc[train_start_pos:train_end_pos + 1].copy()

                vol_thr = None
                if "atr_pct" in df_train_bars.columns:
                    s_train = df_train_bars["atr_pct"].dropna()
                    if len(s_train) > 0:
                        vol_thr = float(s_train.quantile(VOL_GUARD_Q))

                vol_threshold_by_pair = {pair_name: vol_thr} if vol_thr is not None else {}

                # DEBUG (optional; once per fold, not per tau)
                print(f"[vol-guard] {pair_name=} {fold_id=} vol_thr is None? {vol_thr is None}")
                if vol_thr is not None and "atr_pct" in df_test.columns:
                    s_test = df_test["atr_pct"].dropna()
                    if len(s_test) and len(s_train):
                        print(
                            f"    [vol-guard] train atr_pct min/med/max="
                            f"{float(s_train.min()):.6f}/{float(s_train.median()):.6f}/{float(s_train.max()):.6f} | "
                            f"test min/med/max="
                            f"{float(s_test.min()):.6f}/{float(s_test.median()):.6f}/{float(s_test.max()):.6f} | "
                            f"thr(q={VOL_GUARD_Q:.2f})={vol_thr:.6f} frac>thr={float((s_test >= vol_thr).mean()):.3f}"
                        )

                # (optional) remove these if unused; currently you precompute but do not use:
                # tf_sigs, tf_sl_s, tf_tp_s = TF4Strategy().generate_signals(df_test)
                # mr_sigs, mr_sl_s, mr_tp_s = MR42Strategy().generate_signals(df_test)

                # Baseline PhaseAware (tau-independent)
                pip_value = PIP_VALUES_BY_PAIRNAME.get(pair_name, 0.0001)

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
                base_res = backtester.run(df_test, pa_signals, "PhaseAware_TF4_MR42_WF", pa_sl, pa_tp)

                for tau in TAUS:
                    dyn_strategy = StrategySelector_Dynamic(
                        selector_trained={pair_name: selector},
                        tf_strategies={
                        'TF1': TF1Strategy(),
                        'TF2': TF2Strategy(),
                        'TF3': TF3Strategy(),
                        'TF4': TF4Strategy(),
                        'TF5': TF5Strategy(),
                    },
                        mr_strategies={
                        'MR1': MR1Strategy(),
                        'MR2': MR2Strategy(),
                        'MR32': MR32Strategy(),
                        'MR42': MR42Strategy(),
                        'MR5': MR5Strategy(),
                    },
                        default_tf="TF4",
                        default_mr="MR42",
                        tau_enter=tau,
                        tau_exit=max(0.0, tau - 0.05),
                        **DYNAMIC_POLICY_KWARGS,
                        use_vol_guard=True,
                        vol_feature=VOL_FEATURE,
                        vol_threshold_by_pair=vol_threshold_by_pair,
                        vol_guard_mode=VOL_GUARD_MODE
                    )

                    dyn_signals, dyn_sl, dyn_tp, selected_s = dyn_strategy.generate_signals(df_test, pair_name,
                                                                                            return_selected=True)
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



        tau_df = pd.DataFrame(tau_rows)
        if tau_df.empty:
            print("✗ Tau sweep produced no results.")
        else:
            os.makedirs("results", exist_ok=True)
            tau_df.to_csv("results/walkforward_tau_sweep_per_fold.csv", index=False)
            print("Saved: results/walkforward_tau_sweep_per_fold.csv")

            # Global summary per tau

            summary = (
                tau_df.groupby("Tau", as_index=False)
                .agg(**{
                    "Sharpe Δ": ("Sharpe Δ", "mean"),
                    "Return Δ": ("Return Δ", "mean"),
                    "DD Δ": ("DD Δ", "mean"),
                    "DD Δ median": ("DD Δ", "median"),
                    "Confident Bars (%)": ("Confident Bars (%)", "mean"),
                    "Rows": ("Fold", "count"),
                })
            )
            summary.to_csv("results/walkforward_tau_sweep_summary.csv", index=False)
            print("Saved: results/walkforward_tau_sweep_summary.csv")
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

        for pair_name, df_full in processed_data.items():
            print(f"\n  --- {pair_name} ---")

            pair_results_full = hardcoded_results.get(pair_name, {})
            # Sanity: equity curve index must match df_full.index
            for sname, res in pair_results_full.items():
                eq = res.get("equity_curve")
                if eq is None:
                    continue

                if not eq.index.equals(df_full.index):
                    print(f"[WARN] {pair_name} {sname}: eq.index != df_full.index")
                    print("  eq len:", len(eq), "df len:", len(df_full))
                    print("  missing in eq:", len(df_full.index.difference(eq.index)))
                    print("  extra in eq:", len(eq.index.difference(df_full.index)))
                    # show a few concrete timestamps
                    print("  missing examples:", list(df_full.index.difference(eq.index)[:5]))
                    print("  extra examples:", list(eq.index.difference(df_full.index)[:5]))
                    break
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

                label_end_pos = train_end_pos + LABEL_HORIZON_BARS
                if label_end_pos >= len(df_full):
                    continue

                # ---- training data ----
                df_for_labels = df_full.iloc[train_start_pos:label_end_pos + 1].copy()

                strat_results_for_labels = {}
                for sname, res in pair_results_full.items():
                    eq = res.get("equity_curve", None)
                    if eq is None:
                        continue
                    strat_results_for_labels[sname] = dict(res)
                    strat_results_for_labels[sname]["equity_curve"] = eq.reindex(df_for_labels.index).ffill()

                tracker = StrategyPerformanceTracker(window_days=LABEL_HORIZON_BARS)
                training_data_all = tracker.compute_strategy_returns(df_for_labels, strat_results_for_labels)

                train_end_dt = df_full.index[train_end_pos]
                train_start_dt = df_full.index[train_start_pos]
                train_mask = (
                        (training_data_all["date"] >= train_start_dt) &
                        (training_data_all["date"] <= train_end_dt)
                )
                training_data = training_data_all.loc[train_mask].copy()
                if len(training_data) < 200:
                    continue

                selector = StrategySelector()
                selector.train(training_data, do_cv=False)

                # ---- test slice ----
                df_test = df_full.iloc[test_start_pos:test_end_pos + 1].copy()
                if len(df_test) < 50:
                    continue

                # ---- train bars slice (for vol-guard calibration; no leakage) ----
                df_train_bars = df_full.iloc[train_start_pos:train_end_pos + 1].copy()

                vol_thr = None
                if "atr_pct" in df_train_bars.columns:
                    s_train = df_train_bars["atr_pct"].dropna()
                    if len(s_train) > 0:
                        vol_thr = float(s_train.quantile(VOL_GUARD_Q))


                # baseline PhaseAware on test slice (shared across policies)
                pip_value = PIP_VALUES_BY_PAIRNAME.get(pair_name, 0.0001)

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
                base_res = backtester.run(df_test, pa_signals, "PhaseAware_TF4_MR42_WF", pa_sl, pa_tp)

                # ---- vol guard per fold (optional, but consistent with 4f/4g) ----
                vol_threshold_by_pair = {pair_name: vol_thr} if vol_thr is not None else {}

                # run each policy (dynamic selector backtest)
                for pol in POLICIES:
                    dyn_name = f"Dynamic_{pol['name']}"

                    dynamic_strategy = StrategySelector_Dynamic(
                        selector_trained={pair_name: selector},
                        tf_strategies={
                            'TF1': TF1Strategy(),
                            'TF2': TF2Strategy(),
                            'TF3': TF3Strategy(),
                            'TF4': TF4Strategy(),
                            'TF5': TF5Strategy(),
                        },
                        mr_strategies={
                            'MR1': MR1Strategy(),
                            'MR2': MR2Strategy(),
                            'MR32': MR32Strategy(),
                            'MR42': MR42Strategy(),
                            'MR5': MR5Strategy(),
                        },
                        default_tf="TF4",
                        default_mr="MR42",

                        # policy-specific gating params (override defaults)
                        tau_enter=pol["tau_enter"],
                        tau_exit=pol["tau_exit"],
                        min_hold_bars=pol["min_hold_bars"],
                        use_hysteresis=pol["use_hysteresis"],
                        use_min_hold=pol["use_min_hold"],

                        # keep your prob margin settings etc
                        p_margin=DYNAMIC_POLICY_KWARGS.get("p_margin", 0.20),
                        use_prob_margin=DYNAMIC_POLICY_KWARGS.get("use_prob_margin", True),

                        # vol guard (optional)
                        use_vol_guard=True,
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
            os.makedirs("results", exist_ok=True)
            pol_df.to_csv("results/walkforward_policy_sweep_per_fold.csv", index=False)
            print("Saved: results/walkforward_policy_sweep_per_fold.csv")

            summary = (pol_df.groupby("Policy", as_index=False)
                       .agg({
                "Sharpe Δ": "mean",
                "Return Δ": "mean",
                "DD Δ": "mean",
                "Fold": "count",
            })
                       .rename(columns={"Fold": "Rows"}))
            summary.to_csv("results/walkforward_policy_sweep_summary.csv", index=False)
            print("Saved: results/walkforward_policy_sweep_summary.csv")
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

    # ─────────────────────────────────────────
    # DONE
    # ─────────────────────────────────────────
    print('\n' + '=' * 60)
    print('✓ ANALYSIS COMPLETE!')
    print('=' * 60)
    print('\nOutput files:')
    print('  results/results_per_pair.csv')
    print('  results/results_majors.csv')
    print('  results/results_minors.csv')
    print('  results/results_summary.csv')
    print('  results/results_majors_atr.csv')
    print('  results/results_minors_atr.csv')
    print('  results/results_ml.csv')
    print('  figures/phases_overview.png')
    print('  figures/phase_statistics.png')
    print('  figures/phase_distribution_heatmap.png')
    print('  figures/backtest_results.png')
    print('  figures/phase_performance.png')
    print('  figures/group_comparison.png')
    print('  figures/equity_curves_*.png')



if __name__ == '__main__':
    main()