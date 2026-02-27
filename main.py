# main.py

import matplotlib
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
from src.models import PhaseMLExperiment
from src import visualization as viz
from src.visualization import PhaseVisualizer


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
    """Print phase distribution for a single pair."""
    phase_counts = df['phase'].value_counts()
    total = len(df)
    print(f'\n  {pair_name} phase distribution:')
    for phase, count in phase_counts.items():
        pct = count / total * 100
        print(f'    {phase:<20} {count:>5} ({pct:.1f}%)')


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


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
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

    detector = MarketPhaseDetector(
        adx_period=14,
        adx_trend_threshold=25.0,
        atr_period=14,
        vol_rolling_window=252,
        risk_pct=0.01
    )

    processed_data = {}
    for pair_name, df in raw_data.items():
        processed_df = process_pair(pair_name, df, detector)
        if processed_df is not None:
            processed_data[pair_name] = processed_df
            # Save processed data per pair
            processed_df.to_csv(
                f'data/processed/{pair_name}.csv'
            )

    if not processed_data:
        print('✗ No pairs processed successfully. Exiting.')
        return

    # Print phase distribution for each pair
    for pair_name, df in processed_data.items():
        print_phase_distribution(df, pair_name)

    # Update group lists to only include successfully processed pairs
    loaded_majors = [p for p in loaded_majors if p in processed_data]
    loaded_minors = [p for p in loaded_minors if p in processed_data]

    print(f'\n✓ {len(processed_data)} pairs ready for analysis')
    print(f'  Majors: {loaded_majors}')
    print(f'  Minors: {loaded_minors}')

    # ─────────────────────────────────────────
    # 3. RUN ML EXPERIMENTS
    # ─────────────────────────────────────────
    print('\n[3/5] Running ML experiments...')

    ml_results_all = {}

    for pair_name, df in processed_data.items():
        print(f'\n  --- {pair_name} ---')
        try:
            experiment = PhaseMLExperiment(
                n_splits=5,
                random_state=42
            )
            experiment.run_baseline(df)
            experiment.run_phase_features(df)
            experiment.run_phase_models(
                df, min_samples=MIN_PHASE_SAMPLES
            )
            ml_results_all[pair_name] = experiment.compare_results()
        except Exception as e:
            print(f'  ✗ {pair_name}: ML experiment failed — {e}')

    # Save per-pair ML results
    if ml_results_all:
        ml_combined = pd.concat(
            [df.assign(Pair=pair)
             for pair, df in ml_results_all.items()],
            ignore_index=True
        )
        ml_combined.to_csv('results/results_ml.csv', index=False)
        print('\n✓ ML results saved to results/results_ml.csv')

    # ─────────────────────────────────────────
    # 4. RUN BACKTESTS
    # ─────────────────────────────────────────
    print('\n[4/5] Running strategy backtests...')

    all_pair_results = {}

    for pair_name, df in processed_data.items():
        ticker = next(
            (t for t, n in PAIR_NAMES.items() if n == pair_name),
            None
        )
        pip_value = PIP_VALUES.get(ticker, 0.0001)

        print(f'\n  --- {pair_name} (pip={pip_value}) ---')

        results_hardcoded = {}
        results_atr = {}

        try:
            results_hardcoded = run_backtests(
                df=df,
                initial_capital=INITIAL_CAPITAL,
                use_atr_sizing=False
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  ✗ {pair_name}: backtest failed — {e}')

        try:
            results_atr = run_backtests(
                df=df,
                initial_capital=INITIAL_CAPITAL,
                use_atr_sizing=True
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  ✗ {pair_name}: backtest failed — {e}')

        # Only store if at least hardcoded succeeded
        if results_hardcoded:
            all_pair_results[pair_name] = {
                **{f'{k}_hardcoded': v
                   for k, v in results_hardcoded.items()},
                **{f'{k}_atr': v
                   for k, v in results_atr.items()},
            }
            print(f'  ✓ {pair_name}: results stored')
        else:
            print(f'  ✗ {pair_name}: no results stored')

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
    # VISUALIZATIONS
    # ─────────────────────────────────────────
    print('\nCreating visualizations...')
    os.makedirs('src/figures', exist_ok=True)

    visualizer = PhaseVisualizer()

    # Plot phase overview for first available pair
    first_pair = next(iter(processed_data))
    first_df = processed_data[first_pair]

    visualizer.plot_phases_overview(
        first_df, ticker=first_pair
    )
    visualizer.plot_phase_statistics(first_df)

    # Plot cross-pair phase distribution heatmap
    viz.plot_phase_distribution_heatmap(processed_data)

    # Plot backtest comparison for first pair (detailed view)
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

    # Plot group summary comparison
    viz.plot_group_comparison(
        majors_hardcoded,
        minors_hardcoded,
        majors_atr,
        minors_atr
    )

    # Plot equity curves for all pairs, one chart per strategy
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