# src/visualization.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Color scheme for the 4-phase scheme
# ---------------------------------------------------------------------------

PHASE_COLORS = {
    'HV_Trend': '#AA0000',  # Dark red   — high vol trend
    'LV_Trend': '#00AA00',  # Dark green — low vol trend (best)
    'HV_Ranging': '#FF8800',  # Orange     — high vol ranging
    'LV_Ranging': '#4444FF',  # Blue       — low vol ranging
    'Unknown': '#AAAAAA',  # Gray
}

STRATEGY_COLORS = {
    'TrendFollowing': '#1f77b4',  # Blue
    'MeanReversion': '#d62728',  # Red
    'PhaseAware': '#2ca02c',  # Green
}

FIGURES_DIR = Path('figures')


def _ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class PhaseVisualizer:
    """
    Creates per-pair visualizations for market phase analysis.
    """

    def __init__(self, figsize: tuple = (16, 10)):
        self.figsize = figsize
        self.phase_colors = PHASE_COLORS
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_phases_overview(self,
                             df: pd.DataFrame,
                             ticker: str = 'EURUSD') -> None:
        """
        4-panel overview: price colored by phase, ATR%,
        ADX, and phase timeline.

        Panels:
            1. Close price scatter colored by phase
            2. ATR% with rolling median (HV/LV split line)
            3. ADX with trend threshold line
            4. Phase label over time
        """
        _ensure_figures_dir()
        fig, axes = plt.subplots(4, 1,
                                 figsize=(16, 14),
                                 sharex=True)

        # --- Panel 1: Price colored by phase ---
        ax1 = axes[0]
        for phase, color in self.phase_colors.items():
            mask = df['phase'] == phase
            if mask.sum() > 0:
                ax1.scatter(
                    df.index[mask],
                    df['Close'][mask],
                    c=color,
                    s=2,
                    label=phase,
                    alpha=0.8
                )

        ax1.set_title(
            f'{ticker} — Price Colored by Market Phase',
            fontsize=12
        )
        ax1.set_ylabel('Price')
        ax1.legend(
            loc='upper left',
            fontsize=8,
            ncol=2,
            markerscale=3
        )

        # --- Panel 2: ATR% with rolling median ---
        ax2 = axes[1]
        ax2.plot(
            df.index, df['atr_pct'],
            color='blue', linewidth=0.8, alpha=0.7,
            label='ATR%'
        )

        # Rolling median (the HV/LV split line)
        if len(df) >= 126:
            rolling_med = df['atr_pct'].rolling(
                window=252, min_periods=126
            ).median()
            ax2.plot(
                df.index, rolling_med,
                color='red', linewidth=1.2,
                linestyle='--', label='Rolling median (252d)'
            )
            ax2.fill_between(
                df.index, df['atr_pct'], rolling_med,
                where=df['atr_pct'] >= rolling_med,
                alpha=0.25, color='red', label='High Volatility'
            )
            ax2.fill_between(
                df.index, df['atr_pct'], rolling_med,
                where=df['atr_pct'] < rolling_med,
                alpha=0.25, color='blue', label='Low Volatility'
            )

        ax2.set_title('ATR% (Normalized Volatility)', fontsize=12)
        ax2.set_ylabel('ATR / Close (%)')
        ax2.legend(loc='upper right', fontsize=8)

        # --- Panel 3: ADX with threshold ---
        ax3 = axes[2]
        ax3.plot(
            df.index, df['adx'],
            color='black', linewidth=1, label='ADX'
        )
        ax3.plot(
            df.index, df['plus_di'],
            color='green', linewidth=0.8, alpha=0.7, label='+DI'
        )
        ax3.plot(
            df.index, df['minus_di'],
            color='red', linewidth=0.8, alpha=0.7, label='-DI'
        )
        ax3.axhline(
            25, color='orange', linestyle='--',
            linewidth=1, label='Trend threshold (25)'
        )
        ax3.set_title('ADX and Directional Indicators', fontsize=12)
        ax3.set_ylabel('ADX Value')
        ax3.legend(loc='upper right', fontsize=8)

        # --- Panel 4: Phase timeline ---
        ax4 = axes[3]
        phase_list = list(self.phase_colors.keys())
        phase_to_num = {p: i for i, p in enumerate(phase_list)}
        phase_numeric = df['phase'].map(phase_to_num).fillna(0)

        ax4.scatter(
            df.index, phase_numeric,
            c=[self.phase_colors.get(p, '#AAAAAA')
               for p in df['phase']],
            s=2, alpha=0.8
        )
        ax4.set_yticks(range(len(phase_list)))
        ax4.set_yticklabels(phase_list, fontsize=8)
        ax4.set_title('Market Phase Over Time', fontsize=12)
        ax4.set_xlabel('Date')

        plt.tight_layout()
        out_path = FIGURES_DIR / 'phases_overview.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Saved {out_path}')

    def plot_phase_statistics(self, df: pd.DataFrame) -> None:
        """
        4-panel statistical summary for the 4-phase scheme.

        Panels:
            1. Phase frequency distribution
            2. Average next-day return per phase
            3. Directional accuracy per phase
            4. ATR% distribution per phase (violin)
        """
        _ensure_figures_dir()
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # --- 1. Phase frequency ---
        ax1 = axes[0, 0]
        phase_counts = df['phase'].value_counts()
        colors = [self.phase_colors.get(p, '#AAAAAA')
                  for p in phase_counts.index]
        bars = ax1.bar(
            range(len(phase_counts)),
            phase_counts.values,
            color=colors
        )
        ax1.set_xticks(range(len(phase_counts)))
        ax1.set_xticklabels(
            phase_counts.index, rotation=30,
            ha='right', fontsize=9
        )
        ax1.set_title('Phase Frequency Distribution')
        ax1.set_ylabel('Count')

        total = len(df)
        for bar, count in zip(bars, phase_counts.values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height(),
                f'{count / total * 100:.1f}%',
                ha='center', va='bottom', fontsize=8
            )

        # --- 2. Average next-day return per phase ---
        ax2 = axes[0, 1]
        phase_returns = df.groupby('phase')['next_return'].agg(
            ['mean', 'std']
        ).sort_values('mean', ascending=True)

        colors2 = [self.phase_colors.get(p, '#AAAAAA')
                   for p in phase_returns.index]
        ax2.barh(
            range(len(phase_returns)),
            phase_returns['mean'] * 100,
            xerr=phase_returns['std'] * 100,
            color=colors2, alpha=0.8, capsize=3
        )
        ax2.set_yticks(range(len(phase_returns)))
        ax2.set_yticklabels(phase_returns.index, fontsize=9)
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.set_title('Average Next-Day Return by Phase (%)')
        ax2.set_xlabel('Return (%)')

        # --- 3. Directional accuracy per phase ---
        ax3 = axes[1, 0]
        phase_accuracy = df.groupby('phase').apply(
            lambda x: (
                    np.sign(x['next_return']) ==
                    np.sign(x['returns'])
            ).mean()
        ).sort_values(ascending=True)

        colors3 = [self.phase_colors.get(p, '#AAAAAA')
                   for p in phase_accuracy.index]
        ax3.barh(
            range(len(phase_accuracy)),
            phase_accuracy.values * 100,
            color=colors3, alpha=0.8
        )
        ax3.set_yticks(range(len(phase_accuracy)))
        ax3.set_yticklabels(phase_accuracy.index, fontsize=9)
        ax3.axvline(
            50, color='red', linestyle='--',
            linewidth=1, label='Random (50%)'
        )
        ax3.set_title('Directional Accuracy by Phase (%)')
        ax3.set_xlabel('Accuracy (%)')
        ax3.legend(fontsize=8)

        # --- 4. ATR% distribution per phase (violin) ---
        ax4 = axes[1, 1]
        phases_present = df['phase'].unique()
        violin_data = [
            df[df['phase'] == p] ['atr_pct'].dropna().values
            for p in phases_present
        ]
        violin_colors = [
            self.phase_colors.get(p, '#AAAAAA')
            for p in phases_present
        ]

        parts = ax4.violinplot(
            violin_data,
            positions=range(len(phases_present)),
            showmedians=True
        )
        for pc, color in zip(parts['bodies'], violin_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax4.set_xticks(range(len(phases_present)))
        ax4.set_xticklabels(
            phases_present, rotation=30,
            ha='right', fontsize=9
        )
        ax4.set_title('ATR% Distribution by Phase')
        ax4.set_xlabel('Phase')
        ax4.set_ylabel('ATR%')

        plt.tight_layout()
        out_path = FIGURES_DIR / 'phase_statistics.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Saved {out_path}')

    def plot_model_comparison(self,
                              results: dict,
                              phase_results: dict) -> None:
        """
        Compare ML model performance across experiments.

        Panels:
            1. Overall accuracy: baseline vs phase-feature vs per-phase
            2. Per-phase model accuracy breakdown
        """
        _ensure_figures_dir()
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # --- 1. Overall model comparison ---
        ax1 = axes
        models, accuracies, errors, colors = [], [], [], []

        if 'baseline' in results:
            models.append('Baseline\n(No Phases)')
            accuracies.append(results['baseline'] ['accuracy_mean'])
            errors.append(results['baseline'] ['accuracy_std'])
            colors.append('#888888')

        if 'phase_features' in results:
            models.append('Phase\nas Feature')
            accuracies.append(
                results['phase_features'] ['accuracy_mean']
            )
            errors.append(results['phase_features'] ['accuracy_std'])
            colors.append('#4444FF')

        if phase_results:
            phase_accs = [
                v['accuracy_mean'] for v in phase_results.values()
            ]
            models.append('Separate\nPhase Models')
            accuracies.append(np.mean(phase_accs))
            errors.append(np.std(phase_accs))
            colors.append('#44AA44')

        if accuracies:
            bars = ax1.bar(
                models,
                [a * 100 for a in accuracies],
                yerr=[e * 100 for e in errors],
                color=colors, alpha=0.8, capsize=5
            )
            ax1.axhline(
                50, color='red', linestyle='--',
                linewidth=1, label='Random (50%)'
            )
            for bar, acc in zip(bars, accuracies):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 0.5,
                    f'{acc * 100:.2f}%',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold'
                )
            ax1.set_title(
                'ML Model Comparison:\nImpact of Phase Awareness',
                fontsize=12
            )
            ax1.set_ylabel('Directional Accuracy (%)')
            ax1.set_ylim(
                45, max([a * 100 for a in accuracies]) + 5
            )
            ax1.legend(fontsize=8)

        # --- 2. Per-phase model accuracy ---
        ax2 = axes
        if phase_results:
            phase_names = list(phase_results.keys())
            phase_accs  = [
                v['accuracy_mean'] * 100
                for v in phase_results.values()
            ]
            phase_stds  = [
                v['accuracy_std'] * 100
                for v in phase_results.values()
            ]
            phase_colors = [
                self.phase_colors.get(p, '#AAAAAA')
                for p in phase_names
            ]

            sorted_idx   = np.argsort(phase_accs)
            phase_names  = [phase_names[i]  for i in sorted_idx]
            phase_accs   = [phase_accs[i]   for i in sorted_idx]
            phase_stds   = [phase_stds[i]   for i in sorted_idx]
            phase_colors = [phase_colors[i] for i in sorted_idx]

            ax2.barh(
                range(len(phase_names)), phase_accs,
                xerr=phase_stds, color=phase_colors,
                alpha=0.8, capsize=3
            )
            ax2.set_yticks(range(len(phase_names)))
            ax2.set_yticklabels(phase_names, fontsize=9)
            ax2.axvline(
                50, color='red', linestyle='--',
                linewidth=1, label='Random (50%)'
            )
            ax2.set_title(
                'Accuracy per Phase\n(Separate Models)',
                fontsize=12
            )
            ax2.set_xlabel('Directional Accuracy (%)')
            ax2.legend(fontsize=8)

        plt.tight_layout()
        out_path = FIGURES_DIR / 'model_comparison.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Saved {out_path}')


# ---------------------------------------------------------------------------
# Module-level functions (multi-pair visualizations)
# ---------------------------------------------------------------------------

def plot_backtest_results(results: dict,
                          df: pd.DataFrame,
                          title: str = 'Backtest Results') -> None:
    """
    3-panel backtest summary for a single pair.

    Panels:
        1. Equity curves for all strategies
        2. Drawdown curves
        3. Performance metrics bar chart
    """
    _ensure_figures_dir()
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # --- Panel 1: Equity curves ---
    ax1 = axes[0]
    for name, metrics in results.items():
        equity = metrics['equity_curve']
        ax1.plot(
            equity.index, equity.values,
            color=STRATEGY_COLORS.get(name, 'gray'),
            linewidth=1.5,
            label=f"{name} ({metrics['total_return']:.1f}%)"
        )
    ax1.set_title(f'{title} — Equity Curves', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Drawdown ---
    ax2 = axes[1]
    for name, metrics in results.items():
        equity = metrics['equity_curve']
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        ax2.plot(
            drawdown.index, drawdown.values,
            color=STRATEGY_COLORS.get(name, 'gray'),
            linewidth=1,
            label=f"{name} (max: {metrics['max_drawdown']:.1f}%)"
        )
    ax2.set_title('Drawdown Comparison', fontsize=12)
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Performance metrics bar chart ---
    ax3 = axes[2]
    metric_labels = [
        'Total Return (%)',
        'Sharpe × 10',
        'Win Rate (%)',
        'Profit Factor × 10'
    ]
    x     = np.arange(len(metric_labels))
    width = 0.25

    for idx, (name, metrics) in enumerate(results.items()):
        values = [
            metrics['total_return'],
            metrics['sharpe_ratio'] * 10,
            metrics['win_rate'],
            metrics['profit_factor'] * 10
        ]
        ax3.bar(
            x + idx * width, values, width,
            label=name,
            color=STRATEGY_COLORS.get(name, 'gray'),
            alpha=0.8
        )

    ax3.set_xticks(x + width)
    ax3.set_xticklabels(metric_labels)
    ax3.set_title('Performance Metrics Comparison', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = FIGURES_DIR / 'backtest_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved {out_path}')


def plot_phase_performance(results: dict) -> None:
    """
    4-panel per-phase performance breakdown
    for the PhaseAware strategy.

    Panels:
        1. Total PnL per phase
        2. Win rate per phase
        3. Number of trades per phase
        4. Average PnL per trade per phase
    """
    _ensure_figures_dir()

    phase_perf = results.get('PhaseAware', {}).get(
        'phase_performance'
    )
    if phase_perf is None or len(phase_perf) == 0:
        print('  No phase performance data available')
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = [
        PHASE_COLORS.get(p, '#AAAAAA')
        for p in phase_perf.index
    ]

    # --- 1. Total PnL per phase ---
    ax1 = axes[0, 0]
    phase_perf['total_pnl'].sort_values().plot(
        kind='barh', ax=ax1, color=colors
    )
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.set_title('Total PnL by Phase ($)')
    ax1.set_xlabel('PnL ($)')

    # --- 2. Win rate per phase ---
    ax2 = axes[0, 1]
    phase_perf['win_rate'].sort_values().plot(
        kind='barh', ax=ax2, color=colors
    )
    ax2.axvline(
        50, color='red', linestyle='--',
        linewidth=1, label='50% (random)'
    )
    ax2.set_title('Win Rate by Phase (%)')
    ax2.set_xlabel('Win Rate (%)')
    ax2.legend(fontsize=8)

    # --- 3. Number of trades per phase ---
    ax3 = axes[1, 0]
    phase_perf['n_trades'].sort_values().plot(
        kind='barh', ax=ax3, color=colors
    )
    ax3.set_title('Number of Trades by Phase')
    ax3.set_xlabel('N Trades')

    # --- 4. Average PnL per trade ---
    ax4 = axes[1, 1]
    phase_perf['avg_pnl'].sort_values().plot(
        kind='barh', ax=ax4, color=colors
    )
    ax4.axvline(0, color='black', linewidth=0.8)
    ax4.set_title('Average PnL per Trade by Phase ($)')
    ax4.set_xlabel('Avg PnL ($)')

    plt.suptitle(
        'Phase-Aware Strategy: Per-Phase Performance',
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / 'phase_performance.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved {out_path}')


def plot_phase_distribution_heatmap(processed_data: dict) -> None:
    """
    Heatmap showing phase distribution (%) across all pairs.

    Rows = pairs, Columns = phases.
    Useful for checking whether phase detection is consistent
    across different currency pairs.
    """

    print(f"\nDEBUG heatmap:")
    print(f"  processed_data pairs: {list(processed_data.keys())}")
    for pair, df in processed_data.items():
        print(f"  {pair} phases: {df['phase'].unique()}")

    _ensure_figures_dir()

    rows = {}
    for pair_name, df in processed_data.items():
        counts = df['phase'].value_counts(normalize=True) * 100
        rows[pair_name] = counts

    heatmap_df = pd.DataFrame(rows).T.fillna(0)

    # Sort columns by mean frequency descending
    heatmap_df = heatmap_df[
        heatmap_df.mean().sort_values(ascending=False).index
    ]

    fig, ax = plt.subplots(
        figsize=(max(10, len(heatmap_df.columns) * 1.5),
                 max(6, len(heatmap_df) * 0.5))
    )

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': 'Phase frequency (%)'}
    )

    ax.set_title(
        'Phase Distribution Across All Pairs (%)',
        fontsize=13, pad=12
    )
    ax.set_xlabel('Market Phase')
    ax.set_ylabel('Currency Pair')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / 'phase_distribution_heatmap.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved {out_path}')


def plot_group_comparison(majors_hardcoded: pd.DataFrame,
                          minors_hardcoded: pd.DataFrame,
                          majors_atr: pd.DataFrame,
                          minors_atr: pd.DataFrame) -> None:
    """
    Side-by-side comparison of weighted average results
    for majors vs minors, hardcoded vs ATR sizing.

    2 rows × 2 cols:
        Row 1: Hardcoded sizing  — Majors | Minors
        Row 2: ATR sizing        — Majors | Minors
    """
    _ensure_figures_dir()

    metrics = [
        'Total Return (%)',
        'Sharpe Ratio',
        'Win Rate (%)',
        'Profit Factor'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    plot_configs = [
        (axes[0, 0], majors_hardcoded, 'Majors — Hardcoded Sizing'),
        (axes[0, 1], minors_hardcoded, 'Minors — Hardcoded Sizing'),
        (axes[1, 0], majors_atr,       'Majors — ATR Constant Risk'),
        (axes[1, 1], minors_atr,       'Minors — ATR Constant Risk'),
    ]

    for ax, summary_df, title in plot_configs:
        if summary_df.empty:
            ax.text(
                0.5, 0.5, 'No data',
                ha='center', va='center',
                transform=ax.transAxes, fontsize=12
            )
            ax.set_title(title, fontsize=11)
            continue

        strategies = summary_df['Strategy'].tolist()
        x = np.arange(len(metrics))
        width = 0.8 / len(strategies)

        for idx, strategy in enumerate(strategies):
            row = summary_df[
                summary_df['Strategy'] == strategy
            ].iloc[0]

            values = [
                row.get('Total Return (%)', 0),
                row.get('Sharpe Ratio', 0),
                row.get('Win Rate (%)', 0),
                row.get('Profit Factor', 0),
            ]

            ax.bar(
                x + idx * width, values, width,
                label=strategy,
                color=STRATEGY_COLORS.get(strategy, 'gray'),
                alpha=0.8
            )

        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels(metrics, fontsize=8, rotation=15)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.5)

    plt.suptitle(
        'Strategy Comparison: Majors vs Minors, '
        'Hardcoded vs ATR Sizing',
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / 'group_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved {out_path}')


def plot_equity_curves_by_strategy(
        hardcoded_results: dict,
        processed_data: dict,
        loaded_majors: list,
        loaded_minors: list) -> None:
    """
    One figure per strategy showing equity curves for all pairs,
    with majors and minors visually distinguished.

    Produces 3 figures:
        figures/equity_curves_TrendFollowing.png
        figures/equity_curves_MeanReversion.png
        figures/equity_curves_PhaseAware.png

    Within each figure:
        - Solid lines   = major pairs
        - Dashed lines  = minor pairs
        - Bold black    = equally-weighted average across all pairs
    """

    print(f"\nDEBUG equity curves:")
    print(f"  hardcoded_results pairs: {list(hardcoded_results.keys())}")
    print(f"  loaded_majors: {loaded_majors}")
    print(f"  loaded_minors: {loaded_minors}")
    for pair in loaded_majors + loaded_minors:
        pair_result = hardcoded_results.get(pair, {})
        print(f"  {pair} keys: {list(pair_result.keys())}")

    _ensure_figures_dir()

    strategies = ['TrendFollowing', 'MeanReversion', 'PhaseAware']

    # Color palette for individual pairs
    pair_palette = plt.cm.tab20.colors

    for strategy in strategies:
        fig, ax = plt.subplots(figsize=(16, 8))

        all_equity_series = []
        color_idx = 0

        for group_pairs, linestyle, group_label in [
            (loaded_majors, '-',  'Major'),
            (loaded_minors, '--', 'Minor'),
        ]:
            for pair_name in group_pairs:
                pair_result = hardcoded_results.get(pair_name, {})
                metrics = pair_result.get(strategy)

                if metrics is None or 'equity_curve' not in metrics:
                    print(f"  SKIP {pair_name} / {strategy}: metrics={metrics is None}, keys={list(pair_result.keys()) if pair_result else 'empty'}")
                    continue

                equity = metrics['equity_curve']

                # Normalise to 100 for cross-pair comparison
                equity_norm = equity / equity.iloc[0] * 100

                color = pair_palette[color_idx % len(pair_palette)]
                ax.plot(
                    equity_norm.index,
                    equity_norm.values,
                    color=color,
                    linewidth=0.8,
                    linestyle=linestyle,
                    alpha=0.6,
                    label=f'{pair_name} ({group_label})'
                )

                all_equity_series.append(equity_norm)
                color_idx += 1

        # --- Equally-weighted average equity curve ---
        if all_equity_series:
            # Align all series to a common date index
            combined = pd.concat(all_equity_series, axis=1)
            combined = combined.ffill()
            avg_equity = combined.mean(axis=1)

            ax.plot(
                avg_equity.index,
                avg_equity.values,
                color='black',
                linewidth=2.5,
                linestyle='-',
                alpha=1.0,
                label='Average (all pairs)',
                zorder=5
            )

        # Reference line at 100 (starting equity)
        ax.axhline(
            100, color='gray', linestyle=':',
            linewidth=1, alpha=0.7
        )

        ax.set_title(
            f'{strategy} — Equity Curves: All Pairs\n'
            f'(Solid = Majors, Dashed = Minors, '
            f'Black = Average)',
            fontsize=12
        )
        ax.set_ylabel('Normalised Equity (start = 100)')
        ax.set_xlabel('Date')
        ax.legend(
            fontsize=7, ncol=3,
            loc='upper left',
            framealpha=0.7
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = FIGURES_DIR / f'equity_curves_{strategy}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Saved {out_path}')


def plot_sizing_comparison(hardcoded_results: dict,
                           atr_results: dict,
                           pair_name: str) -> None:
    """
    Side-by-side equity curve comparison of hardcoded vs
    ATR-based constant risk sizing for a single pair.

    Useful for directly evaluating whether ATR sizing
    improves risk-adjusted returns.

    Args:
        hardcoded_results: Results dict from run_backtests(use_atr_sizing=False)
        atr_results:       Results dict from run_backtests(use_atr_sizing=True)
        pair_name:         Short pair name for title (e.g. 'EURUSD')
    """
    _ensure_figures_dir()

    strategies = ['TrendFollowing', 'MeanReversion', 'PhaseAware']
    fig, axes = plt.subplots(
        len(strategies), 1,
        figsize=(16, 5 * len(strategies)),
        sharex=True
    )

    for ax, strategy in zip(axes, strategies):
        # Hardcoded sizing
        hc_metrics = hardcoded_results.get(strategy)
        if hc_metrics and 'equity_curve' in hc_metrics:
            eq = hc_metrics['equity_curve']
            eq_norm = eq / eq.iloc[0] * 100
            ax.plot(
                eq_norm.index, eq_norm.values,
                color=STRATEGY_COLORS.get(strategy, 'gray'),
                linewidth=1.5, linestyle='-',
                label=(
                    f'Hardcoded '
                    f'(R={hc_metrics["total_return"]:.1f}%, '
                    f'SR={hc_metrics["sharpe_ratio"]:.2f})'
                )
            )

        # ATR sizing
        atr_metrics = atr_results.get(strategy)
        if atr_metrics and 'equity_curve' in atr_metrics:
            eq = atr_metrics['equity_curve']
            eq_norm = eq / eq.iloc[0] * 100
            ax.plot(
                eq_norm.index, eq_norm.values,
                color=STRATEGY_COLORS.get(strategy, 'gray'),
                linewidth=1.5, linestyle='--',
                label=(
                    f'ATR sizing '
                    f'(R={atr_metrics["total_return"]:.1f}%, '
                    f'SR={atr_metrics["sharpe_ratio"]:.2f})'
                )
            )

        ax.axhline(
            100, color='gray', linestyle=':',
            linewidth=1, alpha=0.7
        )
        ax.set_title(
            f'{strategy} — {pair_name}',
            fontsize=11
        )
        ax.set_ylabel('Normalised Equity')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'{pair_name} — Hardcoded vs ATR Sizing Comparison',
        fontsize=14, y=1.01
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / f'sizing_comparison_{pair_name}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved {out_path}')