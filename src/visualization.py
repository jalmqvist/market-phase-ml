# src/visualization.py
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Strategy color scheme
# TF strategies  — blue family
# MR strategies  — orange/red family
# PhaseAware     — green family
# Special        — distinct colors for key comparison figure

STRATEGY_COLORS = {
    # TF strategies — blue family
    'TF1': '#1f77b4',
    'TF2': '#4a90d9',
    'TF3': '#7eb8e8',
    'TF4': '#0a4f8c',
    'TF5': '#a8c8f0',

    # MR strategies — orange/red family
    'MR1':  '#d62728',
    'MR2':  '#e8703a',
    'MR32': '#c0392b',
    'MR42': '#e74c3c',
    'MR5':  '#f0a500',

    # PhaseAware — green family
    # Best combo gets a distinct bright green
    'PhaseAware_TF4_MR42': '#2ca02c',
    'PhaseAware_TF4_MR5':  '#5cb85c',
    'PhaseAware_TF5_MR42': '#3d9970',
    'PhaseAware_TF2_MR42': '#27ae60',
    'PhaseAware_TF1_MR42': '#82c982',
    'PhaseAware_TF3_MR42': '#a8d8a8',

    # Fallback for remaining PhaseAware combos
    # (generated dynamically below)
}

# Dynamically add remaining PhaseAware combinations
# so nothing falls back to gray
_tf_names = ['TF1', 'TF2', 'TF3', 'TF4', 'TF5']
_mr_names = ['MR1', 'MR2', 'MR3', 'MR32', 'MR42', 'MR5']
_pa_green_shades = plt.cm.Greens(np.linspace(0.3, 0.9, 30))
_pa_idx = 0
for _tf in _tf_names:
    for _mr in _mr_names:
        _key = f'PhaseAware_{_tf}_{_mr}'
        if _key not in STRATEGY_COLORS:
            STRATEGY_COLORS[_key] = matplotlib.colors.to_hex(
                _pa_green_shades[_pa_idx]
            )
            _pa_idx += 1

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

        fig.suptitle(
            'Market Phase Overview',
            fontsize=16,
            fontweight='bold',
            y=1.01
        )
        fig.text(
            0.5, 0.99,
            f'Data: {ticker} — Daily (D1) — Representative example',
            ha='center',
            fontsize=11,
            style='italic',
            color='black'
        )

        # --- Panel 1: Price colored by phase ---
        ax1 = axes[0]
        for phase, color in self.phase_colors.items():
            mask = df['phase'] == phase
            if mask.sum() > 0:
                ax1.scatter(
                    df.index[mask],
                    df['Close'] [mask],
                    c=color,
                    s=2,
                    label=phase,
                    alpha=0.8
                )

        ax1.set_title(
            f'{ticker} — Price Colored by Market Phase',
            fontsize=13
        )
        ax1.set_ylabel('Price', fontsize=13)
        ax1.tick_params(axis='y', labelsize=11)
        ax1.legend(
            loc='upper left',
            fontsize=10,
            ncol=2,
            markerscale=3
        )
        ax1.grid(True, alpha=0.3)

        # --- Panel 2: ATR% with rolling median ---
        ax2 = axes[1]
        ax2.plot(
            df.index, df['atr_pct'],
            color='blue', linewidth=0.8, alpha=0.7,
            label='ATR%'
        )

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

        ax2.set_title(
            'ATR% — Normalised Volatility',
            fontsize=13
        )
        ax2.set_ylabel('ATR / Close (%)', fontsize=13)
        ax2.tick_params(axis='y', labelsize=11)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

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
        ax3.set_title(
            'ADX and Directional Indicators',
            fontsize=13
        )
        ax3.set_ylabel('ADX Value', fontsize=13)
        ax3.tick_params(axis='y', labelsize=11)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)

        # --- Panel 4: Phase timeline ---
        ax4 = axes[3]
        phase_list    = list(self.phase_colors.keys())
        phase_to_num  = {p: i for i, p in enumerate(phase_list)}
        phase_numeric = df['phase'].map(phase_to_num).fillna(0)

        ax4.scatter(
            df.index, phase_numeric,
            c=[self.phase_colors.get(p, '#AAAAAA')
               for p in df['phase']],
            s=2, alpha=0.8
        )
        ax4.set_yticks(range(len(phase_list)))
        ax4.set_yticklabels(phase_list, fontsize=11)
        ax4.set_title('Market Phase Over Time', fontsize=13)
        ax4.set_xlabel('Date', fontsize=13)
        ax4.tick_params(axis='x', labelsize=11)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = FIGURES_DIR / 'phases_overview.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Saved {out_path}')

    def plot_phase_statistics(self, df: pd.DataFrame,
                              ticker: str = 'EURUSD') -> None:
        """
        Statistical summary for the 4-phase scheme.

        Panels:
            1. Phase frequency distribution (with baseline)
            2. Directional accuracy per phase (with 50% baseline)
            3. Combined: Average next-day return + ATR% by phase
               (Phase on y-axis, both metrics on x-axis)

        All panels include baseline comparisons.
        Data source annotated in figure subtitle.
        """
        _ensure_figures_dir()

        # 3-panel layout: freq | accuracy | combined return+ATR
        fig, axes = plt.subplots(
            1, 3,
            figsize=(20, 8)
        )

        fig.suptitle(
            'Market Phase Statistics',
            fontsize=16,
            fontweight='bold',
            y=1.02
        )
        fig.text(
            0.5, 0.98,
            f'Data: {ticker} — Daily (D1) — Representative example',
            ha='center',
            fontsize=11,
            style='italic',
            color='#555555'
        )

        phases_ordered = ['HV_Trend', 'LV_Trend', 'HV_Ranging', 'LV_Ranging']
        phases_present = [p for p in phases_ordered if p in df['phase'].unique()]
        colors         = [self.phase_colors.get(p, '#AAAAAA')
                          for p in phases_present]

        # ── Panel 1: Phase frequency ──────────────────────────────────────
        ax1    = axes[0]
        total  = len(df)

        phase_counts = df['phase'].value_counts().reindex(
            phases_present
        ).fillna(0)

        baseline_pct = 100.0 / len(phases_present)

        bars = ax1.barh(
            range(len(phases_present)),
            phase_counts.values / total * 100,
            color=colors,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5
        )

        # Baseline reference line
        ax1.axvline(
            baseline_pct,
            color='red', linestyle='--',
            linewidth=1.2, alpha=0.7,
            label=f'Equal distribution ({baseline_pct:.0f}%)'
        )

        ax1.set_yticks(range(len(phases_present)))
        ax1.set_yticklabels(phases_present, fontsize=11)
        ax1.set_title('Phase Frequency\nDistribution', fontsize=13)
        ax1.set_xlabel('Frequency (%)', fontsize=13)
        ax1.tick_params(axis='x', labelsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='x')

        for bar, count in zip(bars, phase_counts.values):
            ax1.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f'{count / total * 100:.1f}%',
                ha='left', va='center', fontsize=10
            )


        # ── Panel 2: Directional accuracy ────────────────────────────────
        ax2 = axes[1]

        phase_accuracy = df.groupby('phase').apply(
            lambda x: (
                np.sign(x['next_return']) ==
                np.sign(x['returns'])
            ).mean()
        )
        phase_accuracy = pd.Series(phase_accuracy).reindex(
            phases_present
        ).fillna(0)

        ax2.barh(
            range(len(phases_present)),
            phase_accuracy.values * 100,
            color=colors,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5
        )

        # Baseline: random directional accuracy = 50%
        ax2.axvline(
            50, color='red', linestyle='--',
            linewidth=1.2, alpha=0.7,
            label='Random baseline (50%)'
        )

        ax2.set_yticks(range(len(phases_present)))
        ax2.set_yticklabels([])        # remove labels, keep ticks
        ax2.set_title('Directional Accuracy by Phase', fontsize=13)
        ax2.set_xlabel('Accuracy (%)', fontsize=13)
        ax2.tick_params(axis='x', labelsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for idx, val in enumerate(phase_accuracy.values * 100):
            ax2.text(
                val + 0.3, idx,
                f'{val:.1f}%',
                va='center', fontsize=10
            )

        # ── Panel 3: Scatter — ATR% vs Next-Day Return by Phase ──────────
        ax3 = axes[2]

        phase_returns = df.groupby('phase')['next_return'].mean().reindex(
            phases_present
        ).fillna(0) * 100

        phase_atr = df.groupby('phase')['atr_pct'].median().reindex(
            phases_present
        ).fillna(0)

        # Overall baselines
        baseline_return = df['next_return'].mean() * 100
        baseline_atr    = df['atr_pct'].median()

        # Plot each phase as a labeled point
        for idx, phase in enumerate(phases_present):
            color = self.phase_colors.get(phase, '#AAAAAA')
            x_val = phase_atr[phase]
            y_val = phase_returns[phase]

            ax3.scatter(
                x_val, y_val,
                color=color,
                s=200,              # marker size
                zorder=5,
                edgecolors='white',
                linewidth=1.5
            )

            # Phase label next to each point
            ax3.annotate(
                phase,
                xy=(x_val, y_val),
                xytext=(8, 4),
                textcoords='offset points',
                fontsize=10,
                color=color,
                fontweight='bold'
            )

        # Baseline crosshairs
        ax3.axhline(
            baseline_return,
            color='darkgreen', linestyle='--',
            linewidth=1.2, alpha=0.6,
            label=f'Overall avg return ({baseline_return:.3f}%)'
        )
        ax3.axvline(
            baseline_atr,
            color='darkorange', linestyle='--',
            linewidth=1.2, alpha=0.6,
            label=f'Overall median ATR% ({baseline_atr:.3f})'
        )

        # Quadrant labels — explains what each quadrant means
        x_min, x_max = ax3.get_xlim()
        y_min, y_max = ax3.get_ylim()

        # We'll add quadrant labels after setting limits
        ax3.set_xlabel('Median ATR% (Volatility)', fontsize=13)
        ax3.set_ylabel('', fontsize=13)   # no y-label — shared axis
        ax3.set_yticklabels([])           # no y-tick labels
        ax3.set_title(
            'Volatility vs Next-Day Return\nby Phase',
            fontsize=13
        )
        ax3.tick_params(axis='x', labelsize=11)
        ax3.grid(True, alpha=0.3)

        # Figure-level legend — top right corner
        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            fontsize=9,
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            framealpha=0.8,
            title='Baselines',
            title_fontsize=10
        )

        # ── Quadrant annotations ─────────────────────────────────────────
        # Get axis limits after plotting
        x_min, x_max = ax3.get_xlim()
        y_min, y_max = ax3.get_ylim()
        x_mid = baseline_atr
        y_mid = baseline_return

        quadrant_style = dict(
            fontsize=10,
            alpha=1,
            ha='center',
            va='center',
            style='italic',
            color='black'
        )

        # Top-left: low vol, high return — ideal TF conditions
        ax3.text(
            (x_min + x_mid) / 2, (y_mid + y_max) / 2,
            'Low Vol\nHigh Return\n(LV_Trend)',
            **quadrant_style
        )
        # Top-right: high vol, high return
        ax3.text(
            (x_mid + x_max) / 2, (y_mid + y_max) / 2,
            'High Vol\nHigh Return\n(HV_Trend)',
            **quadrant_style
        )
        # Bottom-left: low vol, low return — ideal MR conditions
        ax3.text(
            (x_min + x_mid) / 2, (y_min + y_mid) / 2,
            'Low Vol\nLow Return\n(LV_Ranging)',
            **quadrant_style
        )
        # Bottom-right: high vol, low return — avoid
        ax3.text(
            (x_mid + x_max) / 2, (y_min + y_mid) / 2,
            'High Vol\nLow Return\n(HV_Ranging)',
            **quadrant_style
        )

        # Add right-side y-axis showing return values
        ax3_right = ax3.twinx()
        ax3_right.set_ylim(ax3.get_ylim())
        ax3_right.set_ylabel(
            'Avg Next-Day Return (%)',
            fontsize=12
        )
        ax3_right.tick_params(axis='y', labelsize=11)
        ax3_right.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.3f}%')
        )

        plt.tight_layout()
        out_path = FIGURES_DIR / 'phase_statistics.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')  # bbox_inches='tight' handles the outside legend
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

def plot_key_results(hardcoded_results: dict,
                     loaded_majors: list,
                     loaded_minors: list) -> None:
    """
    The key results figure — shows the core finding of the project.

    Layout:
        Top row:    Equity curves — Majors average | Minors average
        Bottom row: Metrics table — Majors         | Minors

    Strategies shown:
        - PhaseAware_TF4_MR42  (best phase-aware combo)
        - MR42                 (best standalone MR)
        - TF4                  (best standalone TF)
    """
    _ensure_figures_dir()

    FOCUS_STRATEGIES = {
        'PhaseAware_TF4_MR42': {
            'color':     '#2ca02c',
            'linewidth': 2.5,
            'linestyle': '-',
            'label':     'PhaseAware TF4+MR42 (phase-aware)',
            'zorder':    5,
        },
        'MR42': {
            'color':     '#e74c3c',
            'linewidth': 1.8,
            'linestyle': '--',
            'label':     'MR42 (best standalone MR)',
            'zorder':    4,
        },
        'TF4': {
            'color':     '#1f77b4',
            'linewidth': 1.8,
            'linestyle': '--',
            'label':     'TF4 (best standalone TF)',
            'zorder':    3,
        },
    }

    # Consistent y-axis range for equity curves
    Y_MIN = 70
    Y_MAX = 140    # changed from 180

    fig, axes = plt.subplots(
        2, 2,
        figsize=(18, 12),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    fig.suptitle(
        'Phase-Aware Strategy Selection — Key Results\n'
        'PhaseAware_TF4_MR42 vs Best Standalone Strategies',
        fontsize=16,
        fontweight='bold',
        y=1.01
    )
    fig.text(
        0.5, 0.96,
        'Primary result figure — equally weighted average across '
        '7 major and 7 minor forex pairs, daily (D1) data.',
        ha='center',
        fontsize=11,
        style='italic',
        color='#2c7a2c',       # green to signal this is the main result
        wrap=True
    )
    groups = [
        ('Majors', loaded_majors,  axes[0, 0], axes[1, 0]),
        ('Minors', loaded_minors,  axes[0, 1], axes[1, 1]),
    ]

    for group_label, pairs, ax_equity, ax_table in groups:

        # ── Compute average equity curve per strategy ─────────────────────
        for strat_name, style in FOCUS_STRATEGIES.items():
            equity_series = []

            for pair in pairs:
                pair_result = hardcoded_results.get(pair, {})
                metrics     = pair_result.get(strat_name)

                if metrics is None or 'equity_curve' not in metrics:
                    continue

                equity      = metrics['equity_curve']
                equity_norm = equity / equity.iloc[0] * 100
                equity_series.append(equity_norm)

            if not equity_series:
                continue

            # Align and average
            combined   = pd.concat(equity_series, axis=1).ffill()
            avg_equity = combined.mean(axis=1)

            ax_equity.plot(
                avg_equity.index,
                avg_equity.values,
                color=style['color'],
                linewidth=style['linewidth'],
                linestyle=style['linestyle'],
                label=style['label'],
                zorder=style['zorder'],
                alpha=0.9
            )

        # Reference line at 100
        ax_equity.axhline(
            100, color='gray', linestyle=':',
            linewidth=1, alpha=0.5, label='Starting equity'
        )

        ax_equity.set_title(
            f'{group_label} — Average Equity Curve\n'
            f'({len(pairs)} pairs, equally weighted)',
            fontsize=11
        )
        ax_equity.set_ylabel(
            'Normalised Equity (start = 100)',
            fontsize=12        # increased font size
        )
        ax_equity.set_xlabel(
            'Date',
            fontsize=12        # added x-axis label
        )
        ax_equity.set_ylim(Y_MIN, Y_MAX)
        ax_equity.legend(fontsize=9, loc='upper left', framealpha=0.8)
        ax_equity.grid(True, alpha=0.3)

        # ── Metrics table ─────────────────────────────────────────────────
        table_rows = []
        for strat_name, style in FOCUS_STRATEGIES.items():
            returns, sharpes, drawdowns, win_rates, pfs = [], [], [], [], []

            for pair in pairs:
                pair_result = hardcoded_results.get(pair, {})
                metrics     = pair_result.get(strat_name)
                if metrics is None:
                    continue
                returns.append(metrics['total_return'])
                sharpes.append(metrics['sharpe_ratio'])
                drawdowns.append(metrics['max_drawdown'])
                win_rates.append(metrics['win_rate'])
                pfs.append(metrics['profit_factor'])

            if not returns:
                continue

            table_rows.append([
                style['label'].split('(')[0].strip(),  # short name
                f"{np.mean(returns):.0f}%",            # 32.1% -> 32%
                f"{np.mean(sharpes):.2f}",             # 0.190 -> 0.19
                f"{np.mean(drawdowns):.0f}%",          # -27.4% -> -27%
                f"{np.mean(win_rates):.0f}%",          # 62.4% -> 62%
                f"{np.mean(pfs):.2f}",                 # 1.130 -> 1.13
            ])

        col_labels = [
            'Strategy',
            'Avg Return',
            'Avg Sharpe',
            'Avg Drawdown',
            'Win Rate',
            'Profit Factor'
        ]

        ax_table.axis('off')
        table = ax_table.table(
            cellText=table_rows,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)     # slightly larger font
        table.scale(1, 2.0)       # slightly more row height

        # Color rows — fixed: check rowB0B (strategy name) not full row
        for row_idx, row in enumerate(table_rows):
            for col_idx in range(len(col_labels)):
                cell = table[row_idx + 1, col_idx]
                if 'PhaseAware' in row[0]:     # rowB0B = strategy name
                    cell.set_facecolor('#d4edda')   # light green
                elif 'TF4' in row:
                    cell.set_facecolor('#d6eaf8')   # light blue
                else:
                    cell.set_facecolor('#fde8e8')   # light red

        # Color header row
        for col_idx in range(len(col_labels)):
            table[0, col_idx].set_facecolor('#2c3e50')
            table[0, col_idx].set_text_props(
                color='white', fontweight='bold'
            )

        ax_table.set_title(
            f'{group_label} — Average Metrics ({len(pairs)} pairs)',
            fontsize=11,
            pad=10
        )

    plt.tight_layout()
    out_path = FIGURES_DIR / 'key_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved {out_path}')


# ---------------------------------------------------------------------------
# Module-level functions (multi-pair visualizations)
# ---------------------------------------------------------------------------
def plot_backtest_results(results: dict,
                          df: pd.DataFrame,  # noqa
                          title: str = 'Backtest Results') -> None:
    """
    3-panel backtest summary for a single pair.

    Panels:
        1. Equity curves — top 5 + bottom 3 by Sharpe ratio
        2. Drawdown curves — same strategies as panel 1
        3. Performance metrics — 2x2 subpanels (Return/WinRate
           and Sharpe/ProfitFactor on separate y-axes)

    Strategies selected: top 5 + bottom 3 by Sharpe ratio.
    Equity normalized to 100 at start for comparability.
    Single legend on right side of figure.
    """
    _ensure_figures_dir()

    # ── Select top 5 + bottom 3 strategies by Sharpe ─────────────────────
    sorted_by_sharpe = sorted(
        results.items(),
        key=lambda item: item[1]['sharpe_ratio'],
        reverse=True
    )

    top_5    = sorted_by_sharpe[:5]
    bottom_3 = sorted_by_sharpe[-3:]

    seen     = set()
    selected = []
    for name, metrics in top_5 + bottom_3:
        if name not in seen:
            selected.append((name, metrics))
            seen.add(name)

    # ── Figure layout — 3 rows + legend column ────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    gs  = fig.add_gridspec(
        3, 2,
        width_ratios=[5, 1.2],
        hspace=0.4,
        wspace=0.25
    )

    ax1      = fig.add_subplot(gs[0, 0])
    ax2      = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3      = fig.add_subplot(gs[2, 0])
    ax_leg   = fig.add_subplot(gs[:, 1])
    ax_leg.axis('off')

    fig.suptitle(
        f'{title}\nTop 5 + Bottom 3 Strategies by Sharpe Ratio',
        fontsize=16,
        fontweight='bold',
        y=1.01
    )
    fig.text(
        0.5, 0.96,
        'Note: Single pair results. EURUSD trended strongly 2014–2022, '
        'favouring trend-following strategies.\n'
        'See key_results.png for multi-pair risk-adjusted comparison.',
        ha='center',
        fontsize=10,
        style='italic',
        color='#cc0000',       # red to draw attention
        wrap=True
    )

    # ── Panel 1: Normalised equity curves ────────────────────────────────
    legend_handles = []
    legend_labels  = []

    for name, metrics in selected:
        equity      = metrics['equity_curve']
        equity_norm = equity / equity.iloc[0] * 100
        color       = STRATEGY_COLORS.get(name, '#aaaaaa')
        sharpe      = metrics['sharpe_ratio']

        # Distinguish top 5 vs bottom 3 with linewidth
        is_top = any(name == n for n, _ in top_5)
        lw     = 2.0 if is_top else 1.0
        alpha  = 0.9 if is_top else 0.5

        line, = ax1.plot(
            equity_norm.index,
            equity_norm.values,
            color=color,
            linewidth=lw,
            alpha=alpha,
            label=name
        )
        legend_handles.append(line)
        legend_labels.append(
            f"{name}  |  Sharpe: {sharpe:.2f}  |  "
            f"Return: {metrics['total_return']:.0f}%"
        )

    ax1.axhline(
        100, color='gray', linestyle=':',
        linewidth=1, alpha=0.5
    )
    ax1.set_title('Normalised Equity Curves', fontsize=13)
    ax1.set_ylabel('Normalised Equity (start = 100)', fontsize=13)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.set_ylim(40, 200)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    for name, metrics in selected:
        equity      = metrics['equity_curve']
        rolling_max = equity.expanding().max()
        drawdown    = (equity - rolling_max) / rolling_max * 100
        color       = STRATEGY_COLORS.get(name, '#aaaaaa')
        is_top      = any(name == n for n, _ in top_5)
        lw          = 2.0 if is_top else 1.0
        alpha       = 0.9 if is_top else 0.5

        ax2.plot(
            drawdown.index,
            drawdown.values,
            color=color,
            linewidth=lw,
            alpha=alpha
        )

    ax2.set_title('Drawdown (%)', fontsize=13)
    ax2.set_ylabel('Drawdown (%)', fontsize=13)
    ax2.set_xlabel('Date', fontsize=13)
    ax2.tick_params(axis='y', labelsize=11)
    ax2.tick_params(axis='x', labelsize=11)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Metrics — 2x2 using gridspec within ax3 ─────────────────
    # Remove ax3 and replace with 2x2 subgrid
    ax3.remove()
    gs_inner = gs[2, 0].subgridspec(
        1, 2, wspace=0.35
    )
    ax3a = fig.add_subplot(gs_inner[0, 0])   # Total Return + Win Rate
    ax3b = fig.add_subplot(gs_inner[0, 1])   # Sharpe + Profit Factor

    strategy_names = [name for name, _ in selected]
    x              = np.arange(len(strategy_names))
    width          = 0.35

    # Panel 3a: Total Return and Win Rate
    returns   = [m['total_return']  for _, m in selected]
    win_rates = [m['win_rate']       for _, m in selected]
    colors    = [STRATEGY_COLORS.get(n, '#aaaaaa') for n, _ in selected]

    ax3a.bar(
        x - width / 2, returns, width,
        color=colors, alpha=0.85,
        edgecolor='white', linewidth=0.5,
        label='Total Return (%)'
    )
    ax3a_twin = ax3a.twinx()
    ax3a_twin.bar(
        x + width / 2, win_rates, width,
        color=colors, alpha=0.45,
        edgecolor='white', linewidth=0.5,
        hatch='//',
        label='Win Rate (%)'
    )

    ax3a.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax3a.set_xticks(x)
    ax3a.set_xticklabels([])           # no x labels — legend handles this
    ax3a.set_ylabel('Total Return (%)', fontsize=12)
    ax3a_twin.set_ylabel('Win Rate (%)', fontsize=12)
    ax3a.set_title('Return & Win Rate', fontsize=13)
    ax3a.tick_params(axis='y', labelsize=11)
    ax3a_twin.tick_params(axis='y', labelsize=11)
    ax3a.grid(True, alpha=0.3, axis='y')

    # Panel 3b: Sharpe and Profit Factor
    sharpes = [m['sharpe_ratio']  for _, m in selected]
    pfs     = [m['profit_factor'] for _, m in selected]

    ax3b.bar(
        x - width / 2, sharpes, width,
        color=colors, alpha=0.85,
        edgecolor='white', linewidth=0.5,
        label='Sharpe Ratio'
    )
    ax3b_twin = ax3b.twinx()
    ax3b_twin.bar(
        x + width / 2, pfs, width,
        color=colors, alpha=0.45,
        edgecolor='white', linewidth=0.5,
        hatch='//',
        label='Profit Factor'
    )

    ax3b.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax3b_twin.axhline(
        1.0, color='red', linestyle='--',
        linewidth=1.0, alpha=0.6,
        label='Break-even (PF=1.0)'
    )
    ax3b.set_xticks(x)
    ax3b.set_xticklabels([])
    ax3b.set_ylabel('Sharpe Ratio', fontsize=12)
    ax3b_twin.set_ylabel('Profit Factor', fontsize=12)
    ax3b.set_title('Sharpe & Profit Factor', fontsize=13)
    ax3b.tick_params(axis='y', labelsize=11)
    ax3b_twin.tick_params(axis='y', labelsize=11)
    ax3b.grid(True, alpha=0.3, axis='y')

    # ── Single legend on right side ───────────────────────────────────────
    ax_leg.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center left',
        fontsize=11,           # increased from 9
        framealpha=0.9,
        title='Strategy  |  Sharpe  |  Return',
        title_fontsize=12,     # increased from 10
        ncol=1
    )

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
        figsize=(max(10.0, len(heatmap_df.columns) * 1.5),
                 max(6.0, len(heatmap_df) * 0.5))
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
    Strategy comparison for majors vs minors, hardcoded vs ATR sizing.

    Layout: 4 rows x 2 cols
        Row 1: Total Return (%)
        Row 2: Win Rate (%)
        Row 3: Sharpe Ratio
        Row 4: Profit Factor

        Left col:  Hardcoded sizing
        Right col: ATR constant risk sizing

    Single legend on the right side of the figure.
    Sharpe and Profit Factor have independent y-axes.
    """
    _ensure_figures_dir()

    # Metrics split into two groups for independent y-axis scaling
    METRICS_CONFIG = [
        {
            'col':       'Total Return (%)',
            'ylabel':    'Total Return (%)',
            'row':       0,
        },
        {
            'col':       'Win Rate (%)',
            'ylabel':    'Win Rate (%)',
            'row':       1,
        },
        {
            'col':       'Sharpe Ratio',
            'ylabel':    'Sharpe Ratio',
            'row':       2,
        },
        {
            'col':       'Profit Factor',
            'ylabel':    'Profit Factor',
            'row':       3,
        },
    ]

    # Use gridspec to add a narrow legend column on the right
    fig = plt.figure(figsize=(20, 16))
    gs  = fig.add_gridspec(
        4, 3,
        width_ratios=[5, 5, 1.0],   # two plot cols + one legend col, reduced from 1.5
        hspace=0.45,
        wspace=0.35
    )

    # Create plot axes
    plot_axes = np.array([
        [fig.add_subplot(gs[row, 0]),
         fig.add_subplot(gs[row, 1])]
        for row in range(4)
    ])

    # Legend axis — invisible, just holds the legend
    ax_legend = fig.add_subplot(gs[:, 2])
    ax_legend.axis('off')

    plot_configs = [
        (plot_axes[:, 0], majors_hardcoded, 'Majors — Hardcoded Sizing'),
        (plot_axes[:, 1], minors_hardcoded, 'Minors — Hardcoded Sizing'),
    ]

    # We'll collect legend handles once from the first panel
    legend_handles = []
    legend_labels  = []

    for col_idx, (summary_df, col_title) in enumerate([
        (majors_hardcoded, 'Majors — Hardcoded Sizing'),
        (minors_hardcoded, 'Minors — Hardcoded Sizing'),
    ]):
        for metric_cfg in METRICS_CONFIG:
            row      = metric_cfg['row']
            ax       = plot_axes[row, col_idx]
            metric   = metric_cfg['col']
            ylabel   = metric_cfg['ylabel']

            # Title only on top row
            if row == 0:
                ax.set_title(col_title, fontsize=13, fontweight='bold') # increased from 11

            if summary_df.empty:
                ax.text(
                    0.5, 0.5, 'No data',
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=11
                )
                continue

            strategies = summary_df['Strategy'].tolist()
            x          = np.arange(len(strategies))
            width      = 0.6

            for idx, strategy in enumerate(strategies):
                row_data = summary_df[
                    summary_df['Strategy'] == strategy
                ].iloc[0]

                value = row_data.get(metric, 0)
                color = STRATEGY_COLORS.get(strategy, '#aaaaaa')

                bar = ax.bar(
                    idx, value, width,
                    color=color,
                    alpha=0.85,
                    edgecolor='white',
                    linewidth=0.5
                )

                # Collect legend handles from first column, first metric only
                if col_idx == 0 and row == 0:
                    legend_handles.append(bar)
                    legend_labels.append(strategy)

            ax.set_ylabel(ylabel, fontsize=13)   # increased from 11
            ax.tick_params(axis='y', labelsize=11)
            ax.set_xticks([])          # no x tick labels — legend handles this
            ax.set_xlim(-0.5, len(strategies) - 0.5)
            ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')

            # Add profit factor reference line at 1.0
            if metric == 'Profit Factor':
                ax.axhline(
                    1.0, color='red', linewidth=1.0,
                    linestyle='--', alpha=0.6,
                    label='Break-even (PF=1.0)'
                )

    # Add ATR sizing results as a subtle overlay annotation
    # on the title to keep the figure from getting too complex
    fig.suptitle(
        'Strategy Comparison — Majors vs Minors\n'
        '(Hardcoded Position Sizing)',
        fontsize=16,        # increased from 14
        fontweight='bold',
        y=1.01
    )

    # Place single legend in the right column
    ax_legend.legend(
        handles=[h for h in legend_handles],
        labels=legend_labels,
        loc='center left',
        fontsize=11,            # increased from 9
        framealpha=0.9,
        title='Strategies',
        title_fontsize=12,      # increased from 0
        ncol=1
    )

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

    for pair in loaded_majors + loaded_minors:
        pair_result = hardcoded_results.get(pair, {})
        print(f"  {pair} keys: {list(pair_result.keys())}")

    _ensure_figures_dir()

    strategies = [
        'TF1', 'TF2', 'TF3', 'TF4', 'TF5',
        'MR1', 'MR2', 'MR32', 'MR42', 'MR5',
        'PhaseAware_TF4_MR42',   # best combo — always plot this one
    ]
    # Color palette for individual pairs
    pair_palette = list(plt.colormaps['tab20'].colors)

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

        # Consistent y-axis range across all strategy plots
        ax.set_ylim(40, 200)

        ax.set_title(
            f'{strategy} — Equity Curves: All Pairs\n'
            f'(Solid = Majors, Dashed = Minors, '
            f'Black = Average)',
            fontsize=13,           # increased
            fontweight='bold'
        )
        ax.set_ylabel('Normalised Equity (start = 100)',
                      fontsize=13  # increased
        )
        ax.set_xlabel('Date', fontsize=13)    # increased
        ax.tick_params(axis='y', labelsize=11)
        ax.tick_params(axis='x', labelsize=11)
        ax.legend(
            fontsize=8,            # slightly increased
            ncol=3,
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