# src/__init__.py
# Makes src a Python package

from src.data import MarketDataPipeline
from src.phases import MarketPhaseDetector, MarketPhase
from src.strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    PhaseAwareStrategy,
    Backtester,
    TradeResult
)
from src.visualization import (
    PhaseVisualizer,
    plot_backtest_results,
    plot_phase_performance
)

__all__ = [
    'MarketDataPipeline',
    'MarketPhaseDetector',
    'MarketPhase',
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'PhaseAwareStrategy',
    'Backtester',
    'TradeResult',
    'PhaseVisualizer',
    'plot_backtest_results',
    'plot_phase_performance'
]
