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
from src.models import PhaseMLExperiment
from src.visualization import PhaseVisualizer

__all__ = [
    'MarketDataPipeline',
    'MarketPhaseDetector',
    'MarketPhase',
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'PhaseAwareStrategy',
    'Backtester',
    'TradeResult',
    'PhaseMLExperiment',
    'PhaseVisualizer',
]