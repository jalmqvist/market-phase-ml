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
from src.models import PhaseMLExperiment, PhaseMLPredictor
from src.visualization import PhaseVisualizer
from src.strategy_registry import (
    StrategyCapabilities,
    StrategyDefinition,
    StrategyRegistry,
    EvaluationPolicy,
    EvaluationPolicyRegistry,
    get_default_strategy_registry,
    get_default_policy_registry,
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
    'PhaseMLExperiment',
    'PhaseVisualizer',
    'PhaseMLPredictor',
    'StrategyCapabilities',
    'StrategyDefinition',
    'StrategyRegistry',
    'EvaluationPolicy',
    'EvaluationPolicyRegistry',
    'get_default_strategy_registry',
    'get_default_policy_registry',
]