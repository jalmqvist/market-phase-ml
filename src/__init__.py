# src/__init__.py
# Makes src a Python package.
#
# Public names are re-exported lazily (PEP 562) so that importing a light
# submodule such as ``src.phases`` or ``src.strategies`` does not pull in the
# research stack (models, visualization, data loaders). ``from src import X``
# and ``src.X`` keep working for every previously exported name.

from importlib import import_module as _import_module

_LAZY_EXPORTS = {
    'MarketDataPipeline': 'src.data',
    'MarketPhaseDetector': 'src.phases',
    'MarketPhase': 'src.phases',
    'TrendFollowingStrategy': 'src.strategies',
    'MeanReversionStrategy': 'src.strategies',
    'PhaseAwareStrategy': 'src.strategies',
    'Backtester': 'src.strategies',
    'TradeResult': 'src.strategies',
    'PhaseMLExperiment': 'src.models',
    'PhaseMLPredictor': 'src.models',
    'PhaseVisualizer': 'src.visualization',
    'StrategyCapabilities': 'src.strategy_registry',
    'StrategyDefinition': 'src.strategy_registry',
    'StrategyRegistry': 'src.strategy_registry',
    'EvaluationPolicy': 'src.strategy_registry',
    'EvaluationPolicyRegistry': 'src.strategy_registry',
    'ResolvedPhaseAwareComposition': 'src.strategy_registry',
    'get_default_strategy_registry': 'src.strategy_registry',
    'get_default_policy_registry': 'src.strategy_registry',
    'resolve_phaseaware_configuration': 'src.strategy_registry',
}

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
    'ResolvedPhaseAwareComposition',
    'get_default_strategy_registry',
    'get_default_policy_registry',
    'resolve_phaseaware_configuration',
]


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'src' has no attribute {name!r}")
    value = getattr(_import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
