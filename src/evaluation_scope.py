"""Evaluation scope resolution for Phase G3.

G3 determines which strategies are evaluated.
G2 determines how the resulting StrategyEvaluation objects are ranked.

The EvaluationScope is resolved once per run, before the walk-forward
evaluation begins.  It controls which strategy evidence is generated but
does not modify:

- walk-forward logic
- strategy implementations
- StrategyEvaluation semantics
- Recommendation policy
- Recommendation ranking
- Behavioral Surface definitions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.strategy_registry import (
    DEFAULT_PHASEAWARE_POLICY_ID,
    EvaluationPolicyRegistry,
    StrategyRegistry,
)


@dataclass(frozen=True)
class EvaluationScope:
    """Resolved evaluation scope for a single MPML run.

    Attributes
    ----------
    strategy_ids:
        Ordered tuple of registry strategy IDs that participate in this
        evaluation.  These are the effective IDs actually used for the
        run, not merely the raw CLI arguments.
    source:
        Indicates whether the scope was ``"default"`` (derived from the
        active EvaluationPolicy) or ``"explicit"`` (supplied via
        ``--strategy`` CLI args).
    """

    strategy_ids: tuple[str, ...]
    source: str  # "default" or "explicit"

    def to_manifest_block(self) -> dict[str, Any]:
        """Return a compact manifest representation of this scope.

        The block is intentionally small.  It records the effective
        strategy IDs actually used for the run so that the experiment is
        self-describing and reproducible.
        """
        return {
            "strategy_ids": list(self.strategy_ids),
            "source": self.source,
        }


def resolve_evaluation_scope(
    requested_strategy_ids: list[str] | None,
    *,
    registry: StrategyRegistry,
    policy_registry: EvaluationPolicyRegistry,
    surface_id: str,
    default_policy_id: str = DEFAULT_PHASEAWARE_POLICY_ID,
) -> EvaluationScope:
    """Resolve the effective evaluation scope.

    G3 entrypoint: determines which strategies are evaluated.

    When no strategy IDs are explicitly requested the default scope is
    derived from the active EvaluationPolicy and the existing runtime is
    preserved exactly.  When strategy IDs are explicitly requested they
    are validated against the Strategy Registry and checked for
    compatibility with the active Behavioral Surface.

    Parameters
    ----------
    requested_strategy_ids:
        Strategy IDs supplied via ``--strategy`` CLI args.  Pass
        ``None`` or an empty list to use the default scope.
    registry:
        Strategy Registry to validate IDs against.
    policy_registry:
        Evaluation Policy Registry for resolving the default scope.
    surface_id:
        Active Behavioral Surface ID used for compatibility checking.
    default_policy_id:
        Evaluation policy used to derive the default scope.

    Returns
    -------
    EvaluationScope
        Resolved scope with the effective strategy IDs.

    Raises
    ------
    ValueError
        If any requested strategy ID is unknown in the registry.
    ValueError
        If any requested strategy ID is incompatible with the active
        Behavioral Surface.
    """
    if not requested_strategy_ids:
        # Default scope: strategies from the active EvaluationPolicy.
        # This preserves the existing default evaluation behaviour exactly.
        policy = policy_registry.get(default_policy_id)
        return EvaluationScope(
            strategy_ids=tuple(policy.strategies),
            source="default",
        )

    # Deduplicate while preserving order before validation so that error
    # messages do not contain repeated IDs.
    seen_pre: set[str] = set()
    deduplicated: list[str] = []
    for strategy_id in requested_strategy_ids:
        if strategy_id not in seen_pre:
            seen_pre.add(strategy_id)
            deduplicated.append(strategy_id)

    # Explicit scope: validate each requested ID against the registry.
    unknown = [sid for sid in deduplicated if sid not in registry.available()]
    if unknown:
        raise ValueError(
            f"Configuration error: unknown strategy ID(s) {unknown}. "
            f"Available: {registry.available()}"
        )

    # Check compatibility with the active Behavioral Surface.
    compatible_ids = {
        defn.strategy_id
        for defn in registry.supporting_surface(surface_id)
    }
    incompatible = [sid for sid in deduplicated if sid not in compatible_ids]
    if incompatible:
        raise ValueError(
            f"Configuration error: strategy ID(s) {incompatible} are not "
            f"compatible with Behavioral Surface {surface_id!r}. "
            f"Compatible strategies for this surface: {sorted(compatible_ids)}"
        )

    return EvaluationScope(
        strategy_ids=tuple(deduplicated),
        source="explicit",
    )


def filter_strategy_specs(
    strategy_specs: list[dict[str, Any]],
    scope: EvaluationScope,
) -> list[dict[str, Any]]:
    """Filter strategy_specs to those compatible with the evaluation scope.

    A spec is included when ALL of its ``scope_strategy_ids`` are present
    in the effective scope.  Specs without a ``scope_strategy_ids`` key
    are always included for backward compatibility.

    Parameters
    ----------
    strategy_specs:
        List of strategy specification dicts as consumed by
        ``build_strategy_evaluations``.
    scope:
        Resolved evaluation scope.

    Returns
    -------
    list[dict]
        Filtered list (may be empty).
    """
    scope_ids = frozenset(scope.strategy_ids)
    return [
        spec
        for spec in strategy_specs
        if not spec.get("scope_strategy_ids")
        or frozenset(spec["scope_strategy_ids"]).issubset(scope_ids)
    ]


def compute_standalone_execution_flags(
    scope: EvaluationScope,
    baseline_tf: str,
    baseline_mr: str,
) -> tuple[bool, bool]:
    """Return (run_tf, run_mr) execution flags for standalone WF backtests.

    Default runs skip both backtests entirely.  Explicit runs execute only
    the strategies that are present in the resolved scope.

    Parameters
    ----------
    scope:
        Resolved evaluation scope.
    baseline_tf:
        Registry ID of the trend-following baseline strategy (e.g. "TF4").
    baseline_mr:
        Registry ID of the mean-reversion baseline strategy (e.g. "MR42").

    Returns
    -------
    tuple[bool, bool]
        ``(run_tf, run_mr)`` — True when the corresponding standalone
        backtest should be executed for this fold.
    """
    if scope.source == "default":
        return False, False
    return (baseline_tf in scope.strategy_ids), (baseline_mr in scope.strategy_ids)


def should_run_full_universe_backtests(scope: EvaluationScope) -> bool:
    """Return True when the full-universe legacy backtests should be executed.

    Default runs must retain the existing full-universe benchmark behaviour
    (required by the dynamic strategy selector and historical comparisons).
    Explicit runs request a specific strategy subset and should not execute
    the entire strategy universe as a side-effect.

    Parameters
    ----------
    scope:
        Resolved evaluation scope.

    Returns
    -------
    bool
        ``True`` for default/unscoped runs; ``False`` for explicit runs.
    """
    return scope.source == "default"
