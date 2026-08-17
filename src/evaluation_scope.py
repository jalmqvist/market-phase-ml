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

from mpml.behavioral import registry as behavioral_registry
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
    state_id: str | None = None,
    default_policy_id: str = DEFAULT_PHASEAWARE_POLICY_ID,
) -> EvaluationScope:
    """Resolve the effective evaluation scope.

    G3 entrypoint: determines which strategies are evaluated.

    When no strategy IDs are explicitly requested the default scope is
    derived from the active EvaluationPolicy and the existing runtime is
    preserved exactly.  When strategy IDs are explicitly requested they
    are validated against the Strategy Registry, and the Behavioral Surface
    is validated as a registered surface.

    Architectural note: strategy capability (``supported_surfaces`` on
    :class:`~src.strategy_registry.StrategyCapabilities`) describes what a
    strategy is intrinsically implemented for.  Behavioral Surface / State
    describes the market population on which the strategy is *evaluated*.
    Evaluation Scope determines which strategies participate in a particular
    experiment.  These three concepts are independent.  A strategy does not
    need to declare a Behavioral Surface as a native capability in order to
    be evaluated conditionally within that surface's experiment.

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
        Active Behavioral Surface ID.  Must be a registered surface;
        does not restrict which strategies may participate.
    state_id:
        Optional Behavioral State ID.  When provided it must belong to
        *surface_id*; cross-surface state IDs (e.g. ``"LVTF"`` passed
        with ``surface_id="reactive_jpy"``) are rejected.
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
        If the surface_id is not a registered Behavioral Surface.
    ValueError
        If state_id is provided but does not belong to surface_id.
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

    # Validate the Behavioral Surface itself.  This rejects unknown surface
    # IDs while allowing any valid registry strategy to be evaluated
    # conditionally on a Behavioral Surface — regardless of whether that
    # strategy declares the surface as a native capability.
    #
    # Architectural note
    # ------------------
    # Strategy capability (supported_surfaces on StrategyCapabilities) describes
    # what a strategy is intrinsically implemented for.  Behavioral Surface /
    # State describes the market population on which the strategy is *evaluated*.
    # Evaluation Scope determines which strategies participate in a particular
    # experiment.  These are three distinct concepts and must not be conflated.
    #
    # A TrendFollowing strategy does not need to declare "reactive_jpy" in its
    # supported_surfaces in order to be evaluated conditionally within the
    # Reactive-JPY behavioral surface experiment.  The behavioral surface
    # conditions the evaluation universe; it is not an intrinsic strategy
    # attribute.
    if surface_id not in behavioral_registry:
        raise ValueError(
            f"Configuration error: unknown Behavioral Surface {surface_id!r}. "
            f"Available surfaces: {behavioral_registry.available()}"
        )

    # Validate the Behavioral State when provided.  The state must belong to
    # the active surface — cross-surface state IDs (e.g. "LVTF" with
    # surface_id="reactive_jpy") are rejected here rather than silently
    # producing meaningless results downstream.
    if state_id is not None:
        surface = behavioral_registry.load(surface_id)
        try:
            surface.get_state(state_id)
        except KeyError:
            surface_states = [s.state_id for s in surface.states()]
            raise ValueError(
                f"Configuration error: Behavioral State {state_id!r} does not "
                f"belong to surface {surface_id!r}. "
                f"Valid states for this surface: {sorted(surface_states)}"
            ) from None

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
