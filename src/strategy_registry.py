from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

from mpml.behavioral import registry as behavioral_registry

DEFAULT_PHASEAWARE_POLICY_ID = "phaseaware_default"


@dataclass(frozen=True)
class StrategyCapabilities:
    supported_surfaces: tuple[str, ...] = ()
    supported_states: tuple[str, ...] = ()
    supported_assets: tuple[str, ...] = ()
    required_features: tuple[str, ...] = ()
    required_indicators: tuple[str, ...] = ()
    supported_directions: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    display_name: str
    family: str
    implementation: type[Any]
    capabilities: StrategyCapabilities
    dependencies: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def instantiate(self) -> Any:
        return self.implementation()


@dataclass(frozen=True)
class EvaluationPolicy:
    policy_id: str
    display_name: str
    strategies: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedPhaseAwareComposition:
    policy_id: str
    tf_strategy_id: str
    mr_strategy_id: str

    @property
    def strategy_id(self) -> str:
        return f"PhaseAware_{self.tf_strategy_id}_{self.mr_strategy_id}"

    def to_manifest_block(self) -> dict[str, str]:
        return {
            "policy_id": self.policy_id,
            "phaseaware_tf_strategy": self.tf_strategy_id,
            "phaseaware_mr_strategy": self.mr_strategy_id,
            "phaseaware_strategy": self.strategy_id,
        }


class StrategyRegistry:
    def __init__(self, definitions: Iterable[StrategyDefinition]) -> None:
        self._definitions: dict[str, StrategyDefinition] = {}
        self._validate_and_load(definitions)

    def _validate_and_load(self, definitions: Iterable[StrategyDefinition]) -> None:
        for definition in definitions:
            if not definition.strategy_id:
                raise ValueError("StrategyRegistry: strategy_id is required.")
            if definition.strategy_id in self._definitions:
                raise ValueError(
                    f"StrategyRegistry: duplicate strategy_id {definition.strategy_id!r}."
                )
            if not definition.display_name:
                raise ValueError(
                    f"StrategyRegistry: display_name is required for {definition.strategy_id!r}."
                )
            if not definition.family:
                raise ValueError(
                    f"StrategyRegistry: family is required for {definition.strategy_id!r}."
                )
            if not callable(definition.implementation):
                raise ValueError(
                    f"StrategyRegistry: implementation must be callable for {definition.strategy_id!r}."
                )
            self._validate_capabilities(definition)
            self._definitions[definition.strategy_id] = definition

    @staticmethod
    def _validate_capabilities(definition: StrategyDefinition) -> None:
        capabilities = definition.capabilities
        for surface_id in capabilities.supported_surfaces:
            try:
                behavioral_registry.load(surface_id)
            except KeyError as exc:
                raise ValueError(
                    f"StrategyRegistry: {definition.strategy_id!r} references "
                    f"unknown Behavioral Surface {surface_id!r}."
                ) from exc

        if capabilities.supported_states and not capabilities.supported_surfaces:
            raise ValueError(
                f"StrategyRegistry: {definition.strategy_id!r} declares supported_states "
                "without supported_surfaces."
            )

        for state_id in capabilities.supported_states:
            state_valid = False
            for surface_id in capabilities.supported_surfaces:
                try:
                    surface = behavioral_registry.load(surface_id)
                    resolved = surface.get_state(state_id)
                except KeyError:
                    continue
                if resolved.surface_id == surface_id:
                    state_valid = True
                    break
            if not state_valid:
                raise ValueError(
                    f"StrategyRegistry: {definition.strategy_id!r} references "
                    f"unknown Behavioral State {state_id!r}."
                )

    def all(self) -> list[StrategyDefinition]:
        return [self._definitions[key] for key in sorted(self._definitions)]

    def get(self, strategy_id: str) -> StrategyDefinition:
        if strategy_id not in self._definitions:
            raise KeyError(
                f"StrategyRegistry: unknown strategy_id {strategy_id!r}. "
                f"Available: {self.available()}"
            )
        return self._definitions[strategy_id]

    def available(self) -> list[str]:
        return sorted(self._definitions)

    def by_family(self, family: str) -> list[StrategyDefinition]:
        target = family.casefold()
        return [
            definition
            for definition in self.all()
            if definition.family.casefold() == target
        ]

    def supporting_surface(self, surface_id: str) -> list[StrategyDefinition]:
        target = surface_id.casefold()
        return [
            definition
            for definition in self.all()
            if any(surface.casefold() == target for surface in definition.capabilities.supported_surfaces)
        ]

    def supporting_state(self, state_id: str) -> list[StrategyDefinition]:
        target = state_id.casefold()
        return [
            definition
            for definition in self.all()
            if any(state.casefold() == target for state in definition.capabilities.supported_states)
        ]

    def supporting_asset(self, asset: str) -> list[StrategyDefinition]:
        target = asset.casefold()
        return [
            definition
            for definition in self.all()
            if any(value.casefold() == target for value in definition.capabilities.supported_assets)
        ]

    def families(self) -> list[str]:
        return sorted({definition.family for definition in self._definitions.values()})

    def supported_surfaces(self) -> list[str]:
        surfaces = {
            surface_id
            for definition in self._definitions.values()
            for surface_id in definition.capabilities.supported_surfaces
        }
        return sorted(surfaces)


class EvaluationPolicyRegistry:
    def __init__(
        self,
        policies: Iterable[EvaluationPolicy],
        *,
        strategy_registry: StrategyRegistry,
    ) -> None:
        self._strategy_registry = strategy_registry
        self._policies: dict[str, EvaluationPolicy] = {}
        self._validate_and_load(policies)

    def _validate_and_load(self, policies: Iterable[EvaluationPolicy]) -> None:
        for policy in policies:
            if not policy.policy_id:
                raise ValueError("EvaluationPolicyRegistry: policy_id is required.")
            if policy.policy_id in self._policies:
                raise ValueError(
                    f"EvaluationPolicyRegistry: duplicate policy_id {policy.policy_id!r}."
                )
            if not policy.display_name:
                raise ValueError(
                    f"EvaluationPolicyRegistry: display_name is required for {policy.policy_id!r}."
                )
            if not policy.strategies:
                raise ValueError(
                    f"EvaluationPolicyRegistry: {policy.policy_id!r} must reference at least one strategy."
                )
            for strategy_id in policy.strategies:
                try:
                    self._strategy_registry.get(strategy_id)
                except KeyError as exc:
                    raise ValueError(
                        f"EvaluationPolicyRegistry: {policy.policy_id!r} references "
                        f"unknown strategy_id {strategy_id!r}."
                    ) from exc
            self._policies[policy.policy_id] = policy

    def all(self) -> list[EvaluationPolicy]:
        return [self._policies[key] for key in sorted(self._policies)]

    def available(self) -> list[str]:
        return sorted(self._policies)

    def get(self, policy_id: str) -> EvaluationPolicy:
        if policy_id not in self._policies:
            raise KeyError(
                f"EvaluationPolicyRegistry: unknown policy_id {policy_id!r}. "
                f"Available: {self.available()}"
            )
        return self._policies[policy_id]

    def strategies_for(self, policy_id: str) -> list[StrategyDefinition]:
        return [self._strategy_registry.get(strategy_id) for strategy_id in self.get(policy_id).strategies]


def _definition(
    *,
    strategy_id: str,
    display_name: str,
    family: str,
    implementation: type[Any],
    states: tuple[str, ...],
    indicators: tuple[str, ...],
    features: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> StrategyDefinition:
    return StrategyDefinition(
        strategy_id=strategy_id,
        display_name=display_name,
        family=family,
        implementation=implementation,
        capabilities=StrategyCapabilities(
            supported_surfaces=("trend_vol",),
            supported_states=states,
            supported_assets=("fx",),
            required_features=features,
            required_indicators=indicators,
            supported_directions=("long", "short"),
            tags=tags,
        ),
        dependencies=("pandas", "numpy", "ta"),
        metadata=metadata or {},
    )


def _build_strategy_definitions() -> list[StrategyDefinition]:
    from src.strategies import (
        TF1Strategy,
        TF2Strategy,
        TF3Strategy,
        TF4Strategy,
        TF5Strategy,
        MR1Strategy,
        MR2Strategy,
        MR3Strategy,
        MR32Strategy,
        MR42Strategy,
        MR5Strategy,
    )

    trend_states = ("HVTF", "LVTF")
    ranging_states = ("HVR", "LVR")

    return [
        _definition(
            strategy_id="TF1",
            display_name="TF1",
            family="TrendFollowing",
            implementation=TF1Strategy,
            states=trend_states,
            indicators=("lwma", "stddev"),
            features=("Close",),
            tags=("trend_following", "standalone"),
        ),
        _definition(
            strategy_id="TF2",
            display_name="TF2",
            family="TrendFollowing",
            implementation=TF2Strategy,
            states=trend_states,
            indicators=("sma"),
            features=("Close",),
            tags=("trend_following", "standalone"),
        ),
        _definition(
            strategy_id="TF3",
            display_name="TF3",
            family="TrendFollowing",
            implementation=TF3Strategy,
            states=trend_states,
            indicators=("stochastic",),
            features=("High", "Low", "Close"),
            tags=("trend_following", "standalone"),
        ),
        _definition(
            strategy_id="TF4",
            display_name="TF4",
            family="TrendFollowing",
            implementation=TF4Strategy,
            states=trend_states,
            indicators=("adx", "plus_di", "minus_di"),
            features=("High", "Low", "Close"),
            tags=("trend_following", "benchmark"),
        ),
        _definition(
            strategy_id="TF5",
            display_name="TF5",
            family="TrendFollowing",
            implementation=TF5Strategy,
            states=trend_states,
            indicators=("adx", "plus_di", "minus_di"),
            features=("High", "Low", "Close"),
            tags=("trend_following", "standalone"),
        ),
        _definition(
            strategy_id="MR1",
            display_name="MR1",
            family="MeanReversion",
            implementation=MR1Strategy,
            states=ranging_states,
            indicators=("bollinger_bands",),
            features=("Close",),
            tags=("mean_reversion", "standalone"),
        ),
        _definition(
            strategy_id="MR2",
            display_name="MR2",
            family="MeanReversion",
            implementation=MR2Strategy,
            states=ranging_states,
            indicators=("bollinger_bands", "rsi"),
            features=("Close",),
            tags=("mean_reversion", "standalone"),
        ),
        _definition(
            strategy_id="MR3",
            display_name="MR3",
            family="MeanReversion",
            implementation=MR3Strategy,
            states=ranging_states,
            indicators=("rsi",),
            features=("Close",),
            tags=("mean_reversion", "standalone"),
        ),
        _definition(
            strategy_id="MR32",
            display_name="MR32",
            family="MeanReversion",
            implementation=MR32Strategy,
            states=("LVR",),
            indicators=("rsi", "adx"),
            features=("Close",),
            tags=("mean_reversion", "adx_filtered"),
        ),
        _definition(
            strategy_id="MR42",
            display_name="MR42",
            family="MeanReversion",
            implementation=MR42Strategy,
            states=("LVR",),
            indicators=("stochastic", "adx"),
            features=("High", "Low", "Close"),
            tags=("mean_reversion", "adx_filtered", "benchmark"),
        ),
        _definition(
            strategy_id="MR5",
            display_name="MR5",
            family="MeanReversion",
            implementation=MR5Strategy,
            states=("HVR",),
            indicators=("stochastic",),
            features=("High", "Low", "Close"),
            tags=("mean_reversion", "high_volatility"),
        ),
    ]


def _build_policy_definitions() -> list[EvaluationPolicy]:
    return [
        EvaluationPolicy(
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
            display_name="PhaseAware Default",
            strategies=("TF4", "MR42"),
            metadata={
                "description": "Preserves the legacy PhaseAware benchmark composition.",
                "phaseaware": True,
            },
        ),
    ]


@lru_cache(maxsize=1)
def get_default_strategy_registry() -> StrategyRegistry:
    return StrategyRegistry(_build_strategy_definitions())


@lru_cache(maxsize=1)
def get_default_policy_registry() -> EvaluationPolicyRegistry:
    return EvaluationPolicyRegistry(
        _build_policy_definitions(),
        strategy_registry=get_default_strategy_registry(),
    )


def _resolve_policy_phaseaware_strategy_pair(
    policy_id: str = DEFAULT_PHASEAWARE_POLICY_ID,
    *,
    strategy_registry: StrategyRegistry | None = None,
    policy_registry: EvaluationPolicyRegistry | None = None,
) -> tuple[str, str]:
    strategy_registry = strategy_registry or get_default_strategy_registry()
    policy_registry = policy_registry or get_default_policy_registry()
    definitions = policy_registry.strategies_for(policy_id)
    trend = [
        definition.strategy_id
        for definition in definitions
        if definition.family == "TrendFollowing"
    ]
    mean_reversion = [
        definition.strategy_id
        for definition in definitions
        if definition.family == "MeanReversion"
    ]
    if len(trend) != 1 or len(mean_reversion) != 1:
        raise ValueError(
            f"resolve_phaseaware_strategy_pair: policy {policy_id!r} must resolve exactly one "
            "TrendFollowing and one MeanReversion strategy for PhaseAware."
        )
    strategy_registry.get(trend[0])
    strategy_registry.get(mean_reversion[0])
    return trend[0], mean_reversion[0]


def _validate_phaseaware_override(
    *,
    option_name: str,
    strategy_id: str,
    expected_family: str,
    strategy_registry: StrategyRegistry,
) -> str:
    try:
        definition = strategy_registry.get(strategy_id)
    except KeyError as exc:
        raise ValueError(
            f"Configuration error: {option_name} references unknown strategy_id "
            f"{strategy_id!r}. Available: {strategy_registry.available()}"
        ) from exc
    if definition.family != expected_family:
        raise ValueError(
            f"Configuration error: {option_name} must reference a "
            f"{expected_family} strategy, got {strategy_id!r} "
            f"(family={definition.family!r})."
        )
    return definition.strategy_id


def resolve_phaseaware_configuration(
    policy_id: str = DEFAULT_PHASEAWARE_POLICY_ID,
    *,
    phaseaware_tf_strategy_id: str | None = None,
    phaseaware_mr_strategy_id: str | None = None,
    strategy_registry: StrategyRegistry | None = None,
    policy_registry: EvaluationPolicyRegistry | None = None,
) -> ResolvedPhaseAwareComposition:
    strategy_registry = strategy_registry or get_default_strategy_registry()
    policy_registry = policy_registry or get_default_policy_registry()
    default_tf, default_mr = _resolve_policy_phaseaware_strategy_pair(
        policy_id,
        strategy_registry=strategy_registry,
        policy_registry=policy_registry,
    )
    resolved_tf = (
        _validate_phaseaware_override(
            option_name="--phaseaware-tf",
            strategy_id=phaseaware_tf_strategy_id,
            expected_family="TrendFollowing",
            strategy_registry=strategy_registry,
        )
        if phaseaware_tf_strategy_id is not None
        else default_tf
    )
    resolved_mr = (
        _validate_phaseaware_override(
            option_name="--phaseaware-mr",
            strategy_id=phaseaware_mr_strategy_id,
            expected_family="MeanReversion",
            strategy_registry=strategy_registry,
        )
        if phaseaware_mr_strategy_id is not None
        else default_mr
    )
    return ResolvedPhaseAwareComposition(
        policy_id=policy_id,
        tf_strategy_id=resolved_tf,
        mr_strategy_id=resolved_mr,
    )


def resolve_phaseaware_strategy_pair(
    policy_id: str = DEFAULT_PHASEAWARE_POLICY_ID,
    *,
    phaseaware_tf_strategy_id: str | None = None,
    phaseaware_mr_strategy_id: str | None = None,
    strategy_registry: StrategyRegistry | None = None,
    policy_registry: EvaluationPolicyRegistry | None = None,
) -> tuple[str, str]:
    composition = resolve_phaseaware_configuration(
        policy_id,
        phaseaware_tf_strategy_id=phaseaware_tf_strategy_id,
        phaseaware_mr_strategy_id=phaseaware_mr_strategy_id,
        strategy_registry=strategy_registry,
        policy_registry=policy_registry,
    )
    return composition.tf_strategy_id, composition.mr_strategy_id


def phaseaware_strategy_name(
    policy_id: str = DEFAULT_PHASEAWARE_POLICY_ID,
    *,
    phaseaware_tf_strategy_id: str | None = None,
    phaseaware_mr_strategy_id: str | None = None,
    strategy_registry: StrategyRegistry | None = None,
    policy_registry: EvaluationPolicyRegistry | None = None,
) -> str:
    return resolve_phaseaware_configuration(
        policy_id,
        phaseaware_tf_strategy_id=phaseaware_tf_strategy_id,
        phaseaware_mr_strategy_id=phaseaware_mr_strategy_id,
        strategy_registry=strategy_registry,
        policy_registry=policy_registry,
    ).strategy_id


_REGISTRY_SUMMARY_LOGGED = False


def log_default_registry_summary(print_fn=print) -> None:
    global _REGISTRY_SUMMARY_LOGGED
    if _REGISTRY_SUMMARY_LOGGED:
        return
    strategy_registry = get_default_strategy_registry()
    policy_registry = get_default_policy_registry()
    print_fn("\n[STRATEGY REGISTRY]")
    print_fn(f"Loaded strategies: {len(strategy_registry.all())}")
    print_fn("Families")
    for family in strategy_registry.families():
        print_fn(f"    {family}")
    print_fn("Supported Behavioral Surfaces")
    for surface_id in strategy_registry.supported_surfaces():
        print_fn(f"    {surface_id}")
    print_fn("Evaluation Policies")
    for policy_id in policy_registry.available():
        print_fn(f"    {policy_id}")
    _REGISTRY_SUMMARY_LOGGED = True
