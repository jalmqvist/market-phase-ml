"""Tests for Phase G3: Evaluation Scope and Strategy Selection.

These tests cover:
1.  No --strategy argument preserves the existing default scope.
2.  One --strategy selects exactly one valid strategy.
3.  Repeated --strategy selects exactly the requested set.
4.  Unknown strategy ID fails clearly.
5.  Incompatible strategy selection fails clearly.
6.  The resolved evaluation scope is recorded in the manifest block.
7.  Default runs record the effective default scope.
8.  StrategyEvaluation output is limited to the selected strategies
    (via filter_strategy_specs).
9.  Recommendation generation still works normally after targeted evaluation.
10. G2 recommendation ranking remains unchanged.
11. Repeated strategy arguments are deterministic.
12. Existing tests continue to pass (module import smoke-test).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation_scope import (  # noqa: E402
    EvaluationScope,
    filter_strategy_specs,
    resolve_evaluation_scope,
)
from src.strategy_registry import (  # noqa: E402
    DEFAULT_PHASEAWARE_POLICY_ID,
    get_default_policy_registry,
    get_default_strategy_registry,
    resolve_phaseaware_strategy_pair,
)
from src.evaluation import StrategyEvaluation  # noqa: E402
from src.recommendation import (  # noqa: E402
    DEFAULT_RECOMMENDATION_POLICY,
    recommendations_from_evaluations,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_REGISTRY = get_default_strategy_registry()
_POLICY_REGISTRY = get_default_policy_registry()
_SURFACE_ID = "trend_vol"


def _make_scope(strategy_ids: tuple[str, ...], source: str = "explicit") -> EvaluationScope:
    return EvaluationScope(strategy_ids=strategy_ids, source=source)


def _make_evaluation(strategy_id: str = "PhaseAware_TF4_MR42", *, sharpe: float = 0.5) -> StrategyEvaluation:
    return StrategyEvaluation(
        evaluation_id=f"eval_{strategy_id}",
        surface_id="trend_vol",
        surface_version="1.0.0",
        state_id="LVTF",
        strategy_id=strategy_id,
        expected_return=1.0,
        expected_sharpe=sharpe,
        expected_drawdown=-2.0,
        win_rate=None,
        confidence=None,
        stability=None,
        n_folds=4,
        n_trades=40,
        metadata={},
    )


def _default_strategy_specs(scope_ids: tuple[str, ...]) -> list[dict]:
    return [
        {
            "strategy_id": "PhaseAware_TF4_MR42",
            "scope_strategy_ids": scope_ids,
            "expected_return_col": "Baseline Return (%)",
            "expected_sharpe_col": "Baseline Sharpe",
            "expected_drawdown_col": "Baseline Max DD (%)",
            "n_trades_col": "Baseline Trades",
            "confidence_col": None,
            "strategy_role": "baseline",
        },
        {
            "strategy_id": "StrategySelector_Dynamic_WF",
            "scope_strategy_ids": scope_ids,
            "expected_return_col": "Dynamic Return (%)",
            "expected_sharpe_col": "Dynamic Sharpe",
            "expected_drawdown_col": "Dynamic Max DD (%)",
            "n_trades_col": "Dynamic Trades",
            "confidence_col": "Confident Bars (%)",
            "strategy_role": "dynamic_selector",
        },
    ]


# ---------------------------------------------------------------------------
# Test 1 — Default scope preserves existing behaviour
# ---------------------------------------------------------------------------

class TestDefaultScope(unittest.TestCase):
    def test_no_strategy_returns_default_policy_scope(self):
        """Test 1: No --strategy argument preserves the existing default scope."""
        scope = resolve_evaluation_scope(
            None,
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        policy = _POLICY_REGISTRY.get(DEFAULT_PHASEAWARE_POLICY_ID)
        self.assertEqual(set(scope.strategy_ids), set(policy.strategies))
        self.assertEqual(scope.source, "default")

    def test_empty_list_returns_default_scope(self):
        """Empty list is treated the same as None (no --strategy given)."""
        scope = resolve_evaluation_scope(
            [],
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        self.assertEqual(scope.source, "default")

    def test_default_scope_contains_expected_strategies(self):
        """Default scope must contain TF4 and MR42 (the phaseaware_default policy)."""
        scope = resolve_evaluation_scope(
            None,
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        self.assertIn("TF4", scope.strategy_ids)
        self.assertIn("MR42", scope.strategy_ids)


# ---------------------------------------------------------------------------
# Test 2 — Single --strategy selects exactly one strategy
# ---------------------------------------------------------------------------

class TestSingleStrategySelection(unittest.TestCase):
    def test_single_valid_strategy_tf4(self):
        """Test 2: One --strategy selects exactly one valid strategy."""
        scope = resolve_evaluation_scope(
            ["TF4"],
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        self.assertEqual(scope.strategy_ids, ("TF4",))
        self.assertEqual(scope.source, "explicit")

    def test_single_valid_strategy_mr42(self):
        scope = resolve_evaluation_scope(
            ["MR42"],
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        self.assertEqual(scope.strategy_ids, ("MR42",))
        self.assertEqual(scope.source, "explicit")


# ---------------------------------------------------------------------------
# Test 3 — Repeated --strategy selects exactly the requested set
# ---------------------------------------------------------------------------

class TestMultipleStrategySelection(unittest.TestCase):
    def test_two_strategies_selected(self):
        """Test 3: Repeated --strategy selects exactly the requested set."""
        scope = resolve_evaluation_scope(
            ["TF4", "MR42"],
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        self.assertEqual(set(scope.strategy_ids), {"TF4", "MR42"})
        self.assertEqual(len(scope.strategy_ids), 2)

    def test_multiple_tf_strategies(self):
        scope = resolve_evaluation_scope(
            ["TF1", "TF4"],
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        self.assertEqual(set(scope.strategy_ids), {"TF1", "TF4"})


# ---------------------------------------------------------------------------
# Test 4 — Unknown strategy ID fails clearly
# ---------------------------------------------------------------------------

class TestUnknownStrategyId(unittest.TestCase):
    def test_unknown_id_raises_value_error(self):
        """Test 4: Unknown strategy ID fails clearly."""
        with self.assertRaises(ValueError) as ctx:
            resolve_evaluation_scope(
                ["NONEXISTENT"],
                registry=_REGISTRY,
                policy_registry=_POLICY_REGISTRY,
                surface_id=_SURFACE_ID,
            )
        self.assertIn("NONEXISTENT", str(ctx.exception))
        self.assertIn("Configuration error", str(ctx.exception))

    def test_unknown_id_message_lists_available(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_evaluation_scope(
                ["UNKNOWN_XYZ"],
                registry=_REGISTRY,
                policy_registry=_POLICY_REGISTRY,
                surface_id=_SURFACE_ID,
            )
        msg = str(ctx.exception)
        # Error message should reference available IDs
        self.assertIn("TF4", msg)

    def test_partial_unknown_also_fails(self):
        """Mixed valid + unknown fails clearly."""
        with self.assertRaises(ValueError) as ctx:
            resolve_evaluation_scope(
                ["TF4", "BOGUS_ID"],
                registry=_REGISTRY,
                policy_registry=_POLICY_REGISTRY,
                surface_id=_SURFACE_ID,
            )
        self.assertIn("BOGUS_ID", str(ctx.exception))


# ---------------------------------------------------------------------------
# Test 5 — Incompatible strategy selection fails clearly
# ---------------------------------------------------------------------------

class TestIncompatibleStrategy(unittest.TestCase):
    def test_strategy_incompatible_with_surface(self):
        """Test 5: Strategy incompatible with active surface fails clearly.

        All current registry strategies declare support for 'trend_vol'.
        We test against a non-existent surface to confirm the compatibility
        check fires, since the registry does not currently include a strategy
        that is compatible only with a different surface.
        """
        # MR42 supports trend_vol; requesting it against a fictional surface
        # should raise ValueError.
        with self.assertRaises((ValueError, KeyError)):
            resolve_evaluation_scope(
                ["MR42"],
                registry=_REGISTRY,
                policy_registry=_POLICY_REGISTRY,
                surface_id="nonexistent_surface_xyz",
            )

    def test_strategy_not_on_surface_raises_with_clear_message(self):
        """Incompatible surface error message names the incompatible strategy."""
        # Build a minimal registry containing a strategy that only supports
        # a different surface (simulate incompatibility).
        from src.strategy_registry import (
            StrategyCapabilities,
            StrategyDefinition,
            StrategyRegistry,
            EvaluationPolicy,
            EvaluationPolicyRegistry,
        )
        from src.strategies import TF4Strategy

        # Strategy declares support only for a fictional surface
        fictional_surface_strategy = StrategyDefinition(
            strategy_id="TF4",
            display_name="TF4 test",
            family="TrendFollowing",
            implementation=TF4Strategy,
            capabilities=StrategyCapabilities(
                supported_surfaces=("trend_vol",),
                supported_states=("HVTF", "LVTF"),
                supported_assets=("fx",),
                supported_directions=("long", "short"),
            ),
        )
        mini_registry = StrategyRegistry([fictional_surface_strategy])
        mini_policy_registry = EvaluationPolicyRegistry(
            [EvaluationPolicy(
                policy_id="phaseaware_default",
                display_name="PhaseAware Default",
                strategies=("TF4",),
            )],
            strategy_registry=mini_registry,
        )

        # TF4 supports trend_vol but not reactive_jpy; requesting reactive_jpy
        # surface should yield an incompatible error.
        with self.assertRaises(ValueError) as ctx:
            resolve_evaluation_scope(
                ["TF4"],
                registry=mini_registry,
                policy_registry=mini_policy_registry,
                surface_id="reactive_jpy",
            )
        msg = str(ctx.exception)
        self.assertIn("TF4", msg)
        self.assertIn("reactive_jpy", msg)


# ---------------------------------------------------------------------------
# Test 6 & 7 — Manifest recording
# ---------------------------------------------------------------------------

class TestManifestRecording(unittest.TestCase):
    def test_explicit_scope_manifest_block(self):
        """Test 6: Resolved evaluation scope is recorded in the manifest."""
        scope = resolve_evaluation_scope(
            ["TF4", "MR42"],
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        block = scope.to_manifest_block()
        self.assertIn("strategy_ids", block)
        self.assertIn("source", block)
        self.assertEqual(set(block["strategy_ids"]), {"TF4", "MR42"})
        self.assertEqual(block["source"], "explicit")

    def test_default_scope_manifest_block(self):
        """Test 7: Default runs record the effective default scope."""
        scope = resolve_evaluation_scope(
            None,
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        block = scope.to_manifest_block()
        self.assertIn("strategy_ids", block)
        self.assertIn("source", block)
        self.assertEqual(block["source"], "default")
        # Must record the actual IDs, not just "default"
        self.assertGreater(len(block["strategy_ids"]), 0)
        self.assertIn("TF4", block["strategy_ids"])

    def test_manifest_block_is_json_serialisable(self):
        """Manifest block must be JSON-serialisable without custom encoders."""
        import json
        scope = resolve_evaluation_scope(
            ["TF4"],
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        block = scope.to_manifest_block()
        serialised = json.dumps(block)
        restored = json.loads(serialised)
        self.assertEqual(restored["strategy_ids"], list(scope.strategy_ids))


# ---------------------------------------------------------------------------
# Test 8 — StrategyEvaluation output is limited to selected strategies
# ---------------------------------------------------------------------------

class TestFilterStrategySpecs(unittest.TestCase):
    """Test 8: filter_strategy_specs limits StrategyEvaluation to selected strategies."""

    def _policy_ids(self) -> tuple[str, str]:
        return resolve_phaseaware_strategy_pair()

    def test_full_scope_includes_all_specs(self):
        """When scope contains all policy strategy IDs, all specs are included."""
        tf, mr = self._policy_ids()
        scope = _make_scope((tf, mr))
        specs = _default_strategy_specs((tf, mr))
        result = filter_strategy_specs(specs, scope)
        self.assertEqual(len(result), 2)

    def test_partial_scope_excludes_specs(self):
        """When scope is a strict subset of policy IDs, no spec is included."""
        tf, _mr = self._policy_ids()
        scope = _make_scope((tf,))
        specs = _default_strategy_specs((tf, "MR42"))
        result = filter_strategy_specs(specs, scope)
        # PhaseAware_TF4_MR42 requires both TF4 and MR42; scope only has TF4.
        self.assertEqual(len(result), 0)

    def test_spec_without_scope_ids_is_always_included(self):
        """Specs without scope_strategy_ids are always included (backward compat)."""
        scope = _make_scope(("TF1",))
        specs = [{"strategy_id": "LegacyStrategy", "strategy_role": "legacy"}]
        result = filter_strategy_specs(specs, scope)
        self.assertEqual(len(result), 1)

    def test_empty_scope_ids_on_spec_included(self):
        """Spec with empty scope_strategy_ids is always included."""
        scope = _make_scope(("TF4",))
        specs = [{"strategy_id": "S", "scope_strategy_ids": ()}]
        result = filter_strategy_specs(specs, scope)
        self.assertEqual(len(result), 1)

    def test_filter_preserves_spec_content(self):
        """filter_strategy_specs must not mutate spec content."""
        tf, mr = self._policy_ids()
        scope = _make_scope((tf, mr))
        specs = _default_strategy_specs((tf, mr))
        result = filter_strategy_specs(specs, scope)
        self.assertEqual(result[0]["strategy_id"], "PhaseAware_TF4_MR42")
        self.assertEqual(result[1]["strategy_id"], "StrategySelector_Dynamic_WF")


# ---------------------------------------------------------------------------
# Test 9 — Recommendation generation works normally after targeted evaluation
# ---------------------------------------------------------------------------

class TestRecommendationAfterTargetedEvaluation(unittest.TestCase):
    """Test 9: Recommendation generation still works after targeted evaluation."""

    def test_recommendations_generated_from_filtered_evaluations(self):
        evals = [_make_evaluation("PhaseAware_TF4_MR42", sharpe=0.6)]
        recs = recommendations_from_evaluations(evals)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].rank, 1)

    def test_empty_evaluations_produces_no_recommendations(self):
        recs = recommendations_from_evaluations([])
        self.assertEqual(len(recs), 0)

    def test_recommendations_reference_evaluation_ids(self):
        evals = [_make_evaluation("PhaseAware_TF4_MR42")]
        recs = recommendations_from_evaluations(evals)
        self.assertEqual(recs[0].evaluation_id, "eval_PhaseAware_TF4_MR42")


# ---------------------------------------------------------------------------
# Test 10 — G2 recommendation ranking remains unchanged
# ---------------------------------------------------------------------------

class TestG2RankingUnchanged(unittest.TestCase):
    """Test 10: G2 recommendation ranking (sharpe_rank_v1) remains unchanged."""

    def test_ranking_is_sharpe_descending(self):
        evals = [
            _make_evaluation("low_sharpe", sharpe=0.1),
            _make_evaluation("high_sharpe", sharpe=0.9),
            _make_evaluation("mid_sharpe", sharpe=0.5),
        ]
        recs = recommendations_from_evaluations(evals)
        # Rank 1 must be the highest Sharpe
        rank1_eval_id = recs[0].evaluation_id
        self.assertEqual(rank1_eval_id, "eval_high_sharpe")

    def test_recommendation_policy_is_sharpe_rank_v1(self):
        from src.recommendation import SharpeRankingPolicy, DEFAULT_POLICY
        self.assertIsInstance(DEFAULT_POLICY, SharpeRankingPolicy)
        self.assertEqual(DEFAULT_POLICY.policy_name, "sharpe_rank_v1")

    def test_scope_does_not_change_ranking_policy(self):
        """Strategy scope selection must not affect the recommendation policy."""
        from src.recommendation import DEFAULT_RECOMMENDATION_POLICY
        self.assertEqual(DEFAULT_RECOMMENDATION_POLICY, "sharpe_rank_v1")


# ---------------------------------------------------------------------------
# Test 11 — Repeated strategy arguments are deterministic
# ---------------------------------------------------------------------------

class TestDeterministicScope(unittest.TestCase):
    """Test 11: Repeated strategy arguments produce deterministic results."""

    def test_same_input_same_output(self):
        scope_a = resolve_evaluation_scope(
            ["TF4", "MR42"],
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        scope_b = resolve_evaluation_scope(
            ["TF4", "MR42"],
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        self.assertEqual(scope_a, scope_b)

    def test_order_is_preserved(self):
        """Order of --strategy args is preserved in the resulting scope."""
        scope = resolve_evaluation_scope(
            ["MR42", "TF4"],
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        # First-listed strategy should appear first
        self.assertEqual(scope.strategy_ids[0], "MR42")
        self.assertEqual(scope.strategy_ids[1], "TF4")

    def test_duplicates_are_deduplicated_deterministically(self):
        """Duplicate --strategy values are deduplicated, keeping first occurrence."""
        scope = resolve_evaluation_scope(
            ["TF4", "MR42", "TF4"],
            registry=_REGISTRY,
            policy_registry=_POLICY_REGISTRY,
            surface_id=_SURFACE_ID,
        )
        self.assertEqual(list(scope.strategy_ids).count("TF4"), 1)
        self.assertEqual(len(scope.strategy_ids), 2)
        # TF4 first, as it appeared first in input
        self.assertEqual(scope.strategy_ids[0], "TF4")

    def test_filter_specs_is_deterministic(self):
        """filter_strategy_specs produces the same result on repeated calls."""
        tf, mr = resolve_phaseaware_strategy_pair()
        scope = _make_scope((tf, mr))
        specs = _default_strategy_specs((tf, mr))
        result_a = filter_strategy_specs(specs, scope)
        result_b = filter_strategy_specs(specs, scope)
        self.assertEqual(
            [s["strategy_id"] for s in result_a],
            [s["strategy_id"] for s in result_b],
        )


# ---------------------------------------------------------------------------
# Test 12 — Module import smoke-test (existing tests continue to pass)
# ---------------------------------------------------------------------------

class TestModuleImportSmoke(unittest.TestCase):
    """Test 12: Existing module imports are not broken by G3 additions."""

    def test_evaluation_scope_module_importable(self):
        import src.evaluation_scope  # noqa: F401

    def test_strategy_registry_module_importable(self):
        import src.strategy_registry  # noqa: F401

    def test_evaluation_module_importable(self):
        import src.evaluation  # noqa: F401

    def test_recommendation_module_importable(self):
        import src.recommendation  # noqa: F401

    def test_evaluation_scope_exports(self):
        from src.evaluation_scope import (  # noqa: F401
            EvaluationScope,
            filter_strategy_specs,
            resolve_evaluation_scope,
        )

    def test_default_registry_accessible(self):
        registry = get_default_strategy_registry()
        self.assertIn("TF4", registry.available())
        self.assertIn("MR42", registry.available())

    def test_default_policy_registry_accessible(self):
        policy_registry = get_default_policy_registry()
        self.assertIn(DEFAULT_PHASEAWARE_POLICY_ID, policy_registry.available())


if __name__ == "__main__":
    unittest.main()
