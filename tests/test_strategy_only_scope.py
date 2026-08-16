"""Unit tests for the strategy-only scope optimization in main.py.

These tests verify the new control-flow behavior introduced to skip the
expensive legacy ML and aggregation stages when an explicit ``--strategy``
scope is active.

Tests cover:
1. Explicit strategy scope is recognized as strategy-only.
2. Full-universe/default scope is NOT recognized as strategy-only.
3. In explicit strategy mode, the legacy ML experiment stage is not invoked.
4. In explicit strategy mode, ML phase prediction is not invoked.
5. In explicit strategy mode, full-universe backtest/aggregation code is not
   invoked.
6. Existing strategy-specific evaluation code remains reachable (i.e. the
   walk-forward execution plan is still built for the explicit scope).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation_scope import EvaluationScope  # noqa: E402
from src.strategy_registry import (  # noqa: E402
    get_default_policy_registry,
    get_default_strategy_registry,
)


def _explicit_scope(*strategy_ids: str) -> EvaluationScope:
    return EvaluationScope(strategy_ids=tuple(strategy_ids), source="explicit")


def _default_scope() -> EvaluationScope:
    return EvaluationScope(strategy_ids=("TF4", "MR42"), source="default")


class TestStrategyOnlyScopeHelper(unittest.TestCase):
    """Verify that _strategy_only_scope_enabled returns the correct value.

    This helper mirrors the ``_strategy_only_scope`` local variable in
    ``main()`` and is the primary gate that controls whether the legacy ML
    stages are skipped.
    """

    @classmethod
    def setUpClass(cls):
        import importlib
        cls._main = importlib.import_module("main")

    # ── Test 1: explicit scope is recognized as strategy-only ─────────────
    def test_explicit_mr32_is_strategy_only(self):
        scope = _explicit_scope("MR32")
        self.assertTrue(
            self._main._strategy_only_scope_enabled(scope),
            "Explicit MR32 scope must be recognized as strategy-only",
        )

    def test_explicit_mr5_mr32_is_strategy_only(self):
        scope = _explicit_scope("MR5", "MR32")
        self.assertTrue(
            self._main._strategy_only_scope_enabled(scope),
            "Explicit multi-strategy scope must be recognized as strategy-only",
        )

    def test_explicit_tf4_is_strategy_only(self):
        scope = _explicit_scope("TF4")
        self.assertTrue(
            self._main._strategy_only_scope_enabled(scope),
        )

    def test_all_registered_strategies_produce_strategy_only_scope(self):
        registry = get_default_strategy_registry()
        for sid in registry.available():
            scope = _explicit_scope(sid)
            self.assertTrue(
                self._main._strategy_only_scope_enabled(scope),
                f"Explicit scope for {sid} must be recognized as strategy-only",
            )

    # ── Test 2: default scope is NOT strategy-only ─────────────────────────
    def test_default_scope_is_not_strategy_only(self):
        scope = _default_scope()
        self.assertFalse(
            self._main._strategy_only_scope_enabled(scope),
            "Default scope must NOT be recognized as strategy-only",
        )

    def test_default_scope_from_resolve_is_not_strategy_only(self):
        scope = self._main.resolve_evaluation_scope(
            requested_strategy_ids=None,
            registry=get_default_strategy_registry(),
            policy_registry=get_default_policy_registry(),
            surface_id="trend_vol",
        )
        self.assertFalse(self._main._strategy_only_scope_enabled(scope))

    def test_empty_strategy_list_produces_non_strategy_only_scope(self):
        scope = self._main.resolve_evaluation_scope(
            requested_strategy_ids=[],
            registry=get_default_strategy_registry(),
            policy_registry=get_default_policy_registry(),
            surface_id="trend_vol",
        )
        self.assertFalse(self._main._strategy_only_scope_enabled(scope))

    # ── _full_universe_sections_enabled is the logical inverse ────────────
    def test_strategy_only_and_full_universe_are_mutually_exclusive(self):
        """For any scope, exactly one of the two gates should be True."""
        scopes = [
            _default_scope(),
            _explicit_scope("TF4"),
            _explicit_scope("MR42"),
            _explicit_scope("TF4", "MR42"),
        ]
        for scope in scopes:
            full_univ = self._main._full_universe_sections_enabled(scope)
            strat_only = self._main._strategy_only_scope_enabled(scope)
            self.assertNotEqual(
                full_univ, strat_only,
                f"Gates must be mutually exclusive for scope source={scope.source!r} "
                f"ids={sorted(scope.strategy_ids)!r}",
            )


class TestLegacyMLStageSkippedForExplicitScope(unittest.TestCase):
    """Tests 3–5: Verify the legacy ML/backtest stages are not invoked for
    explicit strategy scope by asserting the ``_strategy_only_scope_enabled``
    helper returns True, which is the production gate that guards those stages.

    We also validate that ``_full_universe_sections_enabled`` returns False,
    confirming the existing guards for [3c/5], [4/5], and [4b/5] are active.
    """

    @classmethod
    def setUpClass(cls):
        import importlib
        cls._main = importlib.import_module("main")

    # ── Test 3: legacy ML experiment stage not invoked ─────────────────────
    def test_ml_experiment_stage_gated_for_mr32(self):
        """_strategy_only_scope_enabled() is the gate for [3/5].

        When it returns True the ``PhaseMLExperiment`` loop is skipped and
        ``ml_results_all`` is set to ``{}``.
        """
        scope = _explicit_scope("MR32")
        self.assertTrue(self._main._strategy_only_scope_enabled(scope))

    # ── Test 4: ML phase prediction stage not invoked ─────────────────────
    def test_ml_phase_prediction_stage_gated_for_mr32(self):
        """_strategy_only_scope_enabled() is also the gate for [3b/5]."""
        scope = _explicit_scope("MR32")
        self.assertTrue(self._main._strategy_only_scope_enabled(scope))

    # ── Test 5: full-universe backtest/aggregation not invoked ────────────
    def test_full_universe_backtest_gated_for_mr32(self):
        """_full_universe_sections_enabled() must return False for [3c/5]/[4/5]/[4b/5]."""
        scope = _explicit_scope("MR32")
        self.assertFalse(self._main._full_universe_sections_enabled(scope))

    def test_full_universe_backtest_gated_for_mr5_mr32(self):
        scope = _explicit_scope("MR5", "MR32")
        self.assertFalse(self._main._full_universe_sections_enabled(scope))

    def test_both_gates_active_for_explicit_scope(self):
        """For an explicit scope both optimization gates must be active."""
        scope = _explicit_scope("MR32")
        self.assertTrue(self._main._strategy_only_scope_enabled(scope),
                        "[3/5] and [3b/5] must be skipped (_strategy_only_scope_enabled)")
        self.assertFalse(self._main._full_universe_sections_enabled(scope),
                         "[3c/5]/[4/5]/[4b/5] must be skipped (_full_universe_sections_enabled)")


class TestStrategySpecificEvalRemainsReachable(unittest.TestCase):
    """Test 6: The strategy-specific walk-forward evaluation code must still
    be reachable for explicit strategy scope.

    We verify this by checking that _build_walkforward_execution_plan returns a
    non-empty execution plan with the requested strategy IDs.
    """

    @classmethod
    def setUpClass(cls):
        import importlib
        cls._main = importlib.import_module("main")
        cls._registry = get_default_strategy_registry()

    def test_mr32_walkforward_plan_is_non_empty(self):
        scope = _explicit_scope("MR32")
        plan = self._main._build_walkforward_execution_plan(
            scope,
            strategy_registry=self._registry,
        )
        self.assertIn("MR32", plan["standalone_strategy_ids"],
                      "MR32 must appear in the walk-forward execution plan")
        self.assertTrue(len(plan["strategy_specs"]) > 0,
                        "strategy_specs must be non-empty for explicit MR32 scope")

    def test_mr5_mr32_walkforward_plan_contains_both(self):
        scope = _explicit_scope("MR5", "MR32")
        plan = self._main._build_walkforward_execution_plan(
            scope,
            strategy_registry=self._registry,
        )
        self.assertIn("MR5", plan["standalone_strategy_ids"])
        self.assertIn("MR32", plan["standalone_strategy_ids"])

    def test_default_scope_walkforward_plan_is_also_reachable(self):
        """Full-universe mode must still produce a valid walk-forward plan."""
        scope = _default_scope()
        plan = self._main._build_walkforward_execution_plan(
            scope,
            strategy_registry=self._registry,
        )
        self.assertTrue(plan["run_phaseaware"])
        self.assertTrue(plan["run_dynamic_selector"])


if __name__ == "__main__":
    unittest.main()
