"""Regression tests: explicit evaluation scope must prevent full-universe execution.

These tests verify the requirements from problem statement section 1 & 4:

1.  Default (unscoped) invocation triggers full-universe legacy backtests.
2.  Explicit TF1 scope prevents execution of TF2-TF5, all MR strategies,
    and full-universe PhaseAware work.
3.  Explicit MR42 scope is also scope-based (not TF1-specific).
4.  The scope decision uses EvaluationScope.source, not strategy-specific
    conditionals.
5.  `set_diagnostics_verbose(False)` suppresses [AWARENESS],
    [TRAINING DIAGNOSTICS], and [DL FEATURE USAGE] output.
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation_scope import (  # noqa: E402
    EvaluationScope,
    compute_standalone_execution_flags,
    should_run_full_universe_backtests,
    resolve_evaluation_scope,
)
from src.strategy_registry import (  # noqa: E402
    get_default_policy_registry,
    get_default_strategy_registry,
    resolve_phaseaware_configuration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _explicit_scope(*strategy_ids: str) -> EvaluationScope:
    return EvaluationScope(strategy_ids=tuple(strategy_ids), source="explicit")


def _default_scope() -> EvaluationScope:
    return EvaluationScope(strategy_ids=("TF4", "MR42"), source="default")


# ---------------------------------------------------------------------------
# should_run_full_universe_backtests
# ---------------------------------------------------------------------------

class TestShouldRunFullUniverseBacktests(unittest.TestCase):
    """Tests for the scope gate that controls legacy full-universe backtests."""

    def test_default_scope_runs_full_universe(self):
        scope = _default_scope()
        self.assertTrue(should_run_full_universe_backtests(scope))

    def test_explicit_tf1_scope_skips_full_universe(self):
        scope = _explicit_scope("TF1")
        self.assertFalse(should_run_full_universe_backtests(scope))

    def test_explicit_mr42_scope_skips_full_universe(self):
        scope = _explicit_scope("MR42")
        self.assertFalse(should_run_full_universe_backtests(scope))

    def test_explicit_tf4_scope_skips_full_universe(self):
        scope = _explicit_scope("TF4")
        self.assertFalse(should_run_full_universe_backtests(scope))

    def test_explicit_multi_strategy_scope_skips_full_universe(self):
        scope = _explicit_scope("TF1", "MR42")
        self.assertFalse(should_run_full_universe_backtests(scope))

    def test_source_is_the_deciding_factor(self):
        # Two scopes with identical strategy_ids but different source.
        scope_default = EvaluationScope(strategy_ids=("TF1",), source="default")
        scope_explicit = EvaluationScope(strategy_ids=("TF1",), source="explicit")
        self.assertTrue(should_run_full_universe_backtests(scope_default))
        self.assertFalse(should_run_full_universe_backtests(scope_explicit))


# ---------------------------------------------------------------------------
# Resolved scope source for CLI equivalents
# ---------------------------------------------------------------------------

class TestResolvedScopeSource(unittest.TestCase):
    """Verify that resolve_evaluation_scope sets the expected source values."""

    def setUp(self):
        self.registry = get_default_strategy_registry()
        self.policy_registry = get_default_policy_registry()

    def test_no_strategy_arg_produces_default_source(self):
        scope = resolve_evaluation_scope(
            requested_strategy_ids=None,
            registry=self.registry,
            policy_registry=self.policy_registry,
            surface_id="trend_vol",
        )
        self.assertEqual(scope.source, "default")
        self.assertTrue(should_run_full_universe_backtests(scope))

    def test_empty_strategy_list_produces_default_source(self):
        scope = resolve_evaluation_scope(
            requested_strategy_ids=[],
            registry=self.registry,
            policy_registry=self.policy_registry,
            surface_id="trend_vol",
        )
        self.assertEqual(scope.source, "default")
        self.assertTrue(should_run_full_universe_backtests(scope))

    def test_explicit_tf1_produces_explicit_source(self):
        scope = resolve_evaluation_scope(
            requested_strategy_ids=["TF1"],
            registry=self.registry,
            policy_registry=self.policy_registry,
            surface_id="trend_vol",
        )
        self.assertEqual(scope.source, "explicit")
        self.assertFalse(should_run_full_universe_backtests(scope))

    def test_explicit_mr42_produces_explicit_source(self):
        scope = resolve_evaluation_scope(
            requested_strategy_ids=["MR42"],
            registry=self.registry,
            policy_registry=self.policy_registry,
            surface_id="trend_vol",
        )
        self.assertEqual(scope.source, "explicit")
        self.assertFalse(should_run_full_universe_backtests(scope))

    def test_explicit_tf1_scope_contains_only_tf1(self):
        scope = resolve_evaluation_scope(
            requested_strategy_ids=["TF1"],
            registry=self.registry,
            policy_registry=self.policy_registry,
            surface_id="trend_vol",
        )
        self.assertEqual(set(scope.strategy_ids), {"TF1"})

    def test_explicit_tf1_scope_excludes_other_tf_strategies(self):
        scope = resolve_evaluation_scope(
            requested_strategy_ids=["TF1"],
            registry=self.registry,
            policy_registry=self.policy_registry,
            surface_id="trend_vol",
        )
        for excluded in ("TF2", "TF3", "TF4", "TF5"):
            self.assertNotIn(excluded, scope.strategy_ids,
                             f"{excluded} must not be in TF1-only scope")

    def test_explicit_tf1_scope_excludes_all_mr_strategies(self):
        scope = resolve_evaluation_scope(
            requested_strategy_ids=["TF1"],
            registry=self.registry,
            policy_registry=self.policy_registry,
            surface_id="trend_vol",
        )
        for excluded in ("MR1", "MR2", "MR32", "MR42", "MR5"):
            self.assertNotIn(excluded, scope.strategy_ids,
                             f"{excluded} must not be in TF1-only scope")

    def test_explicit_mr42_scope_contains_only_mr42(self):
        scope = resolve_evaluation_scope(
            requested_strategy_ids=["MR42"],
            registry=self.registry,
            policy_registry=self.policy_registry,
            surface_id="trend_vol",
        )
        self.assertEqual(set(scope.strategy_ids), {"MR42"})

    def test_explicit_mr42_scope_excludes_all_tf_strategies(self):
        scope = resolve_evaluation_scope(
            requested_strategy_ids=["MR42"],
            registry=self.registry,
            policy_registry=self.policy_registry,
            surface_id="trend_vol",
        )
        for excluded in ("TF1", "TF2", "TF3", "TF4", "TF5"):
            self.assertNotIn(excluded, scope.strategy_ids,
                             f"{excluded} must not be in MR42-only scope")


# ---------------------------------------------------------------------------
# run_backtests is not called for explicit scope
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# compute_standalone_execution_flags for explicit TF1 / MR42
# ---------------------------------------------------------------------------

class TestStandaloneExecutionFlags(unittest.TestCase):
    """Verify that the walk-forward standalone flags are scope-aware."""

    def test_tf1_explicit_scope_runs_tf_not_mr(self):
        scope = _explicit_scope("TF1")
        run_tf, run_mr = compute_standalone_execution_flags(scope, "TF1", "MR42")
        self.assertTrue(run_tf)
        self.assertFalse(run_mr)

    def test_mr42_explicit_scope_runs_mr_not_tf(self):
        scope = _explicit_scope("MR42")
        run_tf, run_mr = compute_standalone_execution_flags(scope, "TF4", "MR42")
        self.assertFalse(run_tf)
        self.assertTrue(run_mr)

    def test_default_scope_runs_neither(self):
        scope = _default_scope()
        run_tf, run_mr = compute_standalone_execution_flags(scope, "TF4", "MR42")
        self.assertFalse(run_tf)
        self.assertFalse(run_mr)


# ---------------------------------------------------------------------------
# Diagnostics verbosity control
# ---------------------------------------------------------------------------

class TestDiagnosticsVerbosity(unittest.TestCase):
    """Verify that set_diagnostics_verbose(False) suppresses noisy output."""

    def setUp(self):
        # Import here to get the real module so we can reset state
        import src.models as _models
        self._models = _models
        # Always restore verbose=True after each test
        self._models.set_diagnostics_verbose(True)

    def tearDown(self):
        self._models.set_diagnostics_verbose(True)

    def test_set_diagnostics_verbose_false_suppresses_dl_feature_usage(self):
        self._models.set_diagnostics_verbose(False)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            self._models._print_dl_feature_usage("EURUSD", ["dl_close_ma_20"])
        self.assertEqual(captured.getvalue(), "")

    def test_set_diagnostics_verbose_true_emits_dl_feature_usage(self):
        self._models.set_diagnostics_verbose(True)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            self._models._print_dl_feature_usage("EURUSD", ["dl_close_ma_20"])
        self.assertIn("[DL FEATURE USAGE]", captured.getvalue())

    def test_set_diagnostics_verbose_false_suppresses_awareness(self):
        import pandas as pd
        self._models.set_diagnostics_verbose(False)
        X = pd.DataFrame({"a": [1, 2, 3]})
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            self._models.emit_awareness_diagnostics(
                X, missing_indicators_enabled=False
            )
        self.assertEqual(captured.getvalue(), "")

    def test_set_diagnostics_verbose_true_emits_awareness(self):
        import pandas as pd
        self._models.set_diagnostics_verbose(True)
        X = pd.DataFrame({"a": [1, 2, 3]})
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            self._models.emit_awareness_diagnostics(
                X, missing_indicators_enabled=False
            )
        self.assertIn("[AWARENESS]", captured.getvalue())

    def test_diagnostics_verbose_default_is_true(self):
        # Verify the module-level default is True by reloading the module in a
        # pristine state (before any set_diagnostics_verbose call).
        import importlib
        import src.models as _m
        # Save and reset to verify the original constant in the module source.
        # The simplest approach: temporarily remove the module from sys.modules
        # so we can reload it and inspect the untouched default.
        import sys
        module_name = _m.__name__
        saved_module = sys.modules.pop(module_name, None)
        try:
            fresh = importlib.import_module(module_name)
            self.assertTrue(fresh._DIAGNOSTICS_VERBOSE,
                            "Module-level _DIAGNOSTICS_VERBOSE default must be True")
        finally:
            # Restore the original module so other tests are not affected.
            if saved_module is not None:
                sys.modules[module_name] = saved_module
            elif module_name in sys.modules:
                del sys.modules[module_name]



# ---------------------------------------------------------------------------
# Orchestration-level regression: _full_universe_sections_enabled in main.py
# ---------------------------------------------------------------------------

class TestFullUniverseOrchestrationGate(unittest.TestCase):
    """Verify the production orchestration gate in main.py.

    These tests exercise ``main._full_universe_sections_enabled``, which is
    the actual production helper called by ``main()`` to establish
    ``_run_full_universe`` for sections 3c/4/4b.

    A test here will fail if someone:
    - removes ``_full_universe_sections_enabled`` from main.py, or
    - changes it to always return True, or
    - breaks the contract that explicit scope returns False.
    """

    @classmethod
    def setUpClass(cls):
        # Import the production orchestration function from main.py.
        # Using importlib keeps the import deferred so heavy module-level side
        # effects of main.py do not run at collection time.
        import importlib
        cls._main = importlib.import_module("main")

    def test_explicit_tf1_disables_sections(self):
        scope = _explicit_scope("TF1")
        self.assertFalse(
            self._main._full_universe_sections_enabled(scope),
            "_full_universe_sections_enabled must return False for explicit TF1 scope",
        )

    def test_default_scope_enables_sections(self):
        scope = _default_scope()
        self.assertTrue(
            self._main._full_universe_sections_enabled(scope),
            "_full_universe_sections_enabled must return True for default scope",
        )

    def test_explicit_scope_for_every_registered_strategy_disables_sections(self):
        registry = get_default_strategy_registry()
        for strategy_id in registry.available():
            scope = resolve_evaluation_scope(
                requested_strategy_ids=[strategy_id],
                registry=registry,
                policy_registry=get_default_policy_registry(),
                surface_id="trend_vol",
            )
            self.assertFalse(
                self._main._full_universe_sections_enabled(scope),
                f"Explicit scope for {strategy_id} must disable full-universe sections",
            )


# ---------------------------------------------------------------------------
# Production walk-forward execution plan
# ---------------------------------------------------------------------------

class TestWalkforwardExecutionPlan(unittest.TestCase):
    """Verify that walk-forward execution is driven by the effective scope."""

    @classmethod
    def setUpClass(cls):
        import importlib
        cls._main = importlib.import_module("main")
        cls._registry = get_default_strategy_registry()

    def test_explicit_tf1_runs_only_tf1(self):
        plan = self._main._build_walkforward_execution_plan(
            _explicit_scope("TF1"),
            strategy_registry=self._registry,
        )
        self.assertEqual(plan["standalone_strategy_ids"], ("TF1",))
        self.assertFalse(plan["run_phaseaware"])
        self.assertFalse(plan["run_dynamic_selector"])
        self.assertEqual(
            [spec["strategy_id"] for spec in plan["strategy_specs"]],
            ["TF1"],
        )

    def test_explicit_mr1_runs_only_mr1(self):
        plan = self._main._build_walkforward_execution_plan(
            _explicit_scope("MR1"),
            strategy_registry=self._registry,
        )
        self.assertEqual(plan["standalone_strategy_ids"], ("MR1",))
        self.assertFalse(plan["run_phaseaware"])
        self.assertFalse(plan["run_dynamic_selector"])
        self.assertEqual(
            [spec["strategy_id"] for spec in plan["strategy_specs"]],
            ["MR1"],
        )

    def test_explicit_tf1_mr1_runs_both_without_composites(self):
        plan = self._main._build_walkforward_execution_plan(
            _explicit_scope("TF1", "MR1"),
            strategy_registry=self._registry,
        )
        self.assertEqual(plan["standalone_strategy_ids"], ("TF1", "MR1"))
        self.assertFalse(plan["run_phaseaware"])
        self.assertFalse(plan["run_dynamic_selector"])
        self.assertEqual(
            {spec["strategy_id"] for spec in plan["strategy_specs"]},
            {"TF1", "MR1"},
        )

    def test_explicit_tf4_alone_excludes_composites(self):
        plan = self._main._build_walkforward_execution_plan(
            _explicit_scope("TF4"),
            strategy_registry=self._registry,
        )
        self.assertEqual(plan["standalone_strategy_ids"], ("TF4",))
        self.assertFalse(plan["run_phaseaware"])
        self.assertFalse(plan["run_dynamic_selector"])

    def test_explicit_mr42_alone_excludes_composites(self):
        plan = self._main._build_walkforward_execution_plan(
            _explicit_scope("MR42"),
            strategy_registry=self._registry,
        )
        self.assertEqual(plan["standalone_strategy_ids"], ("MR42",))
        self.assertFalse(plan["run_phaseaware"])
        self.assertFalse(plan["run_dynamic_selector"])

    def test_explicit_tf4_mr42_includes_canonical_composites(self):
        plan = self._main._build_walkforward_execution_plan(
            _explicit_scope("TF4", "MR42"),
            strategy_registry=self._registry,
        )
        self.assertEqual(plan["standalone_strategy_ids"], ("TF4", "MR42"))
        self.assertTrue(plan["run_phaseaware"])
        self.assertTrue(plan["run_dynamic_selector"])
        self.assertEqual(
            {spec["strategy_id"] for spec in plan["strategy_specs"]},
            {"TF4", "MR42", "PhaseAware_TF4_MR42", "StrategySelector_Dynamic_WF"},
        )

    def test_default_scope_preserves_two_spec_benchmark(self):
        plan = self._main._build_walkforward_execution_plan(
            _default_scope(),
            strategy_registry=self._registry,
        )
        self.assertEqual(plan["standalone_strategy_ids"], ())
        self.assertTrue(plan["run_phaseaware"])
        self.assertTrue(plan["run_dynamic_selector"])
        self.assertEqual(
            [spec["strategy_id"] for spec in plan["strategy_specs"]],
            ["PhaseAware_TF4_MR42", "StrategySelector_Dynamic_WF"],
        )

    def test_default_scope_uses_experiment_local_phaseaware_configuration(self):
        composition = resolve_phaseaware_configuration(
            phaseaware_tf_strategy_id="TF2",
            phaseaware_mr_strategy_id="MR2",
        )
        plan = self._main._build_walkforward_execution_plan(
            _default_scope(),
            strategy_registry=self._registry,
            phaseaware_configuration=composition,
        )
        self.assertEqual(plan["standalone_strategy_ids"], ())
        self.assertTrue(plan["run_phaseaware"])
        self.assertTrue(plan["run_dynamic_selector"])
        self.assertEqual(
            [spec["strategy_id"] for spec in plan["strategy_specs"]],
            ["PhaseAware_TF2_MR2", "StrategySelector_Dynamic_WF"],
        )

    def test_explicit_scope_uses_effective_phaseaware_pair_for_composite_gate(self):
        composition = resolve_phaseaware_configuration(
            phaseaware_tf_strategy_id="TF2",
            phaseaware_mr_strategy_id="MR2",
        )
        plan = self._main._build_walkforward_execution_plan(
            _explicit_scope("TF2", "MR2"),
            strategy_registry=self._registry,
            phaseaware_configuration=composition,
        )
        self.assertEqual(plan["standalone_strategy_ids"], ("TF2", "MR2"))
        self.assertTrue(plan["run_phaseaware"])
        self.assertTrue(plan["run_dynamic_selector"])
        self.assertEqual(
            {spec["strategy_id"] for spec in plan["strategy_specs"]},
            {"TF2", "MR2", "PhaseAware_TF2_MR2", "StrategySelector_Dynamic_WF"},
        )

    def test_phaseaware_override_does_not_redefine_explicit_strategy_scope(self):
        composition = resolve_phaseaware_configuration(
            phaseaware_tf_strategy_id="TF2",
            phaseaware_mr_strategy_id="MR2",
        )
        plan = self._main._build_walkforward_execution_plan(
            _explicit_scope("TF4", "MR42"),
            strategy_registry=self._registry,
            phaseaware_configuration=composition,
        )
        self.assertEqual(plan["standalone_strategy_ids"], ("TF4", "MR42"))
        self.assertFalse(plan["run_phaseaware"])
        self.assertFalse(plan["run_dynamic_selector"])
        self.assertEqual(
            {spec["strategy_id"] for spec in plan["strategy_specs"]},
            {"TF4", "MR42"},
        )


class TestWalkforwardAggregation(unittest.TestCase):
    """Verify scope-aware walk-forward aggregation uses the executed strategy specs."""

    @classmethod
    def setUpClass(cls):
        import importlib
        cls._main = importlib.import_module("main")
        cls._registry = get_default_strategy_registry()

    def _aggregate(self, scope: EvaluationScope, rows: list[dict]):
        plan = self._main._build_walkforward_execution_plan(
            scope,
            strategy_registry=self._registry,
        )
        return self._main._aggregate_walkforward_results(
            pd.DataFrame(rows),
            strategy_specs=list(plan["strategy_specs"]),
        )

    def test_explicit_tf1_aggregates_without_composite_delta_columns(self):
        wf_pair, overall = self._aggregate(
            _explicit_scope("TF1"),
            [
                {"Pair": "EURUSD", "Fold": 1, "TF1 Return (%)": 1.0, "TF1 Sharpe": 0.20, "TF1 Max DD (%)": -4.0, "TF1 Trades": 5},
                {"Pair": "EURUSD", "Fold": 2, "TF1 Return (%)": 3.0, "TF1 Sharpe": 0.60, "TF1 Max DD (%)": -6.0, "TF1 Trades": 7},
                {"Pair": "USDJPY", "Fold": 1, "TF1 Return (%)": 2.0, "TF1 Sharpe": 0.10, "TF1 Max DD (%)": -3.0, "TF1 Trades": 4},
            ],
        )
        self.assertEqual(
            list(wf_pair.columns),
            ["Pair", "TF1 Return (%)", "TF1 Sharpe", "TF1 Max DD (%)", "TF1 Trades", "Folds"],
        )
        eurusd = wf_pair.loc[wf_pair["Pair"] == "EURUSD"].iloc[0]
        self.assertAlmostEqual(eurusd["TF1 Return (%)"], 2.0)
        self.assertAlmostEqual(eurusd["TF1 Sharpe"], 0.4)
        self.assertAlmostEqual(eurusd["TF1 Max DD (%)"], -5.0)
        self.assertAlmostEqual(eurusd["TF1 Trades"], 6.0)
        self.assertEqual(int(eurusd["Folds"]), 2)
        self.assertAlmostEqual(overall["Avg TF1 Return (%)"], 2.0)
        self.assertAlmostEqual(overall["Avg TF1 Sharpe"], 0.3)
        self.assertAlmostEqual(overall["Avg TF1 Max DD (%)"], -13.0 / 3.0)
        self.assertAlmostEqual(overall["Avg TF1 Trades"], 16.0 / 3.0)
        self.assertNotIn("Avg Return Δ", overall)

    def test_explicit_mr1_aggregates_without_composite_delta_columns(self):
        wf_pair, overall = self._aggregate(
            _explicit_scope("MR1"),
            [
                {"Pair": "EURUSD", "Fold": 1, "MR1 Return (%)": -1.0, "MR1 Sharpe": -0.20, "MR1 Max DD (%)": -2.0, "MR1 Trades": 8},
                {"Pair": "EURUSD", "Fold": 2, "MR1 Return (%)": 4.0, "MR1 Sharpe": 0.50, "MR1 Max DD (%)": -5.0, "MR1 Trades": 6},
            ],
        )
        self.assertEqual(
            list(wf_pair.columns),
            ["Pair", "MR1 Return (%)", "MR1 Sharpe", "MR1 Max DD (%)", "MR1 Trades", "Folds"],
        )
        eurusd = wf_pair.iloc[0]
        self.assertAlmostEqual(eurusd["MR1 Return (%)"], 1.5)
        self.assertAlmostEqual(eurusd["MR1 Sharpe"], 0.15)
        self.assertAlmostEqual(eurusd["MR1 Max DD (%)"], -3.5)
        self.assertAlmostEqual(eurusd["MR1 Trades"], 7.0)
        self.assertEqual(int(eurusd["Folds"]), 2)
        self.assertAlmostEqual(overall["Avg MR1 Return (%)"], 1.5)
        self.assertAlmostEqual(overall["Avg MR1 Sharpe"], 0.15)
        self.assertAlmostEqual(overall["Avg MR1 Max DD (%)"], -3.5)
        self.assertAlmostEqual(overall["Avg MR1 Trades"], 7.0)
        self.assertNotIn("Avg Return Δ", overall)

    def test_default_scope_preserves_composite_delta_aggregation(self):
        wf_pair, overall = self._aggregate(
            _default_scope(),
            [
                {"Pair": "EURUSD", "Fold": 1, "Return Δ": 2.0, "Sharpe Δ": 0.10, "DD Δ": -1.0},
                {"Pair": "EURUSD", "Fold": 2, "Return Δ": 4.0, "Sharpe Δ": -0.10, "DD Δ": -3.0},
            ],
        )
        self.assertEqual(list(wf_pair.columns), ["Pair", "Return Δ", "Sharpe Δ", "DD Δ", "Folds"])
        self.assertAlmostEqual(wf_pair.iloc[0]["Return Δ"], 3.0)
        self.assertAlmostEqual(wf_pair.iloc[0]["Sharpe Δ"], 0.0)
        self.assertAlmostEqual(wf_pair.iloc[0]["DD Δ"], -2.0)
        self.assertEqual(int(wf_pair.iloc[0]["Folds"]), 2)
        self.assertEqual(
            set(overall.keys()),
            {"Pairs", "Folds", "Avg Return Δ", "Avg Sharpe Δ", "Avg Max DD Δ", "Folds Sharpe Improved"},
        )
        self.assertAlmostEqual(overall["Avg Return Δ"], 3.0)
        self.assertAlmostEqual(overall["Avg Sharpe Δ"], 0.0)
        self.assertAlmostEqual(overall["Avg Max DD Δ"], -2.0)
        self.assertEqual(overall["Folds Sharpe Improved"], 1)

    def test_explicit_tf4_mr42_preserves_composite_delta_aggregation(self):
        wf_pair, overall = self._aggregate(
            _explicit_scope("TF4", "MR42"),
            [
                {"Pair": "EURUSD", "Fold": 1, "Return Δ": 1.0, "Sharpe Δ": 0.20, "DD Δ": -0.5},
                {"Pair": "USDJPY", "Fold": 1, "Return Δ": -2.0, "Sharpe Δ": -0.10, "DD Δ": -1.5},
            ],
        )
        self.assertEqual(list(wf_pair.columns), ["Pair", "Return Δ", "Sharpe Δ", "DD Δ", "Folds"])
        self.assertIn("Avg Return Δ", overall)
        self.assertNotIn("Avg TF4 Return (%)", overall)


# ---------------------------------------------------------------------------
# Debug flag reversibility
# ---------------------------------------------------------------------------

class TestDebugFlagReversibility(unittest.TestCase):
    """Verify that main(debug=False) restores quiet state after main(debug=True).

    This ensures no persistent global debug state leaks between invocations
    in the same Python process.  All tests call the production
    ``_configure_debug`` helper rather than duplicating its assignments.
    """

    @classmethod
    def setUpClass(cls):
        import importlib
        cls._main = importlib.import_module("main")
        cls._models = importlib.import_module("src.models")

    def _read_flags(self):
        m = self._main
        return {
            "DL_DEBUG_VERBOSE": m.DL_DEBUG_VERBOSE,
            "DEBUG_BASELINE_KEYS": m.DEBUG_BASELINE_KEYS,
            "DEBUG_FEATURE_COLUMNS": m.DEBUG_FEATURE_COLUMNS,
            "DEBUG_SIGNAL_TYPES": m.DEBUG_SIGNAL_TYPES,
            "DEBUG_VOL_GUARD": m.DEBUG_VOL_GUARD,
        }

    def tearDown(self):
        # Restore quiet defaults after each test using the production helper.
        self._main._configure_debug(False)

    def test_debug_true_enables_all_flags(self):
        self._main._configure_debug(True)
        flags = self._read_flags()
        for name, value in flags.items():
            self.assertTrue(value, f"{name} must be True when debug=True")

    def test_debug_false_disables_all_flags(self):
        # First enable, then disable — verifies reversibility.
        self._main._configure_debug(True)
        self._main._configure_debug(False)
        flags = self._read_flags()
        for name, value in flags.items():
            self.assertFalse(value, f"{name} must be False when debug=False after debug=True")

    def test_debug_false_after_true_suppresses_diagnostics(self):
        self._main._configure_debug(True)
        self._main._configure_debug(False)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            self._models._print_dl_feature_usage("EURUSD", ["dl_close_ma_20"])
        self.assertEqual(
            captured.getvalue(), "",
            "Diagnostics must be suppressed after debug=False restores quiet state",
        )

    def test_debug_true_is_reversible_by_debug_false(self):
        """Simulate main(debug=True) followed by main(debug=False) in same process."""
        # First call: debug=True
        self._main._configure_debug(True)
        self.assertTrue(self._main.DL_DEBUG_VERBOSE)
        self.assertTrue(self._models._DIAGNOSTICS_VERBOSE)

        # Second call: debug=False — must reset everything
        self._main._configure_debug(False)
        self.assertFalse(self._main.DL_DEBUG_VERBOSE)
        self.assertFalse(self._models._DIAGNOSTICS_VERBOSE)


class TestWalkforwardDebugOutputGates(unittest.TestCase):
    """Verify `[WALKFORWARD DL]` and `[PHASE WF WINDOW]` use the shared debug gate."""

    @classmethod
    def setUpClass(cls):
        import importlib
        cls._models = importlib.import_module("src.models")

    def _make_df(self):
        import numpy as np
        import pandas as pd

        rows = 12
        idx = pd.date_range("2024-01-01", periods=rows, freq="D")
        phases = ["HV_Ranging", "HV_Trend", "LV_Ranging", "LV_Trend"] * 3
        return pd.DataFrame(
            {
                "Close": np.linspace(1.0, 2.1, rows),
                "feature_a": np.linspace(10.0, 21.0, rows),
                "dl_demo_feature": np.linspace(0.1, 1.2, rows),
                "phase": phases[:rows],
            },
            index=idx,
        )

    def _run_predictor(self, verbose: bool) -> str:
        class _DummyModel:
            def fit(self, X, y):
                return self

            def predict(self, X):
                import numpy as np
                return np.zeros(len(X), dtype=int)

        self._models.set_diagnostics_verbose(verbose)
        saved_dl_signals_enabled = self._models.DL_SIGNALS_ENABLED
        self._models.DL_SIGNALS_ENABLED = True
        predictor = self._models.PhaseMLPredictor(
            train_window=4,
            retrain_freq=1,
            smooth_labels=False,
            seed=7,
            missing_indicators_enabled=False,
        )
        predictor._build_model = lambda: _DummyModel()
        captured = io.StringIO()
        try:
            with patch("sys.stdout", captured):
                predictor.fit_predict(self._make_df())
        finally:
            self._models.DL_SIGNALS_ENABLED = saved_dl_signals_enabled
        return captured.getvalue()

    def tearDown(self):
        self._models.set_diagnostics_verbose(True)

    def test_walkforward_dl_suppressed_when_debug_false(self):
        output = self._run_predictor(False)
        self.assertNotIn("[WALKFORWARD DL]", output)

    def test_walkforward_dl_emitted_when_debug_true(self):
        output = self._run_predictor(True)
        self.assertIn("[WALKFORWARD DL]", output)

    def test_phase_wf_window_suppressed_when_debug_false(self):
        output = self._run_predictor(False)
        self.assertNotIn("[PHASE WF WINDOW]", output)

    def test_phase_wf_window_emitted_when_debug_true(self):
        output = self._run_predictor(True)
        self.assertIn("[PHASE WF WINDOW]", output)


if __name__ == "__main__":
    unittest.main()
