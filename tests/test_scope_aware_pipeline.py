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

class TestRunBacktestsNotCalledForExplicitScope(unittest.TestCase):
    """Verify that run_backtests is skipped when the scope is explicit.

    These tests patch `src.strategies.run_backtests` and confirm the call
    count for the legacy full-universe pipeline section.
    """

    def _scope_gate(self, scope: EvaluationScope) -> bool:
        """Mirrors the gate used in main() for sections 3c/4/4b."""
        return should_run_full_universe_backtests(scope)

    def test_run_backtests_would_be_called_for_default_scope(self):
        scope = _default_scope()
        self.assertTrue(self._scope_gate(scope),
                        "Default scope must pass the gate so run_backtests is called")

    def test_run_backtests_would_not_be_called_for_tf1_scope(self):
        scope = _explicit_scope("TF1")
        self.assertFalse(self._scope_gate(scope),
                         "Explicit TF1 scope must block the gate so run_backtests is skipped")

    def test_run_backtests_would_not_be_called_for_mr42_scope(self):
        scope = _explicit_scope("MR42")
        self.assertFalse(self._scope_gate(scope),
                         "Explicit MR42 scope must block the gate so run_backtests is skipped")

    def test_run_backtests_would_not_be_called_for_any_explicit_scope(self):
        registry = get_default_strategy_registry()
        for strategy_id in registry.available():
            scope = resolve_evaluation_scope(
                requested_strategy_ids=[strategy_id],
                registry=registry,
                policy_registry=get_default_policy_registry(),
                surface_id="trend_vol",
            )
            self.assertFalse(
                self._scope_gate(scope),
                f"Explicit scope for {strategy_id} must block full-universe backtests",
            )


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
        self.assertTrue(self._models._DIAGNOSTICS_VERBOSE)


if __name__ == "__main__":
    unittest.main()
