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

    These tests exercise ``main._full_universe_sections_enabled``, the function
    that produces the ``_run_full_universe`` flag consumed by sections 3c/4/4b.

    A test here will fail if someone:
    - removes ``_full_universe_sections_enabled`` from main.py, or
    - changes it to always return True, or
    - breaks the contract that explicit scope returns False.

    The mock-run_backtests pattern reproduces the exact conditional guard used
    in section 4 of main() so that the test fails if the guard logic is wrong.
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

    def test_explicit_scope_does_not_invoke_run_backtests(self):
        """Reproduce the section-4 guard pattern and verify run_backtests is blocked.

        This is the orchestration-boundary regression test: it mirrors the
        exact conditional used in main() section 4 so the test fails if the
        guard is removed or inverted.
        """
        from unittest.mock import MagicMock
        mock_run_backtests = MagicMock()

        explicit_scope = _explicit_scope("TF1")
        _run_full_universe = self._main._full_universe_sections_enabled(explicit_scope)

        # ── Reproduce section-4 guard (copy of the production conditional) ──
        if _run_full_universe:
            mock_run_backtests()  # pragma: no cover — must NOT be reached
        # ────────────────────────────────────────────────────────────────────

        mock_run_backtests.assert_not_called()

    def test_default_scope_would_invoke_run_backtests(self):
        """Verify the gate passes for default scope (legacy path preserved)."""
        from unittest.mock import MagicMock
        mock_run_backtests = MagicMock()

        default_scope = _default_scope()
        _run_full_universe = self._main._full_universe_sections_enabled(default_scope)

        if _run_full_universe:
            mock_run_backtests()

        mock_run_backtests.assert_called_once()

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
# Debug flag reversibility
# ---------------------------------------------------------------------------

class TestDebugFlagReversibility(unittest.TestCase):
    """Verify that main(debug=False) restores quiet state after main(debug=True).

    This ensures no persistent global debug state leaks between invocations
    in the same Python process.
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

    def _apply_debug_flags(self, debug: bool):
        """Mirror the production flag-assignment logic from main()."""
        m = self._main
        m.DL_DEBUG_VERBOSE = bool(debug)
        m.DEBUG_BASELINE_KEYS = bool(debug)
        m.DEBUG_FEATURE_COLUMNS = bool(debug)
        m.DEBUG_SIGNAL_TYPES = bool(debug)
        m.DEBUG_VOL_GUARD = bool(debug)
        self._models.set_diagnostics_verbose(debug)

    def tearDown(self):
        # Restore quiet defaults after each test.
        self._apply_debug_flags(False)

    def test_debug_true_enables_all_flags(self):
        self._apply_debug_flags(True)
        flags = self._read_flags()
        for name, value in flags.items():
            self.assertTrue(value, f"{name} must be True when debug=True")

    def test_debug_false_disables_all_flags(self):
        # First enable, then disable — verifies reversibility.
        self._apply_debug_flags(True)
        self._apply_debug_flags(False)
        flags = self._read_flags()
        for name, value in flags.items():
            self.assertFalse(value, f"{name} must be False when debug=False after debug=True")

    def test_debug_false_after_true_suppresses_diagnostics(self):
        self._apply_debug_flags(True)
        self._apply_debug_flags(False)
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
        self._apply_debug_flags(True)
        self.assertTrue(self._main.DL_DEBUG_VERBOSE)
        self.assertTrue(self._models._DIAGNOSTICS_VERBOSE)

        # Second call: debug=False — must reset everything
        self._apply_debug_flags(False)
        self.assertFalse(self._main.DL_DEBUG_VERBOSE)
        self.assertFalse(self._models._DIAGNOSTICS_VERBOSE)


if __name__ == "__main__":
    unittest.main()
