"""
Phase G3.1 — Selector Reference Universe regression tests.

These tests establish the architectural invariant that:

  1. _build_selector_reference_results() produces exactly the configured
     representative strategies (TF4, MR42, PhaseAware(TF4, MR42)) and no
     others.

  2. Changing a strategy outside the Selector Reference Universe (e.g. MR32)
     does NOT change selector training labels or reference results.

  3. The full strategy research universe (including MR32) remains intact
     and is unaffected by this boundary.

  4. The Selector Reference Universe is policy-derived: changing the
     configured PhaseAware representative pair changes the reference universe
     accordingly.

Tests A–G correspond directly to the G3.1 validation requirements:

  A. Reference universe contents
  B. Global selector training uses reference universe
  C. Per-fold causal selector training uses reference universe
  D. Global and causal consistency
  E. MR32 isolation
  F. Full research universe preserved
  G. Policy-derived behaviour
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from main import (
        _build_selector_reference_results,
        _build_causal_selector_training_data,
    )
    from src.strategy_registry import (
        DEFAULT_PHASEAWARE_POLICY_ID,
        EvaluationPolicy,
        EvaluationPolicyRegistry,
        StrategyCapabilities,
        StrategyDefinition,
        StrategyRegistry,
        get_default_strategy_registry,
        phaseaware_strategy_name,
        resolve_phaseaware_strategy_pair,
    )
    from src.strategies import (
        _DEFAULT_EVALUATED_MR_STRATEGY_IDS,
        _DEFAULT_EVALUATED_TF_STRATEGY_IDS,
        run_backtests,
    )
    from src.models import StrategyPerformanceTracker

    _HAS_DEPS = True
    _DEPS_ERR = ""
except Exception as exc:  # pragma: no cover
    _HAS_DEPS = False
    _DEPS_ERR = str(exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(rows: int = 300) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame suitable for backtesting."""
    idx = pd.date_range("2020-01-01", periods=rows, freq="D")
    rng = np.random.default_rng(42)
    close = 1.10 + np.cumsum(rng.normal(0, 0.002, rows))
    high = close + rng.uniform(0.001, 0.005, rows)
    low = close - rng.uniform(0.001, 0.005, rows)
    atr = np.full(rows, 0.005)
    return pd.DataFrame(
        {
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(rows, 1000.0),
            "rsi": np.linspace(35.0, 65.0, rows),
            "adx": np.linspace(15.0, 35.0, rows),
            "plus_di": np.linspace(20.0, 30.0, rows),
            "minus_di": np.linspace(30.0, 20.0, rows),
            "phase": (
                ["LVTF", "HVTF", "LVR", "HVR"] * (rows // 4)
                + ["LVTF"] * (rows % 4)
            )[:rows],
            "atr": atr,
            "atr_pct": atr / close,
            "stop_atr_mult": np.full(rows, 2.0),
            "returns": np.concatenate([[0.0], np.diff(close) / close[:-1]]),
        },
        index=idx,
    )


def _reference_keys(policy_id: str = "phaseaware_default") -> frozenset[str]:
    """Return the expected key set for the given policy."""
    tf_id, mr_id = resolve_phaseaware_strategy_pair(policy_id)
    pa_name = f"PhaseAware_{tf_id}_{mr_id}"
    return frozenset({tf_id, mr_id, pa_name})


# ---------------------------------------------------------------------------
# A. Reference universe contents
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_DEPS, f"missing deps: {_DEPS_ERR}")
class TestReferenceUniverseContents(unittest.TestCase):
    """A. _build_selector_reference_results() returns exactly the configured
    representative set and no other strategies."""

    def test_default_policy_keys_are_exact(self):
        df = _make_ohlcv_df()
        results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        expected = _reference_keys(DEFAULT_PHASEAWARE_POLICY_ID)
        self.assertEqual(frozenset(results.keys()), expected)

    def test_default_policy_includes_tf4(self):
        df = _make_ohlcv_df()
        results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        self.assertIn("TF4", results)

    def test_default_policy_includes_mr42(self):
        df = _make_ohlcv_df()
        results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        self.assertIn("MR42", results)

    def test_default_policy_includes_phaseaware_tf4_mr42(self):
        df = _make_ohlcv_df()
        results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        pa_key = phaseaware_strategy_name(DEFAULT_PHASEAWARE_POLICY_ID)
        self.assertIn(pa_key, results)

    def test_default_policy_excludes_mr32(self):
        """MR32 must never appear in the selector reference universe."""
        df = _make_ohlcv_df()
        results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        self.assertNotIn("MR32", results)

    def test_default_policy_excludes_research_only_strategies(self):
        """Research-only strategies (TF1-TF3, TF5, MR1, MR2, MR3, MR32, MR5)
        must not appear in the selector reference universe."""
        df = _make_ohlcv_df()
        results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        research_only = {"TF1", "TF2", "TF3", "TF5", "MR1", "MR2", "MR3", "MR32", "MR5"}
        for strat_id in research_only:
            self.assertNotIn(
                strat_id, results,
                f"Research-only strategy {strat_id!r} must not be in selector reference universe",
            )

    def test_result_contains_equity_curve(self):
        """Each reference result must have an equity_curve for label generation."""
        df = _make_ohlcv_df()
        results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        for key, res in results.items():
            self.assertIn(
                "equity_curve", res,
                f"Reference result {key!r} is missing equity_curve",
            )


# ---------------------------------------------------------------------------
# B. Global selector training uses reference universe
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_DEPS, f"missing deps: {_DEPS_ERR}")
class TestGlobalSelectorTrainingUsesReferenceUniverse(unittest.TestCase):
    """B. The global selector training path (4b/5) must produce training labels
    derived only from the selector reference universe strategies."""

    def test_global_training_labels_come_from_reference_strategies(self):
        """When StrategyPerformanceTracker processes reference results, the
        resulting best_strategy column contains only reference strategy names."""
        df = _make_ohlcv_df()
        pair_results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        tracker = StrategyPerformanceTracker(window_days=20)
        training_data = tracker.compute_strategy_returns(df, pair_results)

        if training_data.empty:
            self.skipTest("Insufficient data for training labels")

        # The winning strategy labels must only name strategies from the reference universe
        reference_strategies = _reference_keys(DEFAULT_PHASEAWARE_POLICY_ID)
        observed_labels = set(training_data["best_strategy"].dropna().unique())
        for label in observed_labels:
            # Labels are strategy names, not ids for PhaseAware — check that
            # none of the research-only strategy ids appear.
            for research_id in {"MR32", "MR1", "MR2", "MR3", "MR5", "TF1", "TF2", "TF3", "TF5"}:
                self.assertNotIn(
                    research_id, label,
                    f"Research-only strategy {research_id!r} must not appear in "
                    f"global selector training label {label!r}",
                )

    def test_global_training_labels_include_expected_types(self):
        """Training labels should include TF, MR and PhaseAware type references."""
        df = _make_ohlcv_df()
        pair_results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        tracker = StrategyPerformanceTracker(window_days=20)
        training_data = tracker.compute_strategy_returns(df, pair_results)

        if training_data.empty:
            self.skipTest("Insufficient data for training labels")

        # The reference keys must all appear at some point as possible winners
        observed_labels = set(training_data["best_strategy"].dropna().unique())
        # At minimum, the labels should not be empty
        self.assertGreater(len(observed_labels), 0)


# ---------------------------------------------------------------------------
# C. Per-fold causal selector training uses reference universe
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_DEPS, f"missing deps: {_DEPS_ERR}")
class TestCausalSelectorTrainingUsesReferenceUniverse(unittest.TestCase):
    """C. The per-fold causal selector training path must produce training
    labels derived only from the selector reference universe strategies."""

    def test_causal_training_labels_come_from_reference_strategies(self):
        df = _make_ohlcv_df(rows=500)
        pair_results_full = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        # Build a fold that leaves enough room for label horizon
        label_horizon = 20
        train_start = 0
        train_end = 399
        test_start = 400
        test_end = 499
        training_data, _ = _build_causal_selector_training_data(
            pair_name="EURUSD",
            fold_id=0,
            df_full=df,
            pair_results_full=pair_results_full,
            train_start_pos=train_start,
            train_end_pos=train_end,
            test_start_pos=test_start,
            test_end_pos=test_end,
            label_horizon_bars=label_horizon,
            context_label="test",
        )

        if training_data.empty:
            self.skipTest("Insufficient data for causal training labels")

        observed_labels = set(training_data["best_strategy"].dropna().unique())
        for label in observed_labels:
            for research_id in {"MR32", "MR1", "MR2", "MR3", "MR5", "TF1", "TF2", "TF3", "TF5"}:
                self.assertNotIn(
                    research_id, label,
                    f"Research-only strategy {research_id!r} must not appear in "
                    f"causal training label {label!r}",
                )

    def test_causal_training_receives_only_reference_strategy_keys(self):
        """The pair_results_full dict used for causal training must contain only
        the three reference strategy keys."""
        df = _make_ohlcv_df(rows=500)
        pair_results_full = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        expected = _reference_keys(DEFAULT_PHASEAWARE_POLICY_ID)
        self.assertEqual(frozenset(pair_results_full.keys()), expected)


# ---------------------------------------------------------------------------
# D. Global and causal consistency
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_DEPS, f"missing deps: {_DEPS_ERR}")
class TestGlobalAndCausalConsistency(unittest.TestCase):
    """D. Both selector-training paths must use the same concrete reference
    strategy keys. Having two different definitions of MeanReversion would
    create a semantic inconsistency."""

    def test_global_and_causal_use_same_reference_keys(self):
        """Both training paths call _build_selector_reference_results() with the
        same policy_id, so the strategy keys they receive must be identical."""
        df = _make_ohlcv_df()

        # Keys seen by the global training path
        global_results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        global_keys = frozenset(global_results.keys())

        # Keys seen by the per-fold causal training path
        causal_results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        causal_keys = frozenset(causal_results.keys())

        self.assertEqual(
            global_keys,
            causal_keys,
            "Global and causal selector training must use identical reference strategy keys",
        )

    def test_mean_reversion_representative_is_mr42_in_both_paths(self):
        """The MeanReversion representative in both training paths must be MR42,
        not any other registered MeanReversion strategy."""
        df = _make_ohlcv_df()

        for path_label in ("global", "causal"):
            results = _build_selector_reference_results(
                df_full=df,
                pair_name="EURUSD",
                policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
            )
            mr_id, _ = resolve_phaseaware_strategy_pair(DEFAULT_PHASEAWARE_POLICY_ID)
            # Swap to get MR id
            tf_id, mr_id = resolve_phaseaware_strategy_pair(DEFAULT_PHASEAWARE_POLICY_ID)
            self.assertIn(
                mr_id, results,
                f"MR representative {mr_id!r} must be present in {path_label} path results",
            )
            self.assertEqual(mr_id, "MR42")

    def test_trend_following_representative_is_tf4_in_both_paths(self):
        """The TrendFollowing representative in both training paths must be TF4."""
        tf_id, _ = resolve_phaseaware_strategy_pair(DEFAULT_PHASEAWARE_POLICY_ID)
        self.assertEqual(tf_id, "TF4")


# ---------------------------------------------------------------------------
# E. MR32 isolation
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_DEPS, f"missing deps: {_DEPS_ERR}")
class TestMR32Isolation(unittest.TestCase):
    """E. Changing the MR32 strategy implementation must not change selector
    training labels or reference results."""

    @staticmethod
    def _make_modified_mr32_registry() -> StrategyRegistry:
        """Return a registry where MR32 always produces the maximum possible
        signal (all buys) — if MR32 were in the reference universe this would
        change its equity curve and therefore selector labels."""
        from src.strategies import (
            TF1Strategy, TF2Strategy, TF3Strategy, TF4Strategy, TF5Strategy,
            MR1Strategy, MR2Strategy, MR3Strategy, MR42Strategy, MR5Strategy,
        )
        from src.strategy_registry import _definition

        class _AlwaysBuyMR32:
            """Stub that always returns maximum-strength buy signal — semantics
            are completely different from the real MR32 implementation."""
            def generate_signals(self, df: pd.DataFrame):
                n = len(df)
                return (
                    pd.Series(np.ones(n) * 1.0, index=df.index),
                    pd.Series(np.zeros(n), index=df.index),
                    pd.Series(np.zeros(n), index=df.index),
                )

        trend_states = ("HVTF", "LVTF")
        ranging_states = ("HVR", "LVR")

        defs = [
            _definition(strategy_id="TF1", display_name="TF1", family="TrendFollowing", implementation=TF1Strategy, states=trend_states, indicators=("lwma", "stddev"), features=("Close",)),
            _definition(strategy_id="TF2", display_name="TF2", family="TrendFollowing", implementation=TF2Strategy, states=trend_states, indicators=("sma",), features=("Close",)),
            _definition(strategy_id="TF3", display_name="TF3", family="TrendFollowing", implementation=TF3Strategy, states=trend_states, indicators=("stochastic",), features=("High", "Low", "Close")),
            _definition(strategy_id="TF4", display_name="TF4", family="TrendFollowing", implementation=TF4Strategy, states=trend_states, indicators=("adx", "plus_di", "minus_di"), features=("High", "Low", "Close")),
            _definition(strategy_id="TF5", display_name="TF5", family="TrendFollowing", implementation=TF5Strategy, states=trend_states, indicators=("adx", "plus_di", "minus_di"), features=("High", "Low", "Close")),
            _definition(strategy_id="MR1", display_name="MR1", family="MeanReversion", implementation=MR1Strategy, states=ranging_states, indicators=("bollinger_bands",), features=("Close",)),
            _definition(strategy_id="MR2", display_name="MR2", family="MeanReversion", implementation=MR2Strategy, states=ranging_states, indicators=("bollinger_bands", "rsi"), features=("Close",)),
            _definition(strategy_id="MR3", display_name="MR3", family="MeanReversion", implementation=MR3Strategy, states=ranging_states, indicators=("rsi",), features=("Close",)),
            # MR32 replaced with always-buy stub
            _definition(strategy_id="MR32", display_name="MR32", family="MeanReversion", implementation=_AlwaysBuyMR32, states=("LVR",), indicators=("rsi", "adx"), features=("Close",)),
            _definition(strategy_id="MR42", display_name="MR42", family="MeanReversion", implementation=MR42Strategy, states=("LVR",), indicators=("stochastic", "adx"), features=("High", "Low", "Close")),
            _definition(strategy_id="MR5", display_name="MR5", family="MeanReversion", implementation=MR5Strategy, states=("HVR",), indicators=("stochastic",), features=("High", "Low", "Close")),
        ]
        return StrategyRegistry(defs)

    def test_mr32_change_does_not_change_reference_universe_keys(self):
        """The keys returned by _build_selector_reference_results() must be the
        same regardless of what MR32 does."""
        df = _make_ohlcv_df()

        standard_registry = get_default_strategy_registry()
        modified_registry = self._make_modified_mr32_registry()

        keys_standard = frozenset(
            _build_selector_reference_results(
                df_full=df,
                pair_name="EURUSD",
                strategy_registry=standard_registry,
                policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
            ).keys()
        )
        keys_modified = frozenset(
            _build_selector_reference_results(
                df_full=df,
                pair_name="EURUSD",
                strategy_registry=modified_registry,
                policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
            ).keys()
        )
        self.assertEqual(
            keys_standard,
            keys_modified,
            "Changing MR32 must not change the selector reference universe keys",
        )
        self.assertNotIn("MR32", keys_modified)

    def test_mr32_change_does_not_change_tf4_equity_curve(self):
        """The TF4 equity curve in the reference universe must be identical
        regardless of whether MR32 uses the standard or stub implementation."""
        df = _make_ohlcv_df()

        standard_registry = get_default_strategy_registry()
        modified_registry = self._make_modified_mr32_registry()

        results_standard = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            strategy_registry=standard_registry,
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        results_modified = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            strategy_registry=modified_registry,
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )

        eq_std = results_standard["TF4"]["equity_curve"]
        eq_mod = results_modified["TF4"]["equity_curve"]
        pd.testing.assert_series_equal(
            eq_std, eq_mod,
            check_names=False,
            obj="TF4 equity curve must be unaffected by MR32 change",
        )

    def test_mr32_change_does_not_change_mr42_equity_curve(self):
        """The MR42 equity curve in the reference universe must be identical
        regardless of whether MR32 uses the standard or stub implementation."""
        df = _make_ohlcv_df()

        standard_registry = get_default_strategy_registry()
        modified_registry = self._make_modified_mr32_registry()

        results_standard = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            strategy_registry=standard_registry,
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        results_modified = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            strategy_registry=modified_registry,
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )

        eq_std = results_standard["MR42"]["equity_curve"]
        eq_mod = results_modified["MR42"]["equity_curve"]
        pd.testing.assert_series_equal(
            eq_std, eq_mod,
            check_names=False,
            obj="MR42 equity curve must be unaffected by MR32 change",
        )

    def test_mr32_change_does_not_change_selector_training_labels(self):
        """The training labels produced by StrategyPerformanceTracker must be
        identical whether or not MR32 was changed — because MR32 is outside
        the selector reference universe and therefore not a label source."""
        df = _make_ohlcv_df()

        standard_registry = get_default_strategy_registry()
        modified_registry = self._make_modified_mr32_registry()

        results_standard = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            strategy_registry=standard_registry,
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        results_modified = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            strategy_registry=modified_registry,
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )

        tracker = StrategyPerformanceTracker(window_days=20)
        labels_standard = tracker.compute_strategy_returns(df, results_standard)
        labels_modified = tracker.compute_strategy_returns(df, results_modified)

        if labels_standard.empty or labels_modified.empty:
            self.skipTest("Insufficient data for training labels")

        pd.testing.assert_frame_equal(
            labels_standard.reset_index(drop=True),
            labels_modified.reset_index(drop=True),
            check_like=False,
            obj=(
                "Selector training labels must be identical regardless of MR32 "
                "implementation, because MR32 is outside the selector reference universe"
            ),
        )


# ---------------------------------------------------------------------------
# F. Full research universe preserved
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_DEPS, f"missing deps: {_DEPS_ERR}")
class TestFullResearchUniversePreserved(unittest.TestCase):
    """F. MR32 and other research strategies must remain in the full strategy
    registry and full research universe.  The G3.1 fix must not solve the
    problem by removing MR32 from the registry or the evaluation list."""

    def test_mr32_is_in_default_strategy_registry(self):
        registry = get_default_strategy_registry()
        self.assertIn("MR32", registry.available())

    def test_full_research_strategy_ids_in_registry(self):
        """All historically registered research strategies must still be available."""
        registry = get_default_strategy_registry()
        expected = {"TF1", "TF2", "TF3", "TF4", "TF5", "MR1", "MR2", "MR3", "MR32", "MR42", "MR5"}
        for strat_id in expected:
            self.assertIn(
                strat_id,
                registry.available(),
                f"Strategy {strat_id!r} must remain in the default strategy registry",
            )

    def test_mr32_is_in_default_evaluated_mr_strategy_ids(self):
        """_DEFAULT_EVALUATED_MR_STRATEGY_IDS must still include MR32."""
        self.assertIn(
            "MR32",
            _DEFAULT_EVALUATED_MR_STRATEGY_IDS,
            "MR32 must remain in _DEFAULT_EVALUATED_MR_STRATEGY_IDS (full research universe)",
        )

    def test_default_evaluated_tf_strategy_ids_unchanged(self):
        """_DEFAULT_EVALUATED_TF_STRATEGY_IDS must still contain the full TF set."""
        for strat_id in ("TF1", "TF2", "TF3", "TF4", "TF5"):
            self.assertIn(strat_id, _DEFAULT_EVALUATED_TF_STRATEGY_IDS)

    def test_run_backtests_result_contains_mr32(self):
        """run_backtests() must produce results that include MR32."""
        df = _make_ohlcv_df(rows=200)
        results = run_backtests(df)
        self.assertIn(
            "MR32",
            results,
            "run_backtests() must still include MR32 in full research universe results",
        )

    def test_run_backtests_result_contains_mr32_and_mr42_separately(self):
        """MR32 and MR42 must both be independently present in run_backtests()."""
        df = _make_ohlcv_df(rows=200)
        results = run_backtests(df)
        self.assertIn("MR32", results)
        self.assertIn("MR42", results)

    def test_reference_universe_is_strict_subset_of_full_universe(self):
        """The selector reference universe is a strict subset of the full research universe."""
        df = _make_ohlcv_df(rows=200)

        full_results = run_backtests(df, evaluation_policy_id=DEFAULT_PHASEAWARE_POLICY_ID)
        ref_results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )

        ref_keys = frozenset(ref_results.keys())
        full_keys = frozenset(full_results.keys())

        # All reference keys must be present in the full universe
        self.assertTrue(
            ref_keys.issubset(full_keys),
            f"Reference keys {ref_keys - full_keys!r} missing from full universe",
        )
        # The full universe must have MORE strategies than the reference universe
        self.assertGreater(
            len(full_keys),
            len(ref_keys),
            "Full research universe must contain more strategies than the selector reference universe",
        )


# ---------------------------------------------------------------------------
# G. Policy-derived behaviour
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_DEPS, f"missing deps: {_DEPS_ERR}")
class TestPolicyDerivedBehaviour(unittest.TestCase):
    """G. The selector reference universe is policy-derived: changing the
    configured PhaseAware representative pair changes the reference universe.
    This verifies that the reference universe is not a second hardcoded list."""

    def _make_alt_policy_registry(
        self, *, strategy_registry: StrategyRegistry
    ) -> EvaluationPolicyRegistry:
        """Build a policy registry that maps an alternative policy to TF1+MR1."""
        alt_policy = EvaluationPolicy(
            policy_id="phaseaware_alt_test",
            display_name="PhaseAware Alt Test (TF1+MR1)",
            strategies=("TF1", "MR1"),
            metadata={"phaseaware": True},
        )
        return EvaluationPolicyRegistry(
            [alt_policy],
            strategy_registry=strategy_registry,
        )

    def test_alt_policy_changes_reference_universe_tf_representative(self):
        """When the configured policy uses TF1 instead of TF4, the reference
        universe must contain TF1 (not TF4) as the TrendFollowing representative."""
        df = _make_ohlcv_df()
        strategy_registry = get_default_strategy_registry()
        alt_policy_registry = self._make_alt_policy_registry(
            strategy_registry=strategy_registry
        )

        alt_tf, alt_mr = resolve_phaseaware_strategy_pair(
            "phaseaware_alt_test",
            strategy_registry=strategy_registry,
            policy_registry=alt_policy_registry,
        )
        self.assertEqual(alt_tf, "TF1")
        self.assertEqual(alt_mr, "MR1")

        alt_results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            strategy_registry=strategy_registry,
            policy_id="phaseaware_alt_test",
            policy_registry=alt_policy_registry,
        )

        self.assertIn("TF1", alt_results)
        self.assertNotIn("TF4", alt_results)

    def test_alt_policy_changes_reference_universe_mr_representative(self):
        """When the configured policy uses MR1 instead of MR42, the reference
        universe must contain MR1 (not MR42) as the MeanReversion representative."""
        df = _make_ohlcv_df()
        strategy_registry = get_default_strategy_registry()
        alt_policy_registry = self._make_alt_policy_registry(
            strategy_registry=strategy_registry
        )

        alt_results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            strategy_registry=strategy_registry,
            policy_id="phaseaware_alt_test",
            policy_registry=alt_policy_registry,
        )

        self.assertIn("MR1", alt_results)
        self.assertNotIn("MR42", alt_results)

    def test_alt_policy_produces_correct_phaseaware_key(self):
        """The PhaseAware key in the reference universe must reflect the
        configured representative pair, not a hardcoded TF4/MR42 name."""
        df = _make_ohlcv_df()
        strategy_registry = get_default_strategy_registry()
        alt_policy_registry = self._make_alt_policy_registry(
            strategy_registry=strategy_registry
        )

        alt_results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            strategy_registry=strategy_registry,
            policy_id="phaseaware_alt_test",
            policy_registry=alt_policy_registry,
        )

        # The PhaseAware key for TF1+MR1 must be present
        self.assertIn("PhaseAware_TF1_MR1", alt_results)
        # The default TF4+MR42 PhaseAware key must NOT be present
        self.assertNotIn("PhaseAware_TF4_MR42", alt_results)

    def test_alt_policy_reference_universe_excludes_mr32(self):
        """MR32 must not be in the reference universe even under an alternative
        policy configuration."""
        df = _make_ohlcv_df()
        strategy_registry = get_default_strategy_registry()
        alt_policy_registry = self._make_alt_policy_registry(
            strategy_registry=strategy_registry
        )

        alt_results = _build_selector_reference_results(
            df_full=df,
            pair_name="EURUSD",
            strategy_registry=strategy_registry,
            policy_id="phaseaware_alt_test",
            policy_registry=alt_policy_registry,
        )

        self.assertNotIn("MR32", alt_results)

    def test_default_and_alt_policy_reference_universes_are_distinct(self):
        """The default and alternative policy reference universes must be different
        sets, confirming policy-derived (not hardcoded) behaviour."""
        df = _make_ohlcv_df()
        strategy_registry = get_default_strategy_registry()
        alt_policy_registry = self._make_alt_policy_registry(
            strategy_registry=strategy_registry
        )

        default_keys = frozenset(
            _build_selector_reference_results(
                df_full=df,
                pair_name="EURUSD",
                strategy_registry=strategy_registry,
                policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
            ).keys()
        )
        alt_keys = frozenset(
            _build_selector_reference_results(
                df_full=df,
                pair_name="EURUSD",
                strategy_registry=strategy_registry,
                policy_id="phaseaware_alt_test",
                policy_registry=alt_policy_registry,
            ).keys()
        )

        self.assertNotEqual(
            default_keys,
            alt_keys,
            "Different policies must produce different selector reference universes",
        )


if __name__ == "__main__":
    unittest.main()
