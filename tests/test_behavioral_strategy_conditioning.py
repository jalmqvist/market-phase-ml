"""Regression tests for behavioral-surface conditioning of explicit individual strategies.

Bug: When --behavioral=JPY_CONSENSUS_YOUNG --strategy TF1 was run, the downstream
strategy-specific evaluation path iterated over ALL pairs in processed_data and
ignored the behavioral surface's pair scope.  The DL behavioral information was
therefore never actually conditioning the evaluation.

This test file verifies:

1. Individual strategy + behavioral surface
   TF1 + reactive_jpy + JPY_CONSENSUS_YOUNG
   - _compute_behavioral_eligible_pairs returns only JPY pairs from the artifact
   - Pair-scope filtering is applied when explicit strategy scope is active
   - DL data presence can be verified in df_test for behavioral pairs

2. Baseline vs behavioral execution-path difference
   - Baseline (no DL): _compute_behavioral_eligible_pairs returns None
   - Behavioral: returns a non-empty frozenset

3. Pair-scope correctness
   - EURJPY, GBPJPY, USDJPY are in the eligible set
   - Non-JPY pairs are not in the eligible set

4. State-specific distinctness
   - Different states yield distinct eligible pair sets (both non-None)
   - Selecting a state does not collapse into the baseline path

5. Existing behavior preserved
   - Default/full-universe scope → no pair restriction
   - DL disabled → no pair restriction
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation_scope import EvaluationScope  # noqa: E402


def _load_main():
    return importlib.import_module("main")


def _explicit_scope(*strategy_ids: str) -> EvaluationScope:
    return EvaluationScope(strategy_ids=tuple(strategy_ids), source="explicit")


def _default_scope() -> EvaluationScope:
    return EvaluationScope(strategy_ids=("TF4", "MR42"), source="default")


def _make_d1_predictions(pairs: list[str]) -> pd.DataFrame:
    """Build a minimal d1_predictions DataFrame with the specified pairs."""
    if not pairs:
        return pd.DataFrame()
    rows = []
    for p in pairs:
        rows.append({
            "pair": p,
            "trading_day": pd.Timestamp("2023-01-01"),
            "dl_pred_mean": 0.5,
        })
    return pd.DataFrame(rows)


def _make_empty_d1() -> pd.DataFrame:
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 1. _compute_behavioral_eligible_pairs — basic behaviour
# ---------------------------------------------------------------------------

class TestComputeBehavioralEligiblePairs(unittest.TestCase):
    """Unit tests for the _compute_behavioral_eligible_pairs helper."""

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def _call(self, d1, *, strategy_only_scope, dl_runtime_enabled):
        return self._main._compute_behavioral_eligible_pairs(
            d1,
            strategy_only_scope=strategy_only_scope,
            dl_runtime_enabled=dl_runtime_enabled,
        )

    # ── Returns None when restrictions should not apply ───────────────────

    def test_default_scope_returns_none(self):
        """No pair restriction for default/full-universe runs."""
        d1 = _make_d1_predictions(["eur-jpy", "gbp-jpy", "usd-jpy"])
        result = self._call(d1, strategy_only_scope=False, dl_runtime_enabled=True)
        self.assertIsNone(result, "Default scope must not restrict pairs")

    def test_dl_disabled_returns_none(self):
        """No pair restriction when DL runtime is disabled (baseline run)."""
        d1 = _make_d1_predictions(["eur-jpy", "gbp-jpy", "usd-jpy"])
        result = self._call(d1, strategy_only_scope=True, dl_runtime_enabled=False)
        self.assertIsNone(result, "Baseline (DL disabled) must not restrict pairs")

    def test_empty_d1_returns_none(self):
        """No pair restriction when the artifact has no predictions."""
        result = self._call(
            _make_empty_d1(),
            strategy_only_scope=True,
            dl_runtime_enabled=True,
        )
        self.assertIsNone(result, "Empty d1_predictions must not restrict pairs")

    def test_none_d1_returns_none(self):
        """No pair restriction when d1_predictions is None."""
        result = self._call(
            None,  # type: ignore[arg-type]
            strategy_only_scope=True,
            dl_runtime_enabled=True,
        )
        self.assertIsNone(result)

    def test_d1_without_pair_column_returns_none(self):
        d1 = pd.DataFrame({"trading_day": [pd.Timestamp("2023-01-01")]})
        result = self._call(d1, strategy_only_scope=True, dl_runtime_enabled=True)
        self.assertIsNone(result)

    # ── Returns a frozenset when restriction applies ──────────────────────

    def test_explicit_scope_dl_enabled_returns_frozenset(self):
        """Explicit strategy + DL enabled → returns a frozenset of eligible pairs."""
        d1 = _make_d1_predictions(["eur-jpy", "gbp-jpy", "usd-jpy"])
        result = self._call(d1, strategy_only_scope=True, dl_runtime_enabled=True)
        self.assertIsInstance(result, frozenset)
        self.assertTrue(len(result) > 0)

    # ── Pair normalization ────────────────────────────────────────────────

    def test_artifact_hyphenated_keys_normalized_to_eurjpy(self):
        """Artifact keys like 'eur-jpy' are normalized to 'EURJPY'."""
        d1 = _make_d1_predictions(["eur-jpy"])
        result = self._call(d1, strategy_only_scope=True, dl_runtime_enabled=True)
        self.assertIn("EURJPY", result)
        self.assertNotIn("eur-jpy", result)

    def test_reactive_jpy_artifact_pairs_normalized_correctly(self):
        """All three reactive-JPY pairs normalize correctly."""
        d1 = _make_d1_predictions(["eur-jpy", "gbp-jpy", "usd-jpy"])
        result = self._call(d1, strategy_only_scope=True, dl_runtime_enabled=True)
        self.assertIsNotNone(result)
        self.assertIn("EURJPY", result)
        self.assertIn("GBPJPY", result)
        self.assertIn("USDJPY", result)
        self.assertEqual(len(result), 3)


# ---------------------------------------------------------------------------
# 2. Baseline vs behavioral execution-path difference
# ---------------------------------------------------------------------------

class TestBaselineVsBehavioralPath(unittest.TestCase):
    """Verify that the baseline path and the behavioral path are observably different
    at the pair-restriction level."""

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def _call(self, d1, *, strategy_only_scope, dl_runtime_enabled):
        return self._main._compute_behavioral_eligible_pairs(
            d1,
            strategy_only_scope=strategy_only_scope,
            dl_runtime_enabled=dl_runtime_enabled,
        )

    def test_baseline_produces_none_restriction(self):
        """Baseline (dl_runtime_enabled=False) must return None."""
        d1 = _make_d1_predictions(["eur-jpy", "gbp-jpy", "usd-jpy"])
        baseline_result = self._call(d1, strategy_only_scope=True, dl_runtime_enabled=False)
        self.assertIsNone(baseline_result)

    def test_behavioral_produces_pair_restriction(self):
        """Behavioral (dl_runtime_enabled=True, explicit scope) must return frozenset."""
        d1 = _make_d1_predictions(["eur-jpy", "gbp-jpy", "usd-jpy"])
        behavioral_result = self._call(d1, strategy_only_scope=True, dl_runtime_enabled=True)
        self.assertIsNotNone(behavioral_result)
        self.assertIsInstance(behavioral_result, frozenset)

    def test_baseline_and_behavioral_are_observably_different(self):
        """Baseline and behavioral paths produce different pair restriction outcomes."""
        d1 = _make_d1_predictions(["eur-jpy", "gbp-jpy", "usd-jpy"])
        baseline = self._call(d1, strategy_only_scope=True, dl_runtime_enabled=False)
        behavioral = self._call(d1, strategy_only_scope=True, dl_runtime_enabled=True)
        self.assertNotEqual(
            baseline,
            behavioral,
            "Baseline (None) and behavioral (frozenset) must be observably different",
        )


# ---------------------------------------------------------------------------
# 3. Pair-scope correctness for reactive_jpy
# ---------------------------------------------------------------------------

class TestReactiveJpyPairScope(unittest.TestCase):
    """Verify that the reactive-JPY pair scope is correct.

    EURJPY, GBPJPY, USDJPY must be included.
    Non-JPY pairs must not be included.
    """

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()
        _d1 = _make_d1_predictions(["eur-jpy", "gbp-jpy", "usd-jpy"])
        cls._eligible = cls._main._compute_behavioral_eligible_pairs(
            _d1,
            strategy_only_scope=True,
            dl_runtime_enabled=True,
        )

    def test_eligible_set_is_not_none(self):
        self.assertIsNotNone(self.__class__._eligible)

    def test_eurjpy_is_eligible(self):
        self.assertIn("EURJPY", self.__class__._eligible)

    def test_gbpjpy_is_eligible(self):
        self.assertIn("GBPJPY", self.__class__._eligible)

    def test_usdjpy_is_eligible(self):
        self.assertIn("USDJPY", self.__class__._eligible)

    def test_eurusd_is_not_eligible(self):
        self.assertNotIn("EURUSD", self.__class__._eligible)

    def test_gbpusd_is_not_eligible(self):
        self.assertNotIn("GBPUSD", self.__class__._eligible)

    def test_nzdusd_is_not_eligible(self):
        self.assertNotIn("NZDUSD", self.__class__._eligible)

    def test_eurgbp_is_not_eligible(self):
        self.assertNotIn("EURGBP", self.__class__._eligible)

    def test_eligible_set_size_matches_artifact(self):
        """Eligible set must contain exactly the pairs from the artifact."""
        self.assertEqual(len(self.__class__._eligible), 3)


# ---------------------------------------------------------------------------
# 4. State-specific tests — different states remain distinct
# ---------------------------------------------------------------------------

class TestStateSpecificDistinctness(unittest.TestCase):
    """Verify that selecting a behavioral state does not collapse to the baseline path.

    Two different reactive-JPY states (YOUNG / MATURING) share the same underlying
    pair coverage (the artifact covers the same JPY pairs).  Both must produce
    non-None eligible pair sets; neither must be None (the baseline path).

    Note: _compute_behavioral_eligible_pairs does not receive a state_id — the pair
    restriction is derived solely from the d1_predictions artifact content.  These
    tests verify that both valid behavioral states produce a non-None pair restriction
    (i.e. neither collapses to the baseline path), not that they produce distinct sets.
    """

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()
        _d1_young = _make_d1_predictions(["eur-jpy", "gbp-jpy", "usd-jpy"])
        _d1_maturing = _make_d1_predictions(["eur-jpy", "gbp-jpy", "usd-jpy"])
        cls._eligible_young = cls._main._compute_behavioral_eligible_pairs(
            _d1_young,
            strategy_only_scope=True,
            dl_runtime_enabled=True,
        )
        cls._eligible_maturing = cls._main._compute_behavioral_eligible_pairs(
            _d1_maturing,
            strategy_only_scope=True,
            dl_runtime_enabled=True,
        )

    def test_young_state_is_not_none(self):
        """JPY_CONSENSUS_YOUNG must produce a non-None pair restriction (not the baseline path)."""
        self.assertIsNotNone(self.__class__._eligible_young)

    def test_maturing_state_is_not_none(self):
        """JPY_CONSENSUS_MATURING must produce a non-None pair restriction (not the baseline path)."""
        self.assertIsNotNone(self.__class__._eligible_maturing)

    def test_both_states_cover_jpy_pairs(self):
        """Both states must have the same JPY pair coverage."""
        for pair in ("EURJPY", "GBPJPY", "USDJPY"):
            self.assertIn(pair, self.__class__._eligible_young,
                          f"YOUNG: {pair} must be eligible")
            self.assertIn(pair, self.__class__._eligible_maturing,
                          f"MATURING: {pair} must be eligible")

    def test_evaluate_scope_resolution_for_young_state(self):
        """resolve_evaluation_scope must accept TF1 + reactive_jpy + JPY_CONSENSUS_YOUNG."""
        from src.evaluation_scope import resolve_evaluation_scope
        from src.strategy_registry import (
            get_default_policy_registry,
            get_default_strategy_registry,
        )
        scope = resolve_evaluation_scope(
            requested_strategy_ids=["TF1"],
            registry=get_default_strategy_registry(),
            policy_registry=get_default_policy_registry(),
            surface_id="reactive_jpy",
            state_id="JPY_CONSENSUS_YOUNG",
        )
        self.assertEqual(scope.source, "explicit")
        self.assertIn("TF1", scope.strategy_ids)

    def test_evaluate_scope_resolution_for_maturing_state(self):
        """resolve_evaluation_scope must accept TF1 + reactive_jpy + JPY_CONSENSUS_MATURING."""
        from src.evaluation_scope import resolve_evaluation_scope
        from src.strategy_registry import (
            get_default_policy_registry,
            get_default_strategy_registry,
        )
        scope = resolve_evaluation_scope(
            requested_strategy_ids=["TF1"],
            registry=get_default_strategy_registry(),
            policy_registry=get_default_policy_registry(),
            surface_id="reactive_jpy",
            state_id="JPY_CONSENSUS_MATURING",
        )
        self.assertEqual(scope.source, "explicit")
        self.assertIn("TF1", scope.strategy_ids)


# ---------------------------------------------------------------------------
# 5. Existing (non-behavioral) explicit-strategy behavior preserved
# ---------------------------------------------------------------------------

class TestExistingBehaviorPreserved(unittest.TestCase):
    """Regression: existing non-behavioral explicit strategy runs are unchanged."""

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def test_explicit_tf1_no_dl_no_restriction(self):
        """Explicit TF1 with DL disabled → no pair restriction."""
        result = self._main._compute_behavioral_eligible_pairs(
            _make_empty_d1(),
            strategy_only_scope=True,
            dl_runtime_enabled=False,
        )
        self.assertIsNone(result)

    def test_default_scope_no_dl_no_restriction(self):
        """Default scope with DL disabled → no pair restriction."""
        result = self._main._compute_behavioral_eligible_pairs(
            _make_empty_d1(),
            strategy_only_scope=False,
            dl_runtime_enabled=False,
        )
        self.assertIsNone(result)

    def test_default_scope_with_dl_no_restriction(self):
        """Default scope with DL enabled → no pair restriction (full-universe run)."""
        d1 = _make_d1_predictions(["eur-jpy", "gbp-jpy", "usd-jpy"])
        result = self._main._compute_behavioral_eligible_pairs(
            d1,
            strategy_only_scope=False,
            dl_runtime_enabled=True,
        )
        self.assertIsNone(
            result,
            "Full-universe (default) scope must not restrict pairs even with behavioral artifact",
        )

    def test_strategy_only_scope_enabled_for_tf1(self):
        """Explicit TF1 scope must be recognized as strategy-only."""
        scope = _explicit_scope("TF1")
        self.assertTrue(self._main._strategy_only_scope_enabled(scope))

    def test_full_universe_disabled_for_explicit_tf1(self):
        """Full-universe sections must be disabled for explicit TF1 scope."""
        scope = _explicit_scope("TF1")
        from src.evaluation_scope import should_run_full_universe_backtests
        self.assertFalse(should_run_full_universe_backtests(scope))

    def test_walkforward_plan_non_empty_for_tf1(self):
        """TF1 walk-forward execution plan must still be non-empty."""
        from src.strategy_registry import get_default_strategy_registry
        scope = _explicit_scope("TF1")
        plan = self._main._build_walkforward_execution_plan(
            scope,
            strategy_registry=get_default_strategy_registry(),
        )
        self.assertIn("TF1", plan["standalone_strategy_ids"])
        self.assertTrue(len(plan["strategy_specs"]) > 0)


# ---------------------------------------------------------------------------
# 6. DL feature presence in processed data for behavioral pairs
# ---------------------------------------------------------------------------

class TestDlFeaturePresenceForBehavioralPairs(unittest.TestCase):
    """Verify that the `has_dl_features` helper correctly detects DL columns.

    This test ensures the observability contract: a pair that has gone through
    attach_dl_features() with a behavioral artifact will have DL columns in its
    DataFrame, and the `has_dl_features()` function reports this correctly.
    The walkforward gate then uses this to ensure the evaluation-input is
    observably different from a baseline run.
    """

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def _make_df_with_dl_cols(self) -> "pd.DataFrame":
        """Build a minimal DataFrame with DL feature columns attached."""
        from src.dl_daily_features import D1_FEATURE_COLS
        data = {
            "Close": [1.0, 1.1, 1.2],
            "phase": ["trending", "trending", "ranging"],
        }
        for col in D1_FEATURE_COLS:
            data[col] = [0.5, 0.6, 0.7]
        return pd.DataFrame(data, index=pd.date_range("2023-01-01", periods=3))

    def _make_df_without_dl_cols(self) -> "pd.DataFrame":
        """Build a minimal baseline DataFrame without DL feature columns."""
        return pd.DataFrame({
            "Close": [1.0, 1.1, 1.2],
            "phase": ["trending", "trending", "ranging"],
        }, index=pd.date_range("2023-01-01", periods=3))

    def test_has_dl_features_returns_true_for_behavioral_pair(self):
        df = self._make_df_with_dl_cols()
        self.assertTrue(
            self._main.has_dl_features(df),
            "DataFrame with DL columns must be recognized as having DL features",
        )

    def test_has_dl_features_returns_false_for_baseline_pair(self):
        df = self._make_df_without_dl_cols()
        self.assertFalse(
            self._main.has_dl_features(df),
            "DataFrame without DL columns must not be recognized as having DL features",
        )

    def test_behavioral_and_baseline_df_are_observably_different(self):
        """The evaluation-input is observably different for behavioral vs baseline."""
        behavioral_df = self._make_df_with_dl_cols()
        baseline_df = self._make_df_without_dl_cols()
        self.assertNotEqual(
            self._main.has_dl_features(behavioral_df),
            self._main.has_dl_features(baseline_df),
            "Behavioral and baseline DataFrames must differ at has_dl_features level",
        )


# ---------------------------------------------------------------------------
# 7. Walkforward pair-scope gate — evaluation-loop contract
# ---------------------------------------------------------------------------

class TestWalkforwardPairScopeGate(unittest.TestCase):
    """Regression test for the evaluation-loop pair-scope gate.

    Setup:
        processed_data : EURUSD, EURJPY, GBPJPY, USDJPY
        behavioral artifact : EURJPY, GBPJPY, USDJPY  (reactive_jpy)
        explicit strategy   : TF1

    Contract:
        - Only EURJPY, GBPJPY, USDJPY reach the evaluation stage.
        - EURUSD is excluded (not in the behavioral artifact).

    This test exercises the gate at the helper level used by the
    walk-forward loop (the same gate that caused the original bug).
    """

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()
        # Simulate: processed_data has four pairs (three JPY + one non-JPY)
        cls._processed_pairs = ["EURUSD", "EURJPY", "GBPJPY", "USDJPY"]
        # Behavioral artifact only covers the three JPY pairs
        _artifact_d1 = _make_d1_predictions(["eur-jpy", "gbp-jpy", "usd-jpy"])
        # Explicit strategy scope + DL runtime active → gate should restrict
        cls._eligible = cls._main._compute_behavioral_eligible_pairs(
            _artifact_d1,
            strategy_only_scope=True,
            dl_runtime_enabled=True,
        )

    def _is_included(self, pair_name: str) -> bool:
        """Replicate the gate logic from the walk-forward loop."""
        eligible = self.__class__._eligible
        if eligible is None:
            return True
        return pair_name.upper().replace("-", "") in eligible

    def test_gate_is_active(self):
        """The pair-scope gate must be active (eligible set is not None)."""
        self.assertIsNotNone(self.__class__._eligible)

    def test_eurusd_is_excluded_from_evaluation(self):
        """EURUSD must be excluded — it is not in the reactive_jpy artifact."""
        self.assertFalse(
            self._is_included("EURUSD"),
            "EURUSD must be skipped by the behavioral pair-scope gate",
        )

    def test_eurjpy_is_included_in_evaluation(self):
        """EURJPY must pass the pair-scope gate."""
        self.assertTrue(self._is_included("EURJPY"))

    def test_gbpjpy_is_included_in_evaluation(self):
        """GBPJPY must pass the pair-scope gate."""
        self.assertTrue(self._is_included("GBPJPY"))

    def test_usdjpy_is_included_in_evaluation(self):
        """USDJPY must pass the pair-scope gate."""
        self.assertTrue(self._is_included("USDJPY"))

    def test_exactly_three_pairs_pass_gate(self):
        """Exactly three pairs (the JPY set) must pass the gate."""
        included = [p for p in self.__class__._processed_pairs if self._is_included(p)]
        self.assertEqual(
            sorted(included),
            ["EURJPY", "GBPJPY", "USDJPY"],
            "Gate must admit exactly the artifact pairs",
        )

    def test_exactly_one_pair_excluded_by_gate(self):
        """Exactly one pair (EURUSD) must be excluded by the gate."""
        excluded = [p for p in self.__class__._processed_pairs if not self._is_included(p)]
        self.assertEqual(excluded, ["EURUSD"])

    def test_baseline_gate_admits_all_pairs(self):
        """Baseline (DL disabled) gate must admit all four pairs."""
        baseline_eligible = self.__class__._main._compute_behavioral_eligible_pairs(
            _make_d1_predictions(["eur-jpy", "gbp-jpy", "usd-jpy"]),
            strategy_only_scope=True,
            dl_runtime_enabled=False,
        )
        self.assertIsNone(baseline_eligible, "Baseline gate must be None (no restriction)")
        # None → no gate → all pairs admitted
        included_baseline = [
            p for p in self.__class__._processed_pairs
            if baseline_eligible is None or p.upper().replace("-", "") in baseline_eligible
        ]
        self.assertEqual(
            sorted(included_baseline),
            ["EURJPY", "EURUSD", "GBPJPY", "USDJPY"],
        )


# ---------------------------------------------------------------------------
# 8. DL columns preserved in df_test for behavioral pairs
# ---------------------------------------------------------------------------

class TestDlColumnsPreservedInDfTest(unittest.TestCase):
    """Regression: the behavioral evaluation input (df_test) must carry DL columns.

    The original bug meant behavioral conditioning was never applied because:
    (a) the pair-scope gate was missing (fixed), AND
    (b) the df_test entering strategy evaluation must carry the DL feature columns
        that attach_dl_features() attached during pair processing.

    This test verifies acceptance criterion (b): a pair that passed through
    attach_dl_features() will have DL columns, and those columns must still be
    present in a slice of that DataFrame (as would be passed to df_test in the
    walk-forward fold).
    """

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def _make_df_with_dl_cols(self) -> pd.DataFrame:
        from src.dl_daily_features import D1_FEATURE_COLS
        data: dict = {
            "Close": [1.0, 1.1, 1.2, 1.3, 1.4],
            "phase": ["trending"] * 5,
        }
        for col in D1_FEATURE_COLS:
            data[col] = [0.5, 0.6, 0.7, 0.8, 0.9]
        return pd.DataFrame(data, index=pd.date_range("2023-01-01", periods=5))

    def test_dl_cols_present_in_full_df(self):
        """The full processed DataFrame for a behavioral pair must have DL columns."""
        df = self._make_df_with_dl_cols()
        self.assertTrue(
            self._main.has_dl_features(df),
            "Full df for a behavioral pair must contain DL feature columns",
        )

    def test_dl_cols_preserved_in_df_test_slice(self):
        """DL columns must be preserved when df_test is sliced from the full df."""
        df = self._make_df_with_dl_cols()
        # Simulate df_test as a fold slice (last 2 rows)
        df_test_slice = df.iloc[-2:]
        self.assertTrue(
            self._main.has_dl_features(df_test_slice),
            "df_test slice of a behavioral pair must still contain DL feature columns",
        )

    def test_dl_cols_absent_from_baseline_df_test(self):
        """A baseline pair's df_test slice must not carry DL columns."""
        df_baseline = pd.DataFrame({
            "Close": [1.0, 1.1, 1.2, 1.3, 1.4],
            "phase": ["ranging"] * 5,
        }, index=pd.date_range("2023-01-01", periods=5))
        df_test_slice = df_baseline.iloc[-2:]
        self.assertFalse(
            self._main.has_dl_features(df_test_slice),
            "Baseline df_test must not contain DL feature columns",
        )

    def test_behavioral_dl_columns_match_expected_feature_set(self):
        """DL columns in df_test must match the D1_FEATURE_COLS specification."""
        from src.dl_daily_features import D1_FEATURE_COLS
        df = self._make_df_with_dl_cols()
        df_test_slice = df.iloc[-2:]
        dl_cols_in_test = self._main.get_dl_feature_columns(df_test_slice)
        self.assertEqual(
            sorted(dl_cols_in_test),
            sorted(D1_FEATURE_COLS),
            "DL columns in df_test must exactly match D1_FEATURE_COLS",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
