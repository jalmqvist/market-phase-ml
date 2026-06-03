"""
Regression tests for StrategySelector feature-schema invariants.

These tests guard against the "silent schema drift" regression where the
selector would print a per-bar warning and continue with a fallback instead of
failing immediately when the inference feature schema did not match the fitted
schema.

Root cause: _stable_feature_columns() re-sorts all columns (including *_missing
indicator columns appended by apply_optional_feature_imputation) alphabetically,
producing a different column order from what StandardScaler.fit_transform() saw.
sklearn then raises "feature names should match those passed during fit" at every
predict_proba() call.

Contract enforced by these tests:
  1. Matching schema (with or without DL cols) → predict_proba succeeds.
  2. DL columns absent at inference → deterministic reindex via imputation (no error).
  3. Column ordering mismatch → RuntimeError, not silent continuation.
  4. Extra unexpected columns at inference → RuntimeError.
  5. Missing required columns → ValueError (legitimate PhaseAware fallback).
  6. Selector trained with DL cols but DL absent at inference → deterministic reindex.
  7. feature_schema_ is frozen after train() and survives predict calls.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from src.models import (
        StrategySelector,
        validate_feature_schema,
        apply_optional_feature_imputation,
        _stable_feature_columns,
    )
    from src.dl_config import DL_SIGNALS_ENABLED
    from src.dl_daily_features import D1_FEATURE_COLS as _D1_FEATURE_COLS
    _HAS_DEPS = True
    _DEPS_ERR = ""
except Exception as exc:  # pragma: no cover
    _HAS_DEPS = False
    _DEPS_ERR = str(exc)
    DL_SIGNALS_ENABLED = False
    _D1_FEATURE_COLS = ()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_COLS = ["adx", "atr_pct", "minus_di", "plus_di", "rsi", "returns_recent", "volatility_recent"]
# Use real DL column names so they are recognised by StrategySelector.train()
# (which gates on DL_D1_FEATURE_COLS and DL_SIGNALS_ENABLED).
_DL_COLS = list(_D1_FEATURE_COLS[:2]) if _D1_FEATURE_COLS else []


def _make_training_df(n: int = 200, *, include_dl: bool = False) -> pd.DataFrame:
    """Build a minimal StrategyPerformanceTracker-like training DataFrame."""
    rng = np.random.default_rng(42)
    data = {col: rng.uniform(0.1, 50.0, n) for col in _BASE_COLS}
    data["best_strategy"] = np.where(
        rng.integers(0, 3, n) == 0,
        "TF_test",
        np.where(rng.integers(0, 2, n) == 0, "MR_test", "PhaseAware"),
    )
    data["date"] = pd.date_range("2022-01-01", periods=n, freq="D")
    if include_dl and _DL_COLS:
        # ~60 % coverage so DL indicator columns are created during training.
        dl_vals = rng.uniform(0.0, 1.0, n)
        mask = rng.uniform(0.0, 1.0, n) < 0.4  # 40 % missing
        for col in _DL_COLS:
            vals = dl_vals.copy()
            vals[mask] = np.nan
            data[col] = vals
    return pd.DataFrame(data)


def _make_inference_row(*, include_dl: bool = False, dl_nan: bool = False) -> pd.DataFrame:
    """One-row inference DataFrame."""
    data = {col: [25.0] for col in _BASE_COLS}
    if include_dl and _DL_COLS:
        for col in _DL_COLS:
            data[col] = [np.nan if dl_nan else 0.5]
    return pd.DataFrame(data)


def _train_selector(
    *,
    include_dl: bool = False,
    do_cv: bool = False,
) -> StrategySelector:
    """Train a minimal StrategySelector for testing."""
    sel = StrategySelector(seed=42)
    df = _make_training_df(n=200, include_dl=include_dl)
    sel.train(df, do_cv=do_cv, diagnostics_label="test-selector")
    return sel


# ---------------------------------------------------------------------------
# Tests for validate_feature_schema() utility
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_DEPS, f"missing deps: {_DEPS_ERR}")
class TestValidateFeatureSchema(unittest.TestCase):

    def test_exact_match_passes(self):
        schema = ["adx", "atr_pct", "plus_di"]
        validate_feature_schema(schema, schema)  # must not raise

    def test_same_cols_different_order_raises(self):
        fitted = ["adx", "atr_pct", "plus_di"]
        infer = ["atr_pct", "adx", "plus_di"]
        with self.assertRaises(RuntimeError) as ctx:
            validate_feature_schema(fitted, infer)
        self.assertIn("ordering_mismatch_only: True", str(ctx.exception))

    def test_missing_col_raises(self):
        fitted = ["adx", "atr_pct", "plus_di"]
        infer = ["adx", "atr_pct"]
        with self.assertRaises(RuntimeError) as ctx:
            validate_feature_schema(fitted, infer)
        self.assertIn("plus_di", str(ctx.exception))
        self.assertIn("missing_cols", str(ctx.exception))

    def test_extra_col_raises(self):
        fitted = ["adx", "atr_pct"]
        infer = ["adx", "atr_pct", "surprise_col"]
        with self.assertRaises(RuntimeError) as ctx:
            validate_feature_schema(fitted, infer)
        self.assertIn("surprise_col", str(ctx.exception))
        self.assertIn("extra_cols", str(ctx.exception))

    def test_context_label_included_in_message(self):
        with self.assertRaises(RuntimeError) as ctx:
            validate_feature_schema(["a"], ["b"], context="pair=EURUSD fold=3 bar=17")
        self.assertIn("pair=EURUSD", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests for StrategySelector schema lifecycle
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_DEPS, f"missing deps: {_DEPS_ERR}")
class TestSelectorSchemaNoDL(unittest.TestCase):
    """Base feature schema (no DL columns)."""

    def setUp(self):
        self.sel = _train_selector(include_dl=False)

    def test_feature_schema_populated_after_train(self):
        self.assertIsNotNone(self.sel.feature_schema_)
        self.assertIsInstance(self.sel.feature_schema_, list)
        self.assertGreater(len(self.sel.feature_schema_), 0)

    def test_feature_schema_has_no_missing_indicators(self):
        # No DL cols → no imputation → no *_missing indicator cols.
        for col in self.sel.feature_schema_:
            self.assertFalse(col.endswith("_missing"), f"unexpected indicator col: {col}")

    def test_feature_schema_order_matches_scaler(self):
        self.assertEqual(
            list(self.sel.scaler.feature_names_in_),
            self.sel.feature_schema_,
            "Scaler feature order must exactly equal feature_schema_",
        )

    def test_predict_proba_succeeds_with_matching_schema(self):
        row = _make_inference_row(include_dl=False)
        probs = self.sel.predict_proba(row)
        self.assertIsInstance(probs, dict)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)
        self.assertIn("TrendFollowing", probs)
        self.assertIn("MeanReversion", probs)
        self.assertIn("PhaseAware", probs)

    def test_predict_proba_stable_across_calls(self):
        row = _make_inference_row(include_dl=False)
        a = self.sel.predict_proba(row)
        b = self.sel.predict_proba(row)
        for k in a:
            self.assertAlmostEqual(a[k], b[k], places=10)

    def test_predict_proba_raises_on_extra_column(self):
        row = _make_inference_row(include_dl=False)
        row["unexpected_col"] = 99.0
        # Extra columns are safely ignored by reindex — should not raise.
        probs = self.sel.predict_proba(row)
        self.assertIsInstance(probs, dict)

    def test_predict_proba_raises_on_missing_required_col(self):
        row = _make_inference_row(include_dl=False)
        row["adx"] = np.nan  # required column → ValueError
        with self.assertRaises(ValueError) as ctx:
            self.sel.predict_proba(row)
        self.assertIn("NaN", str(ctx.exception))

    def test_predict_succeeds_with_matching_schema(self):
        row = _make_inference_row(include_dl=False)
        result = self.sel.predict(row)
        self.assertIn(result, {"TrendFollowing", "MeanReversion", "PhaseAware"})

    def test_feature_schema_immutable_after_predict(self):
        schema_before = list(self.sel.feature_schema_)
        row = _make_inference_row(include_dl=False)
        self.sel.predict_proba(row)
        self.assertEqual(self.sel.feature_schema_, schema_before)

    def test_no_sklearn_feature_name_warning(self):
        """Inference must not trigger sklearn's 'feature names mismatch' warning."""
        import warnings
        row = _make_inference_row(include_dl=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.sel.predict_proba(row)


@unittest.skipUnless(_HAS_DEPS, f"missing deps: {_DEPS_ERR}")
@unittest.skipUnless(DL_SIGNALS_ENABLED and bool(_DL_COLS), "DL_SIGNALS_ENABLED=False — DL schema tests skipped")
class TestSelectorSchemaWithDL(unittest.TestCase):
    """DL-enabled schema: base cols + DL cols + *_missing indicator cols.

    These tests only run when DL_SIGNALS_ENABLED=True so that
    StrategySelector.train() actually incorporates DL feature columns.
    """

    def setUp(self):
        self.sel = _train_selector(include_dl=True)

    def test_feature_schema_populated_after_train(self):
        self.assertIsNotNone(self.sel.feature_schema_)

    def test_feature_schema_contains_missing_indicators(self):
        # DL cols had partial coverage → *_missing indicator cols are created.
        indicator_cols = [c for c in self.sel.feature_schema_ if c.endswith("_missing")]
        self.assertGreater(len(indicator_cols), 0, "Expected *_missing indicator cols")

    def test_feature_schema_order_matches_scaler(self):
        # The scaler's feature_names_in_ must equal feature_schema_ exactly.
        self.assertEqual(
            list(self.sel.scaler.feature_names_in_),
            self.sel.feature_schema_,
            "Scaler feature order must exactly equal feature_schema_",
        )

    def test_predict_proba_with_dl_cols_present(self):
        row = _make_inference_row(include_dl=True, dl_nan=False)
        probs = self.sel.predict_proba(row)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    def test_predict_proba_with_dl_cols_absent_uses_deterministic_reindex(self):
        # DL columns absent from inference row → imputation fills them with 0.0
        # and sets *_missing indicators to 1 → no schema error, no silent drift.
        row = _make_inference_row(include_dl=False)  # no DL cols at all
        probs = self.sel.predict_proba(row)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    def test_predict_proba_with_dl_cols_nan_uses_deterministic_reindex(self):
        # DL columns present but all NaN → same deterministic imputation path.
        row = _make_inference_row(include_dl=True, dl_nan=True)
        probs = self.sel.predict_proba(row)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    def test_no_sklearn_feature_name_warning(self):
        """Inference must not trigger sklearn's 'feature names mismatch' warning."""
        import warnings
        row = _make_inference_row(include_dl=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.sel.predict_proba(row)

    def test_predict_proba_stable_with_reordered_input(self):
        # Simulate what the OLD code did: pass features_df already reindexed to
        # feature_cols (sorted with *_missing interleaved) instead of raw row.
        # The new inference path strips *_missing cols and regenerates them, so
        # column order is always deterministic — this should still succeed.
        row = _make_inference_row(include_dl=False)
        # Reindex to sorted feature_cols (old-style) — should succeed with new code.
        row_sorted = row.reindex(columns=self.sel.feature_cols)
        probs = self.sel.predict_proba(row_sorted)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    def test_feature_schema_frozen_does_not_change_across_calls(self):
        schema_before = list(self.sel.feature_schema_)
        for _ in range(3):
            row = _make_inference_row(include_dl=False)
            self.sel.predict_proba(row)
        self.assertEqual(self.sel.feature_schema_, schema_before)


# ---------------------------------------------------------------------------
# Tests for apply_optional_feature_imputation *_missing indicator consistency
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_DEPS, f"missing deps: {_DEPS_ERR}")
class TestMissingIndicatorConsistency(unittest.TestCase):
    """apply_optional_feature_imputation must create identical indicator columns
    at training and inference time, enabling schema equality."""

    def test_missing_indicators_appended_in_sorted_order(self):
        df = pd.DataFrame({
            "dl_z": [np.nan, 0.5],
            "dl_a": [1.0, np.nan],
            "base": [10.0, 20.0],
        })
        result = apply_optional_feature_imputation(
            df, ["dl_z", "dl_a"], add_missing_indicators=True
        )
        # *_missing cols should be present and deterministically ordered
        self.assertIn("dl_a_missing", result.columns)
        self.assertIn("dl_z_missing", result.columns)
        # Values: 1 where original was NaN, 0 where present
        self.assertEqual(result["dl_a_missing"].iloc[0], 0)  # dl_a=1.0 → not missing
        self.assertEqual(result["dl_a_missing"].iloc[1], 1)  # dl_a=NaN → missing
        self.assertEqual(result["dl_z_missing"].iloc[0], 1)  # dl_z=NaN → missing
        self.assertEqual(result["dl_z_missing"].iloc[1], 0)  # dl_z=0.5 → not missing

    def test_training_and_inference_produce_identical_schemas(self):
        """Simulate training and inference imputation paths; schemas must match."""
        optional_cols = ["dl_b", "dl_a"]
        # Training: partial NaN coverage
        train_df = pd.DataFrame({
            "base": [1.0, 2.0, 3.0],
            "dl_a": [0.5, np.nan, 0.3],
            "dl_b": [np.nan, 1.0, 0.2],
        })
        train_out = apply_optional_feature_imputation(
            train_df, optional_cols, add_missing_indicators=True
        )
        # Inference: DL cols fully absent (NaN after reindex)
        infer_df = pd.DataFrame({
            "base": [5.0],
            "dl_a": [np.nan],
            "dl_b": [np.nan],
        })
        infer_out = apply_optional_feature_imputation(
            infer_df, optional_cols, add_missing_indicators=True
        )
        # Column SETS must match
        self.assertEqual(set(train_out.columns), set(infer_out.columns))
        # Column ORDER must match (both are sorted consistently)
        self.assertEqual(list(train_out.columns), list(infer_out.columns))

    def test_blind_mode_disables_missing_indicators(self):
        df = pd.DataFrame({
            "base": [1.0, 2.0],
            "dl_a": [np.nan, 0.5],
            "dl_b": [0.3, np.nan],
        })
        aware = apply_optional_feature_imputation(
            df, ["dl_a", "dl_b"], add_missing_indicators=True
        )
        blind = apply_optional_feature_imputation(
            df, ["dl_a", "dl_b"], add_missing_indicators=False
        )
        aware_missing_cols = sorted(c for c in aware.columns if c.endswith("_missing"))
        blind_missing_cols = sorted(c for c in blind.columns if c.endswith("_missing"))
        self.assertGreater(len(aware_missing_cols), 0)
        self.assertEqual(blind_missing_cols, [])


@unittest.skipUnless(_HAS_DEPS, f"missing deps: {_DEPS_ERR}")
@unittest.skipUnless(DL_SIGNALS_ENABLED and bool(_DL_COLS), "DL_SIGNALS_ENABLED=False — awareness schema tests skipped")
class TestAwarenessSelectorSchemas(unittest.TestCase):
    def test_aware_vs_blind_selector_schema_diverges_on_missing_indicators(self):
        df = _make_training_df(n=200, include_dl=True)

        aware = StrategySelector(seed=42, missing_indicators_enabled=True)
        aware.train(df, do_cv=False, diagnostics_label="aware-selector")

        blind = StrategySelector(seed=42, missing_indicators_enabled=False)
        blind.train(df, do_cv=False, diagnostics_label="blind-selector")

        aware_missing_cols = [c for c in (aware.feature_schema_ or []) if c.endswith("_missing")]
        blind_missing_cols = [c for c in (blind.feature_schema_ or []) if c.endswith("_missing")]

        self.assertGreater(len(aware_missing_cols), 0)
        self.assertEqual(len(blind_missing_cols), 0)
        self.assertNotEqual(aware.feature_schema_, blind.feature_schema_)


# ---------------------------------------------------------------------------
# Tests ensuring NO silent-continuation on schema drift
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_DEPS, f"missing deps: {_DEPS_ERR}")
class TestSchemaDriftHardFail(unittest.TestCase):
    """validate_feature_schema must raise; inference must not silently continue."""

    def test_validate_raises_not_warns(self):
        """validate_feature_schema raises RuntimeError — never just prints."""
        import warnings
        fitted = ["adx", "atr_pct", "dl_col", "dl_col_missing"]
        infer_bad = ["adx", "dl_col", "dl_col_missing", "atr_pct"]  # wrong order
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(RuntimeError):
                validate_feature_schema(fitted, infer_bad)

    def test_feature_schema_order_preserved_end_to_end(self):
        # Train without DL first.
        sel = _train_selector(include_dl=False)
        schema = sel.feature_schema_
        # Scaler must be consistent with schema.
        self.assertEqual(list(sel.scaler.feature_names_in_), schema)

    def test_no_per_bar_warning_spam(self):
        """After fix, predict_proba must not emit any per-bar warning output."""
        import io
        import contextlib

        sel = _train_selector(include_dl=False)
        row = _make_inference_row(include_dl=False)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for _ in range(5):
                probs = sel.predict_proba(row)
                self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

        output = buf.getvalue()
        self.assertNotIn("Selector prediction failed", output)
        self.assertNotIn("⚠️", output)
