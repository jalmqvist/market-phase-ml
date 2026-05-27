"""
tests/test_dl_conditional.py
=============================
Tests for the DL-conditioned selector analysis modules:

- analysis/diagnostics/selector_diagnostics.py
- analysis/diagnostics/transition_windows.py
- analysis/comparisons/dl_conditional.py
- analysis/parsers/csv_parsers.py (selector_state_timeline parser)

Run with:
    python -m unittest tests/test_dl_conditional.py -v
"""

from __future__ import annotations

import csv
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.diagnostics.selector_diagnostics import (
    compute_selector_entropy,
    compute_selector_entropy_per_pair,
    compute_switch_density,
    compute_switch_density_conditioned,
    compute_confidence_collapse_metrics,
    _parse_bool_field,
)
from analysis.diagnostics.transition_windows import (
    classify_folds_dl_state,
    classify_timeline_dl_state,
    extract_transition_windows,
    summarize_transition_windows,
)
from analysis.comparisons.dl_conditional import (
    build_dl_conditional_analysis,
    _aggregate_fold_metrics,
)
from analysis.parsers.csv_parsers import parse_run_csvs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fold_rows(
    n: int = 10,
    pair: str = "EURUSD",
    *,
    deltas: list[float] | None = None,
) -> list[dict]:
    """Build synthetic walkforward_per_fold rows."""
    if deltas is None:
        deltas = [0.1 * (i - n // 2) for i in range(n)]
    rows = []
    for i, delta in enumerate(deltas):
        rows.append({
            "Pair": pair,
            "Fold": str(i + 1),
            "Sharpe_Dynamic": str(0.5 + delta),
            "Sharpe_Baseline": "0.40",
            "Sharpe_Delta": str(delta),
            "Return_Dynamic": "0.05",
            "Return_Baseline": "0.03",
            "Return_Delta": str(0.02 + delta * 0.1),
            "MaxDD_Dynamic": "-0.10",
            "MaxDD_Baseline": "-0.12",
            "MaxDD_Delta": "0.02",
        })
    return rows


def _make_timeline_rows(
    n: int = 20,
    *,
    switch_every: int = 5,
    dl_active_from: int = 10,
) -> list[dict]:
    """Build synthetic selector_state_timeline rows."""
    strategies = ["TrendFollowing", "MeanReversion", "PhaseAware"]
    rows = []
    for i in range(n):
        strat_idx = (i // switch_every) % len(strategies)
        strat = strategies[strat_idx]
        prev_idx = ((i - 1) // switch_every) % len(strategies) if i > 0 else strat_idx
        prev_strat = strategies[prev_idx]
        is_switch = strat != prev_strat
        is_dl = i >= dl_active_from
        rows.append({
            "timestamp": f"2022-01-{i + 1:02d}",
            "pair": "EURUSD",
            "selected_strategy": strat,
            "selector_confidence": str(0.7 - 0.1 * (i % 3)),
            "dl_active": str(is_dl),
            "dl_missing": str(not is_dl),
            "fallback_active": "False",
            "switch_event": str(is_switch),
            "previous_strategy": prev_strat,
            "current_strategy": strat,
        })
    return rows


def _make_summary(
    run_id: str = "test_run",
    n_folds: int = 10,
    deltas: list[float] | None = None,
    overlap_pct: float | None = 40.0,
    pair: str = "EURUSD",
    timeline_rows: list[dict] | None = None,
) -> dict:
    fold_rows = _make_fold_rows(n=n_folds, pair=pair, deltas=deltas)
    return {
        "run_id": run_id,
        "meta": {},
        "csvs": {
            "walkforward_per_fold": fold_rows,
            "selector_state_timeline": timeline_rows,
        },
        "coverage": {
            "overlap_window": {"overlap_fold_coverage_pct": overlap_pct},
        },
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Tests: selector_diagnostics — fold-level entropy
# ---------------------------------------------------------------------------


class TestComputeSelectorEntropy(unittest.TestCase):

    def test_empty_returns_not_available(self):
        result = compute_selector_entropy([])
        self.assertFalse(result["data_available"])
        self.assertIsNone(result["selector_entropy"])
        self.assertEqual(result["n_folds"], 0)

    def test_none_input_returns_not_available(self):
        result = compute_selector_entropy(None)
        self.assertFalse(result["data_available"])

    def test_all_positive_deltas_low_entropy(self):
        rows = [{"Sharpe_Delta": "0.3"}, {"Sharpe_Delta": "0.5"}, {"Sharpe_Delta": "0.2"}]
        result = compute_selector_entropy(rows)
        self.assertTrue(result["data_available"])
        self.assertEqual(result["outcome_counts"]["positive"], 3)
        self.assertEqual(result["outcome_counts"]["negative"], 0)
        self.assertEqual(result["selector_entropy"], 0.0)
        self.assertEqual(result["normalized_entropy"], 0.0)
        self.assertEqual(result["occupancy_concentration"], 1.0)

    def test_mixed_deltas_higher_entropy(self):
        rows = [
            {"Sharpe_Delta": "0.3"},
            {"Sharpe_Delta": "-0.3"},
            {"Sharpe_Delta": "0.01"},  # near_zero
        ]
        result = compute_selector_entropy(rows)
        self.assertTrue(result["data_available"])
        self.assertGreater(result["selector_entropy"], 0.0)
        self.assertLessEqual(result["normalized_entropy"], 1.0)

    def test_max_entropy_with_equal_distribution(self):
        rows = [
            {"Sharpe_Delta": "0.3"},
            {"Sharpe_Delta": "-0.3"},
            {"Sharpe_Delta": "0.01"},  # near_zero
        ]
        result = compute_selector_entropy(rows)
        # Equal distribution across 3 categories → max entropy
        expected_max = math.log2(3)
        self.assertAlmostEqual(result["selector_entropy"], expected_max, places=3)
        self.assertAlmostEqual(result["normalized_entropy"], 1.0, places=3)

    def test_n_folds_counts_correctly(self):
        rows = [{"Sharpe_Delta": str(v)} for v in [0.1, 0.2, -0.1, 0.0, 0.5]]
        result = compute_selector_entropy(rows)
        self.assertEqual(result["n_folds"], 5)

    def test_fold_sharpe_std_computed(self):
        rows = [{"Sharpe_Delta": "1.0"}, {"Sharpe_Delta": "-1.0"}]
        result = compute_selector_entropy(rows)
        self.assertIsNotNone(result["fold_sharpe_std"])
        self.assertGreater(result["fold_sharpe_std"], 0.0)

    def test_non_metric_col_respected(self):
        rows = [{"Return_Delta": "0.5"}, {"Return_Delta": "-0.5"}, {"Return_Delta": "0.0"}]
        result = compute_selector_entropy(rows, metric_col="Return_Delta")
        self.assertTrue(result["data_available"])
        self.assertEqual(result["n_folds"], 3)

    def test_rows_with_missing_metric_skipped(self):
        rows = [{"Sharpe_Delta": "0.3"}, {"Other": "x"}, {"Sharpe_Delta": "0.1"}]
        result = compute_selector_entropy(rows)
        self.assertEqual(result["n_folds"], 2)

    def test_requires_timeline_is_false(self):
        result = compute_selector_entropy([{"Sharpe_Delta": "0.1"}])
        self.assertFalse(result["requires_timeline"])

    def test_outcome_counts_keys_present(self):
        result = compute_selector_entropy([{"Sharpe_Delta": "0.1"}])
        self.assertIn("positive", result["outcome_counts"])
        self.assertIn("negative", result["outcome_counts"])
        self.assertIn("near_zero", result["outcome_counts"])


class TestComputeSelectorEntropyPerPair(unittest.TestCase):

    def test_groups_by_pair(self):
        rows = [
            {"Pair": "EURUSD", "Sharpe_Delta": "0.1"},
            {"Pair": "EURUSD", "Sharpe_Delta": "0.2"},
            {"Pair": "GBPUSD", "Sharpe_Delta": "-0.1"},
        ]
        result = compute_selector_entropy_per_pair(rows)
        self.assertIn("EURUSD", result)
        self.assertIn("GBPUSD", result)
        self.assertEqual(result["EURUSD"]["n_folds"], 2)
        self.assertEqual(result["GBPUSD"]["n_folds"], 1)

    def test_empty_returns_empty(self):
        result = compute_selector_entropy_per_pair([])
        self.assertEqual(result, {})

    def test_missing_pair_col_uses_unknown(self):
        rows = [{"Sharpe_Delta": "0.1"}]
        result = compute_selector_entropy_per_pair(rows)
        self.assertIn("unknown", result)


# ---------------------------------------------------------------------------
# Tests: selector_diagnostics — switch density
# ---------------------------------------------------------------------------


class TestComputeSwitchDensity(unittest.TestCase):

    def test_empty_returns_not_available(self):
        result = compute_switch_density([])
        self.assertFalse(result["data_available"])
        self.assertTrue(result["requires_timeline"])
        self.assertIsNone(result["switches_per_1000_bars"])

    def test_no_switches(self):
        rows = [{"switch_event": "False"} for _ in range(10)]
        result = compute_switch_density(rows)
        self.assertTrue(result["data_available"])
        self.assertEqual(result["total_switches"], 0)
        self.assertEqual(result["switches_per_1000_bars"], 0.0)

    def test_every_bar_is_a_switch(self):
        rows = [{"switch_event": "True"} for _ in range(10)]
        result = compute_switch_density(rows)
        # Every bar is a switch: 10 switches / 10 bars * 1000 = 1000
        self.assertAlmostEqual(result["switches_per_1000_bars"], 1000.0, places=1)
        self.assertEqual(result["total_switches"], 10)

    def test_switch_every_5_bars(self):
        rows = []
        for i in range(20):
            rows.append({"switch_event": str(i % 5 == 0 and i > 0)})
        result = compute_switch_density(rows)
        # 3 switches in 20 bars → 150 per 1000
        self.assertEqual(result["total_switches"], 3)
        self.assertAlmostEqual(result["switches_per_1000_bars"], 150.0, places=1)

    def test_mean_hold_duration_computed(self):
        # Switches at bars 0, 5, 10 → hold runs of length 5, 5, 10
        rows = [{"switch_event": str(i in (5, 10))} for i in range(20)]
        result = compute_switch_density(rows)
        self.assertIsNotNone(result["mean_hold_duration"])
        self.assertGreater(result["mean_hold_duration"], 0.0)

    def test_median_hold_duration_computed(self):
        rows = [{"switch_event": str(i in (5, 10))} for i in range(20)]
        result = compute_switch_density(rows)
        self.assertIsNotNone(result["median_hold_duration"])


class TestComputeSwitchDensityConditioned(unittest.TestCase):

    def test_returns_three_keys(self):
        rows = _make_timeline_rows(n=20, switch_every=5, dl_active_from=10)
        result = compute_switch_density_conditioned(rows)
        self.assertIn("dl_active", result)
        self.assertIn("dl_missing", result)
        self.assertIn("full", result)

    def test_empty_returns_all_not_available(self):
        result = compute_switch_density_conditioned([])
        for key in ("dl_active", "dl_missing", "full"):
            self.assertFalse(result[key]["data_available"])

    def test_active_and_missing_sum_to_full_bars(self):
        rows = _make_timeline_rows(n=20, switch_every=5, dl_active_from=10)
        result = compute_switch_density_conditioned(rows)
        active_bars = result["dl_active"]["total_bars"]
        missing_bars = result["dl_missing"]["total_bars"]
        full_bars = result["full"]["total_bars"]
        self.assertEqual(active_bars + missing_bars, full_bars)
        self.assertEqual(full_bars, 20)


# ---------------------------------------------------------------------------
# Tests: selector_diagnostics — confidence collapse
# ---------------------------------------------------------------------------


class TestComputeConfidenceCollapseMetrics(unittest.TestCase):

    def test_empty_returns_not_available(self):
        result = compute_confidence_collapse_metrics([])
        self.assertFalse(result["data_available"])
        self.assertTrue(result["requires_timeline"])

    def test_no_collapses_when_confidence_always_high(self):
        rows = [{"selector_confidence": "0.9", "fallback_active": "False"}
                for _ in range(10)]
        result = compute_confidence_collapse_metrics(rows, collapse_threshold=0.5)
        self.assertEqual(result["confidence_collapse_count"], 0)

    def test_single_collapse_detected(self):
        rows = [
            {"selector_confidence": "0.8", "fallback_active": "False"},
            {"selector_confidence": "0.3", "fallback_active": "True"},   # collapse
            {"selector_confidence": "0.3", "fallback_active": "True"},
            {"selector_confidence": "0.8", "fallback_active": "False"},  # recovery
        ]
        result = compute_confidence_collapse_metrics(rows, collapse_threshold=0.5)
        self.assertEqual(result["confidence_collapse_count"], 1)
        self.assertAlmostEqual(result["mean_confidence_recovery_time"], 2.0)

    def test_fallback_entry_exit_rates_computed(self):
        rows = [
            {"selector_confidence": "0.8", "fallback_active": "False"},
            {"selector_confidence": "0.3", "fallback_active": "True"},
            {"selector_confidence": "0.8", "fallback_active": "False"},
        ] * 4
        result = compute_confidence_collapse_metrics(rows, collapse_threshold=0.5)
        self.assertIsNotNone(result["fallback_entry_rate"])
        self.assertIsNotNone(result["fallback_exit_rate"])
        self.assertGreater(result["fallback_entry_rate"], 0.0)

    def test_missing_confidence_col_skipped_gracefully(self):
        rows = [{"fallback_active": "False"} for _ in range(5)]
        result = compute_confidence_collapse_metrics(rows)
        self.assertTrue(result["data_available"])
        self.assertEqual(result["confidence_collapse_count"], 0)


# ---------------------------------------------------------------------------
# Tests: _parse_bool_field
# ---------------------------------------------------------------------------


class TestParseBoolField(unittest.TestCase):

    def test_bool_true(self):
        self.assertTrue(_parse_bool_field(True))

    def test_bool_false(self):
        self.assertFalse(_parse_bool_field(False))

    def test_string_true(self):
        self.assertTrue(_parse_bool_field("True"))
        self.assertTrue(_parse_bool_field("true"))
        self.assertTrue(_parse_bool_field("1"))
        self.assertTrue(_parse_bool_field("yes"))

    def test_string_false(self):
        self.assertFalse(_parse_bool_field("False"))
        self.assertFalse(_parse_bool_field("false"))
        self.assertFalse(_parse_bool_field("0"))

    def test_none_returns_false(self):
        self.assertFalse(_parse_bool_field(None))

    def test_int_nonzero_true(self):
        self.assertTrue(_parse_bool_field(1))
        self.assertFalse(_parse_bool_field(0))


# ---------------------------------------------------------------------------
# Tests: transition_windows — fold classification
# ---------------------------------------------------------------------------


class TestClassifyFoldsDLState(unittest.TestCase):

    def test_empty_returns_empty(self):
        result = classify_folds_dl_state([], overlap_fold_coverage_pct=50.0)
        self.assertEqual(result, [])

    def test_none_coverage_labels_unknown(self):
        rows = _make_fold_rows(n=5)
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=None)
        for r in result:
            self.assertEqual(r["dl_state"], "dl_state_unknown")
            self.assertFalse(r["dl_active"])

    def test_zero_coverage_all_missing(self):
        rows = _make_fold_rows(n=5)
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=0.0)
        for r in result:
            self.assertEqual(r["dl_state"], "dl_missing")
            self.assertFalse(r["dl_active"])

    def test_full_coverage_all_active(self):
        rows = _make_fold_rows(n=5)
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=100.0)
        # All folds should be dl_active (first fold may be dl_transition_enter if k==n).
        for r in result:
            self.assertIn(r["dl_state"], ("dl_active", "dl_transition_enter"))
            self.assertTrue(r["dl_active"])

    def test_40_percent_coverage_correct_split(self):
        rows = _make_fold_rows(n=10)
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=40.0)
        # 40% of 10 → last 4 folds are DL_ACTIVE.
        active = [r for r in result if r["dl_active"]]
        missing = [r for r in result if not r["dl_active"]]
        self.assertEqual(len(active), 4)
        self.assertEqual(len(missing), 6)

    def test_transition_enter_assigned_to_first_active_fold(self):
        rows = _make_fold_rows(n=10)
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=40.0)
        # The first active fold (index 6) should be dl_transition_enter.
        states = [r["dl_state"] for r in sorted(result, key=lambda x: int(x["Fold"]))]
        # Index 6 (fold 7) should be dl_transition_enter.
        self.assertEqual(states[6], "dl_transition_enter")

    def test_preserves_existing_row_fields(self):
        rows = _make_fold_rows(n=3)
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=50.0)
        for r in result:
            self.assertIn("Pair", r)
            self.assertIn("Sharpe_Delta", r)

    def test_multi_pair_classified_independently(self):
        rows = _make_fold_rows(n=5, pair="EURUSD") + _make_fold_rows(n=5, pair="GBPUSD")
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=40.0)
        eur_active = [r for r in result if r["Pair"] == "EURUSD" and r["dl_active"]]
        gbp_active = [r for r in result if r["Pair"] == "GBPUSD" and r["dl_active"]]
        self.assertEqual(len(eur_active), 2)
        self.assertEqual(len(gbp_active), 2)


class TestClassifyTimelineDLState(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(classify_timeline_dl_state([]), [])

    def test_first_bar_active_is_dl_active(self):
        rows = [{"dl_active": "True"}]
        result = classify_timeline_dl_state(rows)
        self.assertEqual(result[0]["dl_state"], "dl_active")

    def test_first_bar_missing_is_dl_missing(self):
        rows = [{"dl_active": "False"}]
        result = classify_timeline_dl_state(rows)
        self.assertEqual(result[0]["dl_state"], "dl_missing")

    def test_transition_enter_detected(self):
        rows = [{"dl_active": "False"}, {"dl_active": "True"}]
        result = classify_timeline_dl_state(rows)
        self.assertEqual(result[1]["dl_state"], "dl_transition_enter")

    def test_transition_exit_detected(self):
        rows = [{"dl_active": "True"}, {"dl_active": "False"}]
        result = classify_timeline_dl_state(rows)
        self.assertEqual(result[1]["dl_state"], "dl_transition_exit")

    def test_sustained_active_stays_dl_active(self):
        rows = [{"dl_active": "True"}] * 5
        result = classify_timeline_dl_state(rows)
        # First bar is dl_active, rest should also be dl_active.
        self.assertEqual(result[0]["dl_state"], "dl_active")
        for r in result[1:]:
            self.assertEqual(r["dl_state"], "dl_active")


# ---------------------------------------------------------------------------
# Tests: transition_windows — extraction and summarization
# ---------------------------------------------------------------------------


class TestExtractTransitionWindows(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(extract_transition_windows([]), [])

    def test_no_transitions_returns_empty(self):
        rows = [{"dl_state": "dl_active"} for _ in range(5)]
        result = extract_transition_windows(rows)
        self.assertEqual(result, [])

    def test_transition_at_boundary_captured(self):
        rows = [
            {"dl_state": "dl_missing"},
            {"dl_state": "dl_missing"},
            {"dl_state": "dl_transition_enter"},
            {"dl_state": "dl_active"},
            {"dl_state": "dl_active"},
        ]
        result = extract_transition_windows(rows, n_before=1, n_after=1)
        # Should capture: index 1, 2, 3.
        self.assertEqual(len(result), 3)

    def test_transition_window_index_assigned(self):
        rows = [
            {"dl_state": "dl_missing"},
            {"dl_state": "dl_transition_enter"},
            {"dl_state": "dl_active"},
        ]
        result = extract_transition_windows(rows, n_before=1, n_after=1)
        self.assertIn("transition_window_index", result[0])
        for r in result:
            self.assertEqual(r["transition_window_index"], 0)

    def test_n_before_n_after_respected(self):
        rows = [{"dl_state": "dl_missing"}] * 5 + [{"dl_state": "dl_transition_enter"}] + [{"dl_state": "dl_active"}] * 5
        result = extract_transition_windows(rows, n_before=2, n_after=3)
        # Should capture: 5-2=3, 4, 5(transition), 6, 7, 8 → 6 rows
        self.assertEqual(len(result), 6)

    def test_overlapping_windows_deduplicated(self):
        # Two transitions close together.
        rows = (
            [{"dl_state": "dl_missing"}] * 2
            + [{"dl_state": "dl_transition_enter"}]
            + [{"dl_state": "dl_transition_exit"}]
            + [{"dl_state": "dl_missing"}] * 2
        )
        result = extract_transition_windows(rows, n_before=1, n_after=1)
        # No duplicate rows.
        indices = [r.get("transition_window_index") for r in result]
        # All row positions should be unique.
        positions = [(r.get("Pair", ""), i) for i, r in enumerate(result)]
        self.assertEqual(len(positions), len(set(i for _, i in positions)))

    def test_custom_transition_states(self):
        rows = [
            {"dl_state": "other_state"},
            {"dl_state": "volatility_spike"},
            {"dl_state": "other_state"},
        ]
        result = extract_transition_windows(rows, transition_states={"volatility_spike"}, n_before=1, n_after=1)
        self.assertEqual(len(result), 3)


class TestSummarizeTransitionWindows(unittest.TestCase):

    def test_empty_not_available(self):
        result = summarize_transition_windows([])
        self.assertFalse(result["data_available"])
        self.assertEqual(result["n_transition_rows"], 0)

    def test_counts_windows_correctly(self):
        rows = [
            {"transition_window_index": 0, "Sharpe_Delta": "0.1"},
            {"transition_window_index": 0, "Sharpe_Delta": "0.2"},
            {"transition_window_index": 1, "Sharpe_Delta": "-0.1"},
        ]
        result = summarize_transition_windows(rows)
        self.assertTrue(result["data_available"])
        self.assertEqual(result["n_windows"], 2)
        self.assertEqual(result["n_transition_rows"], 3)

    def test_metrics_aggregated(self):
        rows = [
            {"transition_window_index": 0, "Sharpe_Delta": "0.2"},
            {"transition_window_index": 0, "Sharpe_Delta": "0.4"},
        ]
        result = summarize_transition_windows(rows, metric_cols=["Sharpe_Delta"])
        self.assertIn("Sharpe_Delta", result["metrics"])
        self.assertAlmostEqual(result["metrics"]["Sharpe_Delta"]["mean"], 0.3, places=3)
        self.assertEqual(result["metrics"]["Sharpe_Delta"]["n"], 2)

    def test_missing_metric_col_ignored(self):
        rows = [{"transition_window_index": 0, "SomethingElse": "x"}]
        result = summarize_transition_windows_safe(rows, metric_cols=["Sharpe_Delta"])
        self.assertNotIn("Sharpe_Delta", result["metrics"])


def summarize_transition_windows_safe(rows, metric_cols=None):
    """Wrapper to avoid NameError in test."""
    return summarize_transition_windows(rows, metric_cols=metric_cols)


# ---------------------------------------------------------------------------
# Tests: dl_conditional — aggregate fold metrics
# ---------------------------------------------------------------------------


class TestAggregateFoldMetrics(unittest.TestCase):

    def test_empty_returns_none_values(self):
        result = _aggregate_fold_metrics([])
        self.assertFalse(result["data_available"])
        self.assertEqual(result["n_folds"], 0)
        self.assertIsNone(result["sharpe_delta"])

    def test_aggregates_mean_correctly(self):
        rows = [
            {"Sharpe_Delta": "0.2", "Return_Delta": "0.03"},
            {"Sharpe_Delta": "0.4", "Return_Delta": "0.07"},
        ]
        result = _aggregate_fold_metrics(rows)
        self.assertTrue(result["data_available"])
        self.assertAlmostEqual(result["sharpe_delta"], 0.3, places=3)
        self.assertAlmostEqual(result["return_delta"], 0.05, places=3)

    def test_n_folds_correct(self):
        rows = _make_fold_rows(n=7)
        result = _aggregate_fold_metrics(rows)
        self.assertEqual(result["n_folds"], 7)

    def test_missing_cols_return_none(self):
        rows = [{"Sharpe_Delta": "0.1"}]
        result = _aggregate_fold_metrics(rows)
        self.assertIsNone(result["return_delta"])
        self.assertIsNone(result["maxdd_delta"])

    def test_non_numeric_values_skipped(self):
        rows = [{"Sharpe_Delta": "N/A"}, {"Sharpe_Delta": "0.3"}]
        result = _aggregate_fold_metrics(rows)
        self.assertAlmostEqual(result["sharpe_delta"], 0.3, places=3)


# ---------------------------------------------------------------------------
# Tests: build_dl_conditional_analysis
# ---------------------------------------------------------------------------


class TestBuildDLConditionalAnalysis(unittest.TestCase):

    def test_empty_summaries_returns_not_available(self):
        result = build_dl_conditional_analysis([])
        self.assertFalse(result["data_available"])
        self.assertGreater(len(result["warnings"]), 0)

    def test_summary_without_fold_data_returns_not_available(self):
        summary = {
            "run_id": "test",
            "meta": {},
            "csvs": {"walkforward_per_fold": None, "selector_state_timeline": None},
            "coverage": {},
            "warnings": [],
        }
        result = build_dl_conditional_analysis([summary])
        self.assertFalse(result["data_available"])

    def test_summary_with_fold_data_returns_available(self):
        result = build_dl_conditional_analysis([_make_summary()])
        self.assertTrue(result["data_available"])

    def test_aggregate_table_has_four_windows(self):
        result = build_dl_conditional_analysis([_make_summary()])
        windows = {row["window"] for row in result["aggregate_table"]}
        self.assertEqual(windows, {"full", "dl_active", "dl_missing", "transition"})

    def test_dl_active_folds_differ_from_missing(self):
        # Use clearly bimodal deltas: first 6 folds negative, last 4 positive.
        deltas = [-0.3] * 6 + [0.5] * 4
        result = build_dl_conditional_analysis([_make_summary(deltas=deltas, overlap_pct=40.0)])
        table = {row["window"]: row for row in result["aggregate_table"]}
        # DL_ACTIVE (last 4 folds) should have positive mean delta.
        self.assertIsNotNone(table["dl_active"]["sharpe_delta"])
        self.assertIsNotNone(table["dl_missing"]["sharpe_delta"])
        self.assertGreater(table["dl_active"]["sharpe_delta"], table["dl_missing"]["sharpe_delta"])

    def test_multiple_runs_produce_rows_per_run(self):
        summaries = [_make_summary(f"run_{i}") for i in range(3)]
        result = build_dl_conditional_analysis(summaries)
        run_ids = {row["run_id"] for row in result["aggregate_table"]}
        self.assertEqual(run_ids, {"run_0", "run_1", "run_2"})
        # 4 windows per run → 12 rows total.
        self.assertEqual(len(result["aggregate_table"]), 12)

    def test_entropy_present_in_aggregate_table(self):
        result = build_dl_conditional_analysis([_make_summary()])
        for row in result["aggregate_table"]:
            self.assertIn("selector_entropy", row)

    def test_per_run_has_correct_keys(self):
        result = build_dl_conditional_analysis([_make_summary()])
        run = result["per_run"][0]
        self.assertIn("data_available", run)
        self.assertIn("conditional_metrics", run)
        self.assertIn("selector_entropy", run)
        self.assertIn("transition_summary", run)
        self.assertIn("overlap_fold_coverage_pct", run)

    def test_timeline_data_populates_switch_density(self):
        timeline = _make_timeline_rows(n=30, switch_every=5, dl_active_from=15)
        summary = _make_summary(timeline_rows=timeline)
        result = build_dl_conditional_analysis([summary])
        # When timeline is present, switches_per_1000_bars should be non-None for 'full'.
        table = {row["window"]: row for row in result["aggregate_table"]}
        self.assertIsNotNone(table["full"]["switches_per_1000_bars"])

    def test_no_timeline_gives_none_switch_density(self):
        result = build_dl_conditional_analysis([_make_summary()])
        for row in result["aggregate_table"]:
            self.assertIsNone(row["switches_per_1000_bars"])

    def test_warnings_propagate(self):
        # Summary with no coverage data → should produce a warning.
        summary = _make_summary(overlap_pct=None)
        result = build_dl_conditional_analysis([summary])
        self.assertGreater(len(result["warnings"]), 0)

    def test_metadata_present_in_result(self):
        result = build_dl_conditional_analysis([_make_summary()])
        self.assertIn("metadata", result)
        self.assertIn("dl_state_assignment_method", result["metadata"])

    def test_metadata_heuristic_when_no_timeline(self):
        result = build_dl_conditional_analysis([_make_summary()])
        self.assertEqual(
            result["metadata"]["dl_state_assignment_method"],
            "heuristic_fold_position",
        )

    def test_metadata_timeline_exact_when_timeline_present(self):
        timeline = _make_timeline_rows(n=20, switch_every=5, dl_active_from=10)
        summary = _make_summary(timeline_rows=timeline)
        result = build_dl_conditional_analysis([summary])
        self.assertEqual(
            result["metadata"]["dl_state_assignment_method"],
            "timeline_exact",
        )

    def test_metadata_unknown_when_empty_summaries(self):
        result = build_dl_conditional_analysis([])
        self.assertEqual(
            result["metadata"]["dl_state_assignment_method"],
            "unknown",
        )

    def test_metadata_unknown_when_no_data(self):
        summary = {
            "run_id": "test",
            "meta": {},
            "csvs": {"walkforward_per_fold": None, "selector_state_timeline": None},
            "coverage": {},
            "warnings": [],
        }
        result = build_dl_conditional_analysis([summary])
        self.assertEqual(
            result["metadata"]["dl_state_assignment_method"],
            "unknown",
        )

    def test_metadata_heuristic_when_mixed_runs(self):
        # One run with timeline (exact), one without (heuristic) → conservative result.
        timeline = _make_timeline_rows(n=20, switch_every=5, dl_active_from=10)
        s_exact = _make_summary("run_exact", timeline_rows=timeline)
        s_heuristic = _make_summary("run_heuristic")
        result = build_dl_conditional_analysis([s_exact, s_heuristic])
        self.assertEqual(
            result["metadata"]["dl_state_assignment_method"],
            "heuristic_fold_position",
        )

    def test_per_run_has_assignment_method(self):
        result = build_dl_conditional_analysis([_make_summary()])
        self.assertIn("dl_state_assignment_method", result["per_run"][0])

    def test_per_run_heuristic_when_no_timeline(self):
        result = build_dl_conditional_analysis([_make_summary()])
        self.assertEqual(
            result["per_run"][0]["dl_state_assignment_method"],
            "heuristic_fold_position",
        )

    def test_per_run_timeline_exact_when_timeline_present(self):
        timeline = _make_timeline_rows(n=20, switch_every=5, dl_active_from=10)
        summary = _make_summary(timeline_rows=timeline)
        result = build_dl_conditional_analysis([summary])
        self.assertEqual(
            result["per_run"][0]["dl_state_assignment_method"],
            "timeline_exact",
        )


# ---------------------------------------------------------------------------
# Tests: csv_parsers — selector_state_timeline parser
# ---------------------------------------------------------------------------


class TestSelectorStateTimelineParser(unittest.TestCase):

    def _write_timeline_csv(self, run_dir: Path, rows: list[dict]) -> None:
        path = run_dir / "selector_state_timeline.csv"
        if not rows:
            path.write_text("timestamp,pair,selected_strategy\n")
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_timeline_parsed_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            # Create the minimal manifest marker.
            (run_dir / "run_manifest_2022.json").write_text(
                '{"run_id": "test", "experiment": {"factors": {}}}'
            )
            rows = [
                {
                    "timestamp": "2022-01-01",
                    "pair": "EURUSD",
                    "selected_strategy": "TrendFollowing",
                    "selector_confidence": "0.8",
                    "dl_active": "True",
                    "dl_missing": "False",
                    "fallback_active": "False",
                    "switch_event": "False",
                }
            ]
            self._write_timeline_csv(run_dir, rows)
            csvs = parse_run_csvs(run_dir)
            timeline = csvs.get("selector_state_timeline")
            self.assertIsNotNone(timeline)
            self.assertEqual(len(timeline), 1)
            row = timeline[0]
            self.assertEqual(row["timestamp"], "2022-01-01")
            self.assertEqual(row["selected_strategy"], "TrendFollowing")
            self.assertAlmostEqual(row["selector_confidence"], 0.8, places=5)
            self.assertTrue(row["dl_active"])
            self.assertFalse(row["dl_missing"])
            self.assertFalse(row["switch_event"])

    def test_timeline_absent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            csvs = parse_run_csvs(run_dir)
            self.assertIsNone(csvs.get("selector_state_timeline"))

    def test_timeline_bool_fields_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            rows = [
                {
                    "timestamp": "2022-01-01",
                    "pair": "EURUSD",
                    "selected_strategy": "MeanReversion",
                    "selector_confidence": "0.55",
                    "dl_active": "False",
                    "dl_missing": "True",
                    "fallback_active": "True",
                    "switch_event": "True",
                }
            ]
            self._write_timeline_csv(run_dir, rows)
            csvs = parse_run_csvs(run_dir)
            row = csvs["selector_state_timeline"][0]
            self.assertFalse(row["dl_active"])
            self.assertTrue(row["dl_missing"])
            self.assertTrue(row["fallback_active"])
            self.assertTrue(row["switch_event"])

    def test_timeline_empty_bool_field_becomes_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            content = "timestamp,pair,dl_active\n2022-01-01,EURUSD,\n"
            (run_dir / "selector_state_timeline.csv").write_text(content)
            csvs = parse_run_csvs(run_dir)
            row = csvs["selector_state_timeline"][0]
            self.assertIsNone(row["dl_active"])

    def test_timeline_selector_state_key_present_in_result(self):
        """selector_state_timeline key must always be in parse_run_csvs output."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            csvs = parse_run_csvs(run_dir)
            self.assertIn("selector_state_timeline", csvs)


# ---------------------------------------------------------------------------
# Integration test: pipeline + conditional analysis
# ---------------------------------------------------------------------------


class TestPipelineConditionalIntegration(unittest.TestCase):

    def test_run_pipeline_with_conditional_flag(self):
        """
        Smoke-test that run_pipeline runs without error when
        --conditional-analysis is enabled and a fold CSV is present.
        """
        import json
        from analysis.pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as archive_tmp:
            with tempfile.TemporaryDirectory() as output_tmp:
                run_dir = Path(archive_tmp) / "test_run"
                run_dir.mkdir()

                # Write a minimal manifest.
                manifest = {
                    "run_id": "test_conditional_run",
                    "experiment": {
                        "generation": "gen1",
                        "variant": "A",
                        "factors": {
                            "dl_enabled": True,
                            "sentiment_enabled": True,
                            "missing_indicators_enabled": True,
                        },
                        "semantic_label": "gen1_A",
                    },
                    "experiment_surface": {
                        "surface_semantics_version": 5,
                        "surface_source": "manifest",
                        "training_pair_family": "persistent",
                        "evaluation_pair_family": "persistent",
                        "sentiment_surface": True,
                        "feature_surface": "trend_vol_only",
                    },
                    "dl_enabled": True,
                }
                (run_dir / "run_manifest_2022.json").write_text(json.dumps(manifest))

                # Write a minimal walkforward_per_fold CSV.
                fold_content = (
                    "Pair,Fold,Sharpe_Dynamic,Sharpe_Baseline,Sharpe_Delta,"
                    "Return_Dynamic,Return_Baseline,Return_Delta,"
                    "MaxDD_Dynamic,MaxDD_Baseline,MaxDD_Delta\n"
                )
                for i in range(10):
                    delta = 0.1 * (i - 5)
                    fold_content += (
                        f"EURUSD,{i + 1},{0.5 + delta:.2f},0.40,{delta:.2f},"
                        f"0.05,0.03,0.02,-0.10,-0.12,0.02\n"
                    )
                (run_dir / "walkforward_results_per_fold__dl_enabled.csv").write_text(fold_content)

                # Write a vol_guard_summary so coverage is non-empty.
                (run_dir / "vol_guard_diagnostics_summary__dl_enabled.csv").write_text(
                    "Pair,Guard_Rate,ATR_Quantile,Vol_Threshold,N_Folds,N_Suppressed,N_Total\n"
                    "EURUSD,0.05,0.90,0.01,10,5,100\n"
                )

                output_dir = Path(output_tmp)
                run_pipeline(
                    archive_root=Path(archive_tmp),
                    output_dir=output_dir,
                    verbose=False,
                    conditional_analysis=True,
                )

                # comparisons.json should exist and have a 'conditional' key.
                comparisons_path = output_dir / "comparisons.json"
                self.assertTrue(comparisons_path.exists())
                comparisons = json.loads(comparisons_path.read_text())
                self.assertIn("conditional", comparisons)
                cond = comparisons["conditional"]
                self.assertIn("aggregate_table", cond)
                self.assertIn("per_run", cond)

    def test_run_pipeline_without_conditional_flag_no_key(self):
        """Without --conditional-analysis, the 'conditional' key must be absent."""
        import json
        from analysis.pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as archive_tmp:
            with tempfile.TemporaryDirectory() as output_tmp:
                run_dir = Path(archive_tmp) / "test_run"
                run_dir.mkdir()
                manifest = {
                    "run_id": "test_no_conditional",
                    "experiment": {"generation": "gen1", "variant": "A", "factors": {}},
                    "experiment_surface": {
                        "surface_semantics_version": 5,
                        "surface_source": "manifest",
                        "training_pair_family": "persistent",
                        "evaluation_pair_family": "persistent",
                        "sentiment_surface": False,
                        "feature_surface": "trend_vol_only",
                    },
                    "dl_enabled": False,
                }
                (run_dir / "run_manifest_2022.json").write_text(json.dumps(manifest))
                output_dir = Path(output_tmp)
                run_pipeline(
                    archive_root=Path(archive_tmp),
                    output_dir=output_dir,
                    verbose=False,
                    conditional_analysis=False,
                )
                comparisons = json.loads((output_dir / "comparisons.json").read_text())
                self.assertNotIn("conditional", comparisons)


# ---------------------------------------------------------------------------
# Tests: per-fold dl_overlap_pct classification
# ---------------------------------------------------------------------------

_DL_STATE_TO_PCT: dict[str, float] = {"active": 100.0, "partial": 60.0, "missing": 0.0}


def _make_fold_rows_with_overlap(
    states: list[str],
    pair: str = "EURUSD",
    pcts: list[float] | None = None,
) -> list[dict]:
    """Build fold rows that carry dl_overlap_pct / dl_overlap_state columns."""
    if pcts is None:
        pcts = [_DL_STATE_TO_PCT[s] for s in states]
    rows = []
    for i, (state, pct) in enumerate(zip(states, pcts)):
        rows.append({
            "Pair": pair,
            "Fold": str(i + 1),
            "Sharpe_Dynamic": "0.5",
            "Sharpe_Baseline": "0.4",
            "Sharpe_Delta": "0.1",
            "Return_Dynamic": "0.05",
            "Return_Baseline": "0.03",
            "Return_Delta": "0.02",
            "MaxDD_Dynamic": "-0.10",
            "MaxDD_Baseline": "-0.12",
            "MaxDD_Delta": "0.02",
            "dl_overlap_pct": pct,
            "dl_overlap_active": state == "active",
            "dl_overlap_state": state,
            "dl_overlap_window": "2021-01-01/2021-06-30",
        })
    return rows


class TestClassifyFoldsPerFoldOverlap(unittest.TestCase):
    """classify_folds_dl_state uses per-fold dl_overlap_pct when present."""

    def test_all_active_folds(self):
        rows = _make_fold_rows_with_overlap(["active"] * 5)
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=None)
        for r in result:
            self.assertTrue(r["dl_active"])
            self.assertIn(r["dl_state"], ("dl_active", "dl_transition_enter"))

    def test_all_missing_folds(self):
        rows = _make_fold_rows_with_overlap(["missing"] * 5)
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=None)
        for r in result:
            self.assertFalse(r["dl_active"])
            self.assertEqual(r["dl_state"], "dl_missing")

    def test_partial_folds_treated_as_active(self):
        rows = _make_fold_rows_with_overlap(["partial"] * 5)
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=None)
        for r in result:
            self.assertTrue(r["dl_active"])

    def test_transition_enter_assigned_on_missing_to_active_change(self):
        rows = _make_fold_rows_with_overlap(["missing", "missing", "missing", "active", "active"])
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=None)
        states = [r["dl_state"] for r in sorted(result, key=lambda x: int(x["Fold"]))]
        self.assertEqual(states[3], "dl_transition_enter")
        self.assertEqual(states[4], "dl_active")

    def test_transition_exit_assigned_on_active_to_missing_change(self):
        rows = _make_fold_rows_with_overlap(["active", "active", "missing", "missing"])
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=None)
        states = [r["dl_state"] for r in sorted(result, key=lambda x: int(x["Fold"]))]
        self.assertEqual(states[2], "dl_transition_exit")

    def test_overlap_pct_arg_ignored_when_per_fold_data_present(self):
        """When dl_overlap_pct column exists, overlap_fold_coverage_pct kwarg is ignored."""
        rows = _make_fold_rows_with_overlap(["active"] * 5)
        # With 0.0 coverage, positional heuristic would label all as missing.
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=0.0)
        for r in result:
            self.assertTrue(r["dl_active"], "Per-fold path should override positional heuristic")

    def test_fallback_to_positional_when_no_per_fold_column(self):
        """When dl_overlap_pct column absent, falls back to positional heuristic."""
        rows = _make_fold_rows(n=10)  # no dl_overlap_pct
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=40.0)
        active = [r for r in result if r["dl_active"]]
        self.assertEqual(len(active), 4)

    def test_preserves_overlap_columns_in_output(self):
        rows = _make_fold_rows_with_overlap(["active", "missing"])
        result = classify_folds_dl_state(rows, overlap_fold_coverage_pct=None)
        for r in result:
            self.assertIn("dl_overlap_pct", r)
            self.assertIn("dl_overlap_state", r)
            self.assertIn("dl_overlap_window", r)


class TestAnalyseRunWithPerFoldOverlap(unittest.TestCase):
    """_analyse_run derives overlap_pct from per-fold data when available."""

    def _make_summary(
        self,
        states: list[str],
        pair: str = "EURUSD",
    ) -> dict:
        rows = _make_fold_rows_with_overlap(states, pair=pair)
        return {
            "run_id": "test_run",
            "csvs": {
                "walkforward_per_fold": rows,
                "selector_state_timeline": None,
            },
            "coverage": {
                "overlap_window": {
                    "overlap_fold_coverage_pct": None,  # not set (simulating old pipeline)
                }
            },
        }

    def test_active_folds_non_empty(self):
        summary = self._make_summary(["missing"] * 6 + ["active"] * 4)
        result = build_dl_conditional_analysis([summary])
        per_run = result["per_run"][0]
        self.assertGreater(per_run["n_folds_dl_active"], 0)

    def test_missing_folds_non_empty(self):
        summary = self._make_summary(["missing"] * 6 + ["active"] * 4)
        result = build_dl_conditional_analysis([summary])
        per_run = result["per_run"][0]
        self.assertGreater(per_run["n_folds_dl_missing"], 0)

    def test_dl_state_assignment_method_is_per_fold(self):
        summary = self._make_summary(["active"] * 5)
        result = build_dl_conditional_analysis([summary])
        per_run = result["per_run"][0]
        self.assertEqual(
            per_run["dl_state_assignment_method"],
            "per_fold_timestamp_overlap",
        )

    def test_aggregate_method_per_fold_when_no_heuristic_run(self):
        summary = self._make_summary(["active"] * 5)
        result = build_dl_conditional_analysis([summary])
        self.assertEqual(
            result["metadata"]["dl_state_assignment_method"],
            "per_fold_timestamp_overlap",
        )

    def test_dl_active_folds_empty_when_all_missing(self):
        summary = self._make_summary(["missing"] * 10)
        result = build_dl_conditional_analysis([summary])
        per_run = result["per_run"][0]
        self.assertEqual(per_run["n_folds_dl_active"], 0)
        self.assertEqual(per_run["n_folds_dl_missing"], 10)


class TestOverlapWindowDiagnosticsWithPerFoldData(unittest.TestCase):
    """_build_overlap_window_diagnostics uses per-fold overlap when available."""

    def _make_fold_csv_content(self, states: list[str]) -> str:
        lines = [
            "Pair,Fold,Sharpe_Dynamic,Sharpe_Baseline,Sharpe_Delta,"
            "Return_Dynamic,Return_Baseline,Return_Delta,"
            "MaxDD_Dynamic,MaxDD_Baseline,MaxDD_Delta,"
            "dl_overlap_pct,dl_overlap_active,dl_overlap_state,dl_overlap_window"
        ]
        for i, state in enumerate(states):
            pct = _DL_STATE_TO_PCT[state]
            active_str = str(state == "active")
            lines.append(
                f"EURUSD,{i + 1},0.5,0.4,0.1,0.05,0.03,0.02,-0.10,-0.12,0.02,"
                f"{pct},{active_str},{state},2021-01-01/2021-06-30"
            )
        return "\n".join(lines) + "\n"

    def test_per_fold_path_used_when_column_present(self):
        import tempfile, json
        from analysis.parsers.csv_parsers import parse_run_csvs
        from analysis.pipeline import _build_overlap_window_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            content = self._make_fold_csv_content(["missing"] * 6 + ["active"] * 4)
            (run_dir / "walkforward_results_per_fold__dl_enabled.csv").write_text(content)
            csvs = parse_run_csvs(run_dir)
            diag = _build_overlap_window_diagnostics(csvs, log=None)
            self.assertEqual(diag["overlap_attribution_source"], "per_fold_timestamp_overlap")
            self.assertIsNotNone(diag["overlap_fold_coverage_pct"])
            # 4 active folds out of 10 → 40%
            self.assertAlmostEqual(diag["overlap_fold_coverage_pct"], 40.0, places=1)

    def test_active_and_partial_both_count(self):
        import tempfile
        from analysis.parsers.csv_parsers import parse_run_csvs
        from analysis.pipeline import _build_overlap_window_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            content = self._make_fold_csv_content(["missing"] * 5 + ["partial"] * 3 + ["active"] * 2)
            (run_dir / "walkforward_results_per_fold__dl_enabled.csv").write_text(content)
            csvs = parse_run_csvs(run_dir)
            diag = _build_overlap_window_diagnostics(csvs, log=None)
            # 5 (partial+active) out of 10 → 50%
            self.assertAlmostEqual(diag["overlap_fold_coverage_pct"], 50.0, places=1)
            self.assertEqual(diag["overlap_active_fold_count"], 2)
            self.assertEqual(diag["overlap_partial_fold_count"], 3)

    def test_all_missing_gives_zero_pct(self):
        import tempfile
        from analysis.parsers.csv_parsers import parse_run_csvs
        from analysis.pipeline import _build_overlap_window_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            content = self._make_fold_csv_content(["missing"] * 5)
            (run_dir / "walkforward_results_per_fold__dl_enabled.csv").write_text(content)
            csvs = parse_run_csvs(run_dir)
            diag = _build_overlap_window_diagnostics(csvs, log=None)
            self.assertAlmostEqual(diag["overlap_fold_coverage_pct"], 0.0, places=5)


class TestWalkforwardFoldCsvParser(unittest.TestCase):
    """_parse_walkforward_per_fold correctly handles dl_overlap_* columns."""

    def _write_fold_csv(self, run_dir: Path, with_overlap: bool = True) -> None:
        header = (
            "Pair,Fold,Sharpe_Dynamic,Sharpe_Baseline,Sharpe_Delta,"
            "Return_Dynamic,Return_Baseline,Return_Delta,"
            "MaxDD_Dynamic,MaxDD_Baseline,MaxDD_Delta"
        )
        row = "EURUSD,1,0.5,0.4,0.1,0.05,0.03,0.02,-0.10,-0.12,0.02"
        if with_overlap:
            header += ",dl_overlap_pct,dl_overlap_active,dl_overlap_state,dl_overlap_window"
            row += ",100.0,True,active,2021-01-01/2021-06-30"
        content = header + "\n" + row + "\n"
        (run_dir / "walkforward_results_per_fold__dl_enabled.csv").write_text(content)

    def test_dl_overlap_pct_parsed_as_float(self):
        import tempfile
        from analysis.parsers.csv_parsers import parse_run_csvs

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_fold_csv(run_dir, with_overlap=True)
            csvs = parse_run_csvs(run_dir)
            fold = csvs["walkforward_per_fold"][0]
            self.assertIsInstance(fold["dl_overlap_pct"], float)
            self.assertAlmostEqual(fold["dl_overlap_pct"], 100.0, places=5)

    def test_dl_overlap_active_parsed_as_bool(self):
        import tempfile
        from analysis.parsers.csv_parsers import parse_run_csvs

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_fold_csv(run_dir, with_overlap=True)
            csvs = parse_run_csvs(run_dir)
            fold = csvs["walkforward_per_fold"][0]
            self.assertIsInstance(fold["dl_overlap_active"], bool)
            self.assertTrue(fold["dl_overlap_active"])

    def test_dl_overlap_state_is_string(self):
        import tempfile
        from analysis.parsers.csv_parsers import parse_run_csvs

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_fold_csv(run_dir, with_overlap=True)
            csvs = parse_run_csvs(run_dir)
            fold = csvs["walkforward_per_fold"][0]
            self.assertEqual(fold["dl_overlap_state"], "active")
            self.assertEqual(fold["dl_overlap_window"], "2021-01-01/2021-06-30")

    def test_csv_without_overlap_columns_parsed_ok(self):
        """Backwards compatibility: CSVs without dl_overlap_* columns still parse."""
        import tempfile
        from analysis.parsers.csv_parsers import parse_run_csvs

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_fold_csv(run_dir, with_overlap=False)
            csvs = parse_run_csvs(run_dir)
            fold = csvs["walkforward_per_fold"][0]
            # dl_overlap_pct should be absent or None (not present in CSV)
            self.assertIsNone(fold.get("dl_overlap_pct"))


if __name__ == "__main__":
    unittest.main()
