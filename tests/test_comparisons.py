"""
tests/test_comparisons.py
===========================
Tests for analysis/comparisons/*.py

Run with:
    python -m unittest tests/test_comparisons.py -v
"""

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.comparisons.sentiment import compare_sentiment_variants
from analysis.comparisons.selector import compare_selector_uplift
from analysis.comparisons.gen_comparison import compare_gen1_gen2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_summary(
    run_id: str,
    dl_enabled: bool = True,
    experiment_gen: str = "gen1",
    run_variant: str | None = None,
    sharpe_delta: float | None = None,
    return_delta: float | None = None,
    maxdd_delta: float | None = None,
    pair: str = "EURUSD",
    dl_coverage: dict | None = None,
) -> dict:
    """Build a minimal summary dict for testing comparisons."""
    if run_variant is None:
        if experiment_gen == "gen1":
            run_variant = "A" if dl_enabled else "B"
        elif experiment_gen == "gen2":
            run_variant = "C" if dl_enabled else "D"
        else:
            run_variant = "U"

    wf_row = {"Pair": pair}
    if sharpe_delta is not None:
        wf_row["Sharpe_Delta"] = sharpe_delta
    if return_delta is not None:
        wf_row["Return_Delta"] = return_delta
    if maxdd_delta is not None:
        wf_row["MaxDD_Delta"] = maxdd_delta

    return {
        "run_id": run_id,
        "meta": {
            "dl_enabled": dl_enabled,
            "experiment_gen": experiment_gen,
            "run_variant": run_variant,
        },
        "csvs": {
            "walkforward_summary": [wf_row],
            "selector_comparison": None,
            "ml_accuracy": None,
            "backtest": None,
            "walkforward_per_pair": None,
            "walkforward_per_fold": None,
            "ablation_aggregate": None,
            "ablation_per_pair": None,
            "vol_guard_summary": None,
            "vol_guard_per_fold": None,
            "results_summary": None,
            "results_per_pair": None,
        },
        "log": {
            "dl_coverage": dl_coverage or {},
        },
        "coverage": {},
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Sentiment comparison tests
# ---------------------------------------------------------------------------


class TestCompareSentimentVariants(unittest.TestCase):

    def test_empty_returns_warnings(self):
        result = compare_sentiment_variants([])
        self.assertIn("warnings", result)
        self.assertTrue(len(result["warnings"]) > 0)

    def test_on_off_split(self):
        summaries = [
            _make_summary("run_on", dl_enabled=True),
            _make_summary("run_off", dl_enabled=False),
        ]
        result = compare_sentiment_variants(summaries)
        self.assertEqual(result["sentiment_on"], ["run_on"])
        self.assertEqual(result["sentiment_off"], ["run_off"])

    def test_delta_table_populated_with_matching_pairs(self):
        summaries = [
            _make_summary("run_on", dl_enabled=True, sharpe_delta=0.13),
            _make_summary("run_off", dl_enabled=False, sharpe_delta=0.05),
        ]
        result = compare_sentiment_variants(summaries)
        delta_rows = result["delta_table"]
        self.assertTrue(len(delta_rows) > 0)
        row = delta_rows[0]
        self.assertIn("delta_on_minus_off", row)
        self.assertAlmostEqual(row["delta_on_minus_off"], 0.08, places=5)

    def test_no_on_runs_warning(self):
        summaries = [_make_summary("run_off", dl_enabled=False)]
        result = compare_sentiment_variants(summaries)
        warnings = result["warnings"]
        self.assertTrue(any("missing variant" in w for w in warnings))

    def test_no_off_runs_warning(self):
        summaries = [_make_summary("run_on", dl_enabled=True)]
        result = compare_sentiment_variants(summaries)
        warnings = result["warnings"]
        self.assertTrue(any("missing variant" in w for w in warnings))

    def test_multiple_pairs(self):
        summaries = [
            _make_summary("run_on", dl_enabled=True, pair="EURUSD", sharpe_delta=0.10),
            _make_summary("run_off", dl_enabled=False, pair="EURUSD", sharpe_delta=0.02),
        ]
        # Add a second pair to run_on
        summaries[0]["csvs"]["walkforward_summary"].append(
            {"Pair": "GBPUSD", "Sharpe_Delta": 0.15}
        )
        result = compare_sentiment_variants(summaries)
        pairs_in_delta = {r["pair"] for r in result["delta_table"]}
        self.assertIn("EURUSD", pairs_in_delta)
        # GBPUSD only in ON runs; delta should be None
        gbp_row = next((r for r in result["delta_table"] if r["pair"] == "GBPUSD"), None)
        if gbp_row:
            self.assertIsNone(gbp_row["delta_on_minus_off"])

    def test_delta_table_empty_when_no_matching_pairs(self):
        """No shared pairs → all deltas should be None."""
        summaries = [
            _make_summary("run_on", dl_enabled=True, pair="EURUSD", sharpe_delta=0.10),
            _make_summary("run_off", dl_enabled=False, pair="GBPUSD", sharpe_delta=0.05),
        ]
        result = compare_sentiment_variants(summaries)
        for row in result["delta_table"]:
            self.assertIsNone(row["delta_on_minus_off"])

    def test_no_cross_generation_mixing(self):
        summaries = [
            _make_summary("gen1_on", dl_enabled=True, experiment_gen="gen1", sharpe_delta=0.10),
            _make_summary("gen2_off", dl_enabled=False, experiment_gen="gen2", sharpe_delta=0.05),
        ]
        result = compare_sentiment_variants(summaries)
        self.assertEqual(result["delta_table"], [])
        self.assertTrue(any("invalid" in w.lower() for w in result["warnings"]))


# ---------------------------------------------------------------------------
# Selector uplift tests
# ---------------------------------------------------------------------------


class TestCompareSelectorUplift(unittest.TestCase):

    def test_empty_has_warnings(self):
        result = compare_selector_uplift([])
        self.assertIn("warnings", result)
        self.assertTrue(len(result["warnings"]) > 0)

    def test_aggregate_populated(self):
        summaries = [
            _make_summary("run_a", sharpe_delta=0.20),
        ]
        result = compare_selector_uplift(summaries)
        agg = result["aggregate"]
        # Data comes from walkforward_summary as fallback
        self.assertIn("EURUSD", agg)

    def test_per_run_structure(self):
        summaries = [
            _make_summary("run_a"),
            _make_summary("run_b"),
        ]
        result = compare_selector_uplift(summaries)
        self.assertEqual(len(result["per_run"]), 2)
        run_ids = {pr["run_id"] for pr in result["per_run"]}
        self.assertIn("run_a", run_ids)
        self.assertIn("run_b", run_ids)

    def test_uses_selector_comparison_csv_when_available(self):
        summary = _make_summary("run_a", sharpe_delta=0.10)
        summary["csvs"]["selector_comparison"] = [
            {"Pair": "EURUSD", "Sharpe_Delta": 0.30, "Return_Delta": 5.0, "MaxDD_Delta": 2.0}
        ]
        result = compare_selector_uplift([summary])
        agg = result["aggregate"]
        self.assertAlmostEqual(agg["EURUSD"]["Sharpe_Delta"], 0.30)


# ---------------------------------------------------------------------------
# Gen1 vs Gen2 comparison tests
# ---------------------------------------------------------------------------


class TestCompareGen1Gen2(unittest.TestCase):

    def test_empty_has_warnings(self):
        result = compare_gen1_gen2([])
        self.assertIn("warnings", result)
        self.assertTrue(len(result["warnings"]) > 0)

    def test_gen_split(self):
        summaries = [
            _make_summary("run_g1", experiment_gen="gen1"),
            _make_summary("run_g2", experiment_gen="gen2"),
        ]
        result = compare_gen1_gen2(summaries)
        self.assertIn("run_g1", result["gen1"])
        self.assertIn("run_g2", result["gen2"])

    def test_delta_table(self):
        summaries = [
            _make_summary("run_g1", experiment_gen="gen1", sharpe_delta=0.10),
            _make_summary("run_g2", experiment_gen="gen2", sharpe_delta=0.20),
        ]
        result = compare_gen1_gen2(summaries)
        delta_rows = result["delta_table"]
        self.assertTrue(len(delta_rows) > 0)
        row = next((r for r in delta_rows if r["metric"] == "sharpe_delta"), None)
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row["delta_gen2_minus_gen1"], 0.10, places=5)

    def test_no_gen1_warning(self):
        summaries = [_make_summary("run_g2", experiment_gen="gen2")]
        result = compare_gen1_gen2(summaries)
        self.assertTrue(any("invalid" in w.lower() for w in result["warnings"]))

    def test_coverage_comparison(self):
        summaries = [
            _make_summary("run_g1", experiment_gen="gen1", dl_coverage={"EURUSD": 80.0}),
            _make_summary("run_g2", experiment_gen="gen2", dl_coverage={"EURUSD": 85.0}),
        ]
        result = compare_gen1_gen2(summaries)
        cov = result["coverage_comparison"]
        self.assertTrue(len(cov) >= 1)
        eurusd = next((r for r in cov if r["pair"] == "EURUSD"), None)
        self.assertIsNotNone(eurusd)
        self.assertAlmostEqual(eurusd["dl_coverage_gen1"], 80.0)
        self.assertAlmostEqual(eurusd["dl_coverage_gen2"], 85.0)

    def test_gen_comparison_respects_sentiment_cohorts(self):
        summaries = [
            _make_summary("run_g1_on", experiment_gen="gen1", dl_enabled=True, sharpe_delta=0.10),
            _make_summary("run_g2_off", experiment_gen="gen2", dl_enabled=False, sharpe_delta=0.30),
        ]
        result = compare_gen1_gen2(summaries)
        self.assertEqual(result["delta_table"], [])
        self.assertTrue(any("invalid" in w.lower() for w in result["warnings"]))


# ---------------------------------------------------------------------------
# Analysis matrix completeness tests
# ---------------------------------------------------------------------------


class TestAnalysisMatrixCompleteness(unittest.TestCase):
    """Verify that comparison matrix correctly identifies present/absent variants."""

    def _full_matrix(self) -> list[dict]:
        """Build a complete A/B/C/D summary set."""
        return [
            _make_summary("fp_gen1_A", experiment_gen="gen1", run_variant="A", sharpe_delta=0.12),
            _make_summary("fp_gen1_B", experiment_gen="gen1", run_variant="B", sharpe_delta=0.02),
            _make_summary("fp_gen2_C", experiment_gen="gen2", run_variant="C", sharpe_delta=0.15),
            _make_summary("fp_gen2_D", experiment_gen="gen2", run_variant="D", sharpe_delta=0.03),
        ]

    def test_full_matrix_has_no_warnings(self):
        """Complete A/B/C/D set must produce valid comparisons with no incomplete-cohort warnings."""
        result_s = compare_sentiment_variants(self._full_matrix())
        result_g = compare_gen1_gen2(self._full_matrix())
        self.assertEqual(result_s["incomplete_comparisons"], [])
        self.assertEqual(result_g["incomplete_comparisons"], [])

    def test_full_matrix_all_valid_comparisons(self):
        """Complete A/B/C/D set must produce all four expected valid comparisons."""
        result_s = compare_sentiment_variants(self._full_matrix())
        result_g = compare_gen1_gen2(self._full_matrix())
        self.assertIn("gen1:A_vs_B", result_s["valid_comparisons"])
        self.assertIn("gen2:C_vs_D", result_s["valid_comparisons"])
        self.assertIn("sentiment_on:A_vs_C", result_g["valid_comparisons"])
        self.assertIn("sentiment_off:B_vs_D", result_g["valid_comparisons"])

    def test_matrix_present_variants_complete(self):
        """Present variants must include A, B, C, D with a full set."""
        result_s = compare_sentiment_variants(self._full_matrix())
        self.assertEqual(sorted(result_s["matrix"]["present_variants"]), ["A", "B", "C", "D"])

    def test_partial_matrix_correctly_identifies_missing(self):
        """With only A and C runs, B and D must be reported as missing."""
        summaries = [
            _make_summary("fp_gen1_A", experiment_gen="gen1", run_variant="A"),
            _make_summary("fp_gen2_C", experiment_gen="gen2", run_variant="C"),
        ]
        result_s = compare_sentiment_variants(summaries)
        self.assertIn("gen1:A_vs_B", result_s["incomplete_comparisons"])
        self.assertIn("gen2:C_vs_D", result_s["incomplete_comparisons"])
        self.assertEqual(sorted(result_s["matrix"]["present_variants"]), ["A", "C"])

    def test_all_gen1_runs_not_collapsed_to_A(self):
        """Sentinel: when A and B runs are present, they must go into separate cohorts."""
        result_s = compare_sentiment_variants(self._full_matrix())
        grouped = result_s["grouped"]
        self.assertEqual(grouped["gen1"]["on"], ["fp_gen1_A"])
        self.assertEqual(grouped["gen1"]["off"], ["fp_gen1_B"])

    def test_all_gen2_runs_not_collapsed_to_C(self):
        """Sentinel: when C and D runs are present, they must go into separate cohorts."""
        result_s = compare_sentiment_variants(self._full_matrix())
        grouped = result_s["grouped"]
        self.assertEqual(grouped["gen2"]["on"], ["fp_gen2_C"])
        self.assertEqual(grouped["gen2"]["off"], ["fp_gen2_D"])

    def test_sentiment_delta_uses_correct_cohorts(self):
        """Sentiment delta must compare A vs B (not B vs B or A vs A)."""
        result_s = compare_sentiment_variants(self._full_matrix())
        # Verify cohort membership is correct: A in on, B in off
        self.assertIn("fp_gen1_A", result_s["grouped"]["gen1"]["on"])
        self.assertIn("fp_gen1_B", result_s["grouped"]["gen1"]["off"])
        self.assertNotIn("fp_gen1_B", result_s["grouped"]["gen1"]["on"])
        self.assertNotIn("fp_gen1_A", result_s["grouped"]["gen1"]["off"])
        # Verify delta is A minus B (0.12 - 0.02 = 0.10)
        gen1_rows = [r for r in result_s["delta_table"] if r["generation"] == "gen1"]
        self.assertTrue(len(gen1_rows) > 0)
        for row in gen1_rows:
            # A=0.12 Sharpe_Delta, B=0.02 Sharpe_Delta → delta = 0.10
            self.assertAlmostEqual(row["delta_on_minus_off"], 0.10, places=5)


if __name__ == "__main__":
    unittest.main()
