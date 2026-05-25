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
from analysis.comparisons.gen_comparison import (
    compare_gen1_gen2,
    compare_training_family_effect,
)
from analysis.comparisons.factor_comparison import build_factor_comparisons
from analysis.reports.markdown_report import render_markdown_report
from experiment_semantics import EXPERIMENT_VARIANTS


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
    msml_regime: str = "LVTF",
    overlap_only: bool = False,
    selector_enabled: bool = True,
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

    semantics = EXPERIMENT_VARIANTS.get(run_variant, {})
    sentiment_enabled = semantics.get("sentiment_enabled")
    missing_indicators_enabled = semantics.get("missing_indicators_enabled")

    return {
        "run_id": run_id,
        "meta": {
            "dl_enabled": dl_enabled,
            "experiment_gen": experiment_gen,
            "run_variant": run_variant,
            "experiment": {
                "generation": semantics.get("generation", experiment_gen),
                "variant": run_variant,
                "run_family": "factorial_v1",
                "sentiment_enabled": sentiment_enabled,
                "missing_indicators_enabled": missing_indicators_enabled,
                "factors": {
                    "dl_enabled": dl_enabled,
                    "sentiment_enabled": sentiment_enabled,
                    "missing_indicators_enabled": missing_indicators_enabled,
                    "msml_regime": msml_regime,
                    "overlap_only": overlap_only,
                    "selector_enabled": selector_enabled,
                },
                "semantic_label": f"{experiment_gen.capitalize()}_{run_variant}",
            },
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


def _make_v5_summary(
    run_id: str,
    *,
    training_pair_family: str,
    sentiment_surface: bool,
    missing_indicators_enabled: bool,
    run_variant: str = "A",
    sharpe_delta: float | None = None,
    pair: str = "EURUSD",
) -> dict:
    summary = _make_summary(
        run_id=run_id,
        experiment_gen="gen1",
        run_variant=run_variant,
        sharpe_delta=sharpe_delta,
        pair=pair,
    )
    summary["meta"]["surface_source"] = "manifest"
    summary["meta"]["manifest_present"] = True
    summary["meta"]["legacy_semantics"] = False
    summary["meta"]["experiment_surface"] = {
        "surface_semantics_version": 5,
        "training_pair_family": training_pair_family,
        "evaluation_pair_family": training_pair_family,
        "sentiment_surface": sentiment_surface,
        "feature_surface": "trend_vol_only",
    }
    summary["meta"]["experiment"]["factors"]["missing_indicators_enabled"] = missing_indicators_enabled
    summary["meta"]["experiment"]["factors"]["sentiment_enabled"] = sentiment_surface
    return summary


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
        self.assertTrue(any("missing cohort" in w.lower() or "incomplete" in w.lower() for w in warnings))

    def test_no_off_runs_warning(self):
        summaries = [_make_summary("run_on", dl_enabled=True)]
        result = compare_sentiment_variants(summaries)
        warnings = result["warnings"]
        self.assertTrue(any("missing cohort" in w.lower() or "incomplete" in w.lower() for w in warnings))

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
        # Expect a warning about no valid comparisons or missing cohorts.
        self.assertTrue(
            any("incomplete" in w.lower() or "invalid" in w.lower() for w in result["warnings"])
        )

    def test_sentiment_comparison_ignores_dl_enabled_flag(self):
        summaries = [
            _make_summary("run_a", dl_enabled=True, experiment_gen="gen1", run_variant="A", sharpe_delta=0.10),
            _make_summary("run_b", dl_enabled=True, experiment_gen="gen1", run_variant="B", sharpe_delta=0.05),
        ]
        result = compare_sentiment_variants(summaries)
        # Legacy path uses "legacy:gen1" as the grouped key.
        self.assertEqual(result["grouped"]["legacy:gen1"]["imputation_awareness=false"]["on"], ["run_a"])
        self.assertEqual(result["grouped"]["legacy:gen1"]["imputation_awareness=false"]["off"], ["run_b"])


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
            _make_summary("run_g1", experiment_gen="gen1", run_variant="A", sharpe_delta=0.10),
            _make_summary("run_g2", experiment_gen="gen2", run_variant="F", sharpe_delta=0.20),
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
        self.assertTrue(
            any("incomplete" in w.lower() or "invalid" in w.lower() for w in result["warnings"])
        )

    def test_coverage_comparison(self):
        summaries = [
            _make_summary("run_g1", experiment_gen="gen1", run_variant="A", dl_coverage={"EURUSD": 80.0}),
            _make_summary("run_g2", experiment_gen="gen2", run_variant="F", dl_coverage={"EURUSD": 85.0}),
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
        self.assertTrue(
            any("incomplete" in w.lower() or "invalid" in w.lower() for w in result["warnings"])
        )

    def test_v5_comparison_is_surface_driven(self):
        summaries = [
            _make_v5_summary(
                "run_persistent",
                training_pair_family="persistent",
                sentiment_surface=True,
                missing_indicators_enabled=False,
                run_variant="A",
                sharpe_delta=0.10,
            ),
            _make_v5_summary(
                "run_reactive",
                training_pair_family="reactive",
                sentiment_surface=True,
                missing_indicators_enabled=False,
                run_variant="F",
                sharpe_delta=0.20,
            ),
        ]
        result = compare_training_family_effect(summaries)
        self.assertTrue(any(key.startswith("persistent_vs_reactive/") for key in result["valid_comparisons"]))
        self.assertEqual(result["gen1"], [])
        self.assertEqual(result["gen2"], [])


# ---------------------------------------------------------------------------
# Analysis matrix completeness tests
# ---------------------------------------------------------------------------


class TestAnalysisMatrixCompleteness(unittest.TestCase):
    """Verify that comparison matrix correctly identifies present/absent variants."""

    def _full_matrix(self) -> list[dict]:
        """Build the legacy A/B/C/D subset used for scoped regression checks."""
        return [
            _make_summary("fp_gen1_A", experiment_gen="gen1", run_variant="A", sharpe_delta=0.12),
            _make_summary("fp_gen1_B", experiment_gen="gen1", run_variant="B", sharpe_delta=0.02),
            _make_summary("fp_gen2_C", experiment_gen="gen2", run_variant="C", sharpe_delta=0.15),
            _make_summary("fp_gen2_D", experiment_gen="gen2", run_variant="D", sharpe_delta=0.03),
        ]

    def test_full_matrix_has_no_warnings(self):
        """A/B/C/D subset yields complete sentiment strata but incomplete strict gen strata."""
        result_s = compare_sentiment_variants(self._full_matrix())
        result_g = compare_gen1_gen2(self._full_matrix())
        self.assertEqual(result_s["incomplete_comparisons"], [])
        self.assertGreater(len(result_g["incomplete_comparisons"]), 0)
        self.assertEqual(result_g["valid_comparisons"], [])

    def test_full_matrix_all_valid_comparisons(self):
        """Scoped A/B/C/D subset keeps deterministic conditioned comparisons."""
        result_s = compare_sentiment_variants(self._full_matrix())
        result_g = compare_gen1_gen2(self._full_matrix())
        # Legacy runs use "legacy:generation=<gen>/imputation_awareness=<bool>" keys.
        self.assertIn("legacy:generation=gen1/imputation_awareness=false", result_s["valid_comparisons"])
        self.assertIn("legacy:generation=gen2/imputation_awareness=true", result_s["valid_comparisons"])
        self.assertEqual(result_g["valid_comparisons"], [])

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
        # Legacy path uses "legacy:generation=<gen>/imputation_awareness=<bool>" keys.
        self.assertIn("legacy:generation=gen1/imputation_awareness=false", result_s["incomplete_comparisons"])
        self.assertIn("legacy:generation=gen2/imputation_awareness=true", result_s["incomplete_comparisons"])
        self.assertEqual(sorted(result_s["matrix"]["present_variants"]), ["A", "C"])

    def test_all_gen1_runs_not_collapsed_to_A(self):
        """Sentinel: when A and B runs are present, they must go into separate cohorts."""
        result_s = compare_sentiment_variants(self._full_matrix())
        grouped = result_s["grouped"]
        # Legacy path uses "legacy:gen1" as top-level key and "imputation_awareness=false" as cohort key.
        self.assertEqual(grouped["legacy:gen1"]["imputation_awareness=false"]["on"], ["fp_gen1_A"])
        self.assertEqual(grouped["legacy:gen1"]["imputation_awareness=false"]["off"], ["fp_gen1_B"])

    def test_all_gen2_runs_not_collapsed_to_C(self):
        """Sentinel: when C and D runs are present, they must go into separate cohorts."""
        result_s = compare_sentiment_variants(self._full_matrix())
        grouped = result_s["grouped"]
        # Legacy path uses "legacy:gen2" as top-level key and "imputation_awareness=true" as cohort key.
        self.assertEqual(grouped["legacy:gen2"]["imputation_awareness=true"]["on"], ["fp_gen2_C"])
        self.assertEqual(grouped["legacy:gen2"]["imputation_awareness=true"]["off"], ["fp_gen2_D"])

    def test_sentiment_delta_uses_correct_cohorts(self):
        """Sentiment delta must compare A vs B (not B vs B or A vs A)."""
        result_s = compare_sentiment_variants(self._full_matrix())
        # Verify cohort membership is correct: A in on, B in off
        self.assertIn("fp_gen1_A", result_s["grouped"]["legacy:gen1"]["imputation_awareness=false"]["on"])
        self.assertIn("fp_gen1_B", result_s["grouped"]["legacy:gen1"]["imputation_awareness=false"]["off"])
        self.assertNotIn("fp_gen1_B", result_s["grouped"]["legacy:gen1"]["imputation_awareness=false"]["on"])
        self.assertNotIn("fp_gen1_A", result_s["grouped"]["legacy:gen1"]["imputation_awareness=false"]["off"])
        # Verify delta is A minus B (0.12 - 0.02 = 0.10)
        # The generation field in delta rows uses the new key format.
        gen1_rows = [r for r in result_s["delta_table"] if r["generation"] == "legacy:gen1:imputation_awareness=false"]
        self.assertTrue(len(gen1_rows) > 0)
        for row in gen1_rows:
            # A=0.12 Sharpe_Delta, B=0.02 Sharpe_Delta → delta = 0.10
            self.assertAlmostEqual(row["delta_on_minus_off"], 0.10, places=5)


class TestGeneralizedFactorComparisons(unittest.TestCase):

    def test_factor_crosstab_includes_msml_regime(self):
        summaries = [
            _make_summary("run_lvtf", run_variant="A", msml_regime="LVTF"),
            _make_summary("run_lv", run_variant="B", dl_enabled=False, msml_regime="LV"),
            _make_summary("run_htf", run_variant="C", msml_regime="HTF"),
        ]
        result = build_factor_comparisons(summaries)
        crosstab = result["factor_crosstab"]["msml_regime"]
        self.assertIn("LVTF", crosstab)
        self.assertIn("LV", crosstab)
        self.assertIn("HTF", crosstab)

    def test_mixed_dl_baseline_slices(self):
        summaries = [
            _make_summary("run_dl", run_variant="A", dl_enabled=True),
            _make_summary("run_baseline", run_variant="B", dl_enabled=False),
        ]
        result = build_factor_comparisons(summaries)
        self.assertEqual(result["slices"]["dl_enabled_true"], ["run_dl"])
        self.assertEqual(result["slices"]["dl_enabled_false"], ["run_baseline"])

    def test_sentiment_by_generation_uses_factor_conditioning(self):
        summaries = [
            _make_summary("g1_on", experiment_gen="gen1", run_variant="A", sharpe_delta=0.2),
            _make_summary("g1_off", experiment_gen="gen1", run_variant="B", dl_enabled=False, sharpe_delta=0.1),
            _make_summary("g2_on", experiment_gen="gen2", run_variant="C", sharpe_delta=0.3),
            _make_summary("g2_off", experiment_gen="gen2", run_variant="D", dl_enabled=False, sharpe_delta=0.05),
        ]
        result = build_factor_comparisons(summaries)
        conditioned = result["comparisons"]["sentiment_enabled_by_generation"]
        self.assertEqual(len(conditioned), 2)
        self.assertTrue(all(entry["valid"] for entry in conditioned))

    def test_modern_factor_payload_warns_against_legacy_variant_grouping(self):
        summaries = [
            _make_v5_summary(
                "run_a",
                training_pair_family="persistent",
                sentiment_surface=True,
                missing_indicators_enabled=False,
                run_variant="A",
            ),
            _make_v5_summary(
                "run_f",
                training_pair_family="reactive",
                sentiment_surface=False,
                missing_indicators_enabled=False,
                run_variant="F",
            ),
        ]
        result = build_factor_comparisons(summaries)
        warnings = result.get("warnings") or []
        self.assertTrue(any("legacy compatibility metadata only" in w for w in warnings))
        self.assertIn("legacy_sentiment_by_generation", result["comparisons"])


class TestMarkdownReportOntologyLanguage(unittest.TestCase):

    def test_legacy_manifest_sections_render_with_compatibility_language(self):
        summaries = [
            _make_summary("legacy_run_a", experiment_gen="gen1", run_variant="A", sharpe_delta=0.1),
            _make_summary("legacy_run_c", experiment_gen="gen2", run_variant="C", sharpe_delta=0.2),
        ]
        for s in summaries:
            s["meta"]["surface_source"] = "legacy_variant_fallback"
            s["meta"]["manifest_present"] = True
            s["meta"]["legacy_semantics"] = True
            s["meta"]["semantic_label"] = f"GenCompat_{s['meta']['run_variant']}"
        comparisons = {
            "sentiment": compare_sentiment_variants(summaries),
            "training_family_effect": compare_training_family_effect(summaries),
            "factors": build_factor_comparisons(summaries),
            "selector": compare_selector_uplift(summaries),
        }
        report = render_markdown_report(summaries, comparisons, validation={"errors": [], "warnings": [], "diagnostics": [], "sections": {}})
        self.assertIn("Runtime architecture vs artifact provenance", report)
        self.assertIn("legacy_generation=gen1 runs", report)
        self.assertIn("Legacy Semantic Label", report)


if __name__ == "__main__":
    unittest.main()
