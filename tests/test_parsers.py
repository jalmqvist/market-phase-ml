"""
tests/test_parsers.py
======================
Tests for analysis/parsers/*.py

Tests use only the Python standard library (no pytest dependency).
Run with:
    python -m unittest tests/test_parsers.py  -v
    # or
    python tests/test_parsers.py
"""

import json
import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.parsers.csv_parsers import parse_run_csvs, _extract_mode_tag
from analysis.parsers.manifest_parser import parse_manifest
from analysis.parsers.log_parser import parse_log
from analysis.parsers.run_discovery import discover_runs


# ---------------------------------------------------------------------------
# Helpers: temporary directory builder
# ---------------------------------------------------------------------------

import tempfile
import os


def _make_run_dir(files: dict[str, str]) -> Path:
    """
    Create a temporary directory with the given filename → content mapping.
    The caller is responsible for cleanup (use as context manager or with
    tempfile.TemporaryDirectory).
    """
    tmp = tempfile.mkdtemp()
    run_dir = Path(tmp)
    for fname, content in files.items():
        (run_dir / fname).write_text(content)
    return run_dir


def _rmtree(path: Path) -> None:
    import shutil
    shutil.rmtree(str(path), ignore_errors=True)


# ---------------------------------------------------------------------------
# CSV parser tests
# ---------------------------------------------------------------------------


class TestExtractModeTag(unittest.TestCase):

    def test_dl_enabled(self):
        self.assertEqual(_extract_mode_tag("results_ml__dl_enabled.csv"), "__dl_enabled")

    def test_baseline(self):
        self.assertEqual(_extract_mode_tag("ablation_summary_aggregate__baseline.csv"), "__baseline")

    def test_no_tag(self):
        self.assertEqual(_extract_mode_tag("results_ml.csv"), "")

    def test_unknown_suffix(self):
        self.assertEqual(_extract_mode_tag("results_ml__foobar.csv"), "__foobar")


class TestParseRunCsvs(unittest.TestCase):

    def setUp(self):
        # Minimal CSV content matching what main.py writes
        ml_csv = (
            "Model,Accuracy,Std,N Samples,Pair\n"
            "Baseline (No Phases),0.57,0.04,5000,EURUSD\n"
            "Phase as Feature,0.58,0.05,5000,EURUSD\n"
        )
        bt_csv = (
            "Pair,Strategy,Total Return (%),Sharpe Ratio,Max Drawdown (%),"
            "Win Rate (%),Profit Factor,Total Trades\n"
            "EURUSD,PhaseAware,28.07,0.228,-20.01,60.29,1.11,549\n"
        )
        self.run_dir = _make_run_dir({
            "results_ml__dl_enabled.csv": ml_csv,
            "results_ml_backtest__dl_enabled.csv": bt_csv,
        })

    def tearDown(self):
        _rmtree(self.run_dir)

    def test_ml_accuracy_parsed(self):
        result = parse_run_csvs(self.run_dir)
        rows = result["ml_accuracy"]
        self.assertIsNotNone(rows)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["Accuracy"], 0.57)
        self.assertEqual(rows[0]["_mode_tag"], "__dl_enabled")

    def test_backtest_parsed(self):
        result = parse_run_csvs(self.run_dir)
        rows = result["backtest"]
        self.assertIsNotNone(rows)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Pair"], "EURUSD")
        self.assertAlmostEqual(rows[0]["Total Return (%)"], 28.07)

    def test_missing_sections_are_none(self):
        result = parse_run_csvs(self.run_dir)
        # These were not written so should be None
        self.assertIsNone(result["walkforward_summary"])
        self.assertIsNone(result["ablation_aggregate"])
        self.assertIsNone(result["selector_comparison"])

    def test_no_errors(self):
        result = parse_run_csvs(self.run_dir)
        self.assertEqual(result["_errors"], [])

    def test_empty_directory(self):
        empty_dir = _make_run_dir({})
        try:
            result = parse_run_csvs(empty_dir)
            self.assertIsNone(result["ml_accuracy"])
            self.assertEqual(result["_files_found"], [])
        finally:
            _rmtree(empty_dir)

    def test_malformed_csv_does_not_raise(self):
        """A file that matches the pattern but has bad/minimal content must not raise.

        Uses mismatched/unexpected column names (a common real-world malformation)
        rather than null bytes, since the csv module silently skips null bytes.
        """
        run_dir = _make_run_dir({
            "walkforward_results_summary__dl_enabled.csv": "WRONG_COL_A,WRONG_COL_B\nfoo,bar\n",
        })
        try:
            # Must not raise regardless of content.
            result = parse_run_csvs(run_dir)
            # Either returns rows (with unrecognised columns) or None — both fine.
            errors = result.get("_errors", [])
            wf = result.get("walkforward_summary")
            self.assertTrue(wf is None or isinstance(wf, list) or len(errors) > 0)
        finally:
            _rmtree(run_dir)

    def test_both_mode_tags_collected(self):
        """Both __baseline and __dl_enabled variants should be merged."""
        ml_csv_base = (
            "Model,Accuracy,Std,N Samples,Pair\n"
            "Baseline (No Phases),0.55,0.03,4000,GBPUSD\n"
        )
        run_dir = _make_run_dir({
            "results_ml__dl_enabled.csv": (
                "Model,Accuracy,Std,N Samples,Pair\n"
                "Baseline (No Phases),0.57,0.04,5000,EURUSD\n"
            ),
            "results_ml__baseline.csv": ml_csv_base,
        })
        try:
            result = parse_run_csvs(run_dir)
            rows = result["ml_accuracy"]
            self.assertIsNotNone(rows)
            self.assertEqual(len(rows), 2)
            mode_tags = {r["_mode_tag"] for r in rows}
            self.assertIn("__dl_enabled", mode_tags)
            self.assertIn("__baseline", mode_tags)
        finally:
            _rmtree(run_dir)


# ---------------------------------------------------------------------------
# Manifest parser tests
# ---------------------------------------------------------------------------


class TestParseManifest(unittest.TestCase):

    def _make_manifest(self, extra: dict | None = None) -> dict:
        base = {
            "dl": {
                "dl_enabled": True,
                "dl_mode_tag": "__dl_enabled",
                "dl_surface": {"model": "mlp", "dl_regime": "LVTF"},
                "dl_surface_string": "mlp/LVTF/h24/price_trend",
                "dl_artifact_path": "/some/path.parquet",
            },
            "run": {
                "run_id": "run_20260512T064904Z",
                "git_sha": "abc123",
                "timestamp_utc": "20260512T064904Z",
                "python_version": "3.12.3",
            },
            "walkforward": {
                "train_years": 7,
                "test_months": 6,
            },
            "flags": {
                "DL_SIGNALS_ENABLED": True,
                "RUN_WALKFORWARD": True,
            },
        }
        if extra:
            base.update(extra)
        return base

    def setUp(self):
        manifest_data = self._make_manifest()
        self.run_dir = _make_run_dir({
            "run_manifest_run_20260512T064904Z__dl_enabled.json": json.dumps(manifest_data),
        })

    def tearDown(self):
        _rmtree(self.run_dir)

    def test_run_id_extracted(self):
        result = parse_manifest(self.run_dir)
        self.assertIsNotNone(result)
        self.assertEqual(result["run_id"], "run_20260512T064904Z")

    def test_dl_enabled_true(self):
        result = parse_manifest(self.run_dir)
        self.assertTrue(result["dl_enabled"])

    def test_walkforward_params(self):
        result = parse_manifest(self.run_dir)
        self.assertEqual(result["walkforward"]["train_years"], 7)

    def test_no_manifest_returns_none(self):
        empty_dir = _make_run_dir({})
        try:
            self.assertIsNone(parse_manifest(empty_dir))
        finally:
            _rmtree(empty_dir)

    def test_prefers_dl_enabled_manifest(self):
        """When both __baseline and __dl_enabled manifests exist, prefer DL."""
        base_manifest = self._make_manifest()
        base_manifest["dl"]["dl_enabled"] = False
        base_manifest["dl"]["dl_mode_tag"] = "__baseline"

        dl_manifest = self._make_manifest()

        run_dir = _make_run_dir({
            "run_manifest_run_1__baseline.json": json.dumps(base_manifest),
            "run_manifest_run_2__dl_enabled.json": json.dumps(dl_manifest),
        })
        try:
            result = parse_manifest(run_dir)
            self.assertTrue(result["dl_enabled"])
        finally:
            _rmtree(run_dir)

    def test_corrupt_manifest_does_not_raise(self):
        run_dir = _make_run_dir({
            "run_manifest_bad__dl_enabled.json": "NOT JSON {{{",
        })
        try:
            result = parse_manifest(run_dir)
            # Should return something; the corrupt file is logged as an error.
            self.assertIsNotNone(result)
            primary = result["primary"]
            self.assertIn("_parse_error", primary)
        finally:
            _rmtree(run_dir)


# ---------------------------------------------------------------------------
# Log parser tests
# ---------------------------------------------------------------------------


class TestParseLog(unittest.TestCase):

    _SAMPLE_LOG = """
--- EURUSD ---
Baseline (No Phases) 0.5736 0.0468 5154
Phase as Feature 0.5692 0.0535 5154
[DL] EURUSD: DL coverage (any col)=94.50%
ML Backtest Results Summary (dl_enabled)
Pair  Total Return (%)  Sharpe  Max DD (%)  Win Rate (%)  N_Trades
---
EURUSD 28.07 0.228 -20.01 60.29 549
[4/5] Next section
"""

    def setUp(self):
        self.run_dir = _make_run_dir({
            "run_log.txt": self._SAMPLE_LOG,
        })

    def tearDown(self):
        _rmtree(self.run_dir)

    def test_dl_coverage_parsed(self):
        result = parse_log(self.run_dir)
        self.assertIsNotNone(result)
        self.assertIn("EURUSD", result["dl_coverage"])
        self.assertAlmostEqual(result["dl_coverage"]["EURUSD"], 94.50)

    def test_ml_results_parsed(self):
        result = parse_log(self.run_dir)
        self.assertEqual(len(result["ml_results"]), 2)
        self.assertEqual(result["ml_results"][0]["pair"], "EURUSD")

    def test_backtest_parsed(self):
        result = parse_log(self.run_dir)
        bt = result["backtests"]
        self.assertEqual(len(bt), 1)
        self.assertEqual(bt[0]["pair"], "EURUSD")
        self.assertAlmostEqual(bt[0]["total_return_pct"], 28.07)

    def test_no_log_returns_none(self):
        empty_dir = _make_run_dir({})
        try:
            self.assertIsNone(parse_log(empty_dir))
        finally:
            _rmtree(empty_dir)


# ---------------------------------------------------------------------------
# Run discovery tests
# ---------------------------------------------------------------------------


class TestDiscoverRuns(unittest.TestCase):

    def test_single_run_dir(self):
        run_dir = _make_run_dir({
            "run_manifest_test__baseline.json": "{}",
        })
        try:
            found = list(discover_runs(run_dir))
            self.assertGreaterEqual(len(found), 1)
            paths = [f[0] for f in found]
            self.assertIn(run_dir, paths)
        finally:
            _rmtree(run_dir)

    def test_archive_with_nested_runs(self):
        import os
        tmp = tempfile.mkdtemp()
        archive = Path(tmp)
        # Create two nested run directories
        run_a = archive / "run_a"
        run_b = archive / "run_b"
        run_a.mkdir()
        run_b.mkdir()
        (run_a / "results_ml__baseline.csv").write_text("Model,Accuracy\n")
        (run_b / "run_manifest_x__dl_enabled.json").write_text("{}")
        try:
            found = list(discover_runs(archive))
            dirs = {f[0] for f in found}
            self.assertIn(run_a, dirs)
            self.assertIn(run_b, dirs)
        finally:
            _rmtree(archive)

    def test_nonexistent_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            list(discover_runs(Path("/nonexistent/path/xyz")))

    def test_gen2_inferred_from_name(self):
        run_dir = _make_run_dir({
            "run_manifest_gen2__dl_enabled.json": "{}",
        })
        try:
            found = list(discover_runs(run_dir))
            gens = {f[1] for f in found}
            self.assertIn("gen2", gens)
        finally:
            _rmtree(run_dir)

    def test_gen1_default(self):
        run_dir = _make_run_dir({
            "run_manifest_run_abc__baseline.json": "{}",
        })
        try:
            found = list(discover_runs(run_dir))
            # The root is yielded; should be gen1
            gens = {f[1] for f in found}
            self.assertIn("gen1", gens)
        finally:
            _rmtree(run_dir)


# ---------------------------------------------------------------------------
# Partial-run handling tests
# ---------------------------------------------------------------------------


class TestPartialRunHandling(unittest.TestCase):
    """
    Ensure the pipeline handles partial runs gracefully — i.e., run
    directories that are missing some expected CSV files.
    """

    def test_run_with_only_manifest(self):
        """A directory with only a manifest should not crash."""
        run_dir = _make_run_dir({
            "run_manifest_partial__baseline.json": json.dumps({
                "run": {"run_id": "partial_run"},
                "dl": {"dl_enabled": False, "dl_mode_tag": "__baseline"},
            }),
        })
        try:
            csvs = parse_run_csvs(run_dir)
            manifest = parse_manifest(run_dir)
            log = parse_log(run_dir)
            self.assertIsNotNone(manifest)
            self.assertIsNone(log)
            self.assertIsNone(csvs["walkforward_summary"])
            self.assertIsNone(csvs["ablation_aggregate"])
        finally:
            _rmtree(run_dir)

    def test_run_with_only_log(self):
        """A directory with only a log file should parse via log fallback."""
        log_content = (
            "--- EURUSD ---\n"
            "[DL] EURUSD: DL coverage (any col)=80.00%\n"
        )
        run_dir = _make_run_dir({"results_run.txt": log_content})
        try:
            log = parse_log(run_dir)
            self.assertIsNotNone(log)
            self.assertAlmostEqual(log["dl_coverage"]["EURUSD"], 80.0)
        finally:
            _rmtree(run_dir)

    def test_run_with_walkforward_but_no_ablation(self):
        """Partial run: walkforward CSV present, ablation absent."""
        wf_csv = (
            "Pair,Sharpe_Dynamic,Sharpe_Baseline,Sharpe_Delta,N_Folds\n"
            "EURUSD,0.35,0.22,0.13,48\n"
        )
        run_dir = _make_run_dir({
            "walkforward_results_summary__dl_enabled.csv": wf_csv,
        })
        try:
            csvs = parse_run_csvs(run_dir)
            self.assertIsNotNone(csvs["walkforward_summary"])
            self.assertIsNone(csvs["ablation_aggregate"])
            self.assertEqual(len(csvs["walkforward_summary"]), 1)
        finally:
            _rmtree(run_dir)


# ---------------------------------------------------------------------------
# Pipeline integration test
# ---------------------------------------------------------------------------


class TestPipelineIntegration(unittest.TestCase):
    """End-to-end: run pipeline on a synthetic archive and check outputs."""

    def setUp(self):
        import tempfile, shutil

        self.archive = Path(tempfile.mkdtemp())
        self.output_dir = Path(tempfile.mkdtemp())

        # Two synthetic run directories
        for run_name, dl_enabled in [("run_baseline", False), ("run_dl", True)]:
            run_dir = self.archive / run_name
            run_dir.mkdir()
            manifest = {
                "run": {"run_id": run_name, "git_sha": "abc123", "timestamp_utc": "T"},
                "dl": {
                    "dl_enabled": dl_enabled,
                    "dl_mode_tag": "__dl_enabled" if dl_enabled else "__baseline",
                },
                "walkforward": {"train_years": 7, "test_months": 6},
                "flags": {},
            }
            (run_dir / f"run_manifest_{run_name}__{'dl_enabled' if dl_enabled else 'baseline'}.json").write_text(
                json.dumps(manifest)
            )
            (run_dir / f"results_ml__{'dl_enabled' if dl_enabled else 'baseline'}.csv").write_text(
                "Model,Accuracy,Std,N Samples,Pair\n"
                "Baseline (No Phases),0.57,0.04,5000,EURUSD\n"
            )
            (run_dir / f"walkforward_results_summary__{'dl_enabled' if dl_enabled else 'baseline'}.csv").write_text(
                "Pair,Sharpe_Dynamic,Sharpe_Baseline,Sharpe_Delta,N_Folds\n"
                "EURUSD,0.35,0.22,0.13,48\n"
            )

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self.archive), ignore_errors=True)
        shutil.rmtree(str(self.output_dir), ignore_errors=True)

    def test_pipeline_produces_report(self):
        from analysis.pipeline import run_pipeline
        run_pipeline(self.archive, self.output_dir, verbose=False)

        report = self.output_dir / "report.md"
        self.assertTrue(report.exists(), "report.md should be created")
        content = report.read_text()
        self.assertIn("MPML Experiment Report", content)

    def test_pipeline_produces_summaries(self):
        from analysis.pipeline import run_pipeline
        run_pipeline(self.archive, self.output_dir, verbose=False)

        summaries_dir = self.output_dir / "summaries"
        self.assertTrue(summaries_dir.exists())
        json_files = list(summaries_dir.glob("*.summary.json"))
        self.assertGreaterEqual(len(json_files), 2)

    def test_pipeline_produces_comparisons(self):
        from analysis.pipeline import run_pipeline
        run_pipeline(self.archive, self.output_dir, verbose=False)

        comp_path = self.output_dir / "comparisons.json"
        self.assertTrue(comp_path.exists())
        comp = json.loads(comp_path.read_text())
        self.assertIn("sentiment", comp)
        self.assertIn("selector", comp)
        self.assertIn("gen", comp)

    def test_pipeline_sentiment_comparison(self):
        from analysis.pipeline import run_pipeline
        run_pipeline(self.archive, self.output_dir, verbose=False)

        comp = json.loads((self.output_dir / "comparisons.json").read_text())
        sentiment = comp["sentiment"]
        # We created one DL-enabled and one baseline run
        self.assertEqual(len(sentiment["sentiment_on"]), 1)
        self.assertEqual(len(sentiment["sentiment_off"]), 1)


if __name__ == "__main__":
    unittest.main()
