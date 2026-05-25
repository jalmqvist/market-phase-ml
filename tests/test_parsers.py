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
from analysis.parsers.run_identity import infer_run_identity
from analysis.validation import validate_summaries, sort_summaries_deterministically
from experiment_semantics import CURRENT_EXPERIMENT_SEMANTICS_VERSION
from experiment_semantics import EXPERIMENT_VARIANTS
from experiment_semantics import build_experiment_metadata_from_variant


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
            "experiment": {
                "run_family": "factorial_v1",
                "generation": "gen1",
                "variant": "A",
                "sentiment_enabled": True,
                "missing_indicators_enabled": False,
                "factors": {
                    "dl_enabled": True,
                    "sentiment_enabled": True,
                    "missing_indicators_enabled": False,
                    "msml_regime": "LVTF",
                    "overlap_only": False,
                    "selector_enabled": True,
                },
                "semantic_label": "Gen1_A",
                "legacy_semantics": False,
                "semantics_version": CURRENT_EXPERIMENT_SEMANTICS_VERSION,
            },
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
            "reproducibility": {
                "experiment_seed": 42,
                "numpy_seed": 42,
                "python_random_seed": 42,
                "torch_seed": 42,
            },
            "feature_ordering": {
                "dl_feature_columns": ["dl_signal_mean_24h"],
                "phase_predictor_by_pair": {
                    "EURUSD": ["adx", "atr_pct", "dl_signal_mean_24h"],
                },
                "strategy_selector_by_pair": {
                    "EURUSD": ["adx", "atr_pct", "plus_di"],
                },
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

    def test_canonical_manifest_filename_supported(self):
        manifest_data = self._make_manifest()
        run_dir = _make_run_dir({
            "run_manifest.json": json.dumps(manifest_data),
        })
        try:
            result = parse_manifest(run_dir)
            self.assertIsNotNone(result)
            self.assertEqual(result["run_id"], "run_20260512T064904Z")
        finally:
            _rmtree(run_dir)

    def test_dl_enabled_true(self):
        result = parse_manifest(self.run_dir)
        self.assertTrue(result["dl_enabled"])

    def test_walkforward_params(self):
        result = parse_manifest(self.run_dir)
        self.assertEqual(result["walkforward"]["train_years"], 7)

    def test_reproducibility_block(self):
        result = parse_manifest(self.run_dir)
        self.assertEqual(result["reproducibility"]["experiment_seed"], 42)

    def test_feature_ordering_block(self):
        result = parse_manifest(self.run_dir)
        self.assertEqual(
            result["feature_ordering"]["phase_predictor_by_pair"]["EURUSD"],
            ["adx", "atr_pct", "dl_signal_mean_24h"],
        )

    def test_factors_block_parsed(self):
        result = parse_manifest(self.run_dir)
        experiment = result["experiment"]
        self.assertEqual(experiment["run_family"], "factorial_v1")
        self.assertTrue(experiment["factors"]["dl_enabled"])
        self.assertEqual(experiment["factors"]["msml_regime"], "LVTF")

    def test_no_manifest_returns_none(self):
        empty_dir = _make_run_dir({})
        try:
            self.assertIsNone(parse_manifest(empty_dir))
        finally:
            _rmtree(empty_dir)

    def test_prefers_dl_enabled_manifest(self):
        """Multiple manifests in one run root must fail loudly."""
        base_manifest = self._make_manifest()
        base_manifest["dl"]["dl_enabled"] = False
        base_manifest["dl"]["dl_mode_tag"] = "__baseline"

        dl_manifest = self._make_manifest()

        run_dir = _make_run_dir({
            "run_manifest_run_1__baseline.json": json.dumps(base_manifest),
            "run_manifest_run_2__dl_enabled.json": json.dumps(dl_manifest),
        })
        try:
            with self.assertRaises(ValueError):
                parse_manifest(run_dir)
        finally:
            _rmtree(run_dir)


class TestCanonicalExperimentMetadata(unittest.TestCase):
    def test_variant_b_metadata_keeps_sentiment_disabled(self):
        experiment = build_experiment_metadata_from_variant("B")
        self.assertEqual(experiment["variant"], "B")
        self.assertEqual(experiment["generation"], "gen1")
        self.assertFalse(experiment["sentiment_enabled"])
        self.assertFalse(experiment["missing_indicators_enabled"])
        self.assertEqual(experiment["semantic_label"], "Gen1_B")

    def test_variant_d_metadata_keeps_sentiment_disabled(self):
        experiment = build_experiment_metadata_from_variant("D")
        self.assertEqual(experiment["variant"], "D")
        self.assertEqual(experiment["generation"], "gen2")
        self.assertFalse(experiment["sentiment_enabled"])
        self.assertTrue(experiment["missing_indicators_enabled"])
        self.assertEqual(experiment["semantic_label"], "Gen2_D")

    def test_variant_e_metadata(self):
        experiment = build_experiment_metadata_from_variant("E")
        self.assertEqual(experiment["variant"], "E")
        self.assertEqual(experiment["generation"], "gen1")
        self.assertTrue(experiment["sentiment_enabled"])
        self.assertTrue(experiment["missing_indicators_enabled"])
        self.assertEqual(experiment["semantic_label"], "Gen1_E")

    def test_variant_f_metadata(self):
        experiment = build_experiment_metadata_from_variant("F")
        self.assertEqual(experiment["variant"], "F")
        self.assertEqual(experiment["generation"], "gen2")
        self.assertTrue(experiment["sentiment_enabled"])
        self.assertFalse(experiment["missing_indicators_enabled"])
        self.assertEqual(experiment["semantic_label"], "Gen2_F")

    def test_corrupt_manifest_raises(self):
        run_dir = _make_run_dir({
            "run_manifest_bad__dl_enabled.json": "NOT JSON {{{",
        })
        try:
            with self.assertRaises(ValueError):
                parse_manifest(run_dir)
        finally:
            _rmtree(run_dir)


class TestRunIdentity(unittest.TestCase):

    def _make_manifest(self, generation, variant, sentiment_enabled, missing_indicators_enabled,
                       ts="20260521T131739Z", run_id=None):
        run_id = run_id or f"run_{ts}"
        return {
            "run_id": run_id,
            "timestamp_utc": ts,
            "dl_enabled": sentiment_enabled,
            "run": {"run_id": run_id, "timestamp_utc": ts},
            "experiment": {
                "generation": generation,
                "variant": variant,
                "sentiment_enabled": sentiment_enabled,
                "missing_indicators_enabled": missing_indicators_enabled,
                "semantic_label": f"{generation.capitalize()}_{variant}",
                "legacy_semantics": False,
                "semantics_version": CURRENT_EXPERIMENT_SEMANTICS_VERSION,
            },
        }

    def _run_identity(self, archive, run_dir_name, manifest):
        run_dir = archive / run_dir_name
        run_dir.mkdir(exist_ok=True)
        return infer_run_identity(
            archive_root=archive,
            run_dir=run_dir,
            experiment_gen=manifest["experiment"]["generation"],
            manifest=manifest,
        )

    # ------------------------------------------------------------------
    # Regression: canonical variant recognition for canonical variants
    # ------------------------------------------------------------------

    def test_fp_gen1_A_recognized_as_variant_A(self):
        """fp_gen1_A with explicit manifest variant=A must be recognized as A."""
        archive = _make_run_dir({})
        try:
            manifest = self._make_manifest("gen1", "A", True, False)
            identity = self._run_identity(archive, "fp_gen1_A", manifest)
            self.assertEqual(identity["run_variant"], "A")
            self.assertEqual(identity["experiment_gen"], "gen1")
            self.assertEqual(identity["semantic_run_name"], "gen1_A")
        finally:
            _rmtree(archive)

    def test_fp_gen1_B_recognized_as_variant_B(self):
        """fp_gen1_B with explicit manifest variant=B must be recognized as B."""
        archive = _make_run_dir({})
        try:
            manifest = self._make_manifest("gen1", "B", False, False)
            identity = self._run_identity(archive, "fp_gen1_B", manifest)
            self.assertEqual(identity["run_variant"], "B")
            self.assertEqual(identity["experiment_gen"], "gen1")
            self.assertEqual(identity["semantic_run_name"], "gen1_B")
        finally:
            _rmtree(archive)

    def test_fp_gen2_C_recognized_as_variant_C(self):
        """fp_gen2_C with explicit manifest variant=C must be recognized as C."""
        archive = _make_run_dir({})
        try:
            manifest = self._make_manifest("gen2", "C", True, True)
            identity = self._run_identity(archive, "fp_gen2_C", manifest)
            self.assertEqual(identity["run_variant"], "C")
            self.assertEqual(identity["experiment_gen"], "gen2")
            self.assertEqual(identity["semantic_run_name"], "gen2_C")
        finally:
            _rmtree(archive)

    def test_fp_gen2_D_recognized_as_variant_D(self):
        """fp_gen2_D with explicit manifest variant=D must be recognized as D."""
        archive = _make_run_dir({})
        try:
            manifest = self._make_manifest("gen2", "D", False, True)
            identity = self._run_identity(archive, "fp_gen2_D", manifest)
            self.assertEqual(identity["run_variant"], "D")
            self.assertEqual(identity["experiment_gen"], "gen2")
            self.assertEqual(identity["semantic_run_name"], "gen2_D")
        finally:
            _rmtree(archive)

    def test_fp_gen1_E_recognized_as_variant_E(self):
        """fp_gen1_E with explicit manifest variant=E must be recognized as E."""
        archive = _make_run_dir({})
        try:
            manifest = self._make_manifest("gen1", "E", True, True)
            identity = self._run_identity(archive, "fp_gen1_E", manifest)
            self.assertEqual(identity["run_variant"], "E")
            self.assertEqual(identity["experiment_gen"], "gen1")
            self.assertEqual(identity["semantic_run_name"], "gen1_E")
        finally:
            _rmtree(archive)

    def test_fp_gen2_F_recognized_as_variant_F(self):
        """fp_gen2_F with explicit manifest variant=F must be recognized as F."""
        archive = _make_run_dir({})
        try:
            manifest = self._make_manifest("gen2", "F", True, False)
            identity = self._run_identity(archive, "fp_gen2_F", manifest)
            self.assertEqual(identity["run_variant"], "F")
            self.assertEqual(identity["experiment_gen"], "gen2")
            self.assertEqual(identity["semantic_run_name"], "gen2_F")
        finally:
            _rmtree(archive)

    # ------------------------------------------------------------------
    # Regression: variants must not collapse to A/C when B/D is explicit
    # ------------------------------------------------------------------

    def test_gen1_B_does_not_collapse_to_A(self):
        """Sentinel regression: gen1 variant=B must NOT resolve to A."""
        archive = _make_run_dir({})
        try:
            manifest = self._make_manifest("gen1", "B", False, False)
            identity = self._run_identity(archive, "fp_gen1_B", manifest)
            self.assertNotEqual(identity["run_variant"], "A",
                                "gen1_B collapsed to A — canonical variant not being read from manifest.")
        finally:
            _rmtree(archive)

    def test_gen2_D_does_not_collapse_to_C(self):
        """Sentinel regression: gen2 variant=D must NOT resolve to C."""
        archive = _make_run_dir({})
        try:
            manifest = self._make_manifest("gen2", "D", False, True)
            identity = self._run_identity(archive, "fp_gen2_D", manifest)
            self.assertNotEqual(identity["run_variant"], "C",
                                "gen2_D collapsed to C — canonical variant not being read from manifest.")
        finally:
            _rmtree(archive)

    # ------------------------------------------------------------------
    # Malformed manifests / missing semantics
    # ------------------------------------------------------------------

    def test_missing_experiment_block_yields_variant_U(self):
        """Manifest without experiment block must yield variant=U and emit warnings."""
        archive = _make_run_dir({})
        run_dir = archive / "legacy_run"
        run_dir.mkdir()
        try:
            identity = infer_run_identity(
                archive_root=archive,
                run_dir=run_dir,
                experiment_gen="unknown",
                manifest={
                    "run_id": "run_20260521T000000Z",
                    "timestamp_utc": "20260521T000000Z",
                    "run": {"run_id": "run_20260521T000000Z"},
                },
            )
            self.assertEqual(identity["run_variant"], "U")
            self.assertGreater(len(identity["identity_warnings"]), 0)
        finally:
            _rmtree(archive)

    def test_missing_variant_is_marked_legacy_unknown(self):
        """Manifest without explicit variant must be legacy_semantics with variant=U."""
        archive = _make_run_dir({})
        run_dir = archive / "partial_manifest"
        run_dir.mkdir()
        try:
            identity = infer_run_identity(
                archive_root=archive,
                run_dir=run_dir,
                experiment_gen="gen1",
                manifest={
                    "run_id": "run_20260521T000000Z",
                    "timestamp_utc": "20260521T000000Z",
                    "run": {"run_id": "run_20260521T000000Z"},
                    "experiment": {
                        "generation": "gen1",
                        "sentiment_enabled": True,
                        "missing_indicators_enabled": False,
                    },
                },
            )
            self.assertEqual(identity["run_variant"], "U")
            self.assertTrue(identity["legacy_semantics"])
        finally:
            _rmtree(archive)

    def test_manifest_experiment_fields_are_used_verbatim(self):
        """Identity must mirror manifest.experiment without semantic re-derivation."""
        archive = _make_run_dir({})
        run_dir = archive / "explicit_variant"
        run_dir.mkdir()
        try:
            identity = infer_run_identity(
                archive_root=archive,
                run_dir=run_dir,
                experiment_gen="gen1",
                manifest={
                    "run_id": "run_20260521T000000Z",
                    "timestamp_utc": "20260521T000000Z",
                    "run": {"run_id": "run_20260521T000000Z"},
                    "experiment": {
                        "generation": "gen1",
                        "variant": "B",
                        "sentiment_enabled": False,
                        "missing_indicators_enabled": False,
                        "semantic_label": "Gen1_B",
                        "legacy_semantics": False,
                        "semantics_version": CURRENT_EXPERIMENT_SEMANTICS_VERSION,
                    },
                },
            )
            self.assertEqual(identity["run_variant"], "B")
            self.assertEqual(identity["experiment_gen"], "gen1")
            self.assertFalse(identity["sentiment_enabled"])
            self.assertFalse(identity["missing_indicators_enabled"])
            self.assertEqual(identity["semantic_label"], "Gen1_B")
        finally:
            _rmtree(archive)

    def test_archive_sentinel_gen1_variant_mismatch_raises(self):
        archive = _make_run_dir({})
        run_dir = archive / "conflict_run"
        run_dir.mkdir()
        try:
            with self.assertRaises(RuntimeError):
                infer_run_identity(
                    archive_root=archive,
                    run_dir=archive / "fp_gen1_B",
                    experiment_gen="gen1",
                    manifest={
                        "run_id": "run_20260521T000000Z",
                        "timestamp_utc": "20260521T000000Z",
                        "run": {"run_id": "run_20260521T000000Z"},
                        "experiment": {
                            "generation": "gen1",
                            "variant": "A",
                            "sentiment_enabled": False,
                            "missing_indicators_enabled": False,
                            "semantic_label": "Gen1_B",
                            "legacy_semantics": False,
                            "semantics_version": CURRENT_EXPERIMENT_SEMANTICS_VERSION,
                        },
                    },
                )
        finally:
            _rmtree(archive)

    def test_archive_sentinel_gen2_variant_mismatch_raises(self):
        archive = _make_run_dir({})
        try:
            with self.assertRaises(RuntimeError):
                infer_run_identity(
                    archive_root=archive,
                    run_dir=archive / "fp_gen2_D",
                    experiment_gen="gen2",
                    manifest={
                        "run_id": "run_20260521T000000Z",
                        "timestamp_utc": "20260521T000000Z",
                        "run": {"run_id": "run_20260521T000000Z"},
                        "experiment": {
                            "generation": "gen2",
                            "variant": "C",
                            "sentiment_enabled": False,
                            "missing_indicators_enabled": True,
                            "semantic_label": "Gen2_D",
                            "legacy_semantics": False,
                            "semantics_version": CURRENT_EXPERIMENT_SEMANTICS_VERSION,
                        },
                    },
                )
        finally:
            _rmtree(archive)

    def test_dl_infrastructure_on_variant_b_remains_sentiment_off(self):
        archive = _make_run_dir({})
        run_dir = archive / "fp_gen1_B_dl_on"
        run_dir.mkdir()
        try:
            identity = infer_run_identity(
                archive_root=archive,
                run_dir=run_dir,
                experiment_gen="gen1",
                manifest={
                    "run_id": "run_20260521T000000Z",
                    "timestamp_utc": "20260521T000000Z",
                    "dl_enabled": True,
                    "run": {"run_id": "run_20260521T000000Z"},
                    "experiment": {
                        "generation": "gen1",
                        "variant": "B",
                        "sentiment_enabled": False,
                        "missing_indicators_enabled": False,
                        "semantic_label": "Gen1_B",
                        "legacy_semantics": False,
                        "semantics_version": CURRENT_EXPERIMENT_SEMANTICS_VERSION,
                    },
                },
            )
            self.assertEqual(identity["run_variant"], "B")
            self.assertFalse(identity["legacy_semantics"])
            self.assertEqual(
                [w for w in identity["identity_warnings"] if "sentiment_enabled" in w],
                [],
            )
        finally:
            _rmtree(archive)

    def test_dl_infrastructure_on_variant_d_remains_sentiment_off(self):
        archive = _make_run_dir({})
        run_dir = archive / "fp_gen2_D_dl_on"
        run_dir.mkdir()
        try:
            identity = infer_run_identity(
                archive_root=archive,
                run_dir=run_dir,
                experiment_gen="gen2",
                manifest={
                    "run_id": "run_20260521T000000Z",
                    "timestamp_utc": "20260521T000000Z",
                    "dl_enabled": True,
                    "run": {"run_id": "run_20260521T000000Z"},
                    "experiment": {
                        "generation": "gen2",
                        "variant": "D",
                        "sentiment_enabled": False,
                        "missing_indicators_enabled": True,
                        "semantic_label": "Gen2_D",
                        "legacy_semantics": False,
                        "semantics_version": CURRENT_EXPERIMENT_SEMANTICS_VERSION,
                    },
                },
            )
            self.assertEqual(identity["run_variant"], "D")
            self.assertFalse(identity["legacy_semantics"])
            self.assertEqual(
                [w for w in identity["identity_warnings"] if "sentiment_enabled" in w],
                [],
            )
        finally:
            _rmtree(archive)

    def test_canonical_identity_includes_archive_path(self):
        archive = _make_run_dir({})
        run_dir = archive / "fp_gen1_A"
        run_dir.mkdir()
        identity = infer_run_identity(
            archive_root=archive,
            run_dir=run_dir,
            experiment_gen="gen1",
            manifest={
                "run_id": "run_20260521T131739Z",
                "timestamp_utc": "20260521T131739Z",
                "dl_enabled": True,
                "primary": {"run": {"run_id": "run_20260521T131739Z"}},
                "experiment": {
                    "generation": "gen1",
                    "variant": "A",
                    "sentiment_enabled": True,
                    "missing_indicators_enabled": False,
                    "semantic_label": "Gen1_A",
                    "legacy_semantics": False,
                    "semantics_version": CURRENT_EXPERIMENT_SEMANTICS_VERSION,
                },
            },
        )
        try:
            self.assertEqual(identity["semantic_run_name"], "gen1_A")
            self.assertEqual(identity["archive_slug"], "fp_gen1_A")
            self.assertEqual(identity["run_id"], "gen1_A__20260521T131739Z__fp_gen1_A")
        finally:
            _rmtree(archive)

    def test_identity_unknown_variant_warns(self):
        archive = _make_run_dir({})
        run_dir = archive / "rerun_copy"
        run_dir.mkdir()
        identity = infer_run_identity(
            archive_root=archive,
            run_dir=run_dir,
            experiment_gen="unknown",
            manifest={
                "run_id": "run_20260521T131739Z",
                "timestamp_utc": "20260521T131739Z",
                "dl_enabled": True,
                "run": {"run_id": "run_20260521T131739Z"},
            },
        )
        try:
            self.assertEqual(identity["run_variant"], "U")
            self.assertGreaterEqual(len(identity["identity_warnings"]), 1)
        finally:
            _rmtree(archive)


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

    def test_single_run_dir_with_canonical_manifest_filename(self):
        run_dir = _make_run_dir({
            "run_manifest.json": json.dumps({"experiment": {"generation": "gen1"}}),
        })
        try:
            found = list(discover_runs(run_dir))
            self.assertGreaterEqual(len(found), 1)
            paths = [f[0] for f in found]
            self.assertIn(run_dir, paths)
        finally:
            _rmtree(run_dir)

    def test_archive_with_nested_runs(self):
        tmp = tempfile.mkdtemp()
        archive = Path(tmp)
        # Create two nested run directories
        run_a = archive / "run_a"
        run_b = archive / "run_b"
        run_a.mkdir()
        run_b.mkdir()
        (run_a / "results_ml__baseline.csv").write_text("Model,Accuracy\n")
        (run_a / ".mpml_legacy_run_root").write_text("legacy")
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

    def test_gen_extracted_from_manifest_explicit_field(self):
        run_dir = _make_run_dir({
            "run_manifest_gen2__dl_enabled.json": json.dumps({"experiment": {"generation": "gen2"}}),
        })
        try:
            found = list(discover_runs(run_dir))
            gens = {f[1] for f in found}
            self.assertIn("gen2", gens)
        finally:
            _rmtree(run_dir)

    def test_gen_defaults_to_unknown_without_explicit_manifest_gen(self):
        run_dir = _make_run_dir({
            "run_manifest_run_abc__baseline.json": "{}",
        })
        try:
            found = list(discover_runs(run_dir))
            gens = {f[1] for f in found}
            self.assertIn("unknown", gens)
        finally:
            _rmtree(run_dir)

    def test_nested_manifest_contamination_prevented(self):
        tmp = tempfile.mkdtemp()
        archive = Path(tmp)
        run_root = archive / "fp_gen1_A"
        nested = run_root / "analysis"
        nested.mkdir(parents=True)
        (run_root / "run_manifest_x.json").write_text(json.dumps({"experiment_gen": "gen1"}))
        (nested / "copied_results.csv").write_text("Pair,Sharpe_Delta\nEURUSD,0.1\n")
        try:
            found = list(discover_runs(archive))
            dirs = [p for p, _ in found]
            self.assertEqual(dirs, [run_root.resolve()])
            csvs = parse_run_csvs(run_root)
            self.assertNotIn("copied_results.csv", csvs.get("_files_found", []))
        finally:
            _rmtree(archive)


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

        # Two synthetic run directories (A/B cohort)
        for run_name, dl_enabled, run_variant in [
            ("run_baseline", False, "B"),
            ("run_dl", True, "A"),
        ]:
            run_dir = self.archive / run_name
            run_dir.mkdir()
            manifest = {
                "run": {
                    "run_id": run_name,
                    "git_sha": "abc123",
                    "timestamp_utc": "20260521T000000Z",
                },
                "experiment": {
                    "generation": "gen1",
                    "variant": run_variant,
                    "sentiment_enabled": dl_enabled,
                    "missing_indicators_enabled": False,
                    "semantic_label": f"Gen1_{run_variant}",
                    "legacy_semantics": False,
                    "semantics_version": CURRENT_EXPERIMENT_SEMANTICS_VERSION,
                },
                "experiment_surface": {
                    "surface_semantics_version": 5,
                    "sentiment_surface": dl_enabled,
                    "training_pair_family": "persistent",
                    "evaluation_pair_family": "persistent",
                    "feature_surface": "unknown",
                    "artifact_source": "unknown",
                },
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
        self.assertIn("factors", comp)

    def test_pipeline_sentiment_comparison(self):
        from analysis.pipeline import run_pipeline
        run_pipeline(self.archive, self.output_dir, verbose=False)

        comp = json.loads((self.output_dir / "comparisons.json").read_text())
        sentiment = comp["sentiment"]
        # We created one DL-enabled and one baseline run
        self.assertEqual(len(sentiment["sentiment_on"]), 1)
        self.assertEqual(len(sentiment["sentiment_off"]), 1)
        self.assertIn("validation", comp)

    def test_variant_roundtrip_integrity_all_variants(self):
        from analysis.pipeline import run_pipeline

        archive = Path(tempfile.mkdtemp())
        output = Path(tempfile.mkdtemp())
        try:
            for variant, semantics in EXPERIMENT_VARIANTS.items():
                run_dir = archive / f"fp_{semantics['generation']}_{variant}"
                run_dir.mkdir()
                manifest = {
                    "run": {
                        "run_id": f"run_{variant}",
                        "git_sha": "abc123",
                        "timestamp_utc": "20260521T000000Z",
                    },
                    "experiment": {
                        "generation": semantics["generation"],
                        "variant": variant,
                        "sentiment_enabled": semantics["sentiment_enabled"],
                        "missing_indicators_enabled": semantics["missing_indicators_enabled"],
                        "semantic_label": semantics["semantic_label"],
                        "legacy_semantics": False,
                        "semantics_version": CURRENT_EXPERIMENT_SEMANTICS_VERSION,
                    },
                    "experiment_surface": {
                        "surface_semantics_version": 5,
                        "sentiment_surface": semantics["sentiment_enabled"],
                        "training_pair_family": "persistent",
                        "evaluation_pair_family": "persistent",
                        "feature_surface": "unknown",
                        "artifact_source": "unknown",
                    },
                    "dl": {"dl_enabled": True, "dl_mode_tag": "__dl_enabled"},
                }
                (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
                (run_dir / "walkforward_results_summary__dl_enabled.csv").write_text(
                    "Pair,Sharpe_Dynamic,Sharpe_Baseline,Sharpe_Delta,N_Folds\n"
                    "EURUSD,0.35,0.22,0.13,48\n"
                )

            run_pipeline(archive, output, verbose=False)
            comparisons = json.loads((output / "comparisons.json").read_text())
            validations = comparisons["validation"]
            self.assertEqual(validations["errors"], [])
            self.assertEqual(
                comparisons["sentiment"]["matrix"]["present_variants"],
                sorted(EXPERIMENT_VARIANTS),
            )
            self.assertEqual(
                comparisons["gen"]["matrix"]["present_variants"],
                sorted(EXPERIMENT_VARIANTS),
            )
            report = (output / "report.md").read_text()
            self.assertIn("gen1_A__20260521T000000Z__fp_gen1_A", report)
            self.assertIn("gen1_B__20260521T000000Z__fp_gen1_B", report)
            self.assertIn("gen2_C__20260521T000000Z__fp_gen2_C", report)
            self.assertIn("gen2_D__20260521T000000Z__fp_gen2_D", report)
            self.assertIn("gen1_E__20260521T000000Z__fp_gen1_E", report)
            self.assertIn("gen2_F__20260521T000000Z__fp_gen2_F", report)
            for summary_path in (output / "summaries").glob("*.summary.json"):
                summary = json.loads(summary_path.read_text())
                experiment = summary["meta"]["experiment"]
                variant = experiment["variant"]
                semantics = EXPERIMENT_VARIANTS[variant]
                self.assertEqual(summary["meta"]["experiment_gen"], experiment["generation"])
                self.assertEqual(summary["meta"]["run_variant"], experiment["variant"])
                self.assertEqual(summary["meta"]["sentiment_enabled"], experiment["sentiment_enabled"])
                self.assertEqual(
                    summary["meta"]["missing_indicators_enabled"],
                    experiment["missing_indicators_enabled"],
                )
                self.assertEqual(summary["meta"]["semantic_label"], experiment["semantic_label"])
                self.assertEqual(experiment["generation"], semantics["generation"])
                self.assertEqual(experiment["sentiment_enabled"], semantics["sentiment_enabled"])
                self.assertEqual(experiment["missing_indicators_enabled"], semantics["missing_indicators_enabled"])
                self.assertEqual(experiment["semantic_label"], semantics["semantic_label"])
        finally:
            _rmtree(archive)
            _rmtree(output)

    def test_variant_b_manifest_stays_b_gen1_through_pipeline(self):
        from analysis.pipeline import run_pipeline

        archive = Path(tempfile.mkdtemp())
        output = Path(tempfile.mkdtemp())
        try:
            manifests = [
                ("fp_gen1_A", "gen1", "A", True, False),
                ("fp_gen1_B", "gen1", "B", False, False),
                ("fp_gen2_C", "gen2", "C", True, True),
                ("fp_gen2_D", "gen2", "D", False, True),
            ]
            for run_name, generation, variant, sentiment_enabled, missing_indicators_enabled in manifests:
                run_dir = archive / run_name
                run_dir.mkdir()
                manifest = {
                    "run": {
                        "run_id": run_name,
                        "git_sha": "abc123",
                        "timestamp_utc": "20260521T010101Z",
                    },
                    "experiment": {
                        "generation": generation,
                        "variant": variant,
                        "sentiment_enabled": sentiment_enabled,
                        "missing_indicators_enabled": missing_indicators_enabled,
                        "semantic_label": f"{generation.capitalize()}_{variant}",
                        "legacy_semantics": False,
                        "semantics_version": CURRENT_EXPERIMENT_SEMANTICS_VERSION,
                    },
                    "experiment_surface": {
                        "surface_semantics_version": 5,
                        "sentiment_surface": sentiment_enabled,
                        "training_pair_family": "persistent",
                        "evaluation_pair_family": "persistent",
                        "feature_surface": "unknown",
                        "artifact_source": "unknown",
                    },
                    "dl": {
                        "dl_enabled": True,
                        "dl_mode_tag": "__dl_enabled",
                    },
                }
                (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
                (run_dir / "walkforward_results_summary__dl_enabled.csv").write_text(
                    "Pair,Sharpe_Dynamic,Sharpe_Baseline,Sharpe_Delta,N_Folds\n"
                    "EURUSD,0.35,0.22,0.13,48\n"
                )

            run_pipeline(archive, output, verbose=False)
            summary_b = json.loads(
                (output / "summaries" / "gen1_B__20260521T010101Z__fp_gen1_B.summary.json").read_text()
            )
            self.assertEqual(summary_b["meta"]["experiment_gen"], "gen1")
            self.assertEqual(summary_b["meta"]["run_variant"], "B")
            self.assertFalse(summary_b["meta"]["sentiment_enabled"])
            self.assertFalse(summary_b["meta"]["missing_indicators_enabled"])
            comparisons = json.loads((output / "comparisons.json").read_text())
            self.assertEqual(
                comparisons["sentiment"]["matrix"]["present_variants"],
                ["A", "B", "C", "D"],
            )
        finally:
            _rmtree(archive)
            _rmtree(output)


class TestValidationAndOrdering(unittest.TestCase):
    def _summary(self, run_id: str, gen: str, variant: str, ts: str, relpath: str) -> dict:
        return {
            "run_id": run_id,
            "meta": {
                "experiment_gen": gen,
                "run_variant": variant,
                "legacy_semantics": False,
                "timestamp_utc": ts,
                "archive_relpath": relpath,
                "manifest_present": True,
                "manifest_diagnostics": {
                    "manifest_count": 1,
                    "manifest_path": f"/tmp/{run_id}.json",
                    "manifest_timestamp": ts,
                    "manifest_run_id": run_id,
                    "dl_mode_tag": "__baseline",
                },
                "files_found": ["walkforward_results_summary__baseline.csv"],
                "dl_enabled": bool((EXPERIMENT_VARIANTS.get(variant) or {}).get("dl_enabled", False)),
                "reproducibility": {
                    "experiment_seed": 42,
                    "numpy_seed": 42,
                    "python_random_seed": 42,
                    "torch_seed": 42,
                },
                "feature_ordering": {
                    "dl_feature_columns": [],
                    "phase_predictor_by_pair": {
                        "EURUSD": ["adx", "atr_pct", "rsi"],
                    },
                    "strategy_selector_by_pair": {
                        "EURUSD": ["adx", "atr_pct", "plus_di"],
                    },
                },
                "identity_warnings": [],
            },
            "csvs": {"walkforward_summary": [{"Pair": "EURUSD", "Sharpe_Delta": 0.1}], "walkforward_per_fold": []},
            "log": None,
            "warnings": [],
        }

    def test_deterministic_ordering(self):
        s1 = self._summary("r2", "gen2", "D", "20260522T010101Z", "b")
        s2 = self._summary("r1", "gen1", "A", "20260521T010101Z", "a")
        ordered = sort_summaries_deterministically([s1, s2])
        self.assertEqual(ordered[0]["meta"]["experiment_gen"], "gen1")
        self.assertEqual(ordered[1]["meta"]["experiment_gen"], "gen2")

    def test_duplicate_run_id_detected(self):
        s1 = self._summary("dup", "gen1", "A", "20260521T010101Z", "a")
        s2 = self._summary("dup", "gen1", "B", "20260521T010102Z", "b")
        validation = validate_summaries([s1, s2])
        self.assertTrue(any("Duplicate canonical run_id" in e for e in validation["errors"]))

    def test_malformed_archive_detected(self):
        summary = {
            "run_id": "bad",
            "meta": {
                "experiment_gen": "gen1",
                "run_variant": "U",
                "timestamp_utc": "unknown_ts",
                "archive_relpath": "bad_run",
                "manifest_present": False,
                "manifest_diagnostics": {
                    "manifest_count": 0,
                    "manifest_path": None,
                    "manifest_timestamp": None,
                    "manifest_run_id": None,
                    "dl_mode_tag": None,
                },
                "files_found": [],
                "dl_enabled": False,
                "reproducibility": {},
                "feature_ordering": {},
                "identity_warnings": [],
            },
            "csvs": {"walkforward_summary": None, "walkforward_per_fold": None},
            "log": None,
            "warnings": [],
        }
        validation = validate_summaries([summary])
        self.assertTrue(any("malformed archive" in e for e in validation["errors"]))

    def test_missing_reproducibility_metadata_warns(self):
        summary = self._summary("r1", "gen1", "A", "20260521T010101Z", "a")
        summary["meta"]["reproducibility"] = {"experiment_seed": 42}
        validation = validate_summaries([summary])
        self.assertTrue(
            any("missing reproducibility metadata" in warning for warning in validation["warnings"])
        )

    def test_feature_order_mismatch_warns(self):
        s1 = self._summary("r1", "gen1", "A", "20260521T010101Z", "a")
        s2 = self._summary("r2", "gen1", "B", "20260521T010102Z", "b")
        s2["meta"]["feature_ordering"]["phase_predictor_by_pair"]["EURUSD"] = [
            "atr_pct",
            "adx",
            "rsi",
        ]
        validation = validate_summaries([s1, s2])
        self.assertTrue(
            any("differing feature column order" in warning for warning in validation["warnings"])
        )

    def test_duplicate_semantic_variant_within_cohort_warns(self):
        """Two Gen1_B runs in the same analysis root must emit a semantic warning."""
        s1 = self._summary("run_b_1", "gen1", "B", "20260521T010101Z", "run_b_1")
        s2 = self._summary("run_b_2", "gen1", "B", "20260521T010102Z", "run_b_2")
        validation = validate_summaries([s1, s2])
        self.assertTrue(
            any("Duplicate semantic variant" in w and "gen1_B" in w for w in validation["warnings"]),
            "Expected warning for duplicate semantic variant gen1_B"
        )

    def test_no_duplicate_variant_warning_for_different_cohorts(self):
        """Gen1_A and Gen2_C are different cohorts — no duplicate warning."""
        s1 = self._summary("r1", "gen1", "A", "20260521T010101Z", "a")
        s2 = self._summary("r2", "gen2", "C", "20260521T010102Z", "b")
        validation = validate_summaries([s1, s2])
        dup_warnings = [w for w in validation["warnings"] if "Duplicate semantic variant" in w]
        self.assertEqual(len(dup_warnings), 0)

    def test_all_canonical_variants_no_duplicate_warning(self):
        """One run per canonical variant must not trigger duplicate variant warnings."""
        summaries = []
        for idx, (variant, semantics) in enumerate(sorted(EXPERIMENT_VARIANTS.items()), start=1):
            summaries.append(
                self._summary(
                    f"r_{variant.lower()}",
                    semantics["generation"],
                    variant,
                    f"20260521T0000{idx:02d}Z",
                    f"fp_{semantics['generation']}_{variant}",
                )
            )
        validation = validate_summaries(summaries)
        dup_warnings = [w for w in validation["warnings"] if "Duplicate semantic variant" in w]
        self.assertEqual(len(dup_warnings), 0)

    def test_invalid_generation_detected(self):
        """Invalid generation value in experiment block must produce semantic error."""
        s = self._summary("r1", "gen1", "A", "20260521T010101Z", "a")
        s["meta"]["experiment"] = {
            "generation": "gen99",  # invalid
            "variant": "A",
            "sentiment_enabled": True,
            "missing_indicators_enabled": False,
            "semantic_label": "Gen1_A",
        }
        validation = validate_summaries([s])
        self.assertTrue(any("invalid experiment generation" in e for e in validation["errors"]))

    def test_invalid_variant_detected(self):
        """Non-canonical variant is tolerated with warning (factor-first semantics)."""
        s = self._summary("r1", "gen1", "A", "20260521T010101Z", "a")
        s["meta"]["experiment"] = {
            "generation": "gen1",
            "variant": "Z",  # invalid
            "sentiment_enabled": True,
            "missing_indicators_enabled": False,
            "semantic_label": "Gen1_Z",
        }
        validation = validate_summaries([s])
        self.assertTrue(any("non-canonical experiment variant" in w for w in validation["warnings"]))

    def test_semantic_conflict_gen1_variant_C_detected(self):
        """Meta generation mismatch against manifest.experiment must be an error."""
        s = self._summary("r1", "gen1", "C", "20260521T010101Z", "a")
        s["meta"]["experiment"] = {
            "generation": "gen2",
            "variant": "C",
            "sentiment_enabled": True,
            "missing_indicators_enabled": True,
            "semantic_label": "Gen2_C",
        }
        validation = validate_summaries([s])
        self.assertTrue(any("identity corruption" in e and "generation" in e for e in validation["errors"]))

    def test_semantic_conflict_gen2_variant_A_detected(self):
        """Meta variant mismatch against manifest.experiment must be an error."""
        s = self._summary("r1", "gen2", "A", "20260521T010101Z", "a")
        s["meta"]["experiment"] = {
            "generation": "gen2",
            "variant": "C",
            "sentiment_enabled": True,
            "missing_indicators_enabled": True,
            "semantic_label": "Gen2_C",
        }
        validation = validate_summaries([s])
        self.assertTrue(any("identity corruption" in e and "run_variant" in e for e in validation["errors"]))

    def test_validation_rejects_variant_b_with_sentiment_true(self):
        s = self._summary("r_bad", "gen1", "B", "20260521T010101Z", "bad")
        s["meta"]["experiment"] = {
            "generation": "gen1",
            "variant": "B",
            "sentiment_enabled": False,
            "missing_indicators_enabled": False,
            "semantic_label": "Gen1_B",
        }
        s["meta"]["sentiment_enabled"] = True
        validation = validate_summaries([s])
        self.assertTrue(
            any("identity corruption" in e and "sentiment_enabled" in e for e in validation["errors"]),
            "Semantic corruption should hard-fail when propagated sentiment metadata diverges from manifest.experiment.",
        )




if __name__ == "__main__":
    unittest.main()
