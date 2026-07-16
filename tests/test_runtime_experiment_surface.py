import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analysis.parsers.manifest_parser import parse_manifest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SURFACE_RUNTIME_PATH = _ROOT / "src" / "experiment_surface_runtime.py"
_SURFACE_RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "mpml_surface_runtime_module",
    _SURFACE_RUNTIME_PATH,
)
if _SURFACE_RUNTIME_SPEC is None or _SURFACE_RUNTIME_SPEC.loader is None:
    raise RuntimeError(f"Unable to load runtime surface module from {_SURFACE_RUNTIME_PATH}")
_SURFACE_RUNTIME_MODULE = importlib.util.module_from_spec(_SURFACE_RUNTIME_SPEC)
_SURFACE_RUNTIME_SPEC.loader.exec_module(_SURFACE_RUNTIME_MODULE)
build_runtime_experiment_surface = _SURFACE_RUNTIME_MODULE.build_runtime_experiment_surface


class TestRuntimeExperimentSurfaceEmission(unittest.TestCase):
    def test_manifest_provenance_includes_market_data_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "run": {"run_id": "run_20260525T090000Z", "timestamp_utc": "20260525T090000Z"},
                "experiment": {"generation": "gen1", "variant": "A", "factors": {}},
                "experiment_surface": build_runtime_experiment_surface(
                    dl_runtime_enabled=False,
                    dl_surface={},
                    dl_artifact_path=None,
                    experiment_factors={},
                    artifact_metadata={},
                ),
                "market_data_source": "broker_csv",
                "market_data_root": "../market-sentiment-ml/data/input/fx",
                "market_data_timezone": "UTC+1",
            }
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            parsed = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(parsed.get("market_data_source"), "broker_csv")
            self.assertEqual(parsed.get("market_data_root"), "../market-sentiment-ml/data/input/fx")
            self.assertEqual(parsed.get("market_data_timezone"), "UTC+1")

    def test_new_manifest_includes_experiment_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            surface = build_runtime_experiment_surface(
                dl_runtime_enabled=True,
                dl_surface={
                    "model": "mlp",
                    "target_horizon": 24,
                    "feature_set": "trend_vol_only",
                    "dl_regime": "LVTF",
                },
                dl_artifact_path=Path("/tmp/artifacts/surface.parquet"),
                experiment_factors={
                    "selector_enabled": True,
                    "overlap_only": False,
                    "msml_regime": "LVTF",
                },
                artifact_metadata={
                    "sentiment_surface": "no_sentiment",
                    "training_pair_family": "persistent",
                    "evaluation_pair_family": "reactive",
                    "feature_surface": "trend_vol_only",
                    "artifact_source": "msml/run_123.parquet",
                    "surface_semantics_version": 5,
                    "target_horizon": 24,
                    "model": "mlp",
                },
            )
            manifest = {
                "run": {"run_id": "run_20260525T090000Z", "timestamp_utc": "20260525T090000Z"},
                "experiment": {
                    "generation": "gen1",
                    "variant": "A",
                    "factors": {
                        "dl_enabled": True,
                        "sentiment_enabled": True,
                        "missing_indicators_enabled": False,
                        "msml_regime": "LVTF",
                        "overlap_only": False,
                        "selector_enabled": True,
                    },
                },
                "experiment_surface": surface,
            }
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            parsed = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("experiment_surface", parsed)
            self.assertEqual(parsed["experiment_surface"]["surface_source"], "artifact_introspection")

    def test_sentiment_surface_is_not_inferred_from_variant(self):
        surface = build_runtime_experiment_surface(
            dl_runtime_enabled=True,
            dl_surface={
                "model": "mlp",
                "target_horizon": 24,
                "feature_set": "trend_vol_only",
                "dl_regime": "LVTF",
            },
            dl_artifact_path=Path("/tmp/artifacts/persistent_dl_nosentiment_blind.parquet"),
            experiment_factors={
                # Runtime factors can still indicate sentiment-enabled architecture.
                "sentiment_enabled": True,
                "selector_enabled": True,
                "overlap_only": False,
                "msml_regime": "LVTF",
            },
            artifact_metadata={"sentiment_surface": "no_sentiment"},
        )
        self.assertEqual(surface["sentiment_surface"], "no_sentiment")

    def test_training_eval_family_and_feature_surface_propagation(self):
        surface = build_runtime_experiment_surface(
            dl_runtime_enabled=True,
            dl_surface={
                "model": "mlp",
                "target_horizon": 24,
                "feature_set": "price_trend",
                "dl_regime": "LVTF",
            },
            dl_artifact_path=Path("/tmp/artifacts/surface.parquet"),
            experiment_factors={
                "selector_enabled": True,
                "overlap_only": False,
                "msml_regime": "LVTF",
            },
            artifact_metadata={
                "training_pair_family": "persistent",
                "evaluation_pair_family": "reactive",
                "feature_surface": "trend_vol_only",
                "sentiment_surface": "sentiment",
            },
        )
        self.assertEqual(surface["training_pair_family"], "persistent")
        self.assertEqual(surface["evaluation_pair_family"], "reactive")
        self.assertEqual(surface["feature_surface"], "trend_vol_only")

    def test_missing_provenance_fields_emit_unknown(self):
        surface = build_runtime_experiment_surface(
            dl_runtime_enabled=True,
            dl_surface={
                "model": "mlp",
                "target_horizon": 24,
                "feature_set": "trend_vol_only",
                "dl_regime": "LVTF",
            },
            dl_artifact_path=None,
            experiment_factors={
                "selector_enabled": True,
                "overlap_only": False,
                "msml_regime": "LVTF",
            },
            artifact_metadata={},
        )
        self.assertEqual(surface["training_pair_family"], "unknown")
        self.assertEqual(surface["evaluation_pair_family"], "unknown")
        self.assertEqual(surface["feature_surface"], "trend_vol_only")
        self.assertEqual(surface["artifact_source"], "unknown")
        self.assertEqual(surface["imputation_awareness"], "blind")

    def test_canonical_sentiment_surface_is_resolved_from_feature_surface(self):
        surface = build_runtime_experiment_surface(
            dl_runtime_enabled=True,
            dl_surface={"model": "mlp", "target_horizon": 24, "feature_set": "price_trend"},
            dl_artifact_path=Path("/tmp/artifacts/persistent_dl_sentiment/file.parquet"),
            experiment_factors={
                "selector_enabled": True,
                "overlap_only": False,
                "msml_regime": "LVTF",
                "missing_indicators_enabled": True,
            },
            artifact_metadata={},
        )
        self.assertEqual(surface["feature_surface"], "price_trend")
        self.assertEqual(surface["sentiment_surface"], "sentiment")
        self.assertEqual(surface["training_pair_family"], "persistent")
        self.assertEqual(surface["imputation_awareness"], "aware")

    def test_sentiment_surface_is_none_when_dl_disabled(self):
        surface = build_runtime_experiment_surface(
            dl_runtime_enabled=False,
            dl_surface={"model": "mlp", "target_horizon": 24, "feature_set": "price_trend"},
            dl_artifact_path=None,
            experiment_factors={
                "selector_enabled": True,
                "overlap_only": False,
                "msml_regime": "LVTF",
                "missing_indicators_enabled": False,
            },
            artifact_metadata={},
        )
        self.assertEqual(surface["sentiment_surface"], "none")
        self.assertEqual(surface["imputation_awareness"], "blind")

    def test_evaluation_pair_family_is_inferred_from_active_pairs(self):
        with mock.patch.dict(
            "os.environ",
            {"ACTIVE_PAIRS": "EURUSD,GBPUSD,NZDUSD,EURGBP,EURAUD"},
            clear=False,
        ):
            surface = build_runtime_experiment_surface(
                dl_runtime_enabled=True,
                dl_surface={"model": "mlp", "target_horizon": 24, "feature_set": "trend_vol_only"},
                dl_artifact_path=Path("/tmp/artifacts/reactive_dl_nosentiment/file.parquet"),
                experiment_factors={
                    "selector_enabled": True,
                    "overlap_only": False,
                    "msml_regime": "LVTF",
                    "missing_indicators_enabled": False,
                },
                artifact_metadata={},
            )
        self.assertEqual(surface["evaluation_pair_family"], "persistent")
        self.assertEqual(surface["training_pair_family"], "reactive")
        self.assertEqual(surface["sentiment_surface"], "no_sentiment")

    def test_transfer_artifact_names_infer_training_and_evaluation_families(self):
        cases = [
            ("persistent_to_reactive_sentiment_LVTF", "persistent", "reactive"),
            ("persistent_to_reactive_nosentiment_HVR", "persistent", "reactive"),
            ("reactive_to_persistent_sentiment_HVTF", "reactive", "persistent"),
            ("reactive_to_persistent_nosentiment_LVR", "reactive", "persistent"),
        ]
        for artifact_name, expected_training, expected_evaluation in cases:
            with self.subTest(artifact_name=artifact_name):
                surface = build_runtime_experiment_surface(
                    dl_runtime_enabled=True,
                    dl_surface={
                        "model": "mlp",
                        "target_horizon": 24,
                        "feature_set": "trend_vol_only",
                        "dl_regime": "LVTF",
                    },
                    dl_artifact_path=(
                        Path(tempfile.gettempdir()) / "artifacts" / artifact_name / "file.parquet"
                    ),
                    experiment_factors={
                        "selector_enabled": True,
                        "overlap_only": False,
                        "msml_regime": "LVTF",
                        "missing_indicators_enabled": False,
                    },
                    artifact_metadata={},
                )
                self.assertEqual(surface["training_pair_family"], expected_training)
                self.assertEqual(surface["evaluation_pair_family"], expected_evaluation)

    def test_legacy_manifest_without_surface_still_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "run": {"run_id": "gen1_A__20240101T120000Z", "timestamp_utc": "20240101T120000Z"},
                        "experiment": {
                            "generation": "gen1",
                            "variant": "A",
                            "factors": {
                                "dl_enabled": True,
                                "sentiment_enabled": True,
                                "missing_indicators_enabled": False,
                                "msml_regime": "LVTF",
                                "overlap_only": False,
                                "selector_enabled": True,
                            },
                            "legacy_semantics": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            parsed = parse_manifest(run_dir)
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed["surface_source"], "legacy_variant_fallback")


# ---------------------------------------------------------------------------
# Phase B: Behavioral Surface metadata propagation through experiment_surface
# ---------------------------------------------------------------------------

class TestBehavioralSurfaceMetadataPropagation(unittest.TestCase):
    """Tests for behavioral_surface, behavioral_surface_version, behavioral_state
    fields added in Phase B."""

    def _base_surface_kwargs(self, **overrides):
        kwargs = dict(
            dl_runtime_enabled=False,
            dl_surface={},
            dl_artifact_path=None,
            experiment_factors={},
            artifact_metadata={},
        )
        kwargs.update(overrides)
        return kwargs

    def test_behavioral_surface_id_propagates(self):
        surface = build_runtime_experiment_surface(
            **self._base_surface_kwargs(behavioral_surface_id="trend_vol")
        )
        self.assertEqual(surface["behavioral_surface"], "trend_vol")

    def test_behavioral_surface_version_propagates(self):
        surface = build_runtime_experiment_surface(
            **self._base_surface_kwargs(
                behavioral_surface_id="trend_vol",
                behavioral_surface_version="1.0.0",
            )
        )
        self.assertEqual(surface["behavioral_surface_version"], "1.0.0")

    def test_behavioral_state_id_propagates(self):
        surface = build_runtime_experiment_surface(
            **self._base_surface_kwargs(
                behavioral_surface_id="trend_vol",
                behavioral_state_id="LVTF",
            )
        )
        self.assertEqual(surface["behavioral_state"], "LVTF")

    def test_behavioral_fields_absent_when_not_provided(self):
        surface = build_runtime_experiment_surface(
            **self._base_surface_kwargs()
        )
        self.assertNotIn("behavioral_surface", surface)
        self.assertNotIn("behavioral_surface_version", surface)
        self.assertNotIn("behavioral_state", surface)

    def test_behavioral_state_absent_when_none(self):
        surface = build_runtime_experiment_surface(
            **self._base_surface_kwargs(
                behavioral_surface_id="reactive_jpy",
                behavioral_surface_version="1.0.0",
                behavioral_state_id=None,
            )
        )
        self.assertEqual(surface["behavioral_surface"], "reactive_jpy")
        self.assertEqual(surface["behavioral_surface_version"], "1.0.0")
        self.assertNotIn("behavioral_state", surface)

    def test_reactive_jpy_surface_propagates_without_state(self):
        """Phase B: reactive_jpy surface emits surface metadata but no state_id."""
        surface = build_runtime_experiment_surface(
            dl_runtime_enabled=False,
            dl_surface={},
            dl_artifact_path=None,
            experiment_factors={},
            behavioral_surface_id="reactive_jpy",
            behavioral_surface_version="1.0.0",
            behavioral_state_id=None,
        )
        self.assertEqual(surface["behavioral_surface"], "reactive_jpy")
        self.assertNotIn("behavioral_state", surface)

    def test_trend_vol_full_propagation(self):
        """Phase B: trend_vol surface emits surface_id, version and state."""
        surface = build_runtime_experiment_surface(
            dl_runtime_enabled=True,
            dl_surface={
                "model": "mlp",
                "target_horizon": 24,
                "feature_set": "price_trend",
                "dl_regime": "LVTF",
            },
            dl_artifact_path=None,
            experiment_factors={"selector_enabled": True, "overlap_only": False},
            behavioral_surface_id="trend_vol",
            behavioral_surface_version="1.0.0",
            behavioral_state_id="LVTF",
        )
        self.assertEqual(surface["behavioral_surface"], "trend_vol")
        self.assertEqual(surface["behavioral_surface_version"], "1.0.0")
        self.assertEqual(surface["behavioral_state"], "LVTF")
        # Legacy compat field preserved
        self.assertIn("msml_regime", surface)


if __name__ == "__main__":
    unittest.main()

