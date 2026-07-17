import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.behavioral_artifact_resolver import resolve_behavioral_artifact_runtime
from src.dl_surface_loader import load_dl_surface


def _base_artifact_df() -> pd.DataFrame:
    ts = pd.to_datetime(["2024-01-02 00:00:00", "2024-01-02 01:00:00"])
    return pd.DataFrame(
        {
            "pair": ["usd-jpy", "eur-jpy"],
            "timestamp": ts,
            "prediction_available_timestamp": ts - pd.Timedelta(hours=1),
            "prediction_generated_timestamp": ts - pd.Timedelta(hours=2),
            "artifact_created_timestamp": ts - pd.Timedelta(hours=3),
            "model": ["mlp", "mlp"],
            "target_horizon": [24, 24],
            "feature_set": ["price_trend", "price_trend"],
            "signal_strength": [0.4, -0.2],
            "schema_version": ["2.0.0", "2.0.0"],
        }
    )


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


class TestBehavioralArtifactResolver(unittest.TestCase):
    def test_resolver_loads_canonical_reactive_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "reactive.parquet"
            df = _base_artifact_df()
            df["surface_id"] = "reactive_jpy"
            df["surface_version"] = "1.0.0"
            df["state_id"] = "JPY_CONSENSUS_YOUNG"
            _write_parquet(artifact, df)

            runtime = resolve_behavioral_artifact_runtime(
                dl_runtime_enabled=True,
                behavioral_surface_id="reactive_jpy",
                behavioral_surface_version="1.0.0",
                dl_surface={"model": "mlp", "target_horizon": 24, "feature_set": "price_trend"},
                behavioral_state_id="JPY_CONSENSUS_YOUNG",
                explicit_artifact_path=str(artifact),
            )

            self.assertTrue(runtime.enabled)
            self.assertEqual(runtime.state_id, "JPY_CONSENSUS_YOUNG")
            self.assertEqual(runtime.surface_selector["surface_id"], "reactive_jpy")
            self.assertFalse(runtime.h1_predictions.empty)
            self.assertFalse(runtime.d1_predictions.empty)

    def test_resolver_rejects_incompatible_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "reactive.parquet"
            df = _base_artifact_df()
            df["surface_id"] = "reactive_jpy"
            df["surface_version"] = "1.0.0"
            df["state_id"] = "JPY_CONSENSUS_YOUNG"
            _write_parquet(artifact, df)

            with self.assertRaisesRegex(ValueError, "no compatible predictions found"):
                resolve_behavioral_artifact_runtime(
                    dl_runtime_enabled=True,
                    behavioral_surface_id="trend_vol",
                    behavioral_surface_version="1.0.0",
                    dl_surface={"model": "mlp", "target_horizon": 24, "feature_set": "price_trend"},
                    behavioral_state_id="LVTF",
                    explicit_artifact_path=str(artifact),
                )


class TestDLSurfaceLoaderCanonicalIdentity(unittest.TestCase):
    def test_loader_adapts_legacy_trendvol_dl_regime(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "trend_legacy.parquet"
            df = _base_artifact_df()
            df["dl_regime"] = "LVTF"
            _write_parquet(artifact, df)

            loaded = load_dl_surface(
                artifact,
                {
                    "model": "mlp",
                    "target_horizon": 24,
                    "feature_set": "price_trend",
                    "dl_regime": "LVTF",
                },
                strict=True,
            )
            self.assertFalse(loaded.empty)

    def test_loader_rejects_missing_canonical_metadata_without_legacy_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "missing_identity.parquet"
            df = _base_artifact_df()
            _write_parquet(artifact, df)

            with self.assertRaisesRegex(ValueError, "missing canonical metadata"):
                load_dl_surface(
                    artifact,
                    {
                        "surface_id": "reactive_jpy",
                        "surface_version": "1.0.0",
                        "state_id": "JPY_CONSENSUS_YOUNG",
                        "model": "mlp",
                        "target_horizon": 24,
                        "feature_set": "price_trend",
                    },
                    strict=True,
                )


if __name__ == "__main__":
    unittest.main()
