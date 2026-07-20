import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.behavioral_artifact_resolver import resolve_behavioral_artifact_runtime
from src.dl_surface_loader import load_dl_surface, validate_dl_artifact


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

    def test_resolver_diagnostics_contain_canonical_identity(self):
        """Diagnostics must include surface_id, surface_version and state_id."""
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

            diag_text = "\n".join(runtime.diagnostics)
            self.assertIn("surface_id", diag_text)
            self.assertIn("reactive_jpy", diag_text)
            self.assertIn("surface_version", diag_text)
            self.assertIn("1.0.0", diag_text)
            self.assertIn("state_id", diag_text)
            self.assertIn("JPY_CONSENSUS_YOUNG", diag_text)

    def test_resolver_diagnostics_include_discovery_mode_explicit(self):
        """Diagnostics must report 'explicit path' when artifact path is given."""
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

            diag_text = "\n".join(runtime.diagnostics)
            self.assertIn("explicit path", diag_text)

    def test_resolver_diagnostics_include_coverage(self):
        """Diagnostics must report h1_rows, d1_rows and pair_overlap."""
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

            diag_text = "\n".join(runtime.diagnostics)
            self.assertIn("h1_rows", diag_text)
            self.assertIn("d1_rows", diag_text)
            self.assertIn("pair_overlap", diag_text)

    def test_resolver_disabled_diagnostics_include_surface_id(self):
        """Disabled runtime diagnostics should still include surface_id."""
        runtime = resolve_behavioral_artifact_runtime(
            dl_runtime_enabled=False,
            behavioral_surface_id="trend_vol",
            behavioral_surface_version="1.0.0",
            dl_surface={"model": "mlp", "target_horizon": 24, "feature_set": "price_trend"},
            behavioral_state_id=None,
            explicit_artifact_path=None,
        )
        diag_text = "\n".join(runtime.diagnostics)
        self.assertIn("trend_vol", diag_text)
        self.assertFalse(runtime.enabled)


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


class TestValidateArtifactDuplicateIdentity(unittest.TestCase):
    """Tests for the strengthened duplicate-identity validation."""

    def _base_df(self) -> pd.DataFrame:
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
                "surface_id": ["trend_vol", "trend_vol"],
                "surface_version": ["1.0.0", "1.0.0"],
                "state_id": ["LVTF", "LVTF"],
            }
        )

    def test_canonical_artifact_no_duplicates_passes(self):
        """A canonical artifact with unique identity rows should pass."""
        df = self._base_df()
        result = validate_dl_artifact(df)
        self.assertIsNotNone(result)

    def test_canonical_duplicate_identity_raises(self):
        """Two rows with identical canonical identity should raise ValueError."""
        df = self._base_df()
        # Duplicate the first row — same pair+timestamp+surface identity
        dup_row = df.iloc[[0]].copy()
        df = pd.concat([df, dup_row], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate artifact identity"):
            validate_dl_artifact(df)

    def test_different_state_ids_same_pair_timestamp_is_valid(self):
        """Two rows for the same pair+timestamp but different state_id should pass.

        This represents a multi-surface/multi-state cube where the same bar
        has predictions for two different Behavioral States.
        """
        ts = pd.to_datetime(["2024-01-02 00:00:00"])
        row_base = {
            "pair": ["usd-jpy"],
            "timestamp": ts,
            "prediction_available_timestamp": ts - pd.Timedelta(hours=1),
            "prediction_generated_timestamp": ts - pd.Timedelta(hours=2),
            "artifact_created_timestamp": ts - pd.Timedelta(hours=3),
            "model": ["mlp"],
            "target_horizon": [24],
            "feature_set": ["price_trend"],
            "signal_strength": [0.4],
            "schema_version": ["2.0.0"],
            "surface_id": ["trend_vol"],
            "surface_version": ["1.0.0"],
        }
        df_lvtf = pd.DataFrame({**row_base, "state_id": ["LVTF"]})
        df_hvtf = pd.DataFrame({**row_base, "state_id": ["HVTF"]})
        df = pd.concat([df_lvtf, df_hvtf], ignore_index=True)
        # Should NOT raise — different state_id distinguishes the rows
        result = validate_dl_artifact(df)
        self.assertEqual(len(result), 2)

    def test_legacy_artifact_duplicate_pair_timestamp_raises(self):
        """Legacy artifact (no canonical cols): duplicate pair+timestamp should raise."""
        ts = pd.to_datetime(["2024-01-02 00:00:00"])
        row = {
            "pair": ["usd-jpy"],
            "timestamp": ts,
            "prediction_available_timestamp": ts - pd.Timedelta(hours=1),
            "prediction_generated_timestamp": ts - pd.Timedelta(hours=2),
            "artifact_created_timestamp": ts - pd.Timedelta(hours=3),
            "model": ["mlp"],
            "target_horizon": [24],
            "feature_set": ["price_trend"],
            "signal_strength": [0.4],
            "schema_version": ["2.0.0"],
            "dl_regime": ["LVTF"],
        }
        df = pd.concat([pd.DataFrame(row), pd.DataFrame(row)], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate artifact identity"):
            validate_dl_artifact(df)


if __name__ == "__main__":
    unittest.main()

