import unittest
import importlib.util
from pathlib import Path


_DL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "src" / "dl_config.py"
_DL_CONFIG_SPEC = importlib.util.spec_from_file_location("dl_config_under_test", _DL_CONFIG_PATH)
_DL_CONFIG_MODULE = importlib.util.module_from_spec(_DL_CONFIG_SPEC)
assert _DL_CONFIG_SPEC is not None and _DL_CONFIG_SPEC.loader is not None
_DL_CONFIG_SPEC.loader.exec_module(_DL_CONFIG_MODULE)
infer_dl_regime_from_artifact_path = _DL_CONFIG_MODULE.infer_dl_regime_from_artifact_path
infer_dl_feature_set_from_artifact_path = _DL_CONFIG_MODULE.infer_dl_feature_set_from_artifact_path


class TestInferDLRegimeFromArtifactPath(unittest.TestCase):
    def test_infers_regime_from_canonical_filename(self):
        path = Path("/tmp/mlp__HVR__24__price_trend__20260524T175056Z.parquet")
        self.assertEqual(infer_dl_regime_from_artifact_path(path), "HVR")

    def test_infers_regime_from_freeform_filename(self):
        path = "/tmp/persistent_to_reactive_sentiment_hvtf_transfer.parquet"
        self.assertEqual(infer_dl_regime_from_artifact_path(path), "HVTF")

    def test_returns_none_when_regime_missing(self):
        path = "/tmp/persistent_to_reactive_sentiment_transfer.parquet"
        self.assertIsNone(infer_dl_regime_from_artifact_path(path))


class TestInferDLFeatureSetFromArtifactPath(unittest.TestCase):
    def test_infers_price_trend_from_canonical_filename(self):
        path = Path("/tmp/mlp__HVR__24__price_trend__20260524T175056Z.parquet")
        self.assertEqual(infer_dl_feature_set_from_artifact_path(path), "price_trend")

    def test_infers_trend_vol_only_from_canonical_filename(self):
        path = Path("/tmp/mlp__HVR__24__trend_vol_only__20260524T175056Z.parquet")
        self.assertEqual(infer_dl_feature_set_from_artifact_path(path), "trend_vol_only")

    def test_infers_trend_vol_only_lvr(self):
        path = "/tmp/mlp__LVR__24__trend_vol_only__20260601T120000Z.parquet"
        self.assertEqual(infer_dl_feature_set_from_artifact_path(path), "trend_vol_only")

    def test_infers_price_trend_lvtf(self):
        path = "/tmp/mlp__LVTF__24__price_trend__20260601T120000Z.parquet"
        self.assertEqual(infer_dl_feature_set_from_artifact_path(path), "price_trend")

    def test_returns_none_when_feature_set_missing(self):
        path = "/tmp/mlp__HVR__24__20260524T175056Z.parquet"
        self.assertIsNone(infer_dl_feature_set_from_artifact_path(path))

    def test_returns_none_for_none_input(self):
        self.assertIsNone(infer_dl_feature_set_from_artifact_path(None))


if __name__ == "__main__":
    unittest.main()
