from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.recommendation import (  # noqa: E402
    RECOMMENDATION_SCHEMA_VERSION,
    Recommendation,
    build_recommendation_id,
    recommendations_from_evaluations,
    recommendations_to_frame,
    write_recommendations_parquet,
)
from src.evaluation import StrategyEvaluation


def _make_evaluation(evaluation_id: str = "eval_abc123") -> StrategyEvaluation:
    return StrategyEvaluation(
        evaluation_id=evaluation_id,
        surface_id="trend_vol",
        surface_version="1.0.0",
        state_id="LVTF",
        strategy_id="PhaseAware",
        expected_return=1.2,
        expected_sharpe=0.5,
        expected_drawdown=-3.0,
        win_rate=None,
        confidence=0.4,
        stability=0.1,
        n_folds=12,
        n_trades=120,
        metadata={"experiment_id": "exp_1"},
    )


class TestRecommendationSchemaVersion(unittest.TestCase):
    def test_schema_version_is_string(self):
        self.assertIsInstance(RECOMMENDATION_SCHEMA_VERSION, str)

    def test_schema_version_format(self):
        parts = RECOMMENDATION_SCHEMA_VERSION.split(".")
        self.assertEqual(len(parts), 3)
        for part in parts:
            self.assertTrue(part.isdigit(), f"Non-numeric part: {part!r}")


class TestBuildRecommendationId(unittest.TestCase):
    def test_deterministic(self):
        first = build_recommendation_id(
            evaluation_id="eval_abc",
            recommendation_policy="identity_v1",
            rank=1,
        )
        second = build_recommendation_id(
            evaluation_id="eval_abc",
            recommendation_policy="identity_v1",
            rank=1,
        )
        self.assertEqual(first, second)

    def test_starts_with_prefix(self):
        rec_id = build_recommendation_id(
            evaluation_id="eval_abc",
            recommendation_policy="identity_v1",
            rank=1,
        )
        self.assertTrue(rec_id.startswith("rec_"))

    def test_schema_aware_identity(self):
        """ID must change when RECOMMENDATION_SCHEMA_VERSION changes (encoded in payload)."""
        payload = {
            "schema_version": RECOMMENDATION_SCHEMA_VERSION,
            "evaluation_id": "eval_abc",
            "recommendation_policy": "identity_v1",
            "rank": 1,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        expected = f"rec_{digest[:24]}"
        actual = build_recommendation_id(
            evaluation_id="eval_abc",
            recommendation_policy="identity_v1",
            rank=1,
        )
        self.assertEqual(actual, expected)

    def test_different_ranks_produce_different_ids(self):
        id1 = build_recommendation_id(
            evaluation_id="eval_abc",
            recommendation_policy="identity_v1",
            rank=1,
        )
        id2 = build_recommendation_id(
            evaluation_id="eval_abc",
            recommendation_policy="identity_v1",
            rank=2,
        )
        self.assertNotEqual(id1, id2)

    def test_different_policies_produce_different_ids(self):
        id1 = build_recommendation_id(
            evaluation_id="eval_abc",
            recommendation_policy="identity_v1",
            rank=1,
        )
        id2 = build_recommendation_id(
            evaluation_id="eval_abc",
            recommendation_policy="other_policy",
            rank=1,
        )
        self.assertNotEqual(id1, id2)

    def test_different_evaluation_ids_produce_different_ids(self):
        id1 = build_recommendation_id(
            evaluation_id="eval_abc",
            recommendation_policy="identity_v1",
            rank=1,
        )
        id2 = build_recommendation_id(
            evaluation_id="eval_xyz",
            recommendation_policy="identity_v1",
            rank=1,
        )
        self.assertNotEqual(id1, id2)


class TestRecommendationSerialization(unittest.TestCase):
    def _make_rec(self) -> Recommendation:
        rec_id = build_recommendation_id(
            evaluation_id="eval_abc123",
            recommendation_policy="identity_v1",
            rank=1,
        )
        return Recommendation(
            recommendation_id=rec_id,
            evaluation_id="eval_abc123",
            rank=1,
            recommendation_policy="identity_v1",
            metadata={"schema_version": RECOMMENDATION_SCHEMA_VERSION},
        )

    def test_to_record_keys(self):
        rec = self._make_rec()
        record = rec.to_record()
        expected_keys = {
            "recommendation_id",
            "evaluation_id",
            "rank",
            "recommendation_policy",
            "metadata",
        }
        self.assertEqual(set(record.keys()), expected_keys)

    def test_to_record_metadata_is_json_string(self):
        rec = self._make_rec()
        record = rec.to_record()
        parsed = json.loads(record["metadata"])
        self.assertIsInstance(parsed, dict)

    def test_from_record_roundtrip(self):
        rec = self._make_rec()
        record = rec.to_record()
        restored = Recommendation.from_record(record)
        self.assertEqual(restored.recommendation_id, rec.recommendation_id)
        self.assertEqual(restored.evaluation_id, rec.evaluation_id)
        self.assertEqual(restored.rank, rec.rank)
        self.assertEqual(restored.recommendation_policy, rec.recommendation_policy)
        self.assertEqual(restored.metadata, rec.metadata)

    def test_from_record_with_dict_metadata(self):
        record = {
            "recommendation_id": "rec_aaa",
            "evaluation_id": "eval_bbb",
            "rank": 2,
            "recommendation_policy": "identity_v1",
            "metadata": {"schema_version": "1.0.0"},
        }
        rec = Recommendation.from_record(record)
        self.assertEqual(rec.metadata["schema_version"], "1.0.0")

    def test_from_record_with_empty_metadata(self):
        record = {
            "recommendation_id": "rec_aaa",
            "evaluation_id": "eval_bbb",
            "rank": 1,
            "recommendation_policy": "identity_v1",
            "metadata": "",
        }
        rec = Recommendation.from_record(record)
        self.assertEqual(rec.metadata, {})


class TestRecommendationsFromEvaluations(unittest.TestCase):
    def test_produces_one_recommendation_per_evaluation(self):
        evals = [
            _make_evaluation("eval_001"),
            _make_evaluation("eval_002"),
        ]
        recs = recommendations_from_evaluations(evals)
        self.assertEqual(len(recs), 2)

    def test_ranks_are_sequential_from_one(self):
        evals = [
            _make_evaluation("eval_001"),
            _make_evaluation("eval_002"),
            _make_evaluation("eval_003"),
        ]
        recs = recommendations_from_evaluations(evals)
        ranks = sorted(r.rank for r in recs)
        self.assertEqual(ranks, [1, 2, 3])

    def test_deterministic_ordering(self):
        evals_a = [_make_evaluation("eval_zzz"), _make_evaluation("eval_aaa")]
        evals_b = [_make_evaluation("eval_aaa"), _make_evaluation("eval_zzz")]
        recs_a = recommendations_from_evaluations(evals_a)
        recs_b = recommendations_from_evaluations(evals_b)
        ids_a = [r.recommendation_id for r in sorted(recs_a, key=lambda r: r.rank)]
        ids_b = [r.recommendation_id for r in sorted(recs_b, key=lambda r: r.rank)]
        self.assertEqual(ids_a, ids_b)

    def test_evaluation_id_reference(self):
        evals = [_make_evaluation("eval_abc")]
        recs = recommendations_from_evaluations(evals)
        self.assertEqual(recs[0].evaluation_id, "eval_abc")

    def test_recommendation_id_is_deterministic(self):
        evals = [_make_evaluation("eval_abc")]
        recs_first = recommendations_from_evaluations(evals)
        recs_second = recommendations_from_evaluations(evals)
        self.assertEqual(recs_first[0].recommendation_id, recs_second[0].recommendation_id)

    def test_empty_evaluations_returns_empty(self):
        recs = recommendations_from_evaluations([])
        self.assertEqual(recs, [])

    def test_policy_is_set(self):
        evals = [_make_evaluation()]
        recs = recommendations_from_evaluations(evals, recommendation_policy="test_policy")
        self.assertEqual(recs[0].recommendation_policy, "test_policy")


class TestRecommendationsToFrame(unittest.TestCase):
    def test_columns(self):
        evals = [_make_evaluation("eval_001"), _make_evaluation("eval_002")]
        recs = recommendations_from_evaluations(evals)
        df = recommendations_to_frame(recs)
        expected_cols = {"recommendation_id", "evaluation_id", "rank", "recommendation_policy", "metadata"}
        self.assertEqual(set(df.columns), expected_cols)

    def test_empty_frame_has_correct_columns(self):
        df = recommendations_to_frame([])
        expected_cols = {"recommendation_id", "evaluation_id", "rank", "recommendation_policy", "metadata"}
        self.assertEqual(set(df.columns), expected_cols)
        self.assertEqual(len(df), 0)

    def test_sorted_by_rank(self):
        evals = [_make_evaluation("eval_zzz"), _make_evaluation("eval_aaa")]
        recs = recommendations_from_evaluations(evals)
        df = recommendations_to_frame(recs)
        self.assertTrue(list(df["rank"]) == sorted(df["rank"]))


class TestRecommendationsParquetRoundtrip(unittest.TestCase):
    def test_write_and_read(self):
        evals = [_make_evaluation("eval_001"), _make_evaluation("eval_002")]
        recs = recommendations_from_evaluations(evals)
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "recommendations.parquet"
            write_recommendations_parquet(recommendations=recs, output_path=path)
            self.assertTrue(path.exists())
            loaded = pd.read_parquet(path)
            self.assertEqual(len(loaded), 2)
            cols = set(loaded.columns)
            self.assertIn("recommendation_id", cols)
            self.assertIn("evaluation_id", cols)
            self.assertIn("rank", cols)
            self.assertIn("recommendation_policy", cols)
            self.assertIn("metadata", cols)

    def test_parquet_roundtrip_preserves_values(self):
        evals = [_make_evaluation("eval_abc")]
        recs = recommendations_from_evaluations(evals)
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "recommendations.parquet"
            write_recommendations_parquet(recommendations=recs, output_path=path)
            loaded = pd.read_parquet(path)
            row = loaded.iloc[0]
            restored = Recommendation.from_record(row)
            self.assertEqual(restored.recommendation_id, recs[0].recommendation_id)
            self.assertEqual(restored.evaluation_id, recs[0].evaluation_id)
            self.assertEqual(restored.rank, recs[0].rank)
            self.assertEqual(restored.recommendation_policy, recs[0].recommendation_policy)

    def test_empty_parquet_roundtrip(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "recommendations.parquet"
            write_recommendations_parquet(recommendations=[], output_path=path)
            self.assertTrue(path.exists())
            loaded = pd.read_parquet(path)
            self.assertEqual(len(loaded), 0)


if __name__ == "__main__":
    unittest.main()
