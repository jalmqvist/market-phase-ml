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
    DEFAULT_POLICY,
    DEFAULT_RECOMMENDATION_POLICY,
    RECOMMENDATION_SCHEMA_VERSION,
    Recommendation,
    RecommendationPolicy,
    SharpeRankingPolicy,
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
        class _NamedPolicy(RecommendationPolicy):
            @property
            def policy_name(self) -> str:
                return "test_policy"

            def rank(self, evaluations):
                return sorted(evaluations, key=lambda e: e.evaluation_id)

        evals = [_make_evaluation()]
        recs = recommendations_from_evaluations(evals, policy=_NamedPolicy())
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


# ---------------------------------------------------------------------------
# Phase G2 tests
# ---------------------------------------------------------------------------


def _make_eval(
    evaluation_id: str,
    expected_sharpe: float = 0.5,
    expected_return: float = 1.0,
) -> StrategyEvaluation:
    return StrategyEvaluation(
        evaluation_id=evaluation_id,
        surface_id="trend_vol",
        surface_version="1.0.0",
        state_id="LVTF",
        strategy_id="PhaseAware",
        expected_return=expected_return,
        expected_sharpe=expected_sharpe,
        expected_drawdown=-3.0,
        win_rate=None,
        confidence=None,
        stability=None,
        n_folds=12,
        n_trades=120,
        metadata={},
    )


class TestDefaultPolicyConstant(unittest.TestCase):
    def test_default_recommendation_policy_name(self):
        self.assertEqual(DEFAULT_RECOMMENDATION_POLICY, "sharpe_rank_v1")

    def test_default_policy_is_sharpe_ranking(self):
        self.assertIsInstance(DEFAULT_POLICY, SharpeRankingPolicy)

    def test_default_policy_name_matches_constant(self):
        self.assertEqual(DEFAULT_POLICY.policy_name, DEFAULT_RECOMMENDATION_POLICY)


class TestSharpeRankingPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = SharpeRankingPolicy()

    def test_policy_name(self):
        self.assertEqual(self.policy.policy_name, "sharpe_rank_v1")

    def test_is_recommendation_policy(self):
        self.assertIsInstance(self.policy, RecommendationPolicy)

    def test_ranks_by_expected_sharpe_descending(self):
        evals = [
            _make_eval("eval_a", expected_sharpe=0.3),
            _make_eval("eval_b", expected_sharpe=1.5),
            _make_eval("eval_c", expected_sharpe=0.7),
        ]
        ranked = self.policy.rank(evals)
        sharpes = [e.expected_sharpe for e in ranked]
        self.assertEqual(sharpes, sorted(sharpes, reverse=True))

    def test_tie_break_by_expected_return_descending(self):
        # Same sharpe, different returns
        evals = [
            _make_eval("eval_a", expected_sharpe=1.0, expected_return=0.5),
            _make_eval("eval_b", expected_sharpe=1.0, expected_return=2.0),
            _make_eval("eval_c", expected_sharpe=1.0, expected_return=1.0),
        ]
        ranked = self.policy.rank(evals)
        ids = [e.evaluation_id for e in ranked]
        self.assertEqual(ids, ["eval_b", "eval_c", "eval_a"])

    def test_tie_break_by_evaluation_id_ascending(self):
        # Same sharpe and return, different ids
        evals = [
            _make_eval("eval_zzz", expected_sharpe=1.0, expected_return=1.0),
            _make_eval("eval_aaa", expected_sharpe=1.0, expected_return=1.0),
            _make_eval("eval_mmm", expected_sharpe=1.0, expected_return=1.0),
        ]
        ranked = self.policy.rank(evals)
        ids = [e.evaluation_id for e in ranked]
        self.assertEqual(ids, ["eval_aaa", "eval_mmm", "eval_zzz"])

    def test_deterministic_same_input(self):
        evals = [
            _make_eval("eval_a", expected_sharpe=1.2, expected_return=1.0),
            _make_eval("eval_b", expected_sharpe=0.8, expected_return=2.0),
            _make_eval("eval_c", expected_sharpe=1.2, expected_return=0.5),
        ]
        result_a = [e.evaluation_id for e in self.policy.rank(evals)]
        result_b = [e.evaluation_id for e in self.policy.rank(evals)]
        self.assertEqual(result_a, result_b)

    def test_deterministic_different_input_order(self):
        evals_fwd = [
            _make_eval("eval_a", expected_sharpe=1.2),
            _make_eval("eval_b", expected_sharpe=0.8),
        ]
        evals_rev = list(reversed(evals_fwd))
        ranked_fwd = [e.evaluation_id for e in self.policy.rank(evals_fwd)]
        ranked_rev = [e.evaluation_id for e in self.policy.rank(evals_rev)]
        self.assertEqual(ranked_fwd, ranked_rev)

    def test_empty_input(self):
        self.assertEqual(self.policy.rank([]), [])

    def test_single_evaluation(self):
        evals = [_make_eval("eval_x", expected_sharpe=0.9)]
        ranked = self.policy.rank(evals)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].evaluation_id, "eval_x")

    def test_preserves_all_evaluations(self):
        evals = [_make_eval(f"eval_{i}", expected_sharpe=float(i)) for i in range(5)]
        ranked = self.policy.rank(evals)
        self.assertEqual(len(ranked), len(evals))
        self.assertEqual(
            {e.evaluation_id for e in ranked},
            {e.evaluation_id for e in evals},
        )


class TestRecommendationsFromEvaluationsG2(unittest.TestCase):
    def test_default_policy_is_sharpe_rank(self):
        evals = [
            _make_eval("eval_a", expected_sharpe=0.5),
            _make_eval("eval_b", expected_sharpe=1.5),
        ]
        recs = recommendations_from_evaluations(evals)
        # Rank 1 must be the higher-sharpe evaluation
        rank1 = next(r for r in recs if r.rank == 1)
        self.assertEqual(rank1.evaluation_id, "eval_b")

    def test_uses_default_policy_name(self):
        evals = [_make_eval("eval_x")]
        recs = recommendations_from_evaluations(evals)
        self.assertEqual(recs[0].recommendation_policy, DEFAULT_RECOMMENDATION_POLICY)

    def test_recommendation_does_not_duplicate_evaluation_evidence(self):
        evals = [_make_eval("eval_a", expected_sharpe=1.0, expected_return=2.0)]
        recs = recommendations_from_evaluations(evals)
        rec_record = recs[0].to_record()
        for forbidden in (
            "expected_return", "expected_sharpe", "expected_drawdown",
            "confidence", "stability", "win_rate", "n_folds", "n_trades",
        ):
            self.assertNotIn(forbidden, rec_record)

    def test_top_n_limits_results(self):
        evals = [_make_eval(f"eval_{i}", expected_sharpe=float(i)) for i in range(5)]
        recs = recommendations_from_evaluations(evals, top_n=3)
        self.assertEqual(len(recs), 3)

    def test_top_n_returns_highest_ranked(self):
        evals = [
            _make_eval("eval_low", expected_sharpe=0.1),
            _make_eval("eval_mid", expected_sharpe=0.5),
            _make_eval("eval_high", expected_sharpe=1.5),
        ]
        recs = recommendations_from_evaluations(evals, top_n=2)
        ids = {r.evaluation_id for r in recs}
        self.assertIn("eval_high", ids)
        self.assertIn("eval_mid", ids)
        self.assertNotIn("eval_low", ids)

    def test_top_n_larger_than_evaluations_returns_all(self):
        evals = [_make_eval("eval_a"), _make_eval("eval_b")]
        recs = recommendations_from_evaluations(evals, top_n=100)
        self.assertEqual(len(recs), 2)

    def test_top_n_exactly_equal_to_count(self):
        evals = [_make_eval("eval_a"), _make_eval("eval_b"), _make_eval("eval_c")]
        recs = recommendations_from_evaluations(evals, top_n=3)
        self.assertEqual(len(recs), 3)

    def test_top_n_ordering_remains_deterministic(self):
        evals = [_make_eval(f"eval_{i}", expected_sharpe=float(i)) for i in range(5)]
        recs_a = recommendations_from_evaluations(evals, top_n=3)
        recs_b = recommendations_from_evaluations(evals, top_n=3)
        self.assertEqual(
            [r.recommendation_id for r in recs_a],
            [r.recommendation_id for r in recs_b],
        )

    def test_top_n_invalid_raises(self):
        evals = [_make_eval("eval_a")]
        with self.assertRaises(ValueError):
            recommendations_from_evaluations(evals, top_n=0)
        with self.assertRaises(ValueError):
            recommendations_from_evaluations(evals, top_n=-1)

    def test_empty_evaluations_top_n(self):
        recs = recommendations_from_evaluations([], top_n=5)
        self.assertEqual(recs, [])

    def test_ranks_are_consecutive_from_one(self):
        evals = [_make_eval(f"eval_{i}") for i in range(4)]
        recs = recommendations_from_evaluations(evals)
        ranks = sorted(r.rank for r in recs)
        self.assertEqual(ranks, [1, 2, 3, 4])

    def test_recommendation_id_deterministic(self):
        evals = [_make_eval("eval_x", expected_sharpe=1.0)]
        recs1 = recommendations_from_evaluations(evals)
        recs2 = recommendations_from_evaluations(evals)
        self.assertEqual(recs1[0].recommendation_id, recs2[0].recommendation_id)

    def test_recommendation_references_correct_evaluation_id(self):
        evals = [_make_eval("eval_specific")]
        recs = recommendations_from_evaluations(evals)
        self.assertEqual(recs[0].evaluation_id, "eval_specific")

    def test_different_policy_objects_produce_distinct_policy_names(self):
        class _OtherPolicy(RecommendationPolicy):
            @property
            def policy_name(self) -> str:
                return "other_v1"

            def rank(self, evaluations):
                return sorted(evaluations, key=lambda e: e.evaluation_id)

        evals = [_make_eval("eval_a")]
        recs_default = recommendations_from_evaluations(evals)
        recs_other = recommendations_from_evaluations(evals, policy=_OtherPolicy())
        self.assertNotEqual(
            recs_default[0].recommendation_policy,
            recs_other[0].recommendation_policy,
        )

    def test_different_policies_distinct_recommendation_ids(self):
        class _OtherPolicy(RecommendationPolicy):
            @property
            def policy_name(self) -> str:
                return "other_v1"

            def rank(self, evaluations):
                return sorted(evaluations, key=lambda e: e.evaluation_id)

        evals = [_make_eval("eval_a")]
        recs_default = recommendations_from_evaluations(evals)
        recs_other = recommendations_from_evaluations(evals, policy=_OtherPolicy())
        self.assertNotEqual(
            recs_default[0].recommendation_id,
            recs_other[0].recommendation_id,
        )

    def test_g1_parquet_roundtrip_still_valid(self):
        """G1 serialization contract must remain valid."""
        evals = [_make_eval("eval_001"), _make_eval("eval_002", expected_sharpe=1.0)]
        recs = recommendations_from_evaluations(evals)
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "recommendations.parquet"
            write_recommendations_parquet(recommendations=recs, output_path=path)
            loaded = pd.read_parquet(path)
            self.assertEqual(len(loaded), 2)
            cols = set(loaded.columns)
            for col in ("recommendation_id", "evaluation_id", "rank", "recommendation_policy", "metadata"):
                self.assertIn(col, cols)
            restored = Recommendation.from_record(loaded.iloc[0])
            self.assertEqual(restored.recommendation_id, recs[0].recommendation_id)


if __name__ == "__main__":
    unittest.main()
