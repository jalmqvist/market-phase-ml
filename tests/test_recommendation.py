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
    SUPPORTED_RECOMMENDATION_SCHEMA_VERSIONS,
    Recommendation,
    RecommendationPolicy,
    RecommendationValidationError,
    SharpeRankingPolicy,
    build_recommendation_id,
    recommendations_from_evaluations,
    recommendations_to_frame,
    validate_recommendation_set,
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


class _AltPolicy(RecommendationPolicy):
    """Alternative policy used in tests to produce a distinct policy_name."""

    @property
    def policy_name(self) -> str:
        return "other_v1"

    def rank(self, evaluations: list[StrategyEvaluation]) -> list[StrategyEvaluation]:
        return sorted(evaluations, key=lambda e: e.evaluation_id)


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
        recs = recommendations_from_evaluations(evals, policy=_AltPolicy())
        self.assertEqual(recs[0].recommendation_policy, "other_v1")


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

    def test_nan_sharpe_ranked_below_finite(self):
        evals = [
            _make_eval("eval_finite", expected_sharpe=0.1),
            _make_eval("eval_nan", expected_sharpe=float("nan")),
        ]
        ranked = self.policy.rank(evals)
        ids = [e.evaluation_id for e in ranked]
        self.assertEqual(ids, ["eval_finite", "eval_nan"])

    def test_inf_sharpe_ranked_below_finite(self):
        evals = [
            _make_eval("eval_finite", expected_sharpe=0.5),
            _make_eval("eval_posinf", expected_sharpe=float("inf")),
            _make_eval("eval_neginf", expected_sharpe=float("-inf")),
        ]
        ranked = self.policy.rank(evals)
        ids = [e.evaluation_id for e in ranked]
        # finite Sharpe ranks first; +inf/-inf both map to -inf, so id tiebreaker applies
        self.assertEqual(ids, ["eval_finite", "eval_neginf", "eval_posinf"])

    def test_nan_return_handled_when_sharpe_ties(self):
        evals = [
            _make_eval("eval_a", expected_sharpe=1.0, expected_return=float("nan")),
            _make_eval("eval_b", expected_sharpe=1.0, expected_return=0.5),
        ]
        ranked = self.policy.rank(evals)
        # finite return should rank above NaN return
        self.assertEqual(ranked[0].evaluation_id, "eval_b")

    def test_nonfinite_return_handled_when_sharpe_ties(self):
        evals = [
            _make_eval("eval_a", expected_sharpe=1.0, expected_return=float("inf")),
            _make_eval("eval_b", expected_sharpe=1.0, expected_return=0.5),
        ]
        ranked = self.policy.rank(evals)
        # finite return should rank above non-finite return
        self.assertEqual(ranked[0].evaluation_id, "eval_b")

    def test_evaluation_id_tiebreaker_with_nonfinite(self):
        # Both have NaN sharpe — evaluation_id is the final tiebreaker
        evals = [
            _make_eval("eval_zzz", expected_sharpe=float("nan")),
            _make_eval("eval_aaa", expected_sharpe=float("nan")),
        ]
        ranked = self.policy.rank(evals)
        self.assertEqual(ranked[0].evaluation_id, "eval_aaa")
        self.assertEqual(ranked[1].evaluation_id, "eval_zzz")

    def test_repeated_ranking_same_order(self):
        evals = [
            _make_eval("eval_a", expected_sharpe=float("nan")),
            _make_eval("eval_b", expected_sharpe=1.0),
            _make_eval("eval_c", expected_sharpe=float("inf")),
        ]
        order_1 = [e.evaluation_id for e in self.policy.rank(evals)]
        order_2 = [e.evaluation_id for e in self.policy.rank(evals)]
        self.assertEqual(order_1, order_2)


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
        evals = [_make_eval("eval_a")]
        recs_default = recommendations_from_evaluations(evals)
        recs_other = recommendations_from_evaluations(evals, policy=_AltPolicy())
        self.assertNotEqual(
            recs_default[0].recommendation_policy,
            recs_other[0].recommendation_policy,
        )

    def test_different_policies_distinct_recommendation_ids(self):
        evals = [_make_eval("eval_a")]
        recs_default = recommendations_from_evaluations(evals)
        recs_other = recommendations_from_evaluations(evals, policy=_AltPolicy())
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


# ---------------------------------------------------------------------------
# Phase G4 — Stable MPML–MRML Recommendation Interface
# ---------------------------------------------------------------------------


def _make_valid_recs(count: int = 2) -> list[Recommendation]:
    """Return a small valid recommendation set for G4 testing."""
    evals = [_make_eval(f"eval_{i:03d}", expected_sharpe=float(count - i)) for i in range(count)]
    return recommendations_from_evaluations(evals)


class TestG4SupportedSchemaVersions(unittest.TestCase):
    def test_supported_versions_contains_current(self):
        self.assertIn(RECOMMENDATION_SCHEMA_VERSION, SUPPORTED_RECOMMENDATION_SCHEMA_VERSIONS)

    def test_supported_versions_is_frozenset(self):
        self.assertIsInstance(SUPPORTED_RECOMMENDATION_SCHEMA_VERSIONS, frozenset)


class TestG4ValidateRecommendationSetValid(unittest.TestCase):
    """Happy-path validation — a well-formed recommendation set passes."""

    def test_valid_set_passes(self):
        recs = _make_valid_recs(3)
        known_ids = {r.evaluation_id for r in recs}
        # Should not raise
        validate_recommendation_set(recs, known_evaluation_ids=known_ids)

    def test_valid_set_passes_without_referential_check(self):
        recs = _make_valid_recs(2)
        validate_recommendation_set(recs)

    def test_empty_set_passes(self):
        validate_recommendation_set([])

    def test_single_recommendation_passes(self):
        recs = _make_valid_recs(1)
        validate_recommendation_set(recs, known_evaluation_ids={recs[0].evaluation_id})


class TestG4SchemaVersion(unittest.TestCase):
    """Schema version checks."""

    def test_supported_schema_version_passes(self):
        recs = _make_valid_recs(1)
        # Default recs use RECOMMENDATION_SCHEMA_VERSION — should pass
        validate_recommendation_set(recs)

    def test_unsupported_schema_version_fails(self):
        rec = Recommendation(
            recommendation_id="rec_abc",
            evaluation_id="eval_abc",
            rank=1,
            recommendation_policy="sharpe_rank_v1",
            metadata={"schema_version": "99.0.0"},
        )
        with self.assertRaises(RecommendationValidationError) as ctx:
            validate_recommendation_set([rec])
        self.assertIn("schema_version", str(ctx.exception))
        self.assertIn("99.0.0", str(ctx.exception))

    def test_missing_schema_version_in_metadata_fails(self):
        rec = Recommendation(
            recommendation_id="rec_abc",
            evaluation_id="eval_abc",
            rank=1,
            recommendation_policy="sharpe_rank_v1",
            metadata={},
        )
        with self.assertRaises(RecommendationValidationError) as ctx:
            validate_recommendation_set([rec])
        self.assertIn("schema_version", str(ctx.exception))
        self.assertIn("missing", str(ctx.exception))


class TestG4MissingRequiredFields(unittest.TestCase):
    """Malformed records with missing required fields must fail clearly."""

    def _base_meta(self) -> dict:
        return {"schema_version": RECOMMENDATION_SCHEMA_VERSION}

    def test_empty_recommendation_id_fails(self):
        rec = Recommendation(
            recommendation_id="",
            evaluation_id="eval_abc",
            rank=1,
            recommendation_policy="sharpe_rank_v1",
            metadata=self._base_meta(),
        )
        with self.assertRaises(RecommendationValidationError) as ctx:
            validate_recommendation_set([rec])
        self.assertIn("recommendation_id", str(ctx.exception))

    def test_empty_evaluation_id_fails(self):
        rec = Recommendation(
            recommendation_id="rec_abc",
            evaluation_id="",
            rank=1,
            recommendation_policy="sharpe_rank_v1",
            metadata=self._base_meta(),
        )
        with self.assertRaises(RecommendationValidationError) as ctx:
            validate_recommendation_set([rec])
        self.assertIn("evaluation_id", str(ctx.exception))

    def test_empty_recommendation_policy_fails(self):
        rec = Recommendation(
            recommendation_id="rec_abc",
            evaluation_id="eval_abc",
            rank=1,
            recommendation_policy="",
            metadata=self._base_meta(),
        )
        with self.assertRaises(RecommendationValidationError) as ctx:
            validate_recommendation_set([rec])
        self.assertIn("recommendation_policy", str(ctx.exception))


class TestG4InvalidRank(unittest.TestCase):
    """Invalid rank values must be rejected."""

    def _base_meta(self) -> dict:
        return {"schema_version": RECOMMENDATION_SCHEMA_VERSION}

    def _make_rec(self, rank: int) -> Recommendation:
        return Recommendation(
            recommendation_id="rec_abc",
            evaluation_id="eval_abc",
            rank=rank,
            recommendation_policy="sharpe_rank_v1",
            metadata=self._base_meta(),
        )

    def test_zero_rank_fails(self):
        with self.assertRaises(RecommendationValidationError) as ctx:
            validate_recommendation_set([self._make_rec(0)])
        self.assertIn("rank", str(ctx.exception))

    def test_negative_rank_fails(self):
        with self.assertRaises(RecommendationValidationError) as ctx:
            validate_recommendation_set([self._make_rec(-1)])
        self.assertIn("rank", str(ctx.exception))

    def test_rank_one_passes(self):
        # Should not raise
        validate_recommendation_set([self._make_rec(1)])

    def test_large_rank_passes(self):
        validate_recommendation_set([self._make_rec(999)])


class TestG4DuplicateRecommendationIds(unittest.TestCase):
    """Duplicate recommendation IDs within a set must be rejected."""

    def test_duplicate_ids_rejected(self):
        meta = {"schema_version": RECOMMENDATION_SCHEMA_VERSION}
        rec1 = Recommendation(
            recommendation_id="rec_same",
            evaluation_id="eval_a",
            rank=1,
            recommendation_policy="sharpe_rank_v1",
            metadata=meta,
        )
        rec2 = Recommendation(
            recommendation_id="rec_same",
            evaluation_id="eval_b",
            rank=2,
            recommendation_policy="sharpe_rank_v1",
            metadata=meta,
        )
        with self.assertRaises(RecommendationValidationError) as ctx:
            validate_recommendation_set([rec1, rec2])
        self.assertIn("rec_same", str(ctx.exception))

    def test_unique_ids_pass(self):
        recs = _make_valid_recs(3)
        validate_recommendation_set(recs)


class TestG4DuplicateRanks(unittest.TestCase):
    """Duplicate ranks within a set must be rejected."""

    def test_duplicate_ranks_rejected(self):
        meta = {"schema_version": RECOMMENDATION_SCHEMA_VERSION}
        rec1 = Recommendation(
            recommendation_id="rec_aaa",
            evaluation_id="eval_a",
            rank=1,
            recommendation_policy="sharpe_rank_v1",
            metadata=meta,
        )
        rec2 = Recommendation(
            recommendation_id="rec_bbb",
            evaluation_id="eval_b",
            rank=1,  # duplicate rank
            recommendation_policy="sharpe_rank_v1",
            metadata=meta,
        )
        with self.assertRaises(RecommendationValidationError) as ctx:
            validate_recommendation_set([rec1, rec2])
        self.assertIn("rank", str(ctx.exception))

    def test_unique_ranks_pass(self):
        recs = _make_valid_recs(4)
        validate_recommendation_set(recs)


class TestG4ReferentialIntegrity(unittest.TestCase):
    """Referential integrity between Recommendation.evaluation_id and StrategyEvaluation."""

    def test_valid_references_pass(self):
        recs = _make_valid_recs(3)
        known = {r.evaluation_id for r in recs}
        validate_recommendation_set(recs, known_evaluation_ids=known)

    def test_missing_evaluation_reference_fails(self):
        recs = _make_valid_recs(2)
        # Pass an empty known set — all references will be missing
        with self.assertRaises(RecommendationValidationError) as ctx:
            validate_recommendation_set(recs, known_evaluation_ids=set())
        self.assertIn("evaluation_id", str(ctx.exception))

    def test_partial_missing_reference_fails(self):
        recs = _make_valid_recs(3)
        # Only include the first evaluation_id — others will be missing
        known = {recs[0].evaluation_id}
        with self.assertRaises(RecommendationValidationError):
            validate_recommendation_set(recs, known_evaluation_ids=known)

    def test_referential_integrity_skipped_when_none(self):
        recs = _make_valid_recs(2)
        # No exception even though we pass no known IDs
        validate_recommendation_set(recs, known_evaluation_ids=None)


class TestG4DeterministicIdentity(unittest.TestCase):
    """Repeated construction with identical inputs produces the same recommendation_id."""

    def test_identical_inputs_same_id(self):
        evals = [_make_eval("eval_det", expected_sharpe=1.5)]
        recs_a = recommendations_from_evaluations(evals)
        recs_b = recommendations_from_evaluations(evals)
        self.assertEqual(recs_a[0].recommendation_id, recs_b[0].recommendation_id)

    def test_identical_inputs_same_ordering(self):
        evals = [
            _make_eval("eval_x", expected_sharpe=2.0),
            _make_eval("eval_y", expected_sharpe=1.0),
        ]
        recs_a = recommendations_from_evaluations(evals)
        recs_b = recommendations_from_evaluations(evals)
        ids_a = [r.recommendation_id for r in sorted(recs_a, key=lambda r: r.rank)]
        ids_b = [r.recommendation_id for r in sorted(recs_b, key=lambda r: r.rank)]
        self.assertEqual(ids_a, ids_b)

    def test_identical_inputs_same_ranks(self):
        evals = [_make_eval(f"eval_{i}", expected_sharpe=float(i)) for i in range(5)]
        recs_a = recommendations_from_evaluations(evals)
        recs_b = recommendations_from_evaluations(evals)
        ranks_a = sorted((r.evaluation_id, r.rank) for r in recs_a)
        ranks_b = sorted((r.evaluation_id, r.rank) for r in recs_b)
        self.assertEqual(ranks_a, ranks_b)


class TestG4ParquetRoundtrip(unittest.TestCase):
    """Full Parquet round-trip: Recommendation -> record -> parquet -> record -> Recommendation."""

    def test_full_roundtrip_lossless(self):
        recs = _make_valid_recs(3)
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "recommendations.parquet"
            write_recommendations_parquet(recommendations=recs, output_path=path)
            loaded = pd.read_parquet(path)
            restored = [Recommendation.from_record(row) for _, row in loaded.iterrows()]
            restored_sorted = sorted(restored, key=lambda r: r.rank)
            original_sorted = sorted(recs, key=lambda r: r.rank)
            for orig, rest in zip(original_sorted, restored_sorted):
                self.assertEqual(rest.recommendation_id, orig.recommendation_id)
                self.assertEqual(rest.evaluation_id, orig.evaluation_id)
                self.assertEqual(rest.rank, orig.rank)
                self.assertEqual(rest.recommendation_policy, orig.recommendation_policy)
                self.assertEqual(rest.metadata, orig.metadata)

    def test_repeated_serialization_equivalent(self):
        """Repeated serialization of the same set produces equivalent content."""
        recs = _make_valid_recs(2)
        with TemporaryDirectory() as tmp_dir:
            path_a = Path(tmp_dir) / "recs_a.parquet"
            path_b = Path(tmp_dir) / "recs_b.parquet"
            write_recommendations_parquet(recommendations=recs, output_path=path_a)
            write_recommendations_parquet(recommendations=recs, output_path=path_b)
            df_a = pd.read_parquet(path_a).sort_values("rank").reset_index(drop=True)
            df_b = pd.read_parquet(path_b).sort_values("rank").reset_index(drop=True)
            pd.testing.assert_frame_equal(df_a, df_b)


class TestG4EvidenceSeparation(unittest.TestCase):
    """Recommendation must not contain duplicated StrategyEvaluation evidence fields."""

    _EVIDENCE_FIELDS = (
        "expected_return",
        "expected_sharpe",
        "expected_drawdown",
        "confidence",
        "stability",
        "win_rate",
        "n_folds",
        "n_trades",
        "surface_id",
        "surface_version",
        "state_id",
        "strategy_id",
    )

    def test_no_evidence_fields_in_dataclass(self):
        rec = _make_valid_recs(1)[0]
        rec_fields = {f.name for f in rec.__dataclass_fields__.values()}
        for field in self._EVIDENCE_FIELDS:
            self.assertNotIn(
                field,
                rec_fields,
                msg=f"Recommendation must not contain StrategyEvaluation evidence field {field!r}.",
            )

    def test_no_evidence_fields_in_record(self):
        rec = _make_valid_recs(1)[0]
        record = rec.to_record()
        for field in self._EVIDENCE_FIELDS:
            self.assertNotIn(
                field,
                record,
                msg=f"Serialized Recommendation must not contain evidence field {field!r}.",
            )


class TestG4Provenance(unittest.TestCase):
    """A Recommendation can be traced through evaluation_id to the supporting StrategyEvaluation."""

    def test_recommendation_references_evaluation_id(self):
        evals = [_make_eval("eval_prov_001", expected_sharpe=1.0)]
        recs = recommendations_from_evaluations(evals)
        self.assertEqual(recs[0].evaluation_id, "eval_prov_001")

    def test_evaluation_id_resolves_to_strategy_evaluation(self):
        evals = [_make_eval("eval_prov_002", expected_sharpe=1.0)]
        recs = recommendations_from_evaluations(evals)
        # Build a lookup map as MRML would
        eval_map = {e.evaluation_id: e for e in evals}
        resolved = eval_map[recs[0].evaluation_id]
        self.assertIsInstance(resolved, StrategyEvaluation)

    def test_evaluation_carries_experiment_id_in_metadata(self):
        """StrategyEvaluation metadata propagates experiment_id for provenance chain."""
        eval_with_exp = StrategyEvaluation(
            evaluation_id="eval_prov_003",
            surface_id="trend_vol",
            surface_version="1.0.0",
            state_id="LVTF",
            strategy_id="PhaseAware",
            expected_return=1.0,
            expected_sharpe=0.8,
            expected_drawdown=-2.0,
            win_rate=None,
            confidence=None,
            stability=None,
            n_folds=10,
            n_trades=100,
            metadata={"experiment_id": "exp_test_001", "source": "walkforward"},
        )
        recs = recommendations_from_evaluations([eval_with_exp])
        eval_map = {"eval_prov_003": eval_with_exp}
        resolved = eval_map[recs[0].evaluation_id]
        self.assertEqual(resolved.metadata["experiment_id"], "exp_test_001")

    def test_default_ranking_policy_is_sharpe_rank_v1(self):
        """Confirm default policy remains sharpe_rank_v1 (G4 regression)."""
        evals = [_make_eval("eval_r", expected_sharpe=1.0)]
        recs = recommendations_from_evaluations(evals)
        self.assertEqual(recs[0].recommendation_policy, "sharpe_rank_v1")


# ---------------------------------------------------------------------------
# G4 production path integration tests
# ---------------------------------------------------------------------------

class TestG4ProductionPathIntegration(unittest.TestCase):
    """Integration tests for the validate → write production path.

    These tests mirror what main.py does: generate Recommendations from
    StrategyEvaluations, validate against the known evaluation ID set, then
    write to parquet.  Valid inputs must succeed and produce an unchanged
    artifact; invalid inputs (unknown evaluation_id) must fail before writing.
    """

    def _make_evaluations(self, count: int = 3) -> list[StrategyEvaluation]:
        return [
            _make_eval(f"eval_prod_{i:03d}", expected_sharpe=float(count - i))
            for i in range(count)
        ]

    def test_valid_set_validates_and_writes(self):
        """Normal valid Recommendation set is validated before serialization."""
        evals = self._make_evaluations(3)
        recs = recommendations_from_evaluations(evals)
        known_ids = frozenset(e.evaluation_id for e in evals)

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "recommendations.parquet"
            # Validate (mirrors main.py)
            validate_recommendation_set(recs, known_evaluation_ids=known_ids)
            write_recommendations_parquet(recommendations=recs, output_path=path)

            self.assertTrue(path.exists())
            loaded = pd.read_parquet(path)
            self.assertEqual(len(loaded), 3)
            for col in ("recommendation_id", "evaluation_id", "rank", "recommendation_policy", "metadata"):
                self.assertIn(col, loaded.columns)

    def test_unknown_evaluation_id_blocked_before_write(self):
        """A Recommendation with an unknown evaluation_id cannot produce the artifact."""
        evals = self._make_evaluations(2)
        recs = recommendations_from_evaluations(evals)
        # Supply an empty known set — referential integrity must fail before write
        known_ids: frozenset[str] = frozenset()

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "recommendations.parquet"
            with self.assertRaises(RecommendationValidationError):
                validate_recommendation_set(recs, known_evaluation_ids=known_ids)
            # validate raised — write must not have been called; file must not exist
            self.assertFalse(path.exists())

    def test_parquet_output_unchanged_for_valid_input(self):
        """Validation must not alter the serialized artifact content for valid inputs."""
        evals = self._make_evaluations(2)
        recs = recommendations_from_evaluations(evals)
        known_ids = frozenset(e.evaluation_id for e in evals)

        with TemporaryDirectory() as tmp_dir:
            # Write without validation (baseline)
            path_baseline = Path(tmp_dir) / "baseline.parquet"
            write_recommendations_parquet(recommendations=recs, output_path=path_baseline)
            df_baseline = pd.read_parquet(path_baseline).sort_values("rank").reset_index(drop=True)

            # Write with validation (production path)
            path_validated = Path(tmp_dir) / "validated.parquet"
            validate_recommendation_set(recs, known_evaluation_ids=known_ids)
            write_recommendations_parquet(recommendations=recs, output_path=path_validated)
            df_validated = pd.read_parquet(path_validated).sort_values("rank").reset_index(drop=True)

            pd.testing.assert_frame_equal(df_baseline, df_validated)

    def test_recommendation_ids_ranks_ordering_unchanged(self):
        """IDs, ranks, and ordering remain byte/value compatible with the pre-G4 path."""
        evals = self._make_evaluations(4)
        recs_before = recommendations_from_evaluations(evals)

        known_ids = frozenset(e.evaluation_id for e in evals)
        validate_recommendation_set(recs_before, known_evaluation_ids=known_ids)
        recs_after = recommendations_from_evaluations(evals)

        self.assertEqual(
            [(r.recommendation_id, r.rank, r.evaluation_id) for r in sorted(recs_before, key=lambda r: r.rank)],
            [(r.recommendation_id, r.rank, r.evaluation_id) for r in sorted(recs_after, key=lambda r: r.rank)],
        )

    def test_top_n_path_validates_and_writes(self):
        """Top-N production path (used in main.py --recommendation-top-n) also validates."""
        evals = self._make_evaluations(5)
        recs = recommendations_from_evaluations(evals, top_n=3)
        known_ids = frozenset(e.evaluation_id for e in evals)

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "recommendations.parquet"
            validate_recommendation_set(recs, known_evaluation_ids=known_ids)
            write_recommendations_parquet(recommendations=recs, output_path=path)
            loaded = pd.read_parquet(path)
            self.assertEqual(len(loaded), 3)
            # All referenced evaluation_ids in the artifact must be within the known set
            for eval_id in loaded["evaluation_id"]:
                self.assertIn(eval_id, known_ids)


if __name__ == "__main__":
    unittest.main()
