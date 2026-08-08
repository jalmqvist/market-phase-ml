from __future__ import annotations

import abc
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.evaluation import StrategyEvaluation

logger = logging.getLogger(__name__)

RECOMMENDATION_SCHEMA_VERSION = "1.0.0"

# Default policy used by Phase G2.
DEFAULT_RECOMMENDATION_POLICY = "sharpe_rank_v1"


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def build_recommendation_id(
    *,
    evaluation_id: str,
    recommendation_policy: str,
    rank: int,
) -> str:
    payload = {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "recommendation_policy": recommendation_policy,
        "rank": rank,
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"rec_{digest[:24]}"


# ---------------------------------------------------------------------------
# Recommendation Policy abstraction
# ---------------------------------------------------------------------------


class RecommendationPolicy(abc.ABC):
    """Abstract base class for recommendation policies.

    A policy is a deterministic mapping from a collection of
    StrategyEvaluation objects to an ordered list of StrategyEvaluation
    objects (highest rank first).  Policies must not perform walk-forward
    evaluation or modify StrategyEvaluation objects.
    """

    @property
    @abc.abstractmethod
    def policy_name(self) -> str:
        """Stable identifier for this policy."""

    @abc.abstractmethod
    def rank(self, evaluations: list[StrategyEvaluation]) -> list[StrategyEvaluation]:
        """Return evaluations in descending preference order (rank 1 first).

        The returned list must contain exactly the same elements as the
        input list (no filtering, no duplication).  Filtering by Top-N
        is performed outside the policy.
        """


def _finite_or_neginf(value: float | None) -> float:
    """Return *value* if it is a finite float, otherwise ``-inf``.

    ``None``, ``NaN``, ``+inf``, and ``-inf`` are all treated as
    missing/worst-ranked so that non-finite inputs sort below any finite
    value in a descending ordering.
    """
    if value is None or not math.isfinite(value):
        return float("-inf")
    return value


class SharpeRankingPolicy(RecommendationPolicy):
    """Default G2 recommendation policy: rank by expected_sharpe descending.

    Tie-breaking (all deterministic):
      1. expected_sharpe descending  (None/NaN/non-finite → worst)
      2. expected_return descending  (None/NaN/non-finite → worst)
      3. evaluation_id ascending (lexicographic)

    This policy is intentionally simple and transparent.  It operates only
    on StrategyEvaluation evidence fields and contains no strategy-specific
    or Behavioral Surface-specific logic.
    """

    @property
    def policy_name(self) -> str:
        return "sharpe_rank_v1"

    def rank(self, evaluations: list[StrategyEvaluation]) -> list[StrategyEvaluation]:
        return sorted(
            evaluations,
            key=lambda e: (
                -_finite_or_neginf(e.expected_sharpe),
                -_finite_or_neginf(e.expected_return),
                e.evaluation_id,
            ),
        )


#: Singleton instance of the default policy used across the runtime.
DEFAULT_POLICY: RecommendationPolicy = SharpeRankingPolicy()


@dataclass(frozen=True)
class Recommendation:
    recommendation_id: str
    evaluation_id: str
    rank: int
    recommendation_policy: str
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "evaluation_id": self.evaluation_id,
            "rank": self.rank,
            "recommendation_policy": self.recommendation_policy,
            "metadata": json.dumps(self.metadata, sort_keys=True, separators=(",", ":"), default=str),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Recommendation":
        metadata_value = record.get("metadata", {})
        if isinstance(metadata_value, str):
            metadata = json.loads(metadata_value) if metadata_value else {}
        elif isinstance(metadata_value, Mapping):
            metadata = dict(metadata_value)
        else:
            metadata = {}

        return cls(
            recommendation_id=str(record["recommendation_id"]),
            evaluation_id=str(record["evaluation_id"]),
            rank=int(record["rank"]),
            recommendation_policy=str(record["recommendation_policy"]),
            metadata=metadata,
        )


def recommendations_from_evaluations(
    evaluations: Iterable[StrategyEvaluation],
    *,
    policy: RecommendationPolicy | None = None,
    top_n: int | None = None,
) -> list[Recommendation]:
    """Build Recommendation objects from an iterable of StrategyEvaluation objects.

    Phase G2: evaluations are ordered by the provided *policy* (default:
    :class:`SharpeRankingPolicy`).  Ranks are assigned in descending
    preference order (rank 1 = most preferred).

    Parameters
    ----------
    evaluations:
        StrategyEvaluation objects to rank.
    policy:
        Recommendation policy to apply.  Defaults to :data:`DEFAULT_POLICY`
        (``sharpe_rank_v1``).
    top_n:
        If provided, only the top *N* recommendations are returned.
        Must be a positive integer.  If *top_n* exceeds the number of
        available evaluations all evaluations are returned.  Ranking and
        rank assignment are always performed over the full set before
        truncation so that rank values remain consistent.
    """
    if top_n is not None and top_n <= 0:
        raise ValueError(f"top_n must be a positive integer, got {top_n!r}")

    active_policy = policy if policy is not None else DEFAULT_POLICY
    all_evals = list(evaluations)
    ranked_evals = active_policy.rank(all_evals)

    recommendations: list[Recommendation] = []
    for rank, evaluation in enumerate(ranked_evals, start=1):
        if top_n is not None and rank > top_n:
            break
        rec_id = build_recommendation_id(
            evaluation_id=evaluation.evaluation_id,
            recommendation_policy=active_policy.policy_name,
            rank=rank,
        )
        recommendations.append(
            Recommendation(
                recommendation_id=rec_id,
                evaluation_id=evaluation.evaluation_id,
                rank=rank,
                recommendation_policy=active_policy.policy_name,
                metadata={
                    "schema_version": RECOMMENDATION_SCHEMA_VERSION,
                },
            )
        )
    logger.info("Generated %d Recommendation objects.", len(recommendations))
    return recommendations


def recommendations_to_frame(
    recommendations: list[Recommendation] | tuple[Recommendation, ...],
) -> pd.DataFrame:
    rows = [rec.to_record() for rec in recommendations]
    if not rows:
        return pd.DataFrame(
            columns=[
                "recommendation_id",
                "evaluation_id",
                "rank",
                "recommendation_policy",
                "metadata",
            ]
        )
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def write_recommendations_parquet(
    *,
    recommendations: list[Recommendation] | tuple[Recommendation, ...],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = recommendations_to_frame(recommendations)
    df.to_parquet(output_path, index=False)
