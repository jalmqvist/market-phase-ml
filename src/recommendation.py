from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.evaluation import StrategyEvaluation

logger = logging.getLogger(__name__)

RECOMMENDATION_SCHEMA_VERSION = "1.0.0"


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
    recommendation_policy: str = "identity_v1",
) -> list[Recommendation]:
    """Build Recommendation objects from an iterable of StrategyEvaluation objects.

    This is a placeholder ordering used in Phase G1 (representation only).
    Evaluations are ordered deterministically by evaluation_id to produce a
    stable rank assignment without introducing any ranking policy.
    """
    sorted_evals = sorted(evaluations, key=lambda e: e.evaluation_id)
    recommendations: list[Recommendation] = []
    for rank, evaluation in enumerate(sorted_evals, start=1):
        rec_id = build_recommendation_id(
            evaluation_id=evaluation.evaluation_id,
            recommendation_policy=recommendation_policy,
            rank=rank,
        )
        recommendations.append(
            Recommendation(
                recommendation_id=rec_id,
                evaluation_id=evaluation.evaluation_id,
                rank=rank,
                recommendation_policy=recommendation_policy,
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
