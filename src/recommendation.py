from __future__ import annotations

import abc
import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype

from src.evaluation import StrategyEvaluation

logger = logging.getLogger(__name__)

RECOMMENDATION_SCHEMA_VERSION = "1.0.0"
RECOMMENDATION_PER_PAIR_SCHEMA_VERSION = RECOMMENDATION_SCHEMA_VERSION

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
    def from_record(cls, record: Mapping[str, Any]) -> Recommendation:
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


@dataclass(frozen=True)
class RecommendationPerPair:
    """Reference a ranked evaluation within one concrete pair and state scope."""

    recommendation_id: str
    evaluation_id: str
    pair: str
    surface_id: str
    state_id: str
    rank: int
    recommendation_policy: str
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        """Return the parquet-compatible representation of this recommendation."""
        return {
            "recommendation_id": self.recommendation_id,
            "evaluation_id": self.evaluation_id,
            "pair": self.pair,
            "surface_id": self.surface_id,
            "state_id": self.state_id,
            "rank": self.rank,
            "recommendation_policy": self.recommendation_policy,
            "metadata": json.dumps(
                self.metadata, sort_keys=True, separators=(",", ":"), default=str
            ),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> RecommendationPerPair:
        """Restore a per-pair recommendation from a serialized record."""
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
            pair=str(record["pair"]),
            surface_id=str(record["surface_id"]),
            state_id=str(record["state_id"]),
            rank=int(record["rank"]),
            recommendation_policy=str(record["recommendation_policy"]),
            metadata=metadata,
        )


def build_per_pair_recommendation_id(
    *,
    evaluation_id: str,
    pair: str,
    surface_id: str,
    state_id: str,
    recommendation_policy: str,
    rank: int,
) -> str:
    """Build a deterministic recommendation ID scoped to its pair and state."""
    payload = {
        "schema_version": RECOMMENDATION_PER_PAIR_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "pair": pair,
        "surface_id": surface_id,
        "state_id": state_id,
        "recommendation_policy": recommendation_policy,
        "rank": rank,
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"rec_{digest[:24]}"


_PER_PAIR_RECOMMENDATION_COLUMNS = [
    "recommendation_id",
    "evaluation_id",
    "pair",
    "surface_id",
    "state_id",
    "rank",
    "recommendation_policy",
    "metadata",
]
_PAIR_PATTERN = re.compile(r"^[A-Z]{6}$")


def _concrete_pair(evaluation: StrategyEvaluation) -> str | None:
    """Return a concrete pair scope, excluding absent and multi-pair evidence."""
    pair = evaluation.metadata.get("pair")
    if not isinstance(pair, str) or not _PAIR_PATTERN.fullmatch(pair):
        return None
    pair_count = evaluation.metadata.get("pair_count")
    try:
        if pair_count is not None and int(pair_count) != 1:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return pair


def generate_per_pair_recommendations(
    evaluations: list[StrategyEvaluation],
    policy: str,
    top_n: int | None = None,
) -> list[RecommendationPerPair]:
    """Rank pair-scoped evaluations independently using an existing policy."""
    if top_n is not None and top_n <= 0:
        raise ValueError(f"top_n must be a positive integer, got {top_n!r}")
    if policy != DEFAULT_RECOMMENDATION_POLICY:
        raise ValueError(f"Unsupported recommendation policy: {policy!r}")

    groups: dict[tuple[str, str, str], list[StrategyEvaluation]] = {}
    skipped_count = 0
    for evaluation in evaluations:
        pair = _concrete_pair(evaluation)
        if pair is None:
            skipped_count += 1
            continue
        key = (pair, evaluation.surface_id, evaluation.state_id)
        groups.setdefault(key, []).append(evaluation)
    if skipped_count:
        logger.warning(
            "Skipped %d StrategyEvaluation object(s) without concrete pair metadata "
            "when generating per-pair recommendations.",
            skipped_count,
        )

    recommendations: list[RecommendationPerPair] = []
    for pair, surface_id, state_id in sorted(groups):
        ranked_evaluations = DEFAULT_POLICY.rank(groups[(pair, surface_id, state_id)])
        for rank, evaluation in enumerate(ranked_evaluations, start=1):
            if top_n is not None and rank > top_n:
                break
            recommendations.append(
                RecommendationPerPair(
                    recommendation_id=build_per_pair_recommendation_id(
                        evaluation_id=evaluation.evaluation_id,
                        pair=pair,
                        surface_id=surface_id,
                        state_id=state_id,
                        recommendation_policy=policy,
                        rank=rank,
                    ),
                    evaluation_id=evaluation.evaluation_id,
                    pair=pair,
                    surface_id=surface_id,
                    state_id=state_id,
                    rank=rank,
                    recommendation_policy=policy,
                    metadata={"schema_version": RECOMMENDATION_PER_PAIR_SCHEMA_VERSION},
                )
            )
    logger.info("Generated %d per-pair Recommendation objects.", len(recommendations))
    return recommendations


def recommendations_per_pair_to_frame(
    recommendations: list[RecommendationPerPair] | tuple[RecommendationPerPair, ...],
) -> pd.DataFrame:
    """Serialize per-pair recommendations with a stable column order."""
    rows = [recommendation.to_record() for recommendation in recommendations]
    if not rows:
        return pd.DataFrame(
            {
                column: pd.Series(dtype="int64" if column == "rank" else "string")
                for column in _PER_PAIR_RECOMMENDATION_COLUMNS
            }
        )
    return (
        pd.DataFrame(rows, columns=_PER_PAIR_RECOMMENDATION_COLUMNS)
        .sort_values(["pair", "surface_id", "state_id", "rank"])
        .reset_index(drop=True)
    )


def validate_per_pair_recommendation_set(
    recommendations: list[RecommendationPerPair] | tuple[RecommendationPerPair, ...],
    *,
    known_evaluation_ids: set[str] | frozenset[str] | None = None,
) -> None:
    """Validate schema, deterministic IDs, ranks, scope uniqueness, and references."""
    validate_per_pair_recommendation_frame(
        recommendations_per_pair_to_frame(recommendations),
        known_evaluation_ids=known_evaluation_ids,
    )


def validate_per_pair_recommendation_frame(
    frame: pd.DataFrame,
    *,
    known_evaluation_ids: set[str] | frozenset[str] | None = None,
) -> None:
    """Validate required parquet columns, dtypes, records, and references."""
    missing_columns = [column for column in _PER_PAIR_RECOMMENDATION_COLUMNS if column not in frame]
    if missing_columns:
        raise RecommendationValidationError(
            f"Per-pair recommendation frame is missing columns: {missing_columns!r}."
        )
    for column in _PER_PAIR_RECOMMENDATION_COLUMNS:
        if column == "rank":
            valid_dtype = is_integer_dtype(frame[column].dtype)
        else:
            valid_dtype = is_string_dtype(frame[column].dtype)
        if not valid_dtype:
            raise RecommendationValidationError(
                f"Per-pair recommendation column {column!r} has invalid dtype "
                f"{frame[column].dtype!s}."
            )
        if frame[column].isna().any():
            raise RecommendationValidationError(
                f"Per-pair recommendation column {column!r} contains null values."
            )
        if column != "rank" and not all(
            isinstance(value, str) for value in frame[column]
        ):
            raise RecommendationValidationError(
                f"Per-pair recommendation column {column!r} must contain only strings."
            )
    recommendations = [
        RecommendationPerPair.from_record(
            {str(key): value for key, value in record.items()}
        )
        for record in frame[_PER_PAIR_RECOMMENDATION_COLUMNS].to_dict(orient="records")
    ]
    # Avoid recursive frame validation while applying record-level constraints.
    _validate_per_pair_recommendation_records(
        recommendations, known_evaluation_ids=known_evaluation_ids
    )


def _validate_per_pair_recommendation_records(
    recommendations: list[RecommendationPerPair],
    *,
    known_evaluation_ids: set[str] | frozenset[str] | None,
) -> None:
    """Apply record-level validation without rebuilding a dataframe."""
    seen_ids: set[str] = set()
    seen_ranks: set[tuple[str, str, str, int]] = set()
    for index, recommendation in enumerate(recommendations):
        position = f"per_pair_recommendation[{index}]"
        for field in (
            "recommendation_id",
            "evaluation_id",
            "pair",
            "surface_id",
            "state_id",
            "recommendation_policy",
        ):
            value = getattr(recommendation, field)
            if not isinstance(value, str) or not value:
                raise RecommendationValidationError(
                    f"{position}: {field} must be a non-empty string."
                )
        if _PAIR_PATTERN.fullmatch(recommendation.pair) is None:
            raise RecommendationValidationError(
                f"{position}: pair must be a concrete six-letter currency pair."
            )
        if (
            not isinstance(recommendation.rank, int)
            or isinstance(recommendation.rank, bool)
            or recommendation.rank < 1
        ):
            raise RecommendationValidationError(f"{position}: rank must be positive.")
        if (
            recommendation.metadata.get("schema_version")
            != RECOMMENDATION_PER_PAIR_SCHEMA_VERSION
        ):
            raise RecommendationValidationError(f"{position}: invalid schema_version.")
        expected_id = build_per_pair_recommendation_id(
            evaluation_id=recommendation.evaluation_id,
            pair=recommendation.pair,
            surface_id=recommendation.surface_id,
            state_id=recommendation.state_id,
            recommendation_policy=recommendation.recommendation_policy,
            rank=recommendation.rank,
        )
        if recommendation.recommendation_id != expected_id:
            raise RecommendationValidationError(f"{position}: invalid recommendation_id.")
        if recommendation.recommendation_id in seen_ids:
            raise RecommendationValidationError(f"{position}: duplicate recommendation_id.")
        seen_ids.add(recommendation.recommendation_id)
        group_rank = (
            recommendation.pair,
            recommendation.surface_id,
            recommendation.state_id,
            recommendation.rank,
        )
        if group_rank in seen_ranks:
            raise RecommendationValidationError(f"{position}: duplicate group rank.")
        seen_ranks.add(group_rank)
        if (
            known_evaluation_ids is not None
            and recommendation.evaluation_id not in known_evaluation_ids
        ):
            raise RecommendationValidationError(f"{position}: unknown evaluation_id.")


def write_recommendations_per_pair_parquet(
    *,
    recommendations: list[RecommendationPerPair] | tuple[RecommendationPerPair, ...],
    output_path: Path,
    known_evaluation_ids: set[str] | frozenset[str] | None = None,
) -> None:
    """Validate and write per-pair recommendations to a parquet artifact."""
    frame = recommendations_per_pair_to_frame(recommendations)
    validate_per_pair_recommendation_frame(
        frame, known_evaluation_ids=known_evaluation_ids
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


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


# ---------------------------------------------------------------------------
# G4 — Recommendation set validation
# ---------------------------------------------------------------------------

#: Schema versions that this runtime supports.
SUPPORTED_RECOMMENDATION_SCHEMA_VERSIONS: frozenset[str] = frozenset(
    {RECOMMENDATION_SCHEMA_VERSION}
)

#: Required serialization fields and their expected Python types.
_REQUIRED_RECORD_FIELDS: dict[str, type] = {
    "recommendation_id": str,
    "evaluation_id": str,
    "rank": int,
    "recommendation_policy": str,
    "metadata": str,
}


class RecommendationValidationError(ValueError):
    """Raised when a Recommendation set fails contract validation."""


def validate_recommendation_set(
    recommendations: list[Recommendation] | tuple[Recommendation, ...],
    *,
    known_evaluation_ids: set[str] | frozenset[str] | None = None,
) -> None:
    """Validate a collection of :class:`Recommendation` objects.

    This is the G4 contract-validation entry point.  It checks:

    - Recommendation schema version is supported (via ``metadata["schema_version"]``).
    - ``recommendation_id`` is present and non-empty.
    - ``evaluation_id`` is present and non-empty.
    - ``recommendation_policy`` is present and non-empty.
    - ``rank`` is a positive integer (≥ 1).
    - ``recommendation_id`` values are unique within the set.
    - ``rank`` values are unique within the set.
    - Every ``evaluation_id`` resolves to a known :class:`StrategyEvaluation`
      when *known_evaluation_ids* is provided.
    - Serialization fields produced by :meth:`Recommendation.to_record` are
      complete and correctly typed.

    Parameters
    ----------
    recommendations:
        The recommendation set to validate.
    known_evaluation_ids:
        Set of :attr:`StrategyEvaluation.evaluation_id` values available in
        the current experiment.  When provided, referential integrity is
        verified — every ``Recommendation.evaluation_id`` must appear in this
        set.  Pass ``None`` to skip referential integrity checks.

    Raises
    ------
    RecommendationValidationError
        When any contract constraint is violated.  The error message
        identifies the specific violation.
    """
    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()

    for i, rec in enumerate(recommendations):
        position = f"recommendation[{i}]"

        # --- required fields present and non-empty ---
        if not rec.recommendation_id:
            raise RecommendationValidationError(
                f"{position}: recommendation_id is missing or empty."
            )
        if not rec.evaluation_id:
            raise RecommendationValidationError(
                f"{position}: evaluation_id is missing or empty."
            )
        if not rec.recommendation_policy:
            raise RecommendationValidationError(
                f"{position}: recommendation_policy is missing or empty."
            )

        # --- rank is a valid positive integer ---
        if not isinstance(rec.rank, int) or isinstance(rec.rank, bool) or rec.rank < 1:
            raise RecommendationValidationError(
                f"{position} (id={rec.recommendation_id!r}): "
                f"rank must be a positive integer (≥ 1), got {rec.rank!r}."
            )

        # --- schema version ---
        schema_version = rec.metadata.get("schema_version")
        if schema_version is None:
            raise RecommendationValidationError(
                f"{position} (id={rec.recommendation_id!r}): "
                f"schema_version is missing from metadata."
            )
        if schema_version not in SUPPORTED_RECOMMENDATION_SCHEMA_VERSIONS:
            raise RecommendationValidationError(
                f"{position} (id={rec.recommendation_id!r}): "
                f"unsupported schema_version {schema_version!r}. "
                f"Supported versions: {sorted(SUPPORTED_RECOMMENDATION_SCHEMA_VERSIONS)}."
            )

        # --- duplicate recommendation IDs ---
        if rec.recommendation_id in seen_ids:
            raise RecommendationValidationError(
                f"Duplicate recommendation_id {rec.recommendation_id!r} "
                f"detected at {position}."
            )
        seen_ids.add(rec.recommendation_id)

        # --- duplicate ranks ---
        if rec.rank in seen_ranks:
            raise RecommendationValidationError(
                f"Duplicate rank {rec.rank!r} detected at {position} "
                f"(recommendation_id={rec.recommendation_id!r})."
            )
        seen_ranks.add(rec.rank)

        # --- referential integrity ---
        if known_evaluation_ids is not None and rec.evaluation_id not in known_evaluation_ids:
            raise RecommendationValidationError(
                f"{position} (id={rec.recommendation_id!r}): "
                f"evaluation_id {rec.evaluation_id!r} does not reference a "
                f"known StrategyEvaluation."
            )

        # --- serialization fields complete and correctly typed ---
        record = rec.to_record()
        for field, expected_type in _REQUIRED_RECORD_FIELDS.items():
            if field not in record:
                raise RecommendationValidationError(
                    f"{position} (id={rec.recommendation_id!r}): "
                    f"serialization record is missing field {field!r}."
                )
            value = record[field]
            if not isinstance(value, expected_type):
                raise RecommendationValidationError(
                    f"{position} (id={rec.recommendation_id!r}): "
                    f"serialization field {field!r} has type "
                    f"{type(value).__name__!r}, expected {expected_type.__name__!r}."
                )
