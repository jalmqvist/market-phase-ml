from __future__ import annotations

import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype, is_string_dtype

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation import (  # noqa: E402
    build_strategy_evaluation_id,
    build_strategy_evaluations,
)
from src.recommendation import (  # noqa: E402
    DEFAULT_RECOMMENDATION_POLICY,
    RECOMMENDATION_PER_PAIR_SCHEMA_VERSION,
    RecommendationPerPair,
    RecommendationValidationError,
    build_per_pair_recommendation_id,
    generate_per_pair_recommendations,
    recommendations_from_evaluations,
    recommendations_per_pair_to_frame,
    validate_per_pair_recommendation_frame,
    validate_per_pair_recommendation_set,
    write_recommendations_parquet,
    write_recommendations_per_pair_parquet,
)

EXPECTED_COLUMNS = [
    "recommendation_id",
    "evaluation_id",
    "pair",
    "surface_id",
    "state_id",
    "rank",
    "recommendation_policy",
    "metadata",
]


def _evaluation(
    evaluation_id: str,
    *,
    pair: str | None = "EURUSD",
    surface_id: str = "trend_vol",
    state_id: str = "LVTF",
    sharpe: float = 1.0,
    expected_return: float = 1.0,
    pair_count: int = 1,
) -> object:
    from src.evaluation import StrategyEvaluation

    metadata = {"pair_count": pair_count}
    if pair is not None:
        metadata["pair"] = pair
    return StrategyEvaluation(
        evaluation_id=evaluation_id,
        surface_id=surface_id,
        surface_version="1.0.0",
        state_id=state_id,
        strategy_id=f"strategy_{evaluation_id}",
        expected_return=expected_return,
        expected_sharpe=sharpe,
        expected_drawdown=-2.0,
        win_rate=None,
        confidence=None,
        stability=None,
        n_folds=3,
        n_trades=12,
        metadata=metadata,
    )


def _groups(recommendations: list[RecommendationPerPair]) -> dict[tuple[str, str, str], list]:
    result: dict[tuple[str, str, str], list] = {}
    for recommendation in recommendations:
        key = (
            recommendation.pair,
            recommendation.surface_id,
            recommendation.state_id,
        )
        result.setdefault(key, []).append(recommendation)
    return result


def test_groups_and_ranks_independently_by_pair_surface_and_state() -> None:
    evaluations = [
        _evaluation("a", pair="EURUSD", surface_id="surface_a", state_id="state_a", sharpe=0.3),
        _evaluation("b", pair="EURUSD", surface_id="surface_a", state_id="state_a", sharpe=0.9),
        _evaluation("c", pair="USDJPY", surface_id="surface_a", state_id="state_a", sharpe=0.4),
        _evaluation("d", pair="EURUSD", surface_id="surface_b", state_id="state_a", sharpe=0.5),
        _evaluation("e", pair="EURUSD", surface_id="surface_a", state_id="state_b", sharpe=0.2),
    ]

    recommendations = generate_per_pair_recommendations(
        evaluations, policy=DEFAULT_RECOMMENDATION_POLICY
    )
    groups = _groups(recommendations)

    assert set(groups) == {
        ("EURUSD", "surface_a", "state_a"),
        ("USDJPY", "surface_a", "state_a"),
        ("EURUSD", "surface_b", "state_a"),
        ("EURUSD", "surface_a", "state_b"),
    }
    assert [item.evaluation_id for item in groups[("EURUSD", "surface_a", "state_a")]] == [
        "b",
        "a",
    ]
    assert [item.rank for item in groups[("EURUSD", "surface_a", "state_a")]] == [1, 2]
    assert groups[("USDJPY", "surface_a", "state_a")][0].rank == 1
    assert groups[("USDJPY", "surface_a", "state_a")][0].evaluation_id == "c"


def test_top_n_applies_independently_to_each_group() -> None:
    evaluations = [
        _evaluation("eur_1", pair="EURUSD", sharpe=1.0),
        _evaluation("eur_2", pair="EURUSD", sharpe=2.0),
        _evaluation("eur_3", pair="EURUSD", sharpe=3.0),
        _evaluation("jpy_1", pair="USDJPY", sharpe=1.5),
        _evaluation("jpy_2", pair="USDJPY", sharpe=0.5),
    ]

    recommendations = generate_per_pair_recommendations(
        evaluations, policy=DEFAULT_RECOMMENDATION_POLICY, top_n=1
    )

    assert {(item.pair, item.rank) for item in recommendations} == {
        ("EURUSD", 1),
        ("USDJPY", 1),
    }
    assert {item.evaluation_id for item in recommendations} == {"eur_3", "jpy_1"}


def test_deterministic_ids_and_output_ordering() -> None:
    evaluations = [
        _evaluation("z", pair="USDJPY", sharpe=0.4),
        _evaluation("b", pair="EURUSD", sharpe=0.4),
        _evaluation("a", pair="EURUSD", sharpe=0.4),
    ]
    first = generate_per_pair_recommendations(
        evaluations, policy=DEFAULT_RECOMMENDATION_POLICY
    )
    second = generate_per_pair_recommendations(
        list(reversed(evaluations)), policy=DEFAULT_RECOMMENDATION_POLICY
    )

    assert first == second
    assert [(item.pair, item.rank, item.evaluation_id) for item in first] == [
        ("EURUSD", 1, "a"),
        ("EURUSD", 2, "b"),
        ("USDJPY", 1, "z"),
    ]
    assert first[0].recommendation_id == build_per_pair_recommendation_id(
        evaluation_id="a",
        pair="EURUSD",
        surface_id="trend_vol",
        state_id="LVTF",
        recommendation_policy=DEFAULT_RECOMMENDATION_POLICY,
        rank=1,
    )


def test_missing_and_multi_pair_metadata_are_skipped_and_logged(caplog) -> None:
    evaluations = [
        _evaluation("aggregate", pair=None, pair_count=4),
        _evaluation("invalid_aggregate", pair="EURUSD", pair_count=4),
        _evaluation("not_a_pair", pair="ALL"),
        _evaluation("concrete", pair="EURUSD"),
    ]

    with caplog.at_level(logging.WARNING, logger="src.recommendation"):
        recommendations = generate_per_pair_recommendations(
            evaluations, policy=DEFAULT_RECOMMENDATION_POLICY
        )

    assert [item.evaluation_id for item in recommendations] == ["concrete"]
    assert "Skipped 3 StrategyEvaluation object(s)" in caplog.text


def test_strategy_evaluation_pair_identity_only_changes_pair_scoped_ids() -> None:
    identity_args = {
        "surface_id": "trend_vol",
        "surface_version": "1.0.0",
        "state_id": "LVTF",
        "strategy_id": "TF1",
        "experiment_id": "exp_1",
    }
    aggregate_id = build_strategy_evaluation_id(**identity_args)
    assert aggregate_id == build_strategy_evaluation_id(**identity_args, pair=None)
    assert build_strategy_evaluation_id(**identity_args, pair="EURUSD") != build_strategy_evaluation_id(
        **identity_args, pair="USDJPY"
    )


def test_pair_scoped_evaluations_use_existing_walkforward_rows() -> None:
    frame = pd.DataFrame(
        {
            "Pair": ["EURUSD", "EURUSD", "USDJPY"],
            "Ret": [1.0, 3.0, 8.0],
            "Sharpe": [0.2, 0.4, 1.2],
            "DD": [-1.0, -3.0, -5.0],
            "Trades": [2, 4, 6],
        }
    )
    specs = [
        {
            "strategy_id": "TF1",
            "expected_return_col": "Ret",
            "expected_sharpe_col": "Sharpe",
            "expected_drawdown_col": "DD",
            "n_trades_col": "Trades",
            "strategy_role": "standalone_strategy",
        }
    ]
    pair_eval = build_strategy_evaluations(
        wf_df=frame[frame["Pair"] == "EURUSD"],
        surface_id="trend_vol",
        surface_version="1.0.0",
        state_id="LVTF",
        experiment_id="exp_1",
        mode_tag="base",
        strategy_specs=specs,
        pair="EURUSD",
    )[0]

    assert pair_eval.metadata["pair"] == "EURUSD"
    assert pair_eval.expected_return == 2.0
    assert pair_eval.expected_sharpe == 0.3
    assert pair_eval.n_trades == 6


def test_single_group_rankings_match_global_policy() -> None:
    evaluations = [
        _evaluation("a", sharpe=0.5, expected_return=1.0),
        _evaluation("b", sharpe=1.2, expected_return=0.1),
        _evaluation("c", sharpe=0.5, expected_return=2.0),
    ]
    global_recommendations = recommendations_from_evaluations(evaluations)
    per_pair_recommendations = generate_per_pair_recommendations(
        evaluations, policy=DEFAULT_RECOMMENDATION_POLICY
    )

    assert [item.evaluation_id for item in per_pair_recommendations] == [
        item.evaluation_id
        for item in sorted(global_recommendations, key=lambda item: item.rank)
    ]
    assert [item.rank for item in per_pair_recommendations] == [
        item.rank for item in sorted(global_recommendations, key=lambda item: item.rank)
    ]


def test_validation_checks_scope_rank_ids_and_referential_integrity() -> None:
    evaluations = [_evaluation("a"), _evaluation("b", sharpe=0.5)]
    recommendations = generate_per_pair_recommendations(
        evaluations, policy=DEFAULT_RECOMMENDATION_POLICY
    )
    validate_per_pair_recommendation_set(
        recommendations, known_evaluation_ids={"a", "b"}
    )

    with pytest.raises(RecommendationValidationError, match="unknown evaluation_id"):
        validate_per_pair_recommendation_set(
            recommendations, known_evaluation_ids={"a"}
        )

    duplicate = recommendations[1]
    duplicate_rank = RecommendationPerPair(
        recommendation_id=build_per_pair_recommendation_id(
            evaluation_id="a",
            pair="EURUSD",
            surface_id="trend_vol",
            state_id="LVTF",
            recommendation_policy=DEFAULT_RECOMMENDATION_POLICY,
            rank=1,
        ),
        evaluation_id="a",
        pair="EURUSD",
        surface_id="trend_vol",
        state_id="LVTF",
        rank=1,
        recommendation_policy=DEFAULT_RECOMMENDATION_POLICY,
        metadata={"schema_version": RECOMMENDATION_PER_PAIR_SCHEMA_VERSION},
    )
    with pytest.raises(RecommendationValidationError, match="duplicate"):
        validate_per_pair_recommendation_set([recommendations[0], duplicate_rank])

    altered_id = RecommendationPerPair(
        **{**recommendations[0].__dict__, "recommendation_id": "rec_invalid"}
    )
    with pytest.raises(RecommendationValidationError, match="invalid recommendation_id"):
        validate_per_pair_recommendation_set([altered_id])


def test_parquet_roundtrip_has_exact_schema_and_contents() -> None:
    evaluations = [
        _evaluation("a", pair="EURUSD", sharpe=1.0),
        _evaluation("b", pair="USDJPY", sharpe=0.5),
    ]
    recommendations = generate_per_pair_recommendations(
        evaluations, policy=DEFAULT_RECOMMENDATION_POLICY
    )
    known_ids = {evaluation.evaluation_id for evaluation in evaluations}

    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "recommendations_per_pair.parquet"
        write_recommendations_per_pair_parquet(
            recommendations=recommendations,
            output_path=path,
            known_evaluation_ids=known_ids,
        )
        loaded = pd.read_parquet(path)
        validate_per_pair_recommendation_frame(
            loaded, known_evaluation_ids=known_ids
        )
        restored = [
            RecommendationPerPair.from_record(row)
            for row in loaded.to_dict(orient="records")
        ]

    assert loaded.columns.tolist() == EXPECTED_COLUMNS
    assert all(is_string_dtype(loaded[column].dtype) for column in EXPECTED_COLUMNS if column != "rank")
    assert is_integer_dtype(loaded["rank"].dtype)
    assert restored == recommendations
    assert all(
        recommendation.metadata["schema_version"]
        == RECOMMENDATION_PER_PAIR_SCHEMA_VERSION
        for recommendation in restored
    )


def test_empty_frame_has_the_artifact_schema() -> None:
    frame = recommendations_per_pair_to_frame([])

    assert frame.columns.tolist() == EXPECTED_COLUMNS
    assert is_integer_dtype(frame["rank"].dtype)
    assert all(is_string_dtype(frame[column].dtype) for column in EXPECTED_COLUMNS if column != "rank")


def test_global_parquet_is_unchanged_when_pair_evaluations_are_added() -> None:
    aggregate_evaluations = [
        _evaluation("aggregate_a", pair=None, pair_count=2, sharpe=1.2),
        _evaluation("aggregate_b", pair=None, pair_count=2, sharpe=0.8),
    ]
    pair_evaluations = [
        _evaluation("pair_a", pair="EURUSD", sharpe=2.0),
        _evaluation("pair_b", pair="USDJPY", sharpe=1.5),
    ]

    with TemporaryDirectory() as temp_dir:
        path_without_flag = Path(temp_dir) / "global_without_flag.parquet"
        path_with_flag = Path(temp_dir) / "global_with_flag.parquet"
        write_recommendations_parquet(
            recommendations=recommendations_from_evaluations(aggregate_evaluations),
            output_path=path_without_flag,
        )
        # This is the same pair-less selection used by main.py for its unchanged
        # global recommendation path.
        global_with_pair_enabled = [
            evaluation
            for evaluation in aggregate_evaluations + pair_evaluations
            if "pair" not in evaluation.metadata
        ]
        write_recommendations_parquet(
            recommendations=recommendations_from_evaluations(global_with_pair_enabled),
            output_path=path_with_flag,
        )

        assert path_without_flag.read_bytes() == path_with_flag.read_bytes()
