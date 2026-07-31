from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

EVALUATION_SCHEMA_VERSION = "1.0.0"


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def build_experiment_id(
    *,
    experiment: Mapping[str, Any],
    experiment_surface: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> str:
    payload: dict[str, Any] = {
        "experiment": dict(experiment),
        "experiment_surface": dict(experiment_surface or {}),
        "seed": int(seed) if seed is not None else None,
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"exp_{digest[:16]}"


def build_strategy_evaluation_id(
    *,
    surface_id: str,
    surface_version: str,
    state_id: str,
    strategy_id: str,
    experiment_id: str,
) -> str:
    payload = {
        "surface_id": surface_id,
        "surface_version": surface_version,
        "state_id": state_id,
        "strategy_id": strategy_id,
        "experiment_id": experiment_id,
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"eval_{digest[:24]}"


@dataclass(frozen=True)
class StrategyEvaluation:
    evaluation_id: str
    surface_id: str
    surface_version: str
    state_id: str
    strategy_id: str
    expected_return: float
    expected_sharpe: float
    expected_drawdown: float
    win_rate: float | None
    confidence: float | None
    stability: float | None
    n_folds: int
    n_trades: int
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "surface_id": self.surface_id,
            "surface_version": self.surface_version,
            "state_id": self.state_id,
            "strategy_id": self.strategy_id,
            "expected_return": self.expected_return,
            "expected_sharpe": self.expected_sharpe,
            "expected_drawdown": self.expected_drawdown,
            "win_rate": self.win_rate,
            "confidence": self.confidence,
            "stability": self.stability,
            "n_folds": self.n_folds,
            "n_trades": self.n_trades,
            "metadata": json.dumps(self.metadata, sort_keys=True, separators=(",", ":"), default=str),
        }


def strategy_evaluations_to_frame(
    evaluations: list[StrategyEvaluation] | tuple[StrategyEvaluation, ...],
) -> pd.DataFrame:
    rows = [evaluation.to_record() for evaluation in evaluations]
    if not rows:
        return pd.DataFrame(
            columns=[
                "evaluation_id",
                "surface_id",
                "surface_version",
                "state_id",
                "strategy_id",
                "expected_return",
                "expected_sharpe",
                "expected_drawdown",
                "win_rate",
                "confidence",
                "stability",
                "n_folds",
                "n_trades",
                "metadata",
            ]
        )
    return pd.DataFrame(rows).sort_values(["surface_id", "state_id", "strategy_id"]).reset_index(drop=True)


def write_strategy_evaluations_parquet(
    *,
    evaluations: list[StrategyEvaluation] | tuple[StrategyEvaluation, ...],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = strategy_evaluations_to_frame(evaluations)
    df.to_parquet(output_path, index=False)
