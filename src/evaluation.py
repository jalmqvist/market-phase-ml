from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

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
        "schema_version": EVALUATION_SCHEMA_VERSION,
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

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "StrategyEvaluation":
        metadata_value = record.get("metadata", {})
        if isinstance(metadata_value, str):
            metadata = json.loads(metadata_value) if metadata_value else {}
        elif isinstance(metadata_value, Mapping):
            metadata = dict(metadata_value)
        else:
            metadata = {}

        return cls(
            evaluation_id=str(record["evaluation_id"]),
            surface_id=str(record["surface_id"]),
            surface_version=str(record["surface_version"]),
            state_id=str(record["state_id"]),
            strategy_id=str(record["strategy_id"]),
            expected_return=float(record["expected_return"]),
            expected_sharpe=float(record["expected_sharpe"]),
            expected_drawdown=float(record["expected_drawdown"]),
            win_rate=None if pd.isna(record.get("win_rate")) else float(record["win_rate"]),
            confidence=None if pd.isna(record.get("confidence")) else float(record["confidence"]),
            stability=None if pd.isna(record.get("stability")) else float(record["stability"]),
            n_folds=int(record["n_folds"]),
            n_trades=int(record["n_trades"]),
            metadata=metadata,
        )


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


def build_strategy_evaluations(
    *,
    wf_df: pd.DataFrame,
    surface_id: str,
    surface_version: str,
    state_id: str,
    experiment_id: str,
    mode_tag: str,
    strategy_specs: Iterable[Mapping[str, Any]],
) -> list[StrategyEvaluation]:
    if wf_df.empty:
        return []

    evaluations: list[StrategyEvaluation] = []
    pair_count = int(wf_df["Pair"].nunique()) if "Pair" in wf_df.columns else 0
    fold_count = int(len(wf_df))

    for spec in strategy_specs:
        expected_return = float(pd.to_numeric(wf_df[spec["expected_return_col"]], errors="coerce").mean())
        expected_sharpe_series = pd.to_numeric(wf_df[spec["expected_sharpe_col"]], errors="coerce")
        expected_sharpe = float(expected_sharpe_series.mean())
        expected_drawdown = float(pd.to_numeric(wf_df[spec["expected_drawdown_col"]], errors="coerce").mean())
        n_trades = int(pd.to_numeric(wf_df[spec["n_trades_col"]], errors="coerce").fillna(0).sum())

        confidence = None
        confidence_col = spec.get("confidence_col")
        if confidence_col and confidence_col in wf_df.columns:
            confidence_value = float(pd.to_numeric(wf_df[confidence_col], errors="coerce").mean())
            if not pd.isna(confidence_value):
                confidence = confidence_value / 100.0

        stability_value = float(expected_sharpe_series.std(ddof=0))
        stability = None if pd.isna(stability_value) else stability_value
        strategy_id = str(spec["strategy_id"])
        evaluation_id = build_strategy_evaluation_id(
            surface_id=surface_id,
            surface_version=surface_version,
            state_id=state_id,
            strategy_id=strategy_id,
            experiment_id=experiment_id,
        )
        metadata = {
            "experiment_id": experiment_id,
            "source": "walkforward",
            "mode_tag": mode_tag,
            "strategy_role": spec["strategy_role"],
            "pair_count": pair_count,
            "fold_count": fold_count,
        }
        evaluations.append(
            StrategyEvaluation(
                evaluation_id=evaluation_id,
                surface_id=surface_id,
                surface_version=surface_version,
                state_id=state_id,
                strategy_id=strategy_id,
                expected_return=expected_return,
                expected_sharpe=expected_sharpe,
                expected_drawdown=expected_drawdown,
                win_rate=None,
                confidence=confidence,
                stability=stability,
                n_folds=fold_count,
                n_trades=n_trades,
                metadata=metadata,
            )
        )
    return evaluations
