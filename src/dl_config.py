"""
Configuration for the DL surface integration (market-sentiment-ml consumer).

All settings have safe defaults so that DL signals are *disabled* by default,
keeping existing pipelines completely unaffected until explicitly opted in.

Override via environment variables
-----------------------------------
DL_SIGNALS_ENABLED          "true" to enable (default: false)
DL_PREDICTION_ARTIFACT_PATH absolute or relative path to one exported
                            DL surface parquet artifact (preferred).
DL_SIGNALS_CUBE_PATH        backward-compatible alias for
                            DL_PREDICTION_ARTIFACT_PATH.
DL_SURFACE_MODEL            model name identifier  (default: lstm)
DL_SURFACE_TARGET_HORIZON   target horizon in bars as an integer (default: 24)
DL_SURFACE_FEATURE_SET      feature set identifier (default: price_trend)
DL_SURFACE_REGIME           dl_regime for surface selection (default: HVTF)
                            Valid values: HVTF, LVTF, HVR, LVR
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Toggle — off by default so existing pipelines are unaffected
# ---------------------------------------------------------------------------
DL_SIGNALS_ENABLED: bool = (
    os.environ.get("DL_SIGNALS_ENABLED", "false").lower() == "true"
)

# ---------------------------------------------------------------------------
# Artifact path — defaults to sibling-repo layout
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent  # src/ -> market-phase-ml/

_DEFAULT_ARTIFACT_PATH = (
    _REPO_ROOT.parent
    / "market-sentiment-ml"
    / "data"
    / "output"
    / "dl_predictions"
)

# Prefer the new env var, keep legacy var as fallback.
_artifact_path_value = os.environ.get("DL_PREDICTION_ARTIFACT_PATH")
if _artifact_path_value is None:
    _artifact_path_value = os.environ.get(
        "DL_SIGNALS_CUBE_PATH",
        str(_DEFAULT_ARTIFACT_PATH),
    )

DL_PREDICTION_ARTIFACT_PATH: Path = Path(_artifact_path_value)

# Backward-compatible alias for existing imports/usages.
DL_SIGNALS_CUBE_PATH: Path = DL_PREDICTION_ARTIFACT_PATH


def resolve_dl_prediction_artifact_path(path: Path | None = None) -> Path | None:
    """
    Resolve the configured DL parquet artifact path.

    Parameters
    ----------
    path : Path | None, default None
        Optional override path. When ``None``, uses
        ``DL_PREDICTION_ARTIFACT_PATH``.

    Returns
    -------
    Path | None
        Resolved parquet file path, or ``None`` when no valid artifact is found.

    If the path points to a directory, pick the newest ``*.parquet`` file.
    Returns ``None`` when no parquet exists in that directory or when a file
    path is provided but does not exist.
    """
    target = Path(path) if path is not None else DL_PREDICTION_ARTIFACT_PATH
    if target.is_dir():
        parquet_files = sorted(target.glob("*.parquet"))
        if not parquet_files:
            return None
        return max(parquet_files, key=lambda p: (p.stat().st_mtime_ns, str(p)))
    return target if target.exists() else None

# ---------------------------------------------------------------------------
# Surface selection dict
#
# Identifies one unique signal surface in the consolidated cube.
# dl_regime is required and must be one of: HVTF, LVTF, HVR, LVR.
# target_horizon is the number of H1 bars (Int64).
# ---------------------------------------------------------------------------
DL_SIGNAL_SURFACE: dict = {
    "model": os.environ.get("DL_SURFACE_MODEL", "lstm"),
    "target_horizon": int(os.environ.get("DL_SURFACE_TARGET_HORIZON", "24")),
    "feature_set": os.environ.get("DL_SURFACE_FEATURE_SET", "price_trend"),
    "dl_regime": os.environ.get("DL_SURFACE_REGIME", "HVTF"),
}

_VALID_DL_REGIMES: frozenset[str] = frozenset({"HVTF", "LVTF", "HVR", "LVR"})
_DL_REGIME_PATTERN = re.compile(
    r"(?<![A-Z0-9])(" + "|".join(sorted(_VALID_DL_REGIMES, reverse=True)) + r")(?![A-Z0-9])"
)


def infer_dl_regime_from_artifact_path(path: Path | str | None) -> str | None:
    """
    Infer dl_regime from an artifact file/path string.

    Supports both canonical names like:
      ``mlp__HVR__24__price_trend__20260524T175056Z.parquet``
    and free-form names containing regime tokens such as:
      ``persistent_to_reactive_sentiment_HVR.parquet``.
    """
    if path is None:
        return None

    name = Path(path).name
    upper_name = name.upper()

    for token in upper_name.split("__"):
        cleaned = token.replace(".PARQUET", "")
        if cleaned in _VALID_DL_REGIMES:
            return cleaned

    match = _DL_REGIME_PATTERN.search(upper_name)
    if match:
        return match.group(1)

    return None
