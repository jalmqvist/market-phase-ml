"""
Configuration for the DL surface integration (market-sentiment-ml consumer).

All settings have safe defaults so that DL signals are *disabled* by default,
keeping existing pipelines completely unaffected until explicitly opted in.

Override via environment variables
-----------------------------------
DL_SIGNALS_ENABLED          "true" to enable (default: false)
DL_SIGNALS_CUBE_PATH        absolute or relative path to the consolidated cube
                            parquet file produced by market-sentiment-ml
                            (default: sibling-repo layout below)
DL_SURFACE_MODEL            model name identifier  (default: lstm)
DL_SURFACE_TARGET_HORIZON   target horizon in bars as an integer (default: 24)
DL_SURFACE_FEATURE_SET      feature set identifier (default: price_trend)
DL_SURFACE_REGIME           dl_regime for surface selection (default: HVTF)
                            Valid values: HVTF, LVTF, HVR, LVR
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Toggle — off by default so existing pipelines are unaffected
# ---------------------------------------------------------------------------
DL_SIGNALS_ENABLED: bool = (
    os.environ.get("DL_SIGNALS_ENABLED", "false").lower() == "true"
)

# ---------------------------------------------------------------------------
# Cube path — defaults to sibling-repo layout
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent  # src/ -> market-phase-ml/

_DEFAULT_CUBE_PATH = (
    _REPO_ROOT.parent
    / "market-sentiment-ml"
    / "data"
    / "output"
    / "dl_signals"
    / "dl_signals_h1_v1.parquet"
)

DL_SIGNALS_CUBE_PATH: Path = Path(
    os.environ.get("DL_SIGNALS_CUBE_PATH", str(_DEFAULT_CUBE_PATH))
)

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
