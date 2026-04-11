# src/mt4_regimes.py
"""
MT4-inspired regime detector for H1 FX data.

Implements a simplified version of the legacy MT4 (MQL) trading system's
regime classification logic, adapted for offline (pandas) analysis.

Regime definitions
------------------
**Volatility** — relative volatility using two ATR windows:
    high_volatility = ATR(10) / ATR(100) >= 1.0

**Trend** — ADX-based with SMA(200) direction (direction ignored for labels):
    trending = ADX(14) > 20
    (If ADX <= 20: consolidation / ranging)

Labels: ``HV_Trend``, ``LV_Trend``, ``HV_Ranging``, ``LV_Ranging``
(same 4-phase scheme used elsewhere in market-phase-ml).

Reference
---------
- Relative volatility concept: *Trading Systems and Methods* (Kaufman), p. 854.
- ADX threshold 20 is the classic Wilder default for "trending vs not".
- SMA(200) side determines optional trend direction but is not used in the
  4-phase label (direction is handled by signal generators, not regime labels).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange


# ── Detector parameters (constants) ─────────────────────────────────
ATR_SHORT: int = 10
ATR_LONG: int = 100
ADX_PERIOD: int = 14
ADX_TREND_THRESHOLD: float = 20.0
SMA_PERIOD: int = 200


def detect_mt4_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MT4-style regime labels on H1 OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``Open``, ``High``, ``Low``, ``Close``.
        Should be sorted chronologically for a single pair.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with additional columns:
        ``atr_short``, ``atr_long``, ``rel_vol``, ``high_volatility``,
        ``adx``, ``sma200``, ``trending``, ``phase``.
    """
    result = df.copy()

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # ── ATR-based relative volatility ────────────────────────────────
    atr_short = AverageTrueRange(
        high=high, low=low, close=close, window=ATR_SHORT,
    ).average_true_range()

    atr_long = AverageTrueRange(
        high=high, low=low, close=close, window=ATR_LONG,
    ).average_true_range()

    result["atr_short"] = atr_short
    result["atr_long"] = atr_long

    # Relative volatility: ATR(10) / ATR(100), safe div-by-zero
    rel_vol = atr_short / atr_long.replace(0, np.nan)
    result["rel_vol"] = rel_vol

    # high_volatility = True when rel_vol >= 1 (or when ATR(100) is NaN
    # during warm-up — default to high-vol to be conservative, matching
    # the MT4 logic that sets Vn=2, Vm=1 on missing data).
    result["high_volatility"] = rel_vol.fillna(2.0) >= 1.0

    # ── ADX trend strength ───────────────────────────────────────────
    adx_indicator = ADXIndicator(
        high=high, low=low, close=close, window=ADX_PERIOD,
    )
    result["adx"] = adx_indicator.adx()

    # ── SMA(200) direction (informational, not used in label) ────────
    result["sma200"] = close.rolling(window=SMA_PERIOD, min_periods=SMA_PERIOD).mean()

    # ── Trending flag ────────────────────────────────────────────────
    result["trending"] = result["adx"] > ADX_TREND_THRESHOLD

    # ── 4-phase labels ───────────────────────────────────────────────
    hv = result["high_volatility"]
    tr = result["trending"]
    conditions = [
        hv & tr,
        ~hv & tr,
        hv & ~tr,
        ~hv & ~tr,
    ]
    labels = ["HV_Trend", "LV_Trend", "HV_Ranging", "LV_Ranging"]
    result["phase"] = pd.Series(
        np.select(conditions, labels, default="Unknown"),
        index=result.index,
    )

    return result


def get_detector_description() -> str:
    """Return a human-readable one-liner for console / CSV metadata."""
    return (
        f"MT4-style: ATR({ATR_SHORT})/ATR({ATR_LONG})>=1 vol, "
        f"ADX({ADX_PERIOD})>{ADX_TREND_THRESHOLD} trend"
    )
