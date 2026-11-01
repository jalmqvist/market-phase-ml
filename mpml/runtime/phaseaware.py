"""
Causal PhaseAware(TF4, MR42) runtime.

Public entry point::

    from mpml.runtime.phaseaware import latest_decision
    decision = latest_decision(history)      # history: completed D1 OHLCV

The runtime evaluates MPML's *canonical* ``MarketPhaseDetector`` and
``PhaseAwareStrategy("TF4", "MR42")`` on the supplied history and returns the
decision for the latest completed bar. No phase or strategy logic is
duplicated here; this module only validates input, orchestrates the canonical
calls and validates the output.

Semantics (see docs/runtime/phaseaware_runtime.md for the full contract)
-----------------------------------------------------------------------
* Timing: the decision is known at the *close* of ``decision.timestamp``.
  ``phase``, ``signal``, ``sl_pct`` and ``tp_pct`` all refer to that bar. The
  signal is NOT shifted; the Backtester's next-bar execution is a separate
  convention.
* ``signal``: sign = direction (+ long, - short, 0 flat); magnitude = the
  canonical phase size multiplier (HV_Trend 0.5, LV_Trend 1.5, HV_Ranging 0.5,
  LV_Ranging 1.0), i.e. a relative size, not a unit count.
* ``sl_pct`` / ``tp_pct``: fractions of entry price (0.025 == 2.5%). 0.0 means
  "no fixed stop / target" (TF4 exits on a stochastic reversal instead). They
  are unrelated to ``stop_atr_mult``, the phase ATR multiplier the research
  Backtester uses for position sizing; the runtime does not emit it.
* History-start dependence: TF4/MR42 signals are forward-filled state, so the
  decision depends on *where the supplied history starts*. The runtime always
  uses the full supplied history from its first bar (no sliding window, no
  truncated warm-up prefix). Callers must supply a consistent start.
"""
from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

COMPOSITION = "PhaseAware_TF4_MR42"
TF_STRATEGY_ID = "TF4"
MR_STRATEGY_ID = "MR42"

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
PRICE_COLUMNS = ("Open", "High", "Low", "Close")

# Column-name prefixes that mark forward-looking research labels
# (e.g. MarketDataPipeline.prepare() adds next_return / next_direction*).
FORBIDDEN_LABEL_PREFIXES = ("next_", "fwd_", "forward", "future", "target", "label")


class PhaseAwareRuntimeError(Exception):
    """Base class for runtime errors."""


class InvalidHistory(PhaseAwareRuntimeError, ValueError):
    """Input history violates the runtime input contract."""


class ForwardLabelError(InvalidHistory):
    """Input contains forward-looking label columns."""


class InsufficientHistory(PhaseAwareRuntimeError, ValueError):
    """Fewer bars than the derived warm-up requirement were supplied."""

    def __init__(self, available: int, required: int):
        self.available = available
        self.required = required
        super().__init__(
            f"Insufficient history: {available} bars supplied, "
            f"{required} required for {COMPOSITION}."
        )


class UnmatchedPhase(PhaseAwareRuntimeError):
    """The latest bar's phase is not one of the canonical phases."""


class NonFiniteOutput(PhaseAwareRuntimeError):
    """A canonical output (signal, SL or TP) for the latest bar is not finite."""


@dataclass(frozen=True)
class PhaseAwareDecision:
    """Decision for the latest completed bar (available at that bar's close)."""

    timestamp: pd.Timestamp
    phase: str
    signal: float
    sl_pct: float
    tp_pct: float
    required_history: int
    input_bars: int
    composition: str
    mpml_revision: str

    @property
    def warmup_complete(self) -> bool:
        return self.input_bars >= self.required_history


def _build_components():
    # Lazy: keeps ``import mpml.runtime.phaseaware`` cheap until first use.
    from src.phases import MarketPhaseDetector
    from src.strategies import PhaseAwareStrategy

    return MarketPhaseDetector(), PhaseAwareStrategy(TF_STRATEGY_ID, MR_STRATEGY_ID)


def required_history() -> int:
    """
    Minimum number of bars, derived from the canonical dependency chain.

    Each term is the number of bars after which the canonical dependency
    provides the latest bar's required value:

    * ADX(adx_period): ``ta`` emits 0.0 placeholders for the first
      ``2*adx_period - 1`` bars -> ``2 * adx_period`` bars.
    * ATR(atr_period): ``atr_period`` bars.
    * Volatility classifier: rolling median of ATR% with
      ``min_periods = vol_rolling_window // 2``. Pandas counts the canonical
      numeric startup placeholders toward ``min_periods``, so this requires
      ``min_periods`` bars. The 252-bar window length itself is not required.
    * TF4: LWMA(lwma_period) and its one-bar lag -> ``lwma_period + 1``;
      Stochastic %K -> ``stoch_k``.
    * MR42: Bollinger(bb_period) -> ``bb_period``; ``prev_close`` -> 2.

    RSI is not consumed by TF4, MR42 or the phase detector and is excluded.
    """
    detector, strategy = _build_components()
    tf, mr = strategy.tf_strategy, strategy.mr_strategy
    median_min_periods = detector.vol_rolling_window // 2
    terms = (
        2 * detector.adx_period,
        detector.atr_period,
        median_min_periods,
        tf.lwma_period + 1,
        tf.stoch_k,
        mr.bb_period,
        2,
    )
    return max(terms)


@lru_cache(maxsize=1)
def mpml_revision() -> str:
    """Git revision of this MPML checkout (``MPML_REVISION`` env overrides)."""
    env = os.environ.get("MPML_REVISION")
    if env:
        return env
    root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _validate_history(history: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame):
        raise InvalidHistory("history must be a pandas DataFrame.")

    forward = [
        c for c in history.columns
        if str(c).lower().startswith(FORBIDDEN_LABEL_PREFIXES)
    ]
    if forward:
        raise ForwardLabelError(
            f"history contains forward-looking label columns: {forward}. "
            "The runtime accepts raw completed-bar OHLCV only."
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in history.columns]
    if missing:
        raise InvalidHistory(f"history is missing required columns: {missing}.")

    index = history.index
    if not isinstance(index, pd.DatetimeIndex):
        raise InvalidHistory("history must be indexed by a pandas DatetimeIndex.")
    if index.hasnans:
        raise InvalidHistory("history index contains NaT.")
    if not index.is_unique:
        raise InvalidHistory("history index contains duplicate timestamps.")
    if not index.is_monotonic_increasing:
        raise InvalidHistory("history index must be sorted in ascending order.")

    ohlc = history[list(REQUIRED_COLUMNS)]
    try:
        ohlc = ohlc.astype("float64")
    except (TypeError, ValueError) as exc:
        raise InvalidHistory(f"OHLCV columns must be numeric: {exc}") from exc
    if not np.isfinite(ohlc.to_numpy()).all():
        raise InvalidHistory(
            "OHLCV contains NaN/inf; the runtime never drops rows because the "
            "latest completed bar must be preserved."
        )
    if (ohlc[list(PRICE_COLUMNS)] <= 0).any().any():
        raise InvalidHistory("OHLC prices must be strictly positive.")
    if (ohlc["High"] < ohlc[["Open", "Close", "Low"]].max(axis=1)).any():
        raise InvalidHistory("High must be >= Open, Close and Low on every bar.")
    if (ohlc["Low"] > ohlc[["Open", "Close", "High"]].min(axis=1)).any():
        raise InvalidHistory("Low must be <= Open, Close and High on every bar.")
    if (ohlc["Volume"] < 0).any():
        raise InvalidHistory("Volume must be non-negative.")

    # Only raw OHLCV reaches the canonical code: strategies prefer a
    # pre-computed 'adx' column, so stray indicator columns must not leak in.
    return history[list(REQUIRED_COLUMNS)].astype("float64").copy()


def latest_decision(history: pd.DataFrame) -> PhaseAwareDecision:
    """
    Return the PhaseAware(TF4, MR42) decision for the latest completed bar.

    Args:
        history: completed D1 OHLCV bars, ascending unique DatetimeIndex
            (naive or tz-aware; the returned timestamp preserves the input's
            tz), columns Open/High/Low/Close/Volume. The latest
            row must be a completed bar. The whole history is used from its
            first bar (see module docstring on history-start dependence).

    Raises:
        InvalidHistory, ForwardLabelError, InsufficientHistory,
        UnmatchedPhase, NonFiniteOutput
    """
    data = _validate_history(history)
    detector, strategy = _build_components()

    required = required_history()
    n_bars = len(data)
    if n_bars < required:
        raise InsufficientHistory(n_bars, required)

    # Canonical causal path: detector on the full history, then the strategy.
    # (Distinct from MarketDataPipeline.prepare()/engineer(), which add
    # forward labels and trim rows.)
    phased = detector.detect_phases(data)
    signals, sl_pct, tp_pct = strategy.generate_signals(phased)

    last = phased.index[-1]
    phase = phased["phase"].iloc[-1]
    if phase not in strategy.PHASE_SIZE_MULTIPLIER:
        raise UnmatchedPhase(f"Unmatched phase {phase!r} at {last}.")

    signal = float(signals.iloc[-1])
    sl = float(sl_pct.iloc[-1])
    tp = float(tp_pct.iloc[-1])
    for name, value in (("signal", signal), ("sl_pct", sl), ("tp_pct", tp)):
        if not math.isfinite(value):
            raise NonFiniteOutput(f"{name} is not finite at {last}: {value!r}.")

    return PhaseAwareDecision(
        timestamp=last,
        phase=str(phase),
        signal=signal,
        sl_pct=sl,
        tp_pct=tp,
        required_history=required,
        input_bars=n_bars,
        composition=strategy.name,
        mpml_revision=mpml_revision(),
    )
