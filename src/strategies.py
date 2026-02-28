# src/strategies.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from ta.volatility import BollingerBands
from ta.trend import SMAIndicator, ADXIndicator
from ta.momentum import StochasticOscillator, RSIIndicator

@dataclass
class TradeResult:
    """Represents a single completed trade."""
    entry_date:      pd.Timestamp
    exit_date:       pd.Timestamp
    entry_price:     float
    exit_price:      float
    direction:       int          # 1 = long, -1 = short
    phase:           str
    strategy:        str
    size_multiplier: float        # hardcoded multiplier (legacy method)
    position_size:   float        # ATR-based constant-risk sizing (new method)
    stop_distance:   float        # ATR * stop_multiplier at entry
    pnl:             float
    pnl_pct:         float
    exit_reason:     str          # 'SL', 'TP', or 'signal'

# ─────────────────────────────────────────────
#  NEW STRATEGIES
# ─────────────────────────────────────────────

# ── Helpers ───────────────────────────────────────────────────────────────────

def _lwma(series: pd.Series, period: int) -> pd.Series:
    """
    Linearly Weighted Moving Average (LWMA).
    Weights: most recent bar has weight=period, oldest has weight=1.
    Not available in `ta`, so implemented manually.
    """
    weights = np.arange(1, period + 1, dtype=float)

    def _apply(window):
        if len(window) < period:
            return np.nan
        return np.dot(window, weights) / weights.sum()

    return series.rolling(window=period).apply(_apply, raw=True)


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Return ADX series. Uses pre-computed df['adx'] if available,
    otherwise falls back to ta.trend.ADXIndicator.
    """
    if 'adx' in df.columns:
        return df['adx']
    from ta.trend import ADXIndicator
    return ADXIndicator(
        high=df['High'], low=df['Low'], close=df['Close'], window=period
    ).adx()


# ── TF1 ───────────────────────────────────────────────────────────────────────

class TF1Strategy:
    """
    LWMA band breakout trend following.

    Entry: Yesterday's close outside LWMA(period) ± σ × StdDev band
           i.e. close just crossed outside the band (crossover, not level)
    Exit:  Close crosses back inside the band

    Reference: Original system by Jonas Almqvist (MQL4)
    """

    def __init__(self,
                 lwma_period: int = 40,
                 std_period: int = 20,
                 std_mult: float = 1.0):
        self.lwma_period = lwma_period
        self.std_period = std_period
        self.std_mult = std_mult
        self.name = 'TF1'

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Returns:
            Tuple of (signals, sl_pct_series, tp_pct_series)
            No SL/TP — exits when price returns inside band.
        """
        lwma = _lwma(df['Close'], self.lwma_period)
        std = df['Close'].rolling(self.std_period).std()
        upper = lwma + self.std_mult * std
        lower = lwma - self.std_mult * std

        close = df['Close']
        prev_close = close.shift(1)

        # Entry: close just moved outside band (crossover)
        long_entry = (close > upper) & (prev_close <= upper)
        short_entry = (close < lower) & (prev_close >= lower)

        # Exit: close crosses back inside band
        long_exit = (close < upper) & (prev_close >= upper)
        short_exit = (close > lower) & (prev_close <= lower)

        raw = pd.Series(np.nan, index=df.index)
        raw[long_exit] = 0
        raw[short_exit] = 0
        raw[long_entry] = 1
        raw[short_entry] = -1

        signals = raw.ffill().fillna(0).astype(int)

        sl_pct_series = pd.Series(0.0, index=df.index)
        tp_pct_series = pd.Series(0.0, index=df.index)

        return signals, sl_pct_series, tp_pct_series


# ── TF2 ───────────────────────────────────────────────────────────────────────

class TF2Strategy:
    """
    Donchian channel breakout trend following.

    Entry: New N-day high/low close (close breaks above/below channel)
    Exit:  Trailing exit — price crosses the opposite channel shadow
           (i.e. long exits when close < N-day low, short when close > N-day high)

    Reference: Original system by Jonas Almqvist (MQL4)
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.name = 'TF2'

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Returns:
            Tuple of (signals, sl_pct_series, tp_pct_series)
            No SL/TP — trailing channel exit only.
        """
        close = df['Close']

        # Channel computed on prior N bars (shift 1 to avoid lookahead)
        upper = close.shift(1).rolling(self.period).max()
        lower = close.shift(1).rolling(self.period).min()

        prev_close = close.shift(1)

        # Entry: close just broke above/below channel
        long_entry = (close > upper) & (prev_close <= upper)
        short_entry = (close < lower) & (prev_close >= lower)

        # Trailing exit: close crosses back through opposite shadow
        long_exit = close < lower  # long exits below lower channel
        short_exit = close > upper  # short exits above upper channel

        raw = pd.Series(np.nan, index=df.index)
        raw[long_exit] = 0
        raw[short_exit] = 0
        raw[long_entry] = 1
        raw[short_entry] = -1

        signals = raw.ffill().fillna(0).astype(int)

        sl_pct_series = pd.Series(0.0, index=df.index)
        tp_pct_series = pd.Series(0.0, index=df.index)

        return signals, sl_pct_series, tp_pct_series


# ── TF3 ───────────────────────────────────────────────────────────────────────

class TF3Strategy:
    """
    Dual moving average crossover.
    SMA(9) crosses SMA(26) — classic trend following.

    Entry: SMA(9) crosses above/below SMA(26)
    Exit:  Opposite crossover

    Reference: Mechanical Trading Systems (2005) pg 50
    """

    def __init__(self,
                 fast_period: int = 9,
                 slow_period: int = 26):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = 'TF3'

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Returns:
            Tuple of (signals, sl_pct_series, tp_pct_series)
            No SL/TP — exits on opposite crossover only.
        """
        fast_ma = SMAIndicator(
            close=df['Close'], window=self.fast_period
        ).sma_indicator()
        slow_ma = SMAIndicator(
            close=df['Close'], window=self.slow_period
        ).sma_indicator()

        prev_fast = fast_ma.shift(1)
        prev_slow = slow_ma.shift(1)

        long_entry = (fast_ma > slow_ma) & (prev_fast <= prev_slow)
        short_entry = (fast_ma < slow_ma) & (prev_fast >= prev_slow)

        raw = pd.Series(np.nan, index=df.index)
        raw[long_entry] = 1
        raw[short_entry] = -1

        signals = raw.ffill().fillna(0).astype(int)

        sl_pct_series = pd.Series(0.0, index=df.index)
        tp_pct_series = pd.Series(0.0, index=df.index)

        return signals, sl_pct_series, tp_pct_series


# ── TF4 ───────────────────────────────────────────────────────────────────────

class TF4Strategy:
    """
    LWMA direction + Stochastic extreme trend following.

    Entry: LWMA(40) is rising/falling AND Stochastic(5,3,1) < 20 or > 80
           (enter on pullback to extreme stochastic in trend direction)
    Exit:  Trailing exit — opposite stochastic extreme crossed

    Reference: Original system by Jonas Almqvist (MQL4)
    """

    def __init__(self,
                 lwma_period: int = 40,
                 stoch_k: int = 5,
                 stoch_d: int = 3,
                 stoch_smooth: int = 1,
                 stoch_low: float = 20.0,
                 stoch_high: float = 80.0):
        self.lwma_period = lwma_period
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.stoch_smooth = stoch_smooth
        self.stoch_low = stoch_low
        self.stoch_high = stoch_high
        self.name = 'TF4'

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Returns:
            Tuple of (signals, sl_pct_series, tp_pct_series)
            No SL/TP — trailing stochastic exit only.
        """
        lwma = _lwma(df['Close'], self.lwma_period)
        lwma_prev = lwma.shift(1)

        stoch = StochasticOscillator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=self.stoch_k,
            smooth_window=self.stoch_d
        )
        stoch_k = stoch.stoch()

        # LWMA direction
        lwma_rising = lwma > lwma_prev
        lwma_falling = lwma < lwma_prev

        # Entry: trend direction + stochastic pullback to extreme
        long_entry = lwma_rising & (stoch_k < self.stoch_low)
        short_entry = lwma_falling & (stoch_k > self.stoch_high)

        # Trailing exit: stochastic crosses opposite extreme
        long_exit = stoch_k > self.stoch_high
        short_exit = stoch_k < self.stoch_low

        raw = pd.Series(np.nan, index=df.index)
        raw[long_exit] = 0
        raw[short_exit] = 0
        raw[long_entry] = 1
        raw[short_entry] = -1

        signals = raw.ffill().fillna(0).astype(int)

        sl_pct_series = pd.Series(0.0, index=df.index)
        tp_pct_series = pd.Series(0.0, index=df.index)

        return signals, sl_pct_series, tp_pct_series


# ── TF5 ───────────────────────────────────────────────────────────────────────

class TF5Strategy:
    """
    Bollinger Band breakout trend following.
    Enter on close outside band, exit at center band.

    Note: Uses sigma=1.0 (not standard 2.0) — tighter bands
    mean more frequent signals but earlier entries.

    Entry: Close outside BB(20, 1.0)
    Exit:  Close crosses back through center band

    Reference: Original system by Jonas Almqvist (MQL4)
    """

    def __init__(self,
                 bb_period: int = 20,
                 bb_std: float = 1.0):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.name = 'TF5'

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Returns:
            Tuple of (signals, sl_pct_series, tp_pct_series)
            No SL/TP — exits when price returns to BB center.
        """
        bb = BollingerBands(
            close=df['Close'],
            window=self.bb_period,
            window_dev=self.bb_std
        )
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_center = bb.bollinger_mavg()

        close = df['Close']
        prev_close = close.shift(1)

        # Entry: close just moved outside band (crossover)
        long_entry = (close > bb_upper) & (prev_close <= bb_upper)
        short_entry = (close < bb_lower) & (prev_close >= bb_lower)

        # Exit: close crosses back through center
        long_exit = (close < bb_center) & (prev_close >= bb_center)
        short_exit = (close > bb_center) & (prev_close <= bb_center)

        raw = pd.Series(np.nan, index=df.index)
        raw[long_exit] = 0
        raw[short_exit] = 0
        raw[long_entry] = 1
        raw[short_entry] = -1

        signals = raw.ffill().fillna(0).astype(int)

        sl_pct_series = pd.Series(0.0, index=df.index)
        tp_pct_series = pd.Series(0.0, index=df.index)

        return signals, sl_pct_series, tp_pct_series


# ── MR1 ───────────────────────────────────────────────────────────────────────

class MR1Strategy:
    """
    LWMA band mean reversion.
    Mirror of TF1 but trades the reversion back inside the band.

    Entry: Yesterday's close outside LWMA ± σ × StdDev band
           (same trigger as TF1, opposite intent)
    Exit:  2% SL / 2% TP

    Reference: Original system by Jonas Almqvist (MQL4)
    """

    def __init__(self,
                 lwma_period: int = 40,
                 std_period: int = 20,
                 std_mult: float = 1.0,
                 sl_pct: float = 0.02,
                 tp_pct: float = 0.02):
        self.lwma_period = lwma_period
        self.std_period = std_period
        self.std_mult = std_mult
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.name = 'MR1'

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Entry: close just moved outside band — expect reversion,
               so trade TOWARD the band (opposite to TF1).
               Long when price breaks BELOW lower band (oversold).
               Short when price breaks ABOVE upper band (overbought).

        Returns:
            Tuple of (signals, sl_pct_series, tp_pct_series)
        """
        lwma = _lwma(df['Close'], self.lwma_period)
        std = df['Close'].rolling(self.std_period).std()
        upper = lwma + self.std_mult * std
        lower = lwma - self.std_mult * std

        close = df['Close']
        prev_close = close.shift(1)

        # Entry: price just crossed outside band — fade the move
        long_entry = (close < lower) & (prev_close >= lower)  # oversold
        short_entry = (close > upper) & (prev_close <= upper)  # overbought

        raw = pd.Series(np.nan, index=df.index)
        raw[long_entry] = 1
        raw[short_entry] = -1

        signals = raw.ffill().fillna(0).astype(int)

        sl_pct_series = pd.Series(self.sl_pct, index=df.index)
        tp_pct_series = pd.Series(self.tp_pct, index=df.index)

        return signals, sl_pct_series, tp_pct_series


# ── MR2 ───────────────────────────────────────────────────────────────────────

class MR2Strategy:
    """
    Stochastic extreme mean reversion.

    Entry: Stochastic(2,3,1) < 15 AND rising  (long)
           Stochastic(2,3,1) > 85 AND falling (short)
    Exit:  Stochastic crosses 50

    Fast stochastic (period=2) is very sensitive — suited to
    short-term mean reversion on daily bars.

    Reference: Original system by Jonas Almqvist (MQL4)
    """

    def __init__(self,
                 stoch_k: int = 2,
                 stoch_d: int = 3,
                 stoch_low: float = 15.0,
                 stoch_high: float = 85.0,
                 stoch_exit: float = 50.0):
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.stoch_low = stoch_low
        self.stoch_high = stoch_high
        self.stoch_exit = stoch_exit
        self.name = 'MR2'

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Entry: stochastic just entered extreme zone AND momentum
               is already turning (rising from low / falling from high).
        Exit:  stochastic crosses 50 (midpoint).

        Returns:
            Tuple of (signals, sl_pct_series, tp_pct_series)
            No SL/TP — indicator-based exit only.
        """
        stoch = StochasticOscillator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=self.stoch_k,
            smooth_window=self.stoch_d
        )
        stoch_k = stoch.stoch()
        prev_stoch_k = stoch_k.shift(1)

        # Entry: in extreme zone AND momentum turning
        long_entry = (stoch_k < self.stoch_low) & (stoch_k > prev_stoch_k)
        short_entry = (stoch_k > self.stoch_high) & (stoch_k < prev_stoch_k)

        # Exit: stochastic crosses 50
        long_exit = (stoch_k > self.stoch_exit) & (prev_stoch_k <= self.stoch_exit)
        short_exit = (stoch_k < self.stoch_exit) & (prev_stoch_k >= self.stoch_exit)

        raw = pd.Series(np.nan, index=df.index)
        raw[long_exit] = 0
        raw[short_exit] = 0
        raw[long_entry] = 1
        raw[short_entry] = -1

        signals = raw.ffill().fillna(0).astype(int)

        sl_pct_series = pd.Series(0.0, index=df.index)
        tp_pct_series = pd.Series(0.0, index=df.index)

        return signals, sl_pct_series, tp_pct_series

    # ── MR3 ───────────────────────────────────────────────────────────────────────

class MR3Strategy:
    """
    RSI crossover mean reversion.
    Entry on RSI crossing back through extreme levels.
    Exit: 1% SL / 3% TP

    Reference: Mechanical Trading Systems (2005) pg 97
    """

    def __init__(self,
                 rsi_period: int = 14,
                 rsi_long: float = 25.0,
                 rsi_short: float = 75.0,
                 sl_pct: float = 0.01,
                 tp_pct: float = 0.03):
        self.rsi_period = rsi_period
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.name = 'MR3'

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Entry: RSI crosses back above 25 (long) or below 75 (short)
               i.e. RSI was below 25 yesterday, now above 25 = buy signal
                    RSI was above 75 yesterday, now below 75 = sell signal

        Returns:
            Tuple of (signals, sl_pct_series, tp_pct_series)
        """
        rsi = df['rsi']
        prev_rsi = rsi.shift(1)

        long_entry = (rsi > self.rsi_long) & (prev_rsi <= self.rsi_long)
        short_entry = (rsi < self.rsi_short) & (prev_rsi >= self.rsi_short)

        raw = pd.Series(np.nan, index=df.index)
        raw[long_entry] = 1
        raw[short_entry] = -1

        signals = raw.ffill().fillna(0).astype(int)

        sl_pct_series = pd.Series(self.sl_pct, index=df.index)
        tp_pct_series = pd.Series(self.tp_pct, index=df.index)

        return signals, sl_pct_series, tp_pct_series

    # ── MR32 ──────────────────────────────────────────────────────────────────────

class MR32Strategy:
    """
    RSI mean reversion with MA(200) trend filter.

    Only trades in the direction of the long-term trend:
        Long:  RSI < 35 AND close > MA(200)  — pullback in uptrend
        Short: RSI > 65 AND close < MA(200)  — rally in downtrend

    Exit: RSI crosses 60 (long exit) or 40 (short exit) + 2.5% SL

    Reference: Original system by Jonas Almqvist (MQL4)
    """

    def __init__(self,
                 rsi_period: int = 14,
                 rsi_long: float = 35.0,
                 rsi_short: float = 65.0,
                 rsi_exit_long: float = 60.0,
                 rsi_exit_short: float = 40.0,
                 ma_period: int = 200,
                 sl_pct: float = 0.025):
        self.rsi_period = rsi_period
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short
        self.rsi_exit_long = rsi_exit_long
        self.rsi_exit_short = rsi_exit_short
        self.ma_period = ma_period
        self.sl_pct = sl_pct
        self.name = 'MR32'

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Returns:
            Tuple of (signals, sl_pct_series, tp_pct_series)
            No TP — RSI exit + SL only.
        """
        rsi = df['rsi']
        prev_rsi = rsi.shift(1)

        ma200 = SMAIndicator(
            close=df['Close'], window=self.ma_period
        ).sma_indicator()

        close = df['Close']

        # Entry: RSI in extreme zone AND price on correct side of MA(200)
        long_entry = (rsi < self.rsi_long) & (close > ma200)
        short_entry = (rsi > self.rsi_short) & (close < ma200)

        # Exit: RSI crosses back through exit level
        long_exit = (rsi > self.rsi_exit_long) & (prev_rsi <= self.rsi_exit_long)
        short_exit = (rsi < self.rsi_exit_short) & (prev_rsi >= self.rsi_exit_short)

        raw = pd.Series(np.nan, index=df.index)
        raw[long_exit] = 0
        raw[short_exit] = 0
        raw[long_entry] = 1
        raw[short_entry] = -1

        signals = raw.ffill().fillna(0).astype(int)

        sl_pct_series = pd.Series(self.sl_pct, index=df.index)
        tp_pct_series = pd.Series(0.0, index=df.index)  # no TP, RSI exit

        return signals, sl_pct_series, tp_pct_series

    # ── MR42 ──────────────────────────────────────────────────────────────────────

class MR42Strategy:
    """
    Bollinger Band mean reversion with ADX filter.
    Only trades when ADX < 20 — explicitly ranging markets.
    Maps directly to LV_Ranging and HV_Ranging phases.

    Entry: Price closes outside BB(20, 2) AND ADX(14) < 20
    Exit:  2.5% SL / 1.25% TP
    """

    def __init__(self,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 adx_filter: float = 20.0,
                 sl_pct: float = 0.025,
                 tp_pct: float = 0.0125):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.adx_filter = adx_filter
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.name = 'MR42'

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Returns:
            Tuple of (signals, sl_pct_series, tp_pct_series)
        """
        bb = BollingerBands(
            close=df['Close'],
            window=self.bb_period,
            window_dev=self.bb_std
        )
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()

        close = df['Close']
        prev_close = close.shift(1)
        adx = _adx(df)

        # Entry: close just crossed outside band AND ADX < filter
        long_entry = (
                (close < bb_lower) &
                (prev_close >= bb_lower) &
                (adx < self.adx_filter)
        )
        short_entry = (
                (close > bb_upper) &
                (prev_close <= bb_upper) &
                (adx < self.adx_filter)
        )

        raw = pd.Series(np.nan, index=df.index)
        raw[long_entry] = 1
        raw[short_entry] = -1

        signals = raw.ffill().fillna(0).astype(int)

        sl_pct_series = pd.Series(self.sl_pct, index=df.index)
        tp_pct_series = pd.Series(self.tp_pct, index=df.index)

        return signals, sl_pct_series, tp_pct_series


# ── MR5 ───────────────────────────────────────────────────────────────────────

class MR5Strategy:
    """
    Bollinger Band mean reversion, high volatility version.
    Similar to MR42 but WITHOUT the ADX filter — suited to
    HV_Ranging phases where ADX may be elevated but price
    is still oscillating within a range.

    Entry: Price closes outside BB(20, 2.0)
    Exit:  2% SL / 2% TP

    Reference: Original system by Jonas Almqvist (MQL4)
    """

    def __init__(self,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 sl_pct: float = 0.02,
                 tp_pct: float = 0.02):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.name = 'MR5'

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Returns:
            Tuple of (signals, sl_pct_series, tp_pct_series)
        """
        bb = BollingerBands(
            close=df['Close'],
            window=self.bb_period,
            window_dev=self.bb_std
        )
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()

        close = df['Close']
        prev_close = close.shift(1)

        # Entry: close just crossed outside band (crossover only)
        long_entry = (close < bb_lower) & (prev_close >= bb_lower)
        short_entry = (close > bb_upper) & (prev_close <= bb_upper)

        raw = pd.Series(np.nan, index=df.index)
        raw[long_entry] = 1
        raw[short_entry] = -1

        signals = raw.ffill().fillna(0).astype(int)

        sl_pct_series = pd.Series(self.sl_pct, index=df.index)
        tp_pct_series = pd.Series(self.tp_pct, index=df.index)

        return signals, sl_pct_series, tp_pct_series





class TrendFollowingStrategy:
    """
    Trend following using ADX + DI crossover.

    Entry: DI crossover while ADX > entry_threshold
    Exit:  Opposite DI crossover OR ADX drops below exit_threshold

    Uses crossover detection + ffill to hold position between
    events — dramatically reduces trade frequency vs level-based
    signals.

    Expected trade frequency on D1: ~3-8 per year per pair.
    """

    def __init__(self,
                 adx_entry_threshold: float = 25.0,
                 adx_exit_threshold: float = 15.0):
        self.adx_entry_threshold = adx_entry_threshold
        self.adx_exit_threshold = adx_exit_threshold
        self.name = 'TrendFollowing'

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate crossover-based trend following signals.

        Entry conditions (crossover only, not level):
            Long:  +DI crosses above -DI AND ADX > entry_threshold
            Short: -DI crosses above +DI AND ADX > entry_threshold

        Exit conditions:
            - Opposite DI crossover occurs
            - ADX drops below exit_threshold (trend fading)

        Position is held (ffill) between entry and exit events.

        Returns:
            Series with values: 1 (long), -1 (short), 0 (flat)
        """
        # Current bar conditions
        long_condition = (
                (df['plus_di'] > df['minus_di']) &
                (df['adx'] > self.adx_entry_threshold)
        )
        short_condition = (
                (df['minus_di'] > df['plus_di']) &
                (df['adx'] > self.adx_entry_threshold)
        )

        # Previous bar conditions (for crossover detection)
        prev_long = long_condition.shift(1).fillna(False)
        prev_short = short_condition.shift(1).fillna(False)

        # Crossover events — condition just became True
        long_entry = long_condition & ~prev_long
        short_entry = short_condition & ~prev_short

        # Exit — ADX fading below exit threshold
        exit_signal = df['adx'] < self.adx_exit_threshold

        # Build raw event series
        # NaN = no event (position held from previous bar)
        # 1   = enter long
        # -1  = enter short
        # 0   = exit
        raw = pd.Series(np.nan, index=df.index)
        raw[exit_signal] = 0
        raw[long_entry] = 1  # entry overwrites exit if same bar
        raw[short_entry] = -1

        # Forward fill holds position between events
        # fillna(0) handles the initial period before first signal
        signals = raw.ffill().fillna(0).astype(int)

        return signals


class MeanReversionStrategy:
    """
    Mean reversion using RSI extremes.

    Entry: RSI crosses below oversold OR crosses above overbought
           (crossover only — not level-based)
    Exit:  RSI crosses back through exit level (default 50)

    Uses crossover detection + ffill to hold position between
    events — avoids re-entering on every bar RSI stays extreme.

    Expected trade frequency on D1: ~6-12 per year per pair.
    """

    def __init__(self,
                 rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0,
                 rsi_exit_long: float = 50.0,
                 rsi_exit_short: float = 50.0):
        """
        Args:
            rsi_oversold:   RSI level triggering long entry
            rsi_overbought: RSI level triggering short entry
            rsi_exit_long:  RSI level to exit long (default 50)
            rsi_exit_short: RSI level to exit short (default 50)
        """
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_exit_long = rsi_exit_long
        self.rsi_exit_short = rsi_exit_short
        self.name = 'MeanReversion'

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate crossover-based mean reversion signals.

        Entry conditions (crossover only):
            Long:  RSI crosses below oversold threshold
                   (RSI was >= oversold, now < oversold)
            Short: RSI crosses above overbought threshold
                   (RSI was <= overbought, now > overbought)

        Exit conditions (crossover only):
            Long exit:  RSI crosses back above rsi_exit_long
            Short exit: RSI crosses back below rsi_exit_short

        Position is held (ffill) between entry and exit events.

        Returns:
            Series with values: 1 (long), -1 (short), 0 (flat)
        """
        rsi = df['rsi']
        prev_rsi = rsi.shift(1)

        # Entry crossovers
        long_entry = (rsi < self.rsi_oversold) & (prev_rsi >= self.rsi_oversold)
        short_entry = (rsi > self.rsi_overbought) & (prev_rsi <= self.rsi_overbought)

        # Exit crossovers
        long_exit = (rsi > self.rsi_exit_long) & (prev_rsi <= self.rsi_exit_long)
        short_exit = (rsi < self.rsi_exit_short) & (prev_rsi >= self.rsi_exit_short)

        # Build raw event series
        raw = pd.Series(np.nan, index=df.index)

        # Exits first, entries overwrite if same bar
        raw[long_exit] = 0
        raw[short_exit] = 0
        raw[long_entry] = 1
        raw[short_entry] = -1

        # Forward fill to hold position, 0 before first signal
        signals = raw.ffill().fillna(0).astype(int)

        return signals

# ─────────────────────────────────────────────────────────────────────────────
#  UPDATED PhaseAwareStrategy
# ─────────────────────────────────────────────────────────────────────────────

class PhaseAwareStrategy:
    """
    Phase-aware strategy selector using the 4-phase scheme.

    Phase -> Strategy mapping:
        HV_Trend:   TF1, TF2, TF3, TF4, TF5  (trend following suite)
        LV_Trend:   TF1, TF2, TF3, TF4, TF5  (trend following suite)
        HV_Ranging: MR1, MR2, MR3, MR5        (mean reversion, no ADX filter)
        LV_Ranging: MR1, MR2, MR3, MR32, MR42 (mean reversion, ADX-filtered)

    The strategy argument selects which specific strategy to run
    within the phase routing. Defaults to the original ADX/RSI
    baseline pair for backward compatibility.
    """

    # Phase -> strategy type routing
    PHASE_STRATEGY = {
        'HV_Trend': 'trend',
        'LV_Trend': 'trend',
        'HV_Ranging': 'mean_reversion',
        'LV_Ranging': 'mean_reversion',
    }

    # Hardcoded size multipliers (legacy method)
    PHASE_SIZE_MULTIPLIER = {
        'HV_Trend': 0.5,
        'LV_Trend': 1.5,
        'HV_Ranging': 0.5,
        'LV_Ranging': 1.0,
    }

    # All available strategies per type
    TF_STRATEGIES = {
        'TF1': TF1Strategy,
        'TF2': TF2Strategy,
        'TF3': TF3Strategy,
        'TF4': TF4Strategy,
        'TF5': TF5Strategy,
    }
    MR_STRATEGIES = {
        'MR1': MR1Strategy,
        'MR2': MR2Strategy,
        'MR32': MR32Strategy,
        'MR42': MR42Strategy,
        'MR5': MR5Strategy,
    }

    def __init__(self,
                 tf_strategy_name: str = 'TF3',
                 mr_strategy_name: str = 'MR3'):
        """
        Args:
            tf_strategy_name: Which TF strategy to use in trend phases.
                              One of: TF1, TF2, TF3, TF4, TF5
                              Default: TF3 (SMA crossover — most robust baseline)
            mr_strategy_name: Which MR strategy to use in ranging phases.
                              One of: MR1, MR2, MR3, MR32, MR42, MR5
                              Default: MR3 (RSI crossover — most robust baseline)
        """
        if tf_strategy_name not in self.TF_STRATEGIES:
            raise ValueError(
                f"Unknown TF strategy '{tf_strategy_name}'. "
                f"Choose from: {list(self.TF_STRATEGIES.keys())}"
            )
        if mr_strategy_name not in self.MR_STRATEGIES:
            raise ValueError(
                f"Unknown MR strategy '{mr_strategy_name}'. "
                f"Choose from: {list(self.MR_STRATEGIES.keys())}"
            )

        self.tf_strategy = self.TF_STRATEGIES[tf_strategy_name]()
        self.mr_strategy = self.MR_STRATEGIES[mr_strategy_name]()
        self.tf_strategy_name = tf_strategy_name
        self.mr_strategy_name = mr_strategy_name
        self.name = f'PhaseAware_{tf_strategy_name}_{mr_strategy_name}'

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        tf_signals, tf_sl, tf_tp = self.tf_strategy.generate_signals(df)
        mr_signals, mr_sl, mr_tp = self.mr_strategy.generate_signals(df)

        signals = pd.Series(0.0,  index=df.index)
        sl_pct  = pd.Series(0.0,  index=df.index)
        tp_pct  = pd.Series(0.0,  index=df.index)

        for phase, strategy_type in self.PHASE_STRATEGY.items():
            mask = df['phase'] == phase
            size = self.PHASE_SIZE_MULTIPLIER[phase]

            if strategy_type == 'trend':
                signals[mask] = tf_signals[mask] * size
                sl_pct[mask]  = tf_sl[mask]
                tp_pct[mask]  = tf_tp[mask]
            else:
                signals[mask] = mr_signals[mask] * size
                sl_pct[mask]  = mr_sl[mask]
                tp_pct[mask]  = mr_tp[mask]

        return signals, sl_pct, tp_pct


class Backtester:
    """
    Backtester supporting both hardcoded and ATR-based position sizing.

    Two sizing modes controlled by use_atr_sizing flag:

    1. Hardcoded (use_atr_sizing=False):
        Position size = initial_capital * size_multiplier
        Simple, comparable to original system.

    2. ATR-based constant risk (use_atr_sizing=True):
        stop_distance = stop_atr_mult * ATR
        position_size = (equity * risk_pct) / stop_distance
        Dollar risk per trade stays approximately constant at
        risk_pct * equity regardless of phase or volatility.

    Assumptions:
        - Daily OHLCV data
        - Enter at next bar's close after signal
        - Spread and slippage costs applied at entry and exit
        - No partial closes or pyramiding
    """

    def __init__(self,
                 initial_capital: float = 10000.0,
                 spread_pips: float = 1.0,
                 pip_value: float = 0.0001,
                 commission_per_trade: float = 0.0,
                 slippage_pips: float = 0.5,
                 use_atr_sizing: bool = False,
                 risk_pct: float = 0.01):
        """
        Args:
            initial_capital:      Starting equity
            spread_pips:          Bid/ask spread in pips
            pip_value:            Value of one pip (0.0001 for most pairs,
                                  0.01 for JPY pairs — set per pair)
            commission_per_trade: Fixed commission per trade
            slippage_pips:        Estimated slippage in pips
            use_atr_sizing:       If True, use ATR-based constant risk sizing.
                                  If False, use hardcoded size multipliers.
            risk_pct:             Target risk per trade as fraction of equity.
                                  Only used when use_atr_sizing=True.
        """
        self.initial_capital = initial_capital
        self.spread_pips = spread_pips
        self.pip_value = pip_value
        self.commission = commission_per_trade
        self.slippage = slippage_pips
        self.use_atr_sizing = use_atr_sizing
        self.risk_pct = risk_pct

    def _calculate_trade_cost(self, price: float) -> float:
        """
        Calculate round-trip cost as fraction of price.

        Includes spread, slippage and commission.
        """
        spread_cost = self.spread_pips * self.pip_value / price
        slippage_cost = self.slippage * self.pip_value / price
        commission_cost = self.commission / self.initial_capital
        return spread_cost + slippage_cost + commission_cost

    def _get_position_size(self,
                           signal: float,
                           equity: float,
                           atr: float,
                           stop_atr_mult: float) -> tuple[float, float]:
        """
        Calculate position size and stop distance based on sizing mode.

        Args:
            signal:        Raw signal value (sign=direction, mag=multiplier)
            equity:        Current equity
            atr:           Current ATR value
            stop_atr_mult: Phase-specific ATR stop
        Returns:
            Tuple of (position_size, stop_distance)
            position_size: units to trade (used for PnL scaling)
            stop_distance: price distance for stop loss
        """
        if self.use_atr_sizing:
            # ATR-based constant risk sizing
            # Dollar risk per trade = equity * risk_pct
            # Position size = dollar_risk / stop_distance
            stop_distance = stop_atr_mult * atr
            if stop_distance <= 0:
                return 0.0, 0.0
            dollar_risk = equity * self.risk_pct
            position_size = dollar_risk / stop_distance
        else:
            # Hardcoded multiplier (legacy method)
            # Signal magnitude encodes the size multiplier
            size_multiplier = abs(signal) if signal != 0 else 1.0
            position_size = self.initial_capital * size_multiplier
            stop_distance = stop_atr_mult * atr

        return position_size, stop_distance

    def run(self,
            df: pd.DataFrame,
            signals: pd.Series,
            strategy_name: str,
            sl_pct_series: Optional[pd.Series] = None,
            tp_pct_series: Optional[pd.Series] = None) -> dict:
        """
        Run backtest for a given signal series.

        Args:
            df:            DataFrame with OHLCV, phase, atr, stop_atr_mult
            signals:       Series with position signals
            strategy_name: Name for reporting
            sl_pct_series: Series of per-bar SL % (0.0 = no SL)
            tp_pct_series: Series of per-bar TP % (0.0 = no TP)

        Returns:
            Dictionary with performance metrics, trade list, equity curve
        """
        # Default to no SL/TP if not provided
        if sl_pct_series is None:
            sl_pct_series = pd.Series(0.0, index=df.index)
        if tp_pct_series is None:
            tp_pct_series = pd.Series(0.0, index=df.index)

        capital = self.initial_capital
        position = 0
        entry_price = 0.0
        entry_date = None
        entry_phase = 'Unknown'
        entry_size_mult = 1.0
        entry_position_size = 0.0
        entry_stop_distance = 0.0
        sl_price = 0.0        # absolute SL price level
        tp_price = 0.0        # absolute TP price level
        use_sl = False        # whether SL is active for current trade
        use_tp = False        # whether TP is active for current trade
        trades = []
        equity_curve = [capital]

        # Verify required columns
        required_cols = ['Close', 'High', 'Low', 'phase', 'atr', 'stop_atr_mult']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        try:
            for i in range(1, len(df)):
                current_date  = df.index[i]
                current_close = float(df['Close'].iloc[i])
                current_high  = float(df['High'].iloc[i])
                current_low   = float(df['Low'].iloc[i])
                signal        = float(signals.iloc[i - 1])  # previous bar signal
                phase         = str(df['phase'].iloc[i])
                atr           = float(df['atr'].iloc[i])
                stop_atr_mult = float(df['stop_atr_mult'].iloc[i])

                exit_price    = None
                exit_reason   = None

                # ── Check SL/TP on current bar (before signal logic) ─────────
                if position != 0:
                    if position == 1:  # long
                        # SL: low breached the stop level
                        if use_sl and current_low <= sl_price:
                            exit_price  = sl_price
                            exit_reason = 'SL'
                        # TP: high reached target (only if SL not hit)
                        elif use_tp and current_high >= tp_price:
                            exit_price  = tp_price
                            exit_reason = 'TP'

                    else:  # short
                        # SL: high breached the stop level
                        if use_sl and current_high >= sl_price:
                            exit_price  = sl_price
                            exit_reason = 'SL'
                        # TP: low reached target (only if SL not hit)
                        elif use_tp and current_low <= tp_price:
                            exit_price  = tp_price
                            exit_reason = 'TP'

                # ── Close position (SL/TP hit) ────────────────────────────────
                if exit_price is not None:
                    exit_cost = self._calculate_trade_cost(exit_price)

                    if position == 1:
                        pnl_pct = (
                            (exit_price - entry_price) /
                            entry_price - exit_cost
                        )
                    else:
                        pnl_pct = (
                            (entry_price - exit_price) /
                            entry_price - exit_cost
                        )

                    if self.use_atr_sizing:
                        price_move = (
                            exit_price - entry_price
                            if position == 1
                            else entry_price - exit_price
                        )
                        pnl = entry_position_size * price_move - (
                            entry_position_size * entry_price * exit_cost
                        )
                    else:
                        pnl = capital * pnl_pct * entry_size_mult

                    capital += pnl

                    trades.append(TradeResult(
                        entry_date=entry_date,
                        exit_date=current_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=position,
                        phase=entry_phase,
                        strategy=strategy_name,
                        size_multiplier=entry_size_mult,
                        position_size=entry_position_size,
                        stop_distance=entry_stop_distance,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason = exit_reason
                    ))

                    position  = 0
                    sl_price  = 0.0
                    tp_price  = 0.0
                    use_sl    = False
                    use_tp    = False

                # ── Close position (signal-based exit) ───────────────────────
                if position != 0 and exit_price is None and (
                    np.sign(signal) != np.sign(position) or
                    signal == 0
                ):
                    exit_cost = self._calculate_trade_cost(current_close)

                    if position == 1:
                        pnl_pct = (
                            (current_close - entry_price) /
                            entry_price - exit_cost
                        )
                    else:
                        pnl_pct = (
                            (entry_price - current_close) /
                            entry_price - exit_cost
                        )

                    if self.use_atr_sizing:
                        price_move = (
                            current_close - entry_price
                            if position == 1
                            else entry_price - current_close
                        )
                        pnl = entry_position_size * price_move - (
                            entry_position_size * entry_price * exit_cost
                        )
                    else:
                        pnl = capital * pnl_pct * entry_size_mult

                    capital += pnl

                    trades.append(TradeResult(
                        entry_date=entry_date,
                        exit_date=current_date,
                        entry_price=entry_price,
                        exit_price=current_close,
                        direction=position,
                        phase=entry_phase,
                        strategy=strategy_name,
                        size_multiplier=entry_size_mult,
                        position_size=entry_position_size,
                        stop_distance=entry_stop_distance,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason = 'signal'
                    ))

                    position  = 0
                    sl_price  = 0.0
                    tp_price  = 0.0
                    use_sl    = False
                    use_tp    = False

                # ── Open new position ─────────────────────────────────────────
                if signal != 0 and position == 0:
                    entry_cost = self._calculate_trade_cost(current_close)
                    capital   -= capital * entry_cost

                    position_size, stop_distance = self._get_position_size(
                        signal, capital, atr, stop_atr_mult
                    )

                    position             = int(np.sign(signal))
                    entry_price          = float(current_close)
                    entry_date           = current_date
                    entry_phase          = str(phase)
                    entry_size_mult      = float(abs(signal) if signal != 0 else 1.0)
                    entry_position_size  = float(position_size)
                    entry_stop_distance  = float(stop_distance)

                    # Compute SL/TP price levels from entry bar's series values
                    bar_sl_pct = float(sl_pct_series.iloc[i])
                    bar_tp_pct = float(tp_pct_series.iloc[i])

                    use_sl = bar_sl_pct > 0.0
                    use_tp = bar_tp_pct > 0.0

                    if position == 1:   # long
                        sl_price = entry_price * (1.0 - bar_sl_pct) if use_sl else 0.0
                        tp_price = entry_price * (1.0 + bar_tp_pct) if use_tp else 0.0
                    else:               # short
                        sl_price = entry_price * (1.0 + bar_sl_pct) if use_sl else 0.0
                        tp_price = entry_price * (1.0 - bar_tp_pct) if use_tp else 0.0
                equity_curve.append(capital)

        except Exception as e:
            import traceback
            traceback.print_exc()

        # ── Close any open position at end of data ────────────────────────────
        if position != 0:
            final_price = float(df['Close'].iloc[-1])
            exit_cost   = self._calculate_trade_cost(final_price)

            if position == 1:
                pnl_pct = (
                    (final_price - entry_price) /
                    entry_price - exit_cost
                )
            else:
                pnl_pct = (
                    (entry_price - final_price) /
                    entry_price - exit_cost
                )

            if self.use_atr_sizing:
                price_move = (
                    final_price - entry_price
                    if position == 1
                    else entry_price - final_price
                )
                pnl = entry_position_size * price_move - (
                    entry_position_size * entry_price * exit_cost
                )
            else:
                pnl = capital * pnl_pct * entry_size_mult

            capital += pnl

            trades.append(TradeResult(
                entry_date=entry_date,
                exit_date=df.index[-1],
                entry_price=entry_price,
                exit_price=final_price,
                direction=position,
                phase=entry_phase,
                strategy=strategy_name,
                size_multiplier=entry_size_mult,
                position_size=entry_position_size,
                stop_distance=entry_stop_distance,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason='end_of_data'
            ))

            equity_curve.append(capital)



        # Build metrics
        metrics = self._calculate_metrics(trades, equity_curve, df.index)
        metrics['strategy']     = strategy_name
        metrics['trades']       = trades
        metrics['equity_curve'] = pd.Series(
            equity_curve[1:],  # drop the initial capital entry
            index=df.index[:len(equity_curve) - 1]
        )
        metrics['sizing_method'] = (
            'atr_constant_risk' if self.use_atr_sizing else 'hardcoded_multiplier'
        )

        return metrics


    def _calculate_metrics(self,
                           trades: list,
                           equity_curve: list,
                           dates: pd.DatetimeIndex) -> dict:
        """
        Calculate comprehensive performance metrics.

        Metrics:
            total_return:   % return over full period
            sharpe_ratio:   Annualized Sharpe (daily returns, 252 trading days)
            max_drawdown:   Maximum peak-to-trough drawdown %
            win_rate:       % of trades with positive PnL
            profit_factor:  Gross profit / gross loss
            n_trades:       Total number of completed trades
            avg_trade_pnl:  Average PnL per trade
            gross_profit:   Sum of winning trade PnLs
            gross_loss:     Sum of losing trade PnLs (absolute)
            phase_performance: Per-phase breakdown of trade statistics
        """
        if len(trades) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'n_trades': 0,
                'avg_trade_pnl': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0,
                'phase_performance': pd.DataFrame()
            }

        trades_df = pd.DataFrame([{
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'direction': t.direction,
            'phase': t.phase,
            'strategy': t.strategy,
            'size_multiplier': t.size_multiplier,
            'position_size': t.position_size,
            'stop_distance': t.stop_distance,
            'exit_reason': t.exit_reason,
        } for t in trades])

        equity = pd.Series(equity_curve)
        returns = equity.pct_change().dropna()

        start_equity = float(equity_curve[0])
        end_equity = float(equity_curve[-1])

        total_return = (
                (end_equity - start_equity) /
                start_equity * 100
        )

        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252)
            if returns.std() > 0 else 0.0
        )

        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100

        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0 else np.inf
        )

        win_rate = (
                len(trades_df[trades_df['pnl'] > 0]) /
                len(trades_df) * 100
        )

        phase_performance = trades_df.groupby('phase').agg(
            n_trades=('pnl', 'count'),
            total_pnl=('pnl', 'sum'),
            avg_pnl=('pnl', 'mean'),
            win_rate=('pnl', lambda x: (x > 0).mean() * 100),
            avg_position_size=('position_size', 'mean'),
            avg_stop_distance=('stop_distance', 'mean'),
        ).round(4)

        # Calculate years in data
        n_years = (dates[-1] - dates[0]).days / 365.25

        trades_per_year = round(len(trades_df) / n_years, 2) if n_years > 0 else 0.0
        avg_pnl_per_trade = round(trades_df['pnl'].mean(), 4) if len(trades_df) > 0 else 0.0

        # Bars in market (time spent in a position)
        if len(trades_df) > 0:
            trades_df['trade_duration'] = (
                pd.to_datetime(trades_df['exit_date']) -
                pd.to_datetime(trades_df['entry_date'])
            ).dt.days
            avg_trade_duration = round(trades_df['trade_duration'].mean(), 1)
            pct_time_in_market = round(
                trades_df['trade_duration'].sum() /
                ((dates[-1] - dates[0]).days) * 100, 1
            )
        else:
            avg_trade_duration = 0.0
            pct_time_in_market = 0.0

        return {
            'total_return': round(total_return, 2),
            'sharpe_ratio': round(sharpe, 4),
            'max_drawdown': round(max_drawdown, 2),
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 4),
            'n_trades': len(trades_df),
            'trades_per_year': trades_per_year,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'avg_trade_duration': avg_trade_duration,
            'pct_time_in_market': pct_time_in_market,
            'avg_trade_pnl': round(trades_df['pnl'].mean(), 4),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'phase_performance': phase_performance
        }

# ─────────────────────────────────────────────────────────────────────────────
#  UPDATED run_backtests
# ─────────────────────────────────────────────────────────────────────────────

def run_backtests(df: pd.DataFrame,
                  initial_capital: float = 10000.0,
                  use_atr_sizing:  bool  = False,
                  tf_strategy_name: str  = 'TF3',
                  mr_strategy_name: str  = 'MR3') -> dict:
    """
    Run all strategies and return combined results.

    Runs:
        A) Individual TF strategies
        B) Individual MR strategies
        C) PhaseAware with selected TF + MR pair

    Args:
        df:               DataFrame with OHLCV, indicators and phase columns.
                          Must include: rsi, adx, plus_di, minus_di, phase,
                                        atr, stop_atr_mult, High, Low, Close
        initial_capital:  Starting equity for all strategies
        use_atr_sizing:   If True, use ATR constant-risk sizing.
                          If False, use hardcoded size multipliers.
        tf_strategy_name: TF strategy to use in PhaseAware (default: TF3)
        mr_strategy_name: MR strategy to use in PhaseAware (default: MR3)

    Returns:
        Dictionary keyed by strategy name, each containing
        metrics dict from Backtester.run()
    """
    # Verify required columns
    required_cols = [
        'rsi', 'adx', 'plus_di', 'minus_di',
        'phase', 'atr', 'stop_atr_mult',
        'High', 'Low', 'Close'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    backtester = Backtester(
        initial_capital=initial_capital,
        use_atr_sizing=use_atr_sizing
    )

    tf_strategies = {
        'TF1': TF1Strategy(),
        'TF2': TF2Strategy(),
        'TF3': TF3Strategy(),
        'TF4': TF4Strategy(),
        'TF5': TF5Strategy(),
    }
    mr_strategies = {
        'MR1':  MR1Strategy(),
        'MR2':  MR2Strategy(),
        'MR32': MR32Strategy(),
        'MR42': MR42Strategy(),
        'MR5':  MR5Strategy(),
    }

    results = {}

    # ── [A] Individual TF strategies ─────────────────────────────────────────
    for name, strategy in tf_strategies.items():
        print(f'[TF] Running {name}...')
        signals, sl_pct, tp_pct = strategy.generate_signals(df)
        results[name] = backtester.run(df, signals, name, sl_pct, tp_pct)

    # ── [B] Individual MR strategies ─────────────────────────────────────────
    for name, strategy in mr_strategies.items():
        print(f'[MR] Running {name}...')
        signals, sl_pct, tp_pct = strategy.generate_signals(df)
        results[name] = backtester.run(df, signals, name, sl_pct, tp_pct)

    # ── [C] All PhaseAware combinations ──────────────────────────────────────
    for tf_name in tf_strategies.keys():
        for mr_name in mr_strategies.keys():
            pa = PhaseAwareStrategy(tf_name, mr_name)
            print(f'[PA] Running PhaseAware ({tf_name} + {mr_name})...')
            pa_signals, pa_sl, pa_tp = pa.generate_signals(df)
            results[pa.name] = backtester.run(
                df, pa_signals, pa.name, pa_sl, pa_tp
            )

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STRATEGY COMPARISON RESULTS')
    print('=' * 70)

    summary_rows = []
    for name, result in results.items():
        summary_rows.append({
            'Strategy':         name,
            'Sizing':           result.get('sizing_method', 'unknown'),
            'Total Return (%)': result['total_return'],
            'Sharpe Ratio':     result['sharpe_ratio'],
            'Max Drawdown (%)': result['max_drawdown'],
            'Win Rate (%)':     result['win_rate'],
            'Profit Factor':    result['profit_factor'],
            'N Trades':         result['n_trades'],
            'Trades/Year':      result['trades_per_year'],
            'Avg PnL/Trade':    result['avg_pnl_per_trade'],
            'Avg Duration (days)': result['avg_trade_duration'],
            '% Time in Market': result['pct_time_in_market'],
        })

    summary_df = pd.DataFrame(summary_rows).set_index('Strategy')
    print(summary_df.to_string())

    # ── PhaseAware per-phase breakdown (best by Sharpe) ──────────────────────
    pa_keys = [k for k in results.keys() if k.startswith('PhaseAware_')]
    if pa_keys:
        best_pa_key = max(pa_keys, key=lambda k: results[k] ['sharpe_ratio'])
        print('\n' + '=' * 70)
        print(f'BEST PHASE-AWARE COMBO ({best_pa_key}): PER-PHASE PERFORMANCE')
        print('=' * 70)
        if results[best_pa_key] ['phase_performance'] is not None:
            print(results[best_pa_key] ['phase_performance'].to_string())

    return results