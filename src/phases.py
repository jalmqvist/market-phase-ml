# src/phases.py

import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
from dataclasses import dataclass


@dataclass
class MarketPhase:
    """
    Represents a simplified 4-phase market classification.

    Phases:
        HV_Trend    - High volatility + trending (ADX high)
        LV_Trend    - Low volatility  + trending (ADX high)
        HV_Ranging  - High volatility + ranging  (ADX low)
        LV_Ranging  - Low volatility  + ranging  (ADX low)

    Volatility is measured as ATR% of price, split by 252-day rolling median.
    Direction (up/down) is intentionally excluded: a downtrend in EURUSD
    is equivalent to an uptrend in USDEUR — direction is handled by
    the signal generator, not the phase classifier.
    """
    trending: bool = False
    high_volatility: bool = False

    def __str__(self) -> str:
        vol = 'HV' if self.high_volatility else 'LV'
        trend = 'Trend' if self.trending else 'Ranging'
        return f'{vol}_{trend}'


class MarketPhaseDetector:
    """
    Detects market phases using a simplified 4-phase scheme.

    Volatility measure:
        ATR% = ATR(atr_period) / Close * 100
        Split by 252-day rolling median of ATR% (adaptive per pair/period).
        This normalizes across pairs with very different price levels
        (e.g. USDJPY ~150 vs EURUSD ~1.10).

    Trend measure:
        ADX(adx_period) > adx_trend_threshold => Trending
        ADX(adx_period) <= adx_trend_threshold => Ranging

    Position sizing:
        Targets 1% of equity risk per trade.
        Stop distance is ATR-based and scales with volatility.
        Position size = (equity * risk_pct) / stop_distance
        where stop_distance = atr_stop_multiplier * ATR

        Phase-specific stop multipliers:
            HV_Trend:   2.0x ATR  (wide stop, volatile trend)
            LV_Trend:   1.0x ATR  (normal stop, clean trend)
            HV_Ranging: 2.0x ATR  (wide stop, choppy market)
            LV_Ranging: 0.5x ATR  (tight stop, low movement)

        This means position size is INVERSELY proportional to volatility,
        naturally producing smaller sizes in high-vol phases — consistent
        with constant dollar risk per trade.
    """

    # Stop-loss ATR multipliers per phase
    STOP_MULTIPLIERS = {
        'HV_Trend':   2.0,
        'LV_Trend':   1.0,
        'HV_Ranging': 2.0,
        'LV_Ranging': 0.5,
    }

    def __init__(self,
                 adx_period: int = 14,
                 adx_trend_threshold: float = 25.0,
                 atr_period: int = 14,
                 vol_rolling_window: int = 252,
                 risk_pct: float = 0.01):
        """
        Args:
            adx_period:          ADX lookback period (default 14)
            adx_trend_threshold: ADX level separating trend from ranging
                                 (default 25 — slightly stricter than
                                 the classic 20 to reduce false trends)
            atr_period:          ATR lookback period for volatility and
                                 stop-loss calculation (default 14)
            vol_rolling_window:  Rolling window for ATR% median split,
                                 252 = 1 trading year (default 252)
            risk_pct:            Target risk per trade as fraction of
                                 equity (default 0.01 = 1%)
        """
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.atr_period = atr_period
        self.vol_rolling_window = vol_rolling_window
        self.risk_pct = risk_pct

    def calculate_atr_pct(self,
                          high: pd.Series,
                          low: pd.Series,
                          close: pd.Series) -> pd.Series:
        """
        Calculate ATR as a percentage of closing price.

        ATR% = ATR(n) / Close * 100

        This normalizes volatility across pairs with different price
        levels (e.g. USDJPY ~150 vs EURUSD ~1.10), making the
        rolling median split comparable across all pairs.

        Returns:
            pd.Series of ATR% values
        """
        atr = AverageTrueRange(
            high=high,
            low=low,
            close=close,
            window=self.atr_period
        ).average_true_range()

        # Protect against division by zero
        close_safe = close.replace(0, np.nan)
        atr_pct = (atr / close_safe) * 100

        return atr_pct.ffill()

    def calculate_adx(self,
                      high: pd.Series,
                      low: pd.Series,
                      close: pd.Series) -> pd.Series:
        """
        Calculate ADX for trend strength detection.

        Returns:
            pd.Series of ADX values
        """
        adx_indicator = ADXIndicator(
            high=high,
            low=low,
            close=close,
            window=self.adx_period
        )
        return adx_indicator.adx()

    def classify_volatility(self, atr_pct: pd.Series) -> pd.Series:
        """
        Classify each bar as high or low volatility using a rolling
        median of ATR% over vol_rolling_window bars.

        High volatility: ATR% >= rolling median
        Low volatility:  ATR% <  rolling median

        Using rolling median (not fixed threshold) means:
        - Always ~50/50 HV/LV split regardless of pair or time period
        - Adapts to structural volatility changes (e.g. post-2008 vs 2020s)
        - No look-ahead bias (only uses past data)

        Returns:
            pd.Series of bool (True = high volatility)
        """
        rolling_median = atr_pct.rolling(
            window=self.vol_rolling_window,
            min_periods=self.vol_rolling_window // 2
        ).median()

        return atr_pct >= rolling_median

    def detect_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market phases for entire DataFrame.

        Args:
            df: DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with added columns:
                atr          - raw ATR values (used for stop sizing)
                atr_pct      - ATR as % of price
                adx          - ADX values
                high_vol     - bool, True if above rolling median ATR%
                trending     - bool, True if ADX > threshold
                phase        - str, one of: HV_Trend, LV_Trend,
                                            HV_Ranging, LV_Ranging
                stop_atr_mult - float, ATR multiplier for stop-loss
        """
        result = df.copy()

        # --- Raw ATR (for position sizing calculations) ---
        atr = AverageTrueRange(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=self.atr_period
        ).average_true_range()
        result['atr'] = atr

        # --- ATR% (normalized volatility) ---
        atr_pct = self.calculate_atr_pct(df['High'], df['Low'], df['Close'])
        result['atr_pct'] = atr_pct

        # --- ADX (trend strength) ---
        adx = self.calculate_adx(df['High'], df['Low'], df['Close'])
        result['adx'] = adx

        # --- Volatility classification ---
        high_vol = self.classify_volatility(atr_pct)
        result['high_vol'] = high_vol

        # --- Trend classification ---
        trending = adx > self.adx_trend_threshold
        result['trending'] = trending

        # --- Phase labels ---
        result['phase'] = self._create_phase_labels(trending, high_vol)

        # --- Stop-loss ATR multiplier per phase ---
        result['stop_atr_mult'] = result['phase'].map(self.STOP_MULTIPLIERS)

        return result

    def _create_phase_labels(self,
                              trending: pd.Series,
                              high_vol: pd.Series) -> pd.Series:
        """
        Create phase labels from trend and volatility classifications.

        Four possible phases:
            HV_Trend:   high vol + trending
            LV_Trend:   low vol  + trending
            HV_Ranging: high vol + ranging
            LV_Ranging: low vol  + ranging

        Returns:
            pd.Series of phase label strings
        """
        conditions = [
            high_vol  &  trending,   # HV_Trend
            ~high_vol &  trending,   # LV_Trend
            high_vol  & ~trending,   # HV_Ranging
            ~high_vol & ~trending,   # LV_Ranging
        ]

        labels = [
            'HV_Trend',
            'LV_Trend',
            'HV_Ranging',
            'LV_Ranging',
        ]

        return pd.Series(
            np.select(conditions, labels, default='Unknown'),
            index=trending.index
        )

    def calculate_position_size(self,
                                 equity: float,
                                 atr: float,
                                 stop_atr_mult: float) -> float:
        """
        Calculate position size targeting 1% equity risk per trade.

        Formula:
            stop_distance = stop_atr_mult * ATR
            position_size = (equity * risk_pct) / stop_distance

        This means:
            - HV phases: wider stop => smaller position size
            - LV phases: tighter stop => larger position size
            - Dollar risk per trade stays approximately constant

        Example at $10,000 equity, ATR=0.0080:
            HV_Trend:   stop=0.0160, size = 100/0.0160 =  6,250 units
            LV_Trend:   stop=0.0080, size = 100/0.0080 = 12,500 units
            HV_Ranging: stop=0.0160, size = 100/0.0160 =  6,250 units
            LV_Ranging: stop=0.0040, size = 100/0.0040 = 25,000 units

        Args:
            equity:         Current account equity in base currency
            atr:            Current ATR value (in price units)
            stop_atr_mult:  ATR multiplier for stop distance (from phase)

        Returns:
            float: Position size in units
        """
        if atr <= 0 or stop_atr_mult <= 0:
            return 0.0

        stop_distance = stop_atr_mult * atr
        dollar_risk = equity * self.risk_pct
        position_size = dollar_risk / stop_distance

        return position_size

    def get_phase_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize phase distribution for a processed DataFrame.

        Useful for quickly checking phase balance across pairs.

        Args:
            df: DataFrame after detect_phases() has been called

        Returns:
            DataFrame with columns: phase, count, pct, avg_atr_pct
        """
        if 'phase' not in df.columns:
            raise ValueError("DataFrame must have 'phase' column. "
                             "Run detect_phases() first.")

        summary = (
            df.groupby('phase')
            .agg(
                count=('phase', 'count'),
                avg_atr_pct=('atr_pct', 'mean'),
                avg_adx=('adx', 'mean')
            )
            .reset_index()
        )

        summary['pct'] = (summary['count'] / len(df) * 100).round(1)
        summary = summary.sort_values('count', ascending=False)

        return summary