# src/data.py

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange


# ---------------------------------------------------------------------------
# Currency pair configuration
# ---------------------------------------------------------------------------

MAJORS = [
    'EURUSD=X',
    'GBPUSD=X',
    'USDJPY=X',
    'USDCHF=X',
    'AUDUSD=X',
    'USDCAD=X',
    'NZDUSD=X',
]

MINORS = [
    'EURGBP=X',
    'EURJPY=X',
    'EURCHF=X',
    'GBPJPY=X',
    'AUDJPY=X',
    'EURAUD=X',
    'GBPAUD=X',
]

ALL_PAIRS = MAJORS + MINORS

# Pip value per pair.
# JPY pairs: 1 pip = 0.01
# All others: 1 pip = 0.0001
# Used by Backtester to calculate spread/slippage costs correctly.
PIP_VALUES = {
    'EURUSD=X': 0.0001,
    'GBPUSD=X': 0.0001,
    'USDJPY=X': 0.01,
    'USDCHF=X': 0.0001,
    'AUDUSD=X': 0.0001,
    'USDCAD=X': 0.0001,
    'NZDUSD=X': 0.0001,
    'EURGBP=X': 0.0001,
    'EURJPY=X': 0.01,
    'EURCHF=X': 0.0001,
    'GBPJPY=X': 0.01,
    'AUDJPY=X': 0.01,
    'EURAUD=X': 0.0001,
    'GBPAUD=X': 0.0001,
}

# Human-readable short names for reporting and file naming
PAIR_NAMES = {
    'EURUSD=X': 'EURUSD',
    'GBPUSD=X': 'GBPUSD',
    'USDJPY=X': 'USDJPY',
    'USDCHF=X': 'USDCHF',
    'AUDUSD=X': 'AUDUSD',
    'USDCAD=X': 'USDCAD',
    'NZDUSD=X': 'NZDUSD',
    'EURGBP=X': 'EURGBP',
    'EURJPY=X': 'EURJPY',
    'EURCHF=X': 'EURCHF',
    'GBPJPY=X': 'GBPJPY',
    'AUDJPY=X': 'AUDJPY',
    'EURAUD=X': 'EURAUD',
    'GBPAUD=X': 'GBPAUD',
}


class MarketDataPipeline:
    """
    Downloads, cleans and engineers features for multiple forex pairs.

    Workflow:
        1. download()  - fetch raw OHLCV from Yahoo Finance, cache to CSV
        2. prepare()   - clean, add returns and prediction targets
        3. engineer()  - add technical indicators needed by phases.py
                         and strategies.py
        4. run()       - convenience method: runs all three steps for
                         a list of pairs, returns dict of DataFrames

    Data is cached to data/raw/<PAIR>.csv and data/processed/<PAIR>.csv
    so repeated runs don't re-download unnecessarily.

    Technical indicators added by engineer():
        rsi          - RSI(14), used by MeanReversionStrategy
        adx          - ADX(14), used by PhaseAwareStrategy + phases.py
        plus_di      - +DI(14), used by TrendFollowingStrategy
        minus_di     - -DI(14), used by TrendFollowingStrategy
        atr          - ATR(14) raw value, used for position sizing
        atr_pct      - ATR as % of close, used by phases.py for vol split

    Note: phases.py recalculates atr and atr_pct internally using its
    own parameters. The versions added here use default periods and are
    stored for convenience/inspection. The authoritative values used in
    backtesting come from phases.py detect_phases().
    """

    def __init__(self,
                 start: str = '2005-01-01',
                 end: str = '2024-12-31',
                 data_dir: str = 'data',
                 rsi_period: int = 14,
                 adx_period: int = 14,
                 atr_period: int = 14,
                 use_cache: bool = True):
        """
        Args:
            start:       Start date for download (YYYY-MM-DD)
            end:         End date for download (YYYY-MM-DD)
            data_dir:    Root directory for cached CSV files
            rsi_period:  RSI lookback period (default 14)
            adx_period:  ADX/DI lookback period (default 14)
            atr_period:  ATR lookback period (default 14)
            use_cache:   If True, load from CSV if file exists.
                         If False, always re-download.
        """
        self.start = start
        self.end = end
        self.data_dir = Path(data_dir)
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.use_cache = use_cache

        # Create data directories if they don't exist
        (self.data_dir / 'raw').mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'processed').mkdir(parents=True, exist_ok=True)

    def download(self, ticker: str) -> pd.DataFrame:
        """
        Download raw OHLCV data for a single ticker.

        Checks cache first if use_cache=True. Saves to
        data/raw/<PAIR>.csv after download.

        Args:
            ticker: Yahoo Finance ticker (e.g. 'EURUSD=X')

        Returns:
            Raw OHLCV DataFrame, or empty DataFrame if download fails.
        """
        pair_name = PAIR_NAMES.get(ticker, ticker.replace('=X', ''))
        cache_path = self.data_dir / 'raw' / f'{pair_name}.csv'

        # Load from cache if available
        if self.use_cache and cache_path.exists():
            df = pd.read_csv(
                cache_path,
                index_col=0,
                parse_dates=True
            )
            print(f'  ✓ {pair_name}: loaded {len(df)} rows from cache')
            return df

        # Download from Yahoo Finance
        df = yf.download(
            ticker,
            start=self.start,
            end=self.end,
            progress=False,
            auto_adjust=True
        )

        if len(df) == 0:
            print(f'  ✗ {pair_name}: no data returned')
            return pd.DataFrame()

        # Flatten MultiIndex columns if present (yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Save to cache
        df.to_csv(cache_path)
        print(f'  ✓ {pair_name}: downloaded {len(df)} rows')

        return df

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw OHLCV data and add return columns.

        Adds:
            returns               - simple daily return
            log_returns           - log daily return
            next_return           - next bar's return (ML target)
            next_direction        - sign of next_return (-1, 0, 1)
            next_direction_binary - 1 if next_return > 0, else 0

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            Cleaned DataFrame with return columns added.
        """
        df = df.copy()

        # Keep only OHLCV columns, drop any extras from yfinance
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available = [c for c in ohlcv_cols if c in df.columns]
        df = df[available]

        # Drop rows with any missing OHLCV values
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

        # Ensure index is DatetimeIndex with no timezone
        df.index = pd.to_datetime(df.index).tz_localize(None)

        # Sort chronologically
        df = df.sort_index()

        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(
            df['Close'] / df['Close'].shift(1)
        )

        # Forward returns (ML prediction targets)
        df['next_return'] = df['returns'].shift(-1)
        df['next_direction'] = np.sign(df['next_return'])
        df['next_direction_binary'] = (
                df['next_return'] > 0
        ).astype(int)

        # Drop NaN rows from pct_change and shift
        df = df.dropna()

        return df

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators required by phases.py and strategies.py.

        All indicators use default periods set in __init__.
        Note: phases.py will recalculate atr and atr_pct with its own
        parameters during detect_phases() — these versions are stored
        for inspection and ML feature use only.

        Adds:
            rsi      - RSI(rsi_period)
            adx      - ADX(adx_period)
            plus_di  - +DI(adx_period)
            minus_di - -DI(adx_period)
            atr      - ATR(atr_period), raw price units
            atr_pct  - ATR as % of close price (normalized)

        Args:
            df: Prepared DataFrame from prepare()

        Returns:
            DataFrame with indicators added, NaN rows dropped.
        """
        df = df.copy()

        # --- RSI ---
        df['rsi'] = RSIIndicator(
            close=df['Close'],
            window=self.rsi_period
        ).rsi()

        # --- ADX, +DI, -DI ---
        adx_indicator = ADXIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=self.adx_period
        )
        df['adx'] = adx_indicator.adx()
        df['plus_di'] = adx_indicator.adx_pos()
        df['minus_di'] = adx_indicator.adx_neg()

        # --- ATR (raw) ---
        df['atr'] = AverageTrueRange(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=self.atr_period
        ).average_true_range()

        # --- ATR% (normalized for cross-pair comparison) ---
        df['atr_pct'] = (df['atr'] / df['Close']) * 100

        # Drop NaN rows introduced by indicator lookback periods
        df = df.dropna()

        return df

    def run(self,
            pairs: list = None,
            save_processed: bool = True) -> dict:
        """
        Run the full pipeline for a list of pairs.

        Steps per pair:
            1. download()  - fetch or load from cache
            2. prepare()   - clean + add returns
            3. engineer()  - add technical indicators

        Args:
            pairs:           List of Yahoo Finance tickers.
                             Defaults to ALL_PAIRS if None.
            save_processed:  If True, save processed DataFrame to
                             data/processed/<PAIR>.csv

        Returns:
            Dictionary keyed by short pair name (e.g. 'EURUSD'),
            each value is a fully processed DataFrame ready for
            phases.py and strategies.py.

            Also returns two sub-dicts for convenience:
                result['_majors'] - list of successfully loaded major names
                result['_minors'] - list of successfully loaded minor names
        """
        if pairs is None:
            pairs = ALL_PAIRS

        results = {}
        loaded_majors = []
        loaded_minors = []

        major_tickers = set(MAJORS)
        minor_tickers = set(MINORS)

        for ticker in pairs:
            pair_name = PAIR_NAMES.get(ticker, ticker.replace('=X', ''))
            print(f'Processing {pair_name}...')

            # Step 1: Download
            raw_df = self.download(ticker)
            if raw_df.empty:
                print(f'  ✗ {pair_name}: skipping — no data')
                continue

            # Step 2: Prepare
            try:
                prepared_df = self.prepare(raw_df)
            except Exception as e:
                print(f'  ✗ {pair_name}: prepare() failed — {e}')
                continue

            if len(prepared_df) < 300:
                print(
                    f'  ✗ {pair_name}: skipping — '
                    f'only {len(prepared_df)} rows after prepare()'
                )
                continue

            # Step 3: Engineer features
            try:
                processed_df = self.engineer(prepared_df)
            except Exception as e:
                print(f'  ✗ {pair_name}: engineer() failed — {e}')
                continue

            if len(processed_df) < 300:
                print(
                    f'  ✗ {pair_name}: skipping — '
                    f'only {len(processed_df)} rows after engineer()'
                )
                continue

            # Save processed data
            if save_processed:
                processed_path = (
                        self.data_dir / 'processed' / f'{pair_name}.csv'
                )
                processed_df.to_csv(processed_path)

            results[pair_name] = processed_df
            print(
                f'  ✓ {pair_name}: {len(processed_df)} rows ready'
            )

            # Track which group this pair belongs to
            if ticker in major_tickers:
                loaded_majors.append(pair_name)
            elif ticker in minor_tickers:
                loaded_minors.append(pair_name)

        # Store group membership for use in main.py
        results['_majors'] = loaded_majors
        results['_minors'] = loaded_minors

        print(f'\n✓ Pipeline complete:')
        print(f'  Majors loaded: {loaded_majors}')
        print(f'  Minors loaded: {loaded_minors}')
        print(f'  Total pairs:   {len(results) - 2}')  # exclude _majors/_minors

        return results

def get_pip_value(ticker: str) -> float:
    """
    Convenience function to get pip value for a ticker.

    Args:
        ticker: Yahoo Finance ticker (e.g. 'USDJPY=X')

    Returns:
        Pip value (0.01 for JPY pairs, 0.0001 for all others)
    """
    return PIP_VALUES.get(ticker, 0.0001)

def summarize_dataset(data: dict) -> pd.DataFrame:
    """
    Print a summary table of all loaded pairs.

    Useful for quickly checking data quality and date ranges
    across all pairs before running analysis.

    Args:
        data: Dict returned by MarketDataPipeline.run()

    Returns:
        Summary DataFrame with one row per pair.
    """
    rows = []
    for pair_name, df in data.items():
        # Skip metadata keys
        if pair_name.startswith('_'):
            continue

        rows.append({
            'Pair': pair_name,
            'Start': df.index.date,
            'End': df.index[-1].date,
            'Rows': len(df),
            'Group': (
                'Major' if pair_name in [
                    PAIR_NAMES[t] for t in MAJORS
                ] else 'Minor'
            ),
            'Avg ATR%': round(df['atr_pct'].mean(), 4),
            'Missing': df.isnull().sum().sum(),
        })

    summary = pd.DataFrame(rows).set_index('Pair')
    print('\n' + '=' * 60)
    print('DATASET SUMMARY')
    print('=' * 60)
    print(summary.to_string())
    print('=' * 60 + '\n')

    return summary