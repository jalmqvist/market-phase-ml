# src/sentiment_loader.py
"""
Utilities for loading broker-exported H1 FX price data.

These functions mirror the loading and pair-normalization conventions
used by market-sentiment-ml (build_fx_sentiment_dataset.py) so that
timestamps and pair names are directly join-compatible with the
canonical sentiment dataset.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def normalize_price_pair_from_filename(path: Path) -> str:
    """
    Extract FX pair from a broker-exported filename and normalize to
    the lowercase ``xxx-yyy`` format used by market-sentiment-ml.

    Examples
    --------
    >>> normalize_price_pair_from_filename(Path("USDJPY_H1.csv"))
    'usd-jpy'
    >>> normalize_price_pair_from_filename(Path("EURUSD60.csv"))
    'eur-usd'
    >>> normalize_price_pair_from_filename(Path("gbpjpy.csv"))
    'gbp-jpy'
    """
    stem = path.stem.upper()

    # Remove common timeframe suffixes (H1, 60, M60)
    stem = re.sub(r'(_?H1|_?60|_?M60)$', '', stem, flags=re.IGNORECASE)

    # Keep letters only
    letters = re.sub(r'[^A-Z]', '', stem)

    if len(letters) < 6:
        raise ValueError(
            f"Could not infer FX symbol from filename: {path.name}"
        )

    symbol = letters[:6]
    return f"{symbol[:3].lower()}-{symbol[3:6].lower()}"


def load_broker_h1_prices_from_dir(price_dir: Path) -> pd.DataFrame:
    """
    Load all broker-exported H1 FX CSVs from *price_dir* into a single
    DataFrame with consistent column names and tz-naive UTC timestamps.

    Expected CSV columns (broker export format):
        ``time_utc, open, high, low, close, tick_volume``

    Returns a DataFrame with columns:
        ``timestamp, Open, High, Low, Close, Volume, pair``

    ``Open/High/Low/Close`` are capitalized so that the DataFrame can
    be passed directly to ``MarketPhaseDetector.detect_phases()``.

    Parameters
    ----------
    price_dir : Path
        Directory containing ``*_H1.csv`` (or similar) broker exports.

    Returns
    -------
    pd.DataFrame
        Combined price data for all pairs, sorted by (pair, timestamp).
    """
    files = sorted(price_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No H1 CSV files found in {price_dir.resolve()}"
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        pair = normalize_price_pair_from_filename(path)
        df = _load_one_broker_h1_file(path, pair)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("No valid price data could be loaded.")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["pair", "timestamp"]).reset_index(drop=True)
    prices = prices.drop_duplicates(
        subset=["pair", "timestamp"], keep="last"
    )
    return prices


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _load_one_broker_h1_file(path: Path, pair: str) -> pd.DataFrame:
    """Load a single broker-exported H1 CSV and return a cleaned DF."""
    df = pd.read_csv(path)

    required = {"time_utc", "open", "high", "low", "close", "tick_volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    df = df[["time_utc", "open", "high", "low", "close", "tick_volume"]].copy()

    # Parse and strip timezone to tz-naive UTC
    df["timestamp"] = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    for col in ["open", "high", "low", "close", "tick_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Capitalize OHLCV to match MarketPhaseDetector expectations
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "Volume",
    })

    df["pair"] = pair

    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume", "pair"]]
    df = df.dropna(subset=["timestamp", "Open", "High", "Low", "Close"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df
