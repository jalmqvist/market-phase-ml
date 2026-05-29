import pandas as pd
import yfinance as yf


class YahooLoader:
    """Yahoo Finance OHLCV loader."""

    def load(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
