from pathlib import Path

import pandas as pd


class BrokerCSVLoader:
    """Load broker-exported H1 CSV files and aggregate to D1 OHLCV."""

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)

    def _resolve_csv_path(self, pair_name: str) -> Path:
        candidates = [
            self.data_root / f"{pair_name}60.csv",
            self.data_root / f"{pair_name}_H1.csv",
            self.data_root / f"{pair_name}.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No broker CSV found for {pair_name} under {self.data_root}"
        )

    @staticmethod
    def _find_timestamp_column(columns: list[str]) -> str:
        candidates = ("timestamp", "time_utc", "time", "datetime", "date")
        for candidate in candidates:
            if candidate in columns:
                return candidate
        raise ValueError(
            "Broker CSV is missing a timestamp column. "
            f"Expected one of: timestamp, time_utc, time, datetime, date. Found: {columns}"
        )

    def load(self, pair_name: str, start: str, end: str) -> pd.DataFrame:
        csv_path = self._resolve_csv_path(pair_name)
        raw = pd.read_csv(csv_path)
        raw.columns = [str(c).strip().lower() for c in raw.columns]

        ts_col = self._find_timestamp_column(list(raw.columns))
        volume_col = "volume" if "volume" in raw.columns else "tick_volume"
        required = {ts_col, "open", "high", "low", "close"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(
                f"Broker CSV {csv_path} missing required OHLC columns: {sorted(missing)}"
            )
        if volume_col not in raw.columns:
            raise ValueError(
                f"Broker CSV {csv_path} missing required volume column "
                "(expected 'volume' or 'tick_volume')."
            )

        h1 = raw[[ts_col, "open", "high", "low", "close", volume_col]].copy()
        h1.columns = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
        h1["timestamp"] = pd.to_datetime(
            h1["timestamp"],
            errors="coerce",
            utc=True,
        )
        h1 = h1.dropna(subset=["timestamp", "Open", "High", "Low", "Close"])
        h1 = h1.set_index("timestamp").sort_index()
        # Input timestamps are normalized to UTC above; strip tz to match
        # downstream MPML daily pipeline expectations (timezone-naive index).
        h1.index = h1.index.tz_localize(None)

        daily = h1.resample("1D").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        daily = daily.dropna(subset=["Open", "High", "Low", "Close"])

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return daily.loc[(daily.index >= start_ts) & (daily.index <= end_ts)]
