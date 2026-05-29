import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.data import MarketDataPipeline, resolve_market_data_source
from src.data_sources.broker_csv_loader import BrokerCSVLoader
from src.data_sources.yahoo_loader import YahooLoader


def _write_broker_csv(
    root: Path,
    pair: str,
    start: str,
    periods: int,
    *,
    freq: str = "1h",
) -> None:
    ts = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": range(1, periods + 1),
            "high": [v + 0.4 for v in range(1, periods + 1)],
            "low": [v - 0.4 for v in range(1, periods + 1)],
            "close": [v + 0.1 for v in range(1, periods + 1)],
            "volume": [10] * periods,
        }
    )
    frame.to_csv(root / f"{pair}60.csv", index=False)


class TestMarketDataSourceSelection(unittest.TestCase):
    def test_default_source_is_yfinance(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = MarketDataPipeline(data_dir=tmp, use_cache=False)
            self.assertEqual(pipeline.source, "yfinance")
            self.assertIsInstance(pipeline.loader, YahooLoader)
            self.assertEqual(pipeline.raw_cache_dir.name, "raw")
            self.assertEqual(pipeline.processed_cache_dir.name, "processed")

    def test_env_override_takes_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {"MPML_DATA_SOURCE": "broker_csv"}, clear=False):
                pipeline = MarketDataPipeline(
                    data_dir=tmp,
                    source="yfinance",
                    broker_data_dir=tmp,
                    use_cache=False,
                )
            self.assertEqual(pipeline.source, "broker_csv")
            self.assertIsInstance(pipeline.loader, BrokerCSVLoader)
            self.assertEqual(pipeline.raw_cache_dir.name, "raw_broker_csv")
            self.assertEqual(pipeline.processed_cache_dir.name, "processed_broker_csv")

    def test_invalid_source_raises(self):
        with self.assertRaises(ValueError):
            resolve_market_data_source("unsupported_source")


class TestBrokerCsvLoader(unittest.TestCase):
    def test_h1_to_d1_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df = pd.DataFrame(
                {
                    "timestamp": [
                        "2024-01-01 00:00:00",
                        "2024-01-01 01:00:00",
                        "2024-01-01 23:00:00",
                        "2024-01-02 00:00:00",
                    ],
                    "open": [1.0, 2.0, 3.0, 4.0],
                    "high": [1.5, 2.5, 3.5, 4.5],
                    "low": [0.5, 1.5, 2.5, 3.5],
                    "close": [1.1, 2.1, 3.1, 4.1],
                    "volume": [10, 20, 30, 40],
                }
            )
            df.to_csv(root / "EURUSD60.csv", index=False)

            loader = BrokerCSVLoader(root)
            daily = loader.load("EURUSD", start="2024-01-01", end="2024-01-02")

            self.assertEqual(len(daily), 2)
            self.assertEqual(daily.loc[pd.Timestamp("2024-01-01"), "Open"], 1.0)
            self.assertEqual(daily.loc[pd.Timestamp("2024-01-01"), "High"], 3.5)
            self.assertEqual(daily.loc[pd.Timestamp("2024-01-01"), "Low"], 0.5)
            self.assertEqual(daily.loc[pd.Timestamp("2024-01-01"), "Close"], 3.1)
            self.assertEqual(daily.loc[pd.Timestamp("2024-01-01"), "Volume"], 60)

    def test_differing_pair_start_dates_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_broker_csv(root, "EURUSD", "2024-01-01 00:00:00", periods=48)
            _write_broker_csv(root, "GBPUSD", "2024-01-03 00:00:00", periods=48)

            loader = BrokerCSVLoader(root)
            eur = loader.load("EURUSD", start="2024-01-01", end="2024-01-10")
            gbp = loader.load("GBPUSD", start="2024-01-01", end="2024-01-10")

            self.assertEqual(eur.index.min(), pd.Timestamp("2024-01-01"))
            self.assertEqual(gbp.index.min(), pd.Timestamp("2024-01-03"))

    def test_pipeline_download_uses_broker_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_broker_csv(root, "EURUSD", "2024-01-01 00:00:00", periods=72)

            pipeline = MarketDataPipeline(
                data_dir=tmp,
                source="broker_csv",
                broker_data_dir=root,
                start="2024-01-01",
                end="2024-01-10",
                use_cache=False,
            )
            daily = pipeline.download("EURUSD=X")
            self.assertFalse(daily.empty)
            self.assertTrue({"Open", "High", "Low", "Close", "Volume"}.issubset(daily.columns))


if __name__ == "__main__":
    unittest.main()
