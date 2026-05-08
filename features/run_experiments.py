import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from features.experiments import EXPERIMENTS
from features.assembler import assemble_features, attach_dl_signals


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",  # up from /features
        "..",  # up from /market-phase-ml
        "market-sentiment-ml",
        "data",
        "output",
        "features",
        "sentiment_features_h1_v1.parquet"
    )
)

INPUT_PATH = "../../market-sentiment-ml/data/output/master_research_dataset_core.csv"


# --------------------------------------------------
# Helper: build behavioral feature
# --------------------------------------------------

def add_behavioral_signal(df):
    df = df.copy()

    df["pair_lower"] = df["pair"].str.lower()

    df["JPY_behavioral_signal"] = (
        df["pair_lower"].str.contains("jpy") &
        (df["extreme_streak_70"] >= 3) &
        (
            (
                (df["trend_alignment_12b"] == -1) &
                (df["trend_strength_bucket_12b"].isin(["medium", "strong"]))
            )
            |
            (
                (df["trend_alignment_12b"] == 1) &
                (df["trend_strength_bucket_12b"] == "extreme")
            )
        )
    ).astype(int)

    return df


# --------------------------------------------------
# Simple time split
# --------------------------------------------------

def train_test_split_xy(X, y):
    split = int(len(X) * 0.7)

    return (
        X.iloc[:split],
        X.iloc[split:],
        y.iloc[:split],
        y.iloc[split:]
    )

# --------------------------------------------------
# Walk forward
# --------------------------------------------------

def run_walk_forward(df, feature_groups, target_col):
    df = df.copy()

    df["year"] = pd.to_datetime(df["entry_time"]).dt.year

    years = sorted(df["year"].dropna().unique())

    results = []

    for year in years:
        train = df[df["year"] < year]
        test = df[df["year"] == year]

        # Skip first year (no training data)
        if len(train) == 0 or len(test) == 0:
            continue

        for name, groups in feature_groups.items():

            X_train = assemble_features(train, groups)
            y_train = train[target_col]

            X_test = assemble_features(test, groups)
            y_test = test[target_col]

            # Drop NA rows (train)
            mask_train = X_train.notna().all(axis=1) & y_train.notna()
            X_train = X_train[mask_train]
            y_train = y_train[mask_train]

            # Drop NA rows (test)
            mask_test = X_test.notna().all(axis=1) & y_test.notna()
            X_test = X_test[mask_test]
            y_test = y_test[mask_test]

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            # Financial metrics
            returns = test.loc[X_test.index, "contrarian_ret_12b"]
            strategy_returns = returns[preds == 1]

            mean_ret = strategy_returns.mean()
            hit_rate = (strategy_returns > 0).mean()
            sharpe = (
                strategy_returns.mean() / strategy_returns.std()
                if strategy_returns.std() > 0 else np.nan
            )

            results.append({
                "year": year,
                "experiment": name,
                "accuracy": acc,
                "mean_return": mean_ret,
                "hit_rate": hit_rate,
                "sharpe": sharpe,
                "n_trades": len(strategy_returns)
            })

    return pd.DataFrame(results)

# --------------------------------------------------
# DL signal integration helpers
# --------------------------------------------------

def _maybe_attach_dl_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and attach DL surface signals when ``DL_SIGNALS_ENABLED=True``.

    Returns *df* unchanged (default) when DL signals are disabled.
    On any loading failure the function warns and returns *df* unchanged so
    the existing pipeline is never blocked.
    """
    import sys
    import os

    # Ensure repo root is on sys.path so 'src' package is importable when
    # this file is run directly (e.g. python features/run_experiments.py)
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from src.dl_config import DL_SIGNALS_ENABLED, DL_SIGNALS_CUBE_PATH, DL_SIGNAL_SURFACE
    from src.dl_surface_loader import load_dl_surface

    if not DL_SIGNALS_ENABLED:
        return df

    print(f"Loading DL signals from: {DL_SIGNALS_CUBE_PATH}")
    surface_df = load_dl_surface(DL_SIGNALS_CUBE_PATH, DL_SIGNAL_SURFACE, strict=False)

    if surface_df.empty:
        print("  [warn] DL surface is empty; continuing without DL signals.")
        return df

    df = attach_dl_signals(df, surface_df)
    n_matched = int(df["dl_signal_strength"].notna().sum())
    print(f"  DL signals attached: {n_matched:,} rows with signal data.")
    return df


def _build_experiments() -> dict:
    """
    Return the experiment dict for this run.

    The ``baseline_plus_dl`` variant is excluded when DL signals are
    disabled to avoid spurious NaN-only feature columns in non-DL runs.
    """
    import sys
    import os

    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    try:
        from src.dl_config import DL_SIGNALS_ENABLED
    except ImportError:
        DL_SIGNALS_ENABLED = False

    experiments = dict(EXPERIMENTS)
    if not DL_SIGNALS_ENABLED:
        experiments.pop("baseline_plus_dl", None)
    return experiments

# --------------------------------------------------
# Main experiment loop
# --------------------------------------------------

def run():

    print("Loading data...")
    df = pd.read_csv(INPUT_PATH)

    df["entry_time"] = pd.to_datetime(df["entry_time"])

    df["target"] = (df["contrarian_ret_12b"] > 0).astype(int)
    TARGET = "target"

    df = add_behavioral_signal(df)

    # Optionally attach DL surface signals (no-op when DL_SIGNALS_ENABLED=False)
    df = _maybe_attach_dl_signals(df)

    experiments = _build_experiments()

    print("Running walk-forward experiments...")

    wf_results = run_walk_forward(df, experiments, TARGET)

    print("\nWalk-forward results:")
    print(wf_results)

    print("\nSummary (mean over folds):")
    print(
        wf_results.groupby("experiment")[["accuracy", "mean_return", "hit_rate", "sharpe"]]
        .mean()
        .sort_values("mean_return", ascending=False)
    )

    return wf_results


if __name__ == "__main__":
    res = run()
    print("\nSummary:")
    print(res)
