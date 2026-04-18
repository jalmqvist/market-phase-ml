import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from features.experiments import EXPERIMENTS
from features.assembler import assemble_features


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
# Main experiment loop
# --------------------------------------------------

def run():

    print("Loading data...")
    #df = pd.read_parquet(INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)
    print(df.columns.tolist())
    df["entry_time"] = pd.to_datetime(df["entry_time"])

    df["target"] = (df["contrarian_ret_12b"] > 0).astype(int)
    TARGET = "target"

    # Add behavioral feature
    df = add_behavioral_signal(df)

    results = []

    for name, groups in EXPERIMENTS.items():

        print(f"\nRunning experiment: {name}")

        X = assemble_features(df, groups)
        y = df[TARGET]

        # Drop NA rows
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        train_X, test_X, train_y, test_y = train_test_split_xy(X, y)

        # Simple model (interpretable baseline)
        model = LogisticRegression(max_iter=1000)

        model.fit(train_X, train_y)

        preds = model.predict(test_X)

        acc = accuracy_score(test_y, preds)

        print(f"Accuracy: {acc:.4f}")

        results.append({
            "experiment": name,
            "accuracy": acc,
            "n_features": train_X.shape[1],
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    res = run()
    print("\nSummary:")
    print(res)
