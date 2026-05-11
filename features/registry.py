FEATURES = {
    "baseline": [
        "trend_12b",
        "trend_strength_12b",
        "trend_alignment_12b",  # interaction signal
    ],

    "sentiment_core": [
        "net_sentiment",
        "abs_sentiment",
        "extreme_70",
    ],

    "sentiment_behavioral": [
        "JPY_behavioral_signal",
    ],

    # DL signal surface features (market-sentiment-ml integration).
    # dl_signal_strength is signal_strength renamed to avoid collisions.
    # dl_confidence and dl_pred_prob_up are optional; absent columns are filled
    # with NaN by assemble_features() so pipelines continue to run when the
    # DL artifact is unavailable.
    "dl_signal": [
        "dl_signal_strength",
        "dl_confidence",
        "dl_pred_prob_up",
    ],
}

# Columns belonging to the DL signal feature group.
# assemble_features() fills these with NaN when they are absent from the
# DataFrame rather than raising, allowing existing pipelines to run unchanged
# when DL signals are disabled (DL_SIGNALS_ENABLED=False).
DL_FEATURE_COLUMNS: frozenset = frozenset(FEATURES["dl_signal"])