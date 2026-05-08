EXPERIMENTS = {
    "baseline": ["baseline"],
    "baseline_plus_sentiment": ["baseline", "sentiment_core"],
    "baseline_plus_behavioral": ["baseline", "sentiment_behavioral"],
    # DL surface integration — only active when DL_SIGNALS_ENABLED=True.
    # When DL signals are disabled the run_experiments pipeline omits this
    # variant automatically.
    "baseline_plus_dl": ["baseline", "dl_signal"],
}
