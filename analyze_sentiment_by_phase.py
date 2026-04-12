#!/usr/bin/env python3
"""
Top-level wrapper — delegates to ``analysis/analyze_sentiment_by_phase.py``.

Run from the repository root:

    python analyze_sentiment_by_phase.py
"""
from analysis.analyze_sentiment_by_phase import main

if __name__ == "__main__":
    main()
