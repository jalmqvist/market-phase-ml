"""
analysis/comparisons/__init__.py
==================================
Comparison generators for the MPML analysis framework v2.
"""

from analysis.comparisons.sentiment import compare_sentiment_variants
from analysis.comparisons.selector import compare_selector_uplift
from analysis.comparisons.gen_comparison import compare_gen1_gen2

__all__ = [
    "compare_sentiment_variants",
    "compare_selector_uplift",
    "compare_gen1_gen2",
]
