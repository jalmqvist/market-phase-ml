"""Import-time numeric launch environment conversions."""

import os


def min_dl_train_coverage_pct() -> float:
    return float(os.environ.get("MIN_DL_TRAIN_COVERAGE_PCT", "5.0"))


def dl_surface_target_horizon() -> int:
    return int(os.environ.get("DL_SURFACE_TARGET_HORIZON", "24"))
