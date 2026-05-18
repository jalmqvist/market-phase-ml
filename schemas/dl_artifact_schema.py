"""
Centralized DL artifact contract constants (shared with MSML).

MPML imports these names instead of hard-coding DL artifact column strings so
producer/consumer schema updates stay synchronized across repositories.
"""

DL_SCHEMA_VERSION = "2.0.0"
DL_TIMESTAMP_COL = "timestamp"
DL_AVAILABLE_TS_COL = "prediction_available_timestamp"
DL_GENERATED_TS_COL = "prediction_generated_timestamp"
DL_ARTIFACT_CREATED_COL = "artifact_created_timestamp"
DL_PAIR_COL = "pair"
