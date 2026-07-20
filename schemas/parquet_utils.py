"""
Shared parquet utility helpers for MPML.

These helpers are imported by any module that reads parquet key-value metadata,
avoiding repeated implementations of the same decode loop.
"""
from __future__ import annotations

from pathlib import Path


def read_parquet_kv_metadata(path: Path) -> dict[str, str]:
    """Read file-level parquet key-value metadata as decoded strings.

    Tries to import ``pyarrow.parquet`` at call-time so that callers that
    cannot install pyarrow still import without error (they just receive an
    empty dict).

    Parameters
    ----------
    path : Path
        Path to the parquet file.

    Returns
    -------
    dict[str, str]
        File-level metadata decoded as UTF-8 strings.
        Returns an empty dict when the file cannot be read, when pyarrow is
        unavailable, or when the file has no embedded metadata.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}
    try:
        raw_metadata = pq.read_metadata(path).metadata
    except Exception:
        return {}
    if not raw_metadata:
        return {}
    decoded: dict[str, str] = {}
    for key, value in raw_metadata.items():
        try:
            decoded[key.decode("utf-8")] = value.decode("utf-8")
        except (AttributeError, UnicodeDecodeError):
            continue
    return decoded
