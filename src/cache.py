# src/cache.py

import pickle
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Optional

CACHE_DIR = Path('data/cache')


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Generate a hash from a DataFrame's contents."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:12]


def _hash_dict_of_dataframes(data: dict) -> str:
    """Generate a single hash from a dict of DataFrames."""
    combined = ''.join(
        f'{k}:{_hash_dataframe(v)}'
        for k, v in sorted(data.items())
    )
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def _hash_params(**kwargs) -> str:
    """Generate a hash from arbitrary parameters."""
    param_str = str(sorted(kwargs.items()))
    return hashlib.md5(param_str.encode()).hexdigest()[:12]


def _cache_path(name: str, data_hash: str, param_hash: str) -> Path:
    return CACHE_DIR / f'{name}__{data_hash}__{param_hash}.pkl'


def save_cache(name: str,
               data: Any,
               data_hash: str,
               param_hash: str) -> None:
    """Save data to cache."""
    _ensure_cache_dir()
    path = _cache_path(name, data_hash, param_hash)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f'  ✓ Cache saved: {path.name}')


def load_cache(name: str,
               data_hash: str,
               param_hash: str) -> Optional[Any]:
    """
    Load data from cache if it exists.
    Returns None if cache miss.
    """
    path = _cache_path(name, data_hash, param_hash)
    if path.exists():
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f'  ✓ Cache hit: {path.name}')
        return data
    return None


def clear_cache(name: Optional[str] = None) -> None:
    """
    Clear cache files.
    If name is provided, only clear files matching that name.
    Otherwise clear all cache files.
    """
    _ensure_cache_dir()
    pattern = f'{name}__*.pkl' if name else '*.pkl'
    removed = 0
    for f in CACHE_DIR.glob(pattern):
        f.unlink()
        removed += 1
    print(f'  ✓ Cleared {removed} cache file(s)')