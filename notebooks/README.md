# Notebooks

This folder contains the “portfolio” notebook(s) that explain the project end-to-end using **saved artifacts** (CSVs) produced by `python main.py`.

## Recommended workflow

1) Generate artifacts:

```bash
python main.py
```

2) Open the main notebook:

- `01_regime_gating_walkforward.ipynb`

The notebook reads from `../results/` (walk-forward summaries + fold-level debug CSVs) and produces plots + narrative without re-training models.

## Expected inputs (from `results/`)

- `walkforward_results_summary.csv`
- `walkforward_results_per_fold.csv`
- `walkforward_results_per_pair.csv`
- A few debug files like:
  - `equity_debug_<PAIR>_fold<N>.csv`
  - (optional) `selected_series_<PAIR>_fold<N>.csv` if you save those

If you don’t have debug CSVs yet, enable them in `main.py` via the `DEBUG_SAVE_EQUITY_SERIES` and `DEBUG_SELECTED_PAIRS` flags and re-run.