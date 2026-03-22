# Notebooks

This folder contains `01_regime_gating_walkforward.ipynb` — an end-to-end engineering
case study of the regime-aware mixture-of-experts gating system.  The notebook reads
**saved artifacts** (CSVs produced by `python main.py`) and produces plots + narrative
without re-training any models.

## How to reproduce

### Step 1 — Generate artifacts

Run the pipeline from the repository root.  This produces all CSV artifacts and,
when `DEBUG_SAVE_EQUITY_SERIES` is enabled in `main.py`, the per-fold debug exports
required by Part C of the notebook:

```bash
python main.py
```

Expected output files in `results/`:

| File | Required by |
|---|---|
| `walkforward_results_summary.csv` | Part B |
| `walkforward_results_per_fold.csv` | Part B |
| `walkforward_results_per_pair.csv` | Part B |
| `equity_debug_GBPJPY_fold8.csv` | Part C (default case study) |

> If `equity_debug_*.csv` files are missing, set `DEBUG_SAVE_EQUITY_SERIES = True`
> in `main.py` and re-run.

### Step 2 — Open the notebook

```bash
jupyter notebook notebooks/01_regime_gating_walkforward.ipynb
```

The notebook is **read-only on artifacts** — it loads CSVs and produces figures without
re-training any models or modifying the `results/` directory.

### Changing the case-study pair or fold

The case study in Part C is parameterised at the top of that section:

```python
pair = "GBPJPY"
fold = 8
```

Change these two variables to inspect any other pair/fold for which a
`results/equity_debug_{pair}_fold{fold}.csv` exists, then re-run cells from that
point forward.

## Notebook structure

| Part | Content |
|---|---|
| **Part A** | What / Why / How — executive summary, problem framing, fold definition |
| **Part B** | Evidence — fold-level distributions (Sharpe Delta, DD Delta), per-pair breakdown |
| **Part C** | Failure modes — GBPJPY fold 8 case study: volatility guard, equity comparison, selection timeline |
| **Part D** | Engineering highlights + next steps |

## Expected inputs (from `results/`)

| File | Description |
|---|---|
| `walkforward_results_summary.csv` | Run-level aggregate metrics |
| `walkforward_results_per_fold.csv` | Per-fold deltas — primary analysis unit |
| `walkforward_results_per_pair.csv` | Per-instrument averages |
| `equity_debug_<PAIR>_fold<N>.csv` | Bar-by-bar equity + spike + selection state (debug export) |

> **Note on debug CSVs:** if they are missing, enable them in `main.py` via the
> `DEBUG_SAVE_EQUITY_SERIES` flag and re-run.

## Metric conventions

- `DD Delta = Dynamic Max DD (%) - Baseline Max DD (%)`.
- Max DD (%) is stored as a **negative number** (e.g., -30 %).
- Therefore **positive DD Delta means Dynamic had a less negative drawdown -> better**.
- Same sign convention applies: positive Sharpe Delta and Return Delta are straightforwardly better.

## Utility module

`utils.py` (in this folder) provides DataFrame-based plotting and loading functions
used by the notebook:

- `load_results_csv(filename)` — load a results CSV from `../results/`
- `load_equity_debug(pair, fold)` — load `equity_debug_{pair}_fold{fold}.csv`
- `plot_equity_vs_spikes(df, title)` — 2-panel equity + drawdown plot with spike shading
- `plot_selected_timeline(df, title)` — policy-selection step chart with spike shading
