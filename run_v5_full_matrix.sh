#!/usr/bin/env bash

set -eo pipefail

mkdir -p logs
mkdir -p results_archive_v5_canonical

export EXPERIMENT_SEED=42

# =========================================================
# pair cohorts
# =========================================================

PERSISTENT_PAIRS="EURUSD,GBPUSD,NZDUSD,EURGBP,EURAUD"
REACTIVE_PAIRS="USDJPY,EURJPY,GBPJPY,EURCHF,USDCHF"

# =========================================================
# parquet registry
# =========================================================

PERSISTENT_SENTIMENT="../market-sentiment-ml/artifacts_v5/persistent_dl_sentiment/mlp__LVTF__24__price_trend__20260524T175056Z.parquet"

PERSISTENT_NOSENTIMENT="../market-sentiment-ml/artifacts_v5/persistent_dl_nosentiment/mlp__LVTF__24__trend_vol_only__20260524T175101Z.parquet"

REACTIVE_SENTIMENT="../market-sentiment-ml/artifacts_v5/reactive_dl_sentiment/mlp__LVTF__24__price_trend__20260524T175106Z.parquet"

REACTIVE_NOSENTIMENT="../market-sentiment-ml/artifacts_v5/reactive_dl_nosentiment/mlp__LVTF__24__trend_vol_only__20260524T175111Z.parquet"

# =========================================================
# helper
# =========================================================

run_mpml_experiment () {

    local run_id="$1"
    local generation="$2"
    local active_pairs="$3"
    local dl_enabled="$4"
    local parquet_path="$5"

    echo "========================================================"
    echo "RUNNING MPML EXPERIMENT: ${run_id}"
    echo "========================================================"

    export ACTIVE_PAIRS="${active_pairs}"

    if [ "${dl_enabled}" = "true" ]; then

        export DL_SIGNALS_ENABLED=true
        export DL_MODEL=mlp
        export DL_REGIME=LVTF
        export DL_PREDICTION_ARTIFACT_PATH="${parquet_path}"

    else

        export DL_SIGNALS_ENABLED=false

        unset DL_MODEL
        unset DL_REGIME
        unset DL_PREDICTION_ARTIFACT_PATH

    fi

    if [ "${generation}" = "gen1" ]; then

    if [ "${dl_enabled}" = "true" ]; then
        variant="A"
    else
        variant="B"
    fi

else

    if [ "${dl_enabled}" = "true" ]; then
        variant="C"
    else
        variant="D"
    fi

fi

python -u main.py \
  --experiment-variant "${variant}" \
  --output-dir "results_archive_v5_canonical/${run_id}" \
  > "logs/${run_id}.log" 2>&1

    echo "✓ completed: ${run_id}"
    echo
}

# =========================================================
# persistence-family evaluation
# =========================================================

run_mpml_experiment persistent_dl_sentiment_blind gen1 "${PERSISTENT_PAIRS}" true "${PERSISTENT_SENTIMENT}"

run_mpml_experiment persistent_dl_sentiment_aware gen2 "${PERSISTENT_PAIRS}" true "${PERSISTENT_SENTIMENT}"

run_mpml_experiment persistent_dl_nosentiment_blind gen1 "${PERSISTENT_PAIRS}" true "${PERSISTENT_NOSENTIMENT}"

run_mpml_experiment persistent_dl_nosentiment_aware gen2 "${PERSISTENT_PAIRS}" true "${PERSISTENT_NOSENTIMENT}"

run_mpml_experiment persistent_nodl_blind gen1 "${PERSISTENT_PAIRS}" false "none"

run_mpml_experiment persistent_nodl_aware gen2 "${PERSISTENT_PAIRS}" false "none"

# =========================================================
# reactive-family evaluation
# =========================================================

run_mpml_experiment reactive_dl_sentiment_blind gen1 "${REACTIVE_PAIRS}" true "${REACTIVE_SENTIMENT}"

run_mpml_experiment reactive_dl_sentiment_aware gen2 "${REACTIVE_PAIRS}" true "${REACTIVE_SENTIMENT}"

run_mpml_experiment reactive_dl_nosentiment_blind gen1 "${REACTIVE_PAIRS}" true "${REACTIVE_NOSENTIMENT}"

run_mpml_experiment reactive_dl_nosentiment_aware gen2 "${REACTIVE_PAIRS}" true "${REACTIVE_NOSENTIMENT}"

run_mpml_experiment reactive_nodl_blind gen1 "${REACTIVE_PAIRS}" false "none"

run_mpml_experiment reactive_nodl_aware gen2 "${REACTIVE_PAIRS}" false "none"

echo "========================================================"
echo "ALL MPML V5 PHASE-1 EXPERIMENTS COMPLETE"
echo "========================================================"

echo
echo "Run analysis with:"
echo "python analysis/pipeline.py results_archive_v5_canonical"
