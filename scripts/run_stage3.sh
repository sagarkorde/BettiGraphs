#!/usr/bin/env bash
# Finish the remaining stages. Written once and NOT edited while running.
#
# Already complete and not repeated here: dataset validation, the literature build, H1, H4, the
# wallet retrieval experiment, the ablation, the scalability measurement with the axis fix, and the
# transaction comparison for logistic, XGBoost and LightGBM (24 cells, 240 records).
#
# Stages run one at a time: each saturates all cores, and the wall-clock timings feeding the cost
# table are meaningless under contention. The chain continues past a failure so one broken stage
# does not cost the rest of the run.

set -u
cd "$(dirname "$0")/.." || exit 1
PY="./.venv/Scripts/python.exe"
LOG_DIR="results/raw"
mkdir -p "$LOG_DIR"

declare -a NAMES=()
declare -a CODES=()

run_stage() {
    local name="$1"; shift
    local log="$LOG_DIR/${name}.log"
    echo "=== [$(date +%H:%M:%S)] $name ==="
    "$PY" -u "$@" > "$log" 2>&1
    local code=$?
    NAMES+=("$name"); CODES+=("$code")
    if [ $code -eq 0 ]; then
        echo "=== [$(date +%H:%M:%S)] $name OK ==="
    else
        echo "=== [$(date +%H:%M:%S)] $name FAILED (exit $code) ==="
        tail -20 "$log"
    fi
}

run_stage "selection"        experiments/betti/select_topology_settings.py --stability-n 200
run_stage "trajectories"     experiments/temporal/entity_trajectories.py \
                               --min-observations 3 --max-addresses 3000
run_stage "interpretability" experiments/hybrid/run_interpretability.py --model lightgbm
run_stage "figures_a"        scripts/make_figures.py
run_stage "schematics_a"     scripts/make_schematics.py

# The fourth model. Long, resumable, and only adds random forest to the existing 24 cells.
run_stage "tx_rf"            experiments/baseline/run_transaction_experiment.py --radius 3 \
                               --models random_forest
run_stage "analysis_final"   scripts/analyse_transaction_results.py
run_stage "figures_b"        scripts/make_figures.py

echo ""
echo "================ SUMMARY ================"
for i in "${!NAMES[@]}"; do
    if [ "${CODES[$i]}" -eq 0 ]; then status="ok"; else status="FAILED (${CODES[$i]})"; fi
    printf "  %-18s %s\n" "${NAMES[$i]}" "$status"
done
echo "========================================="
