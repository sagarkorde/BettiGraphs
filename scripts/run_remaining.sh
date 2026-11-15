#!/usr/bin/env bash
# Run the remaining stages sequentially, unattended.
#
# Sequential is not a stylistic choice: every stage saturates all 32 cores, and two concurrent
# stages make the machine slower rather than faster. It also keeps the wall-clock timings that feed
# the cost table uncontended, which a parallel run would silently corrupt.
#
# Each stage logs to results/raw/<stage>.log and the chain continues past a failure so that one
# broken stage does not cost the rest of the night. Stage outcomes are summarised at the end.

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
    NAMES+=("$name")
    CODES+=("$code")
    if [ $code -eq 0 ]; then
        echo "=== [$(date +%H:%M:%S)] $name OK ==="
    else
        echo "=== [$(date +%H:%M:%S)] $name FAILED (exit $code), continuing ==="
        tail -15 "$log"
    fi
}

# Wait for the wallet experiment to land before touching the machine.
echo "waiting for the wallet experiment to finish..."
for _ in $(seq 1 240); do
    [ -f results/processed/wallet_retrieval.csv ] && break
    grep -q "Traceback" "$LOG_DIR/wallet_run.log" 2>/dev/null && break
    sleep 15
done
echo "proceeding at $(date +%H:%M:%S)"

# Gate 6 selection belongs before evaluation per the pre-registration. It was omitted from the
# first version of this script, which is why it ran out of order in practice; keeping it here so a
# clean reproduction does it in the right sequence.
run_stage "selection"      experiments/betti/select_topology_settings.py --stability-n 200
run_stage "ablation"       experiments/ablation/run_topological_ablation.py --radius 3
run_stage "scalability"    scripts/measure_scalability.py --n-sample 400
run_stage "tx_fast"        experiments/baseline/run_transaction_experiment.py --radius 3 \
                             --models logistic xgboost lightgbm
run_stage "analysis_part1" scripts/analyse_transaction_results.py
run_stage "tx_rf"          experiments/baseline/run_transaction_experiment.py --radius 3 \
                             --models random_forest
run_stage "trajectories"   experiments/temporal/entity_trajectories.py
run_stage "analysis_final" scripts/analyse_transaction_results.py
run_stage "figures"        scripts/make_figures.py
run_stage "schematics"     scripts/make_schematics.py

echo ""
echo "================ SUMMARY ================"
for i in "${!NAMES[@]}"; do
    if [ "${CODES[$i]}" -eq 0 ]; then status="ok"; else status="FAILED (${CODES[$i]})"; fi
    printf "  %-16s %s\n" "${NAMES[$i]}" "$status"
done
echo "========================================="
