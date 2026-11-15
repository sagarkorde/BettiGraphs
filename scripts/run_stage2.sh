#!/usr/bin/env bash
# Finish the study. Written once and NOT edited while running.
#
# The previous runner was edited mid-execution to insert a stage. Bash reads a script lazily by
# byte offset, so the insertion shifted the offsets underneath the running shell: it resumed at the
# wrong position, tried to execute a Python file as shell, then looped back through two stages it
# had already completed. Nothing was corrupted on disk, but two hours of machine time went on
# repeats. Never edit a shell script that is executing.
#
# Order: everything cheap and informative first, the three-hour random-forest pass last, so a
# complete set of results exists as early as possible.
#
# Stages run one at a time. Each saturates all cores, and the wall-clock timings that feed the
# cost table are meaningless under contention.

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

# 1. Verdicts for H2, H6 and H7 from the 24 completed cells. Cheap, and the most valuable output.
run_stage "analysis_fast"  scripts/analyse_transaction_results.py

# 2. Cost table, with the axis fix. Needs an idle machine, which it now has.
run_stage "scalability2"   scripts/measure_scalability.py --n-sample 400

# 3. Gate 6 selection, required by the pre-registration.
run_stage "selection"      experiments/betti/select_topology_settings.py --stability-n 200

# 4. H5 on the multi-observation address cohort.
run_stage "trajectories"   experiments/temporal/entity_trajectories.py \
                             --min-observations 3 --max-addresses 3000

# 5. Does a hybrid model reach for anything topological?
run_stage "interpretability" experiments/hybrid/run_interpretability.py --model lightgbm

# 6. Figures from everything so far, before the long stage.
run_stage "figures_a"      scripts/make_figures.py
run_stage "schematics_a"   scripts/make_schematics.py

# 7. The fourth model. Resumable, and only adds random forest to the 24 existing cells.
run_stage "tx_rf"          experiments/baseline/run_transaction_experiment.py --radius 3 \
                             --models random_forest

# 8. Final analysis and figures over all four models.
run_stage "analysis_final" scripts/analyse_transaction_results.py
run_stage "figures_b"      scripts/make_figures.py

echo ""
echo "================ SUMMARY ================"
for i in "${!NAMES[@]}"; do
    if [ "${CODES[$i]}" -eq 0 ]; then status="ok"; else status="FAILED (${CODES[$i]})"; fi
    printf "  %-18s %s\n" "${NAMES[$i]}" "$status"
done
echo "========================================="
