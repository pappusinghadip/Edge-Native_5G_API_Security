#!/usr/bin/env bash
# Round-9 review, recommended experiment, on the Asus PC (WSL Ubuntu, CPU only): the alpha = 0.1 partition
# draws that collapsed on the Mac (342, 442) and the thesis draw (42), each over seeds 101-103, so that
# partition and training-seed variation are measured together on one platform. Federated 60N
# (20 rounds x 3 local epochs), one job at a time, finished runs skipped, stops starting jobs after
# MAX_HOURS. Partitions are regenerated first if missing (deterministic from the partition seed).
# Start: nohup bash scripts/run_round9.sh >/dev/null 2>&1 &      progress: results/metrics/round9.log
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
L=results/metrics
THREADS=${THREADS:-6}
DEADLINE=$(( $(date +%s) + ${MAX_HOURS:-12} * 3600 ))
export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES="" \
       TF_NUM_INTRAOP_THREADS=$THREADS TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=$THREADS
LOG=$L/round9.log
say () { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
job () {   # job <estimated minutes> <name> <command...>
  local est=$1 name=$2; shift 2
  if [ $(( $(date +%s) + est * 60 )) -gt "$DEADLINE" ]; then say "SKIP  $name (needs ~$est min before the deadline)"; return; fi
  say "start $name (~$est min)"
  if "$@" >> "$L/round9_$name.log" 2>&1; then say "done  $name"; else say "FAIL  $name"; fi
}
say "batch start: $THREADS threads, stops starting jobs after ${MAX_HOURS:-12} h"
for d in 342 442; do
  [ -f "data/partitions_draw$d/client_0_dir_0p1.npz" ] || \
    $PY -m src.data.partitioner --config configs_draw$d/fl.yaml --paths configs_draw$d/paths.yaml --alpha 0.1 >> "$LOG" 2>&1
done
[ -f data/partitions/client_0_dir_0p1.npz ] || \
  $PY -m src.data.partitioner --config configs/fl.yaml --paths configs/paths.yaml --alpha 0.1 >> "$LOG" 2>&1
for d in 342 442 42; do
  cfg=configs; [ $d != 42 ] && cfg=configs_draw$d
  for s in 101 102 103; do
    [ -f "$L/repeated_federated_dirichlet_r9draw${d}_a0p1_pc_s$s.json" ] || job 35 "draw${d}_s$s" env THESIS_CFG=$cfg \
      $PY scripts/repeated_runs.py --arm federated --mode dirichlet --alpha 0.1 --rounds 20 --local-epochs 3 \
        --seeds $s --suffix _r9draw${d}_a0p1_pc_s$s
  done
done
say "batch end"
