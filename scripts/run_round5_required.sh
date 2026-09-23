#!/usr/bin/env bash
# Round-5 required reruns (review of 12 Sep 2026): exposure-matched training with the
# class-dependent focal loss. 25 jobs, five seeds each:
#
#   centralized  20 epochs, no early stopping      exposure 20N   (matches federated 20 x 1)
#   centralized  60 epochs, no early stopping      exposure 60N   (matches federated 20 x 3)
#   federated    20 rounds x 1 local epoch         exposure 20N
#   federated    20 rounds x 3 local epochs        exposure 60N   (the schedule in the thesis)
#   isolated     20 rounds x 3 = 60 local epochs   exposure 60N
#
# Run on the Asus under tmux; about 8-10 h with three jobs in parallel:
#   tmux new -s round5
#   JOBS=3 bash scripts/run_round5_required.sh 2>&1 | tee results/metrics/run_round5.log
#
# Idempotent: finished jobs are skipped, so it can be interrupted and restarted.
set -u
cd "$(dirname "$0")/.."

PY=.venv/bin/python
L=results/metrics
JOBS=${JOBS:-3}
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
THREADS=$(( CORES / JOBS )); [ "$THREADS" -lt 1 ] && THREADS=1

export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2
# same device as the first PC run: CPU only (the GPU measured slower for this model)
export CUDA_VISIBLE_DEVICES=""
export TF_NUM_INTRAOP_THREADS=$THREADS OMP_NUM_THREADS=$THREADS TF_NUM_INTEROP_THREADS=1
mkdir -p "$L/joblogs_r5"

grep -q "focal_class_weighted: true" configs/model.yaml || { echo "configs/model.yaml lacks the focal-loss fix; copy the updated configs/ first"; exit 1; }

JOBFILE=$L/joblist_r5.txt; : > "$JOBFILE"
add () {   # add <output-file> <command...>
  local out="$1"; shift
  if [ -f "$L/$out" ]; then echo "skip (done): $out"; else echo "$out|$*" >> "$JOBFILE"; fi
}
RR="$PY scripts/repeated_runs.py"
SIM="$PY -m src.fl.simulate --config configs/fl.yaml --model configs/model.yaml --paths configs/paths.yaml"

for s in 101 102 103 104 105; do
  # longest jobs first in each seed, so three slow arms never queue behind short ones
  add "repeated_federated_iid_r5_le3_s$s.json" $RR --arm federated --rounds 20 --local-epochs 3 --seeds $s --suffix _r5_le3_s$s
  add "fl_isolated_iid_r5_isolated_s$s.json"    $SIM --mode iid --experiment isolated --rounds 20 --local-epochs 3 --seed $s --tag r5_isolated_s$s
  add "repeated_centralized_r5_e60_s$s.json"    $RR --arm centralized --epochs 60 --no-early-stopping --seeds $s --suffix _r5_e60_s$s
  add "repeated_federated_iid_r5_le1_s$s.json"  $RR --arm federated --rounds 20 --local-epochs 1 --seeds $s --suffix _r5_le1_s$s
  add "repeated_centralized_r5_e20_s$s.json"    $RR --arm centralized --epochs 20 --no-early-stopping --seeds $s --suffix _r5_e20_s$s
done

N=$(wc -l < "$JOBFILE" | tr -d ' ')
echo "[$(date +%H:%M:%S)] $N jobs, $JOBS in parallel, $THREADS threads each"
[ "$N" -eq 0 ] && { echo "nothing to do"; exit 0; }

run_one () {
  local out="${1%%|*}" cmd="${1#*|}"
  local log="results/metrics/joblogs_r5/${out%.json}.log"
  echo "[$(date +%H:%M:%S)] start $out"
  if bash -c "$cmd" > "$log" 2>&1 && [ -f "results/metrics/$out" ]; then
    echo "[$(date +%H:%M:%S)] done  $out"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $out  (see $log)"
  fi
}
export -f run_one
tr '\n' '\0' < "$JOBFILE" | xargs -0 -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

echo "[$(date +%H:%M:%S)] finished. Missing outputs, if any:"
while IFS='|' read -r out _; do [ -f "$L/$out" ] || echo "  MISSING $out"; done < "$JOBFILE"
echo "Copy results/metrics back to the Mac (USB), as in HANDOFF.md."
