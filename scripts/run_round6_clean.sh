#!/usr/bin/env bash
# Round-6 review, sensitivity to the cleaned feature set (scripts/preprocess_clean.py): the principal
# 60N pair (centralized 60 epochs, federated 20 rounds x 3 local epochs) over five seeds on
# data/processed_clean, through configs_clean/. Idempotent: finished runs are skipped.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
L=results/metrics
JOBS=${JOBS:-1}
THREADS=${THREADS:-4}   # cap per job, to keep the Mac cool: 4 threads ran an epoch in 35 s against 26 s on all 12 cores
export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 THESIS_CFG=configs_clean
export TF_NUM_INTRAOP_THREADS=$THREADS TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=$THREADS
JOBFILE=$L/joblist_r6clean.txt; : > "$JOBFILE"
add () { local out="$1"; shift; [ -f "$L/$out" ] && { echo "skip: $out"; return; }; echo "$out|$*" >> "$JOBFILE"; }
for s in 101 102 103 104 105; do
  add "repeated_federated_iid_r6clean_le3_s$s.json" $PY scripts/repeated_runs.py --arm federated --rounds 20 --local-epochs 3 --seeds $s --suffix _r6clean_le3_s$s
  add "repeated_centralized_r6clean_e60_s$s.json"   $PY scripts/repeated_runs.py --arm centralized --epochs 60 --no-early-stopping --seeds $s --suffix _r6clean_e60_s$s
done
N=$(wc -l < "$JOBFILE" | tr -d ' '); echo "[$(date +%H:%M:%S)] $N jobs, $JOBS in parallel"
[ "$N" -eq 0 ] && exit 0
run_one () {
  local out="${1%%|*}" cmd="${1#*|}"
  echo "[$(date +%H:%M:%S)] start $out"
  bash -c "$cmd" > "results/metrics/joblogs_r6clean_${out%.json}.log" 2>&1 && [ -f "results/metrics/$out" ] \
    && echo "[$(date +%H:%M:%S)] done  $out" || echo "[$(date +%H:%M:%S)] FAIL  $out"
}
export -f run_one
tr '\n' '\0' < "$JOBFILE" | xargs -0 -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}
echo "[$(date +%H:%M:%S)] cleaned-data reruns finished"
