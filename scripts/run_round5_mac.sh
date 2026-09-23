#!/usr/bin/env bash
# Mac side of the round-5 reruns, so the cross-platform check uses the fixed focal loss too:
# the 60N matched pair (centralized 60 epochs, federated 20 rounds x 3) over five seeds.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=results/metrics
JOBS=${JOBS:-2}
export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2
JOBFILE=$L/joblist_r5mac.txt; : > "$JOBFILE"
add () { local out="$1"; shift; [ -f "$L/$out" ] && { echo "skip: $out"; return; }; echo "$out|$*" >> "$JOBFILE"; }
for s in 101 102 103 104 105; do
  add "repeated_federated_iid_r5mac_le3_s$s.json" $PY scripts/repeated_runs.py --arm federated --rounds 20 --local-epochs 3 --seeds $s --suffix _r5mac_le3_s$s
  add "repeated_centralized_r5mac_e60_s$s.json"   $PY scripts/repeated_runs.py --arm centralized --epochs 60 --no-early-stopping --seeds $s --suffix _r5mac_e60_s$s
done
N=$(wc -l < "$JOBFILE" | tr -d ' '); echo "[$(date +%H:%M:%S)] $N jobs, $JOBS in parallel"
[ "$N" -eq 0 ] && exit 0
run_one () {
  local out="${1%%|*}" cmd="${1#*|}"
  echo "[$(date +%H:%M:%S)] start $out"
  bash -c "$cmd" > "results/metrics/joblogs_r5mac_${out%.json}.log" 2>&1 && [ -f "results/metrics/$out" ] \
    && echo "[$(date +%H:%M:%S)] done  $out" || echo "[$(date +%H:%M:%S)] FAIL  $out"
}
export -f run_one
tr '\n' '\0' < "$JOBFILE" | xargs -0 -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}
echo "[$(date +%H:%M:%S)] mac reruns finished"
