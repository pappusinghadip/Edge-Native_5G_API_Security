#!/usr/bin/env bash
# Re-run the 5 isolated arms so they save per-client probabilities and weights
# (the original run predated the simulate.py patch and saved neither).
#
# Launched detached (setsid) so a stray Ctrl-C in any terminal cannot reach it.
# Existing outputs are NOT moved aside: simulate.py writes its JSON only on
# completion, so an interrupted run leaves the previous results in place.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=results/metrics
JOBS=3
THREADS=5

export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=""
export TF_NUM_INTRAOP_THREADS=$THREADS TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=$THREADS
mkdir -p "$L/joblogs" results/models

JOBFILE=$L/joblist_isolated.txt
: > "$JOBFILE"
: > "$L/progress_isolated.log"
for s in 101 102 103 104 105; do
  t="isolated_s${s}"
  : > "$L/joblogs/${t}_rerun.log"
  echo "$PY -m src.fl.simulate --config configs/fl.yaml --model configs/model.yaml --paths configs/paths.yaml --rounds 20 --tag $t --mode iid --experiment isolated --seed $s >> $L/joblogs/${t}_rerun.log 2>&1 && echo OK $t >> $L/progress_isolated.log || echo FAILED $t >> $L/progress_isolated.log" >> "$JOBFILE"
done

echo "[$(date +%H:%M:%S)] running 5 isolated arms, $JOBS concurrent (detached)"
tr '\n' '\0' < "$JOBFILE" | xargs -0 -P "$JOBS" -n 1 bash -c
echo "[$(date +%H:%M:%S)] finished"
echo "  OK:     $(grep -c '^OK' "$L/progress_isolated.log" 2>/dev/null || true)"
echo "  FAILED: $(grep -c '^FAILED' "$L/progress_isolated.log" 2>/dev/null || true)"
