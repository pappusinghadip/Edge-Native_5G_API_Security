#!/usr/bin/env bash
# Round-6 step 5 on the Mac, kept cool (4 threads, one job at a time, deadline guard, skips finished runs):
# CNN feature study at 20N (original order, a fixed random order, and without Time_To_Live; seeds 101-103)
# and FedAvg against FedBN on the Dirichlet 0.1 partition (scripts/fedbn_sensitivity.py).
# Start: nohup caffeinate -i -s bash scripts/run_step5.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=results/metrics
THREADS=${THREADS:-4}
T6=$(date -v6H -v0M -v0S +%s); [ "$(date +%s)" -ge "$T6" ] && T6=$(date -v+1d -v6H -v0M -v0S +%s)
DEADLINE=${DEADLINE:-$T6}
export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=$THREADS TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=$THREADS
LOG=$L/step5_r6.log
say () { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
job () {   # job <estimated minutes> <name> <command...>
  local est=$1 name=$2; shift 2
  if [ $(( $(date +%s) + est * 60 )) -gt "$DEADLINE" ]; then say "SKIP  $name (needs ~$est min before the deadline)"; return; fi
  say "start $name (~$est min)"
  if "$@" >> "$L/step5_r6_$name.log" 2>&1; then say "done  $name"; else say "FAIL  $name"; fi
}
say "batch start: $THREADS threads, deadline $(date -r "$DEADLINE" +%H:%M)"
[ -d data/processed_perm ] || $PY scripts/make_feature_variants.py >> "$LOG" 2>&1
for s in 101 102 103; do
  for v in orig perm nottl; do
    cfg=configs; [ $v != orig ] && cfg=configs_$v
    [ -f "$L/repeated_centralized_r6feat_${v}_e20_s$s.json" ] || job 15 "feat_${v}_s$s" env THESIS_CFG=$cfg \
      $PY scripts/repeated_runs.py --arm centralized --epochs 20 --no-early-stopping --seeds $s --suffix _r6feat_${v}_e20_s$s
  done
done
job 70 fedbn $PY scripts/fedbn_sensitivity.py --seeds 101 102 103
say "batch end"
