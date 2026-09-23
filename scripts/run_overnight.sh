#!/usr/bin/env bash
# Round-6 overnight batch on the Mac, kept cool: 4 threads per job (TF and OpenMP) and one job at a time.
# A job starts only if its estimated duration still fits before the deadline (default: next 06:00), so
# nothing is left half-finished at the deadline. Finished outputs are skipped, so the script can be rerun.
# Start: nohup caffeinate -i -s bash scripts/run_overnight.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=results/metrics
THREADS=${THREADS:-4}
T6=$(date -v6H -v0M -v0S +%s); [ "$(date +%s)" -ge "$T6" ] && T6=$(date -v+1d -v6H -v0M -v0S +%s)
DEADLINE=${DEADLINE:-$T6}
export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=$THREADS TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=$THREADS
LOG=$L/overnight_r6.log
say () { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
job () {   # job <estimated minutes> <name> <command...>
  local est=$1 name=$2; shift 2
  if [ $(( $(date +%s) + est * 60 )) -gt "$DEADLINE" ]; then say "SKIP  $name (needs ~$est min before the deadline)"; return; fi
  say "start $name (~$est min)"
  if "$@" >> "$L/overnight_r6_$name.log" 2>&1; then say "done  $name"; else say "FAIL  $name"; fi
}
say "batch start: $THREADS threads, deadline $(date -r "$DEADLINE" +%H:%M)"
job 20 baselines_original $PY scripts/baselines_seeds.py --data original --permutation
job 6  baselines_clean    $PY scripts/baselines_seeds.py --data clean --models XGBoost MLP
for s in 101 102 103 104 105; do
  [ -f "$L/repeated_federated_iid_r6clean_le3_s$s.json" ] || job 50 "fed_clean_s$s" env THESIS_CFG=configs_clean \
    $PY scripts/repeated_runs.py --arm federated --rounds 20 --local-epochs 3 --seeds $s --suffix _r6clean_le3_s$s
  [ -f "$L/repeated_centralized_r6clean_e60_s$s.json" ] || job 40 "cen_clean_s$s" env THESIS_CFG=configs_clean \
    $PY scripts/repeated_runs.py --arm centralized --epochs 60 --no-early-stopping --seeds $s --suffix _r6clean_e60_s$s
done
job 25 deployment_sessions $PY scripts/deployment_costs.py --measure-only --sessions 3
say "batch end"
