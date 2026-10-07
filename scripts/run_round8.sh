#!/usr/bin/env bash
# Round-8 review on the Mac, kept cool (THREADS threads, one job at a time, deadline guard, skips finished runs):
# capture-sensitive feature test, 20N, seeds 101-103: the 1D-CNN without Time_To_Live and Header_Length
# (configs_nocap, from scripts/make_feature_variants.py nocap), centralized and federated. The originals it is
# paired with are the r6feat_orig (centralized) and r7fed20_orig (federated) runs.
# Start: THREADS=2 nohup caffeinate -i -s nice -n 19 bash scripts/run_round8.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=results/metrics
THREADS=${THREADS:-4}
T6=$(date -v6H -v0M -v0S +%s); [ "$(date +%s)" -ge "$T6" ] && T6=$(date -v+1d -v6H -v0M -v0S +%s)
DEADLINE=${DEADLINE:-$T6}
export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=$THREADS TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=$THREADS
LOG=$L/round8.log
say () { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
job () {   # job <estimated minutes> <name> <command...>
  local est=$1 name=$2; shift 2
  if [ $(( $(date +%s) + est * 60 )) -gt "$DEADLINE" ]; then say "SKIP  $name (needs ~$est min before the deadline)"; return; fi
  say "start $name (~$est min)"
  if "$@" >> "$L/round8_$name.log" 2>&1; then say "done  $name"; else say "FAIL  $name"; fi
}
say "batch start: $THREADS threads, deadline $(date -r "$DEADLINE" +%H:%M)"
for s in 101 102 103; do
  [ -f "$L/repeated_centralized_r8nocap_e20_s$s.json" ] || job 20 "cen20_nocap_s$s" env THESIS_CFG=configs_nocap \
    $PY scripts/repeated_runs.py --arm centralized --epochs 20 --no-early-stopping --seeds $s --suffix _r8nocap_e20_s$s
  [ -f "$L/repeated_federated_iid_r8nocap_s$s.json" ] || job 20 "fed20_nocap_s$s" env THESIS_CFG=configs_nocap \
    $PY scripts/repeated_runs.py --arm federated --rounds 20 --local-epochs 1 --seeds $s --suffix _r8nocap_s$s
done
say "batch end"
