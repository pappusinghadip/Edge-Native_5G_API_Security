#!/usr/bin/env bash
# Round-7 review on the Mac, kept cool (4 threads, one job at a time, deadline guard, skips finished runs):
#   A1  federated 1D-CNN at 20N with and without Time_To_Live, seeds 101-103 (pairs with the centralized
#       r6feat runs of step 5)
#   A2  one corrected-loss centralized 1D-CNN (60N, seed 102) that also saves validation scores, so the
#       exploratory ensembles can be refitted on it
#   A3  five Dirichlet partition draws (seed 42 = the thesis draw, then 142/242/342/442) at alpha 0.1 and
#       0.5, federated 60N, training seed 101
# Partitions for the draws come from configs_draw<seed>/ (python -m src.data.partitioner ... --alpha a).
# Start: nohup caffeinate -i -s bash scripts/run_round7.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=results/metrics
THREADS=${THREADS:-4}
T6=$(date -v6H -v0M -v0S +%s); [ "$(date +%s)" -ge "$T6" ] && T6=$(date -v+1d -v6H -v0M -v0S +%s)
DEADLINE=${DEADLINE:-$T6}
export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=$THREADS TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=$THREADS
LOG=$L/round7.log
say () { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
job () {   # job <estimated minutes> <name> <command...>
  local est=$1 name=$2; shift 2
  if [ $(( $(date +%s) + est * 60 )) -gt "$DEADLINE" ]; then say "SKIP  $name (needs ~$est min before the deadline)"; return; fi
  say "start $name (~$est min)"
  if "$@" >> "$L/round7_$name.log" 2>&1; then say "done  $name"; else say "FAIL  $name"; fi
}
say "batch start: $THREADS threads, deadline $(date -r "$DEADLINE" +%H:%M)"

# A1: federated 20N, original features and without Time_To_Live
for s in 101 102 103; do
  for v in orig nottl; do
    cfg=configs; [ $v != orig ] && cfg=configs_$v
    [ -f "$L/repeated_federated_iid_r7fed20_${v}_s$s.json" ] || job 15 "fed20_${v}_s$s" env THESIS_CFG=$cfg \
      $PY scripts/repeated_runs.py --arm federated --rounds 20 --local-epochs 1 --seeds $s --suffix _r7fed20_${v}_s$s
  done
done

# A2: corrected-loss centralized CNN with validation scores, for the ensemble refit
[ -f "$L/repeated_centralized_r7ens_e60_s102.json" ] || job 30 cen60_ens_s102 \
  $PY scripts/repeated_runs.py --arm centralized --epochs 60 --no-early-stopping --seeds 102 --suffix _r7ens_e60_s102

# A3: partition draws, alpha 0.1 first (the more skewed level), then 0.5
for a in 0.1 0.5; do
  at=${a/./p}
  for d in 42 142 242 342 442; do
    cfg=configs; [ $d != 42 ] && cfg=configs_draw$d
    [ -f "$L/repeated_federated_dirichlet_r7draw${d}_a${at}_s101.json" ] || job 30 "draw${d}_a${at}" env THESIS_CFG=$cfg \
      $PY scripts/repeated_runs.py --arm federated --mode dirichlet --alpha $a --rounds 20 --local-epochs 3 \
        --seeds 101 --suffix _r7draw${d}_a${at}_s101
  done
done
say "batch end"
