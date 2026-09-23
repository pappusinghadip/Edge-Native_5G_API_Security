#!/usr/bin/env bash
# Parallel driver for the run_all_arms.sh workload.
#
# Every run here is overhead-bound, not resource-bound: the model is (10,1) at batch
# 256, so a step is ~20 ms of framework overhead over microseconds of arithmetic, and
# one run leaves most of the CPU idle. Measured on the RTX 4060 laptop: 148 s/round
# alone, ~80 s/round effective with three concurrent (1.84x throughput).
#
#   JOBS=3 bash scripts/run_all_arms_parallel.sh 2>&1 | tee results/metrics/run_all_arms.log
#
# Phase A runs the training arms JOBS at a time.
# Phase B runs the scalability sweep strictly ALONE, because it records
# mean_round_seconds and CPU contention would inflate that measurement.
#
# Idempotent: any arm whose output already exists is skipped, so this can be
# interrupted and restarted freely.
set -u
cd "$(dirname "$0")/.."

PY=.venv/bin/python
L=results/metrics
JOBS=${JOBS:-3}
SEEDS="101 102 103 104 105"
ROUNDS=20
CORES=$(nproc)
THREADS=$(( CORES / JOBS )); [ "$THREADS" -lt 1 ] && THREADS=1

export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2
# CPU is measurably faster than the GPU for a model this small (148 s vs 172 s per
# round). Every arm must use the SAME device: a CPU/GPU mix across arms risks a
# device-dependent shift in a non-inferiority margin as tight as 0.0025.
export CUDA_VISIBLE_DEVICES=""

mkdir -p "$L" "$L/joblogs" results/models
JOBFILE=$L/joblist.txt
: > "$JOBFILE"

log () { echo "[$(date +%H:%M:%S)] $*"; }
have () { [ -f "$1" ]; }

# repeated_runs.py writes its JSON after EVERY seed, so mere existence does not mean
# the arm finished. Require the full seed count.
seeds_done () {   # <json> <expected>
  [ -f "$1" ] || return 1
  local n
  n=$($PY -c 'import json,sys
try: print(len(json.load(open(sys.argv[1])).get("runs",[])))
except Exception: print(0)' "$1" 2>/dev/null)
  [ "${n:-0}" -ge "$2" ]
}

scal_done () {    # <clients> <rounds> <seed>
  [ -f "$L/scalability.json" ] || return 1
  $PY -c 'import json,sys
p,k,r,s=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
try: d=json.load(open(p))
except Exception: sys.exit(1)
sys.exit(0 if any(x.get("clients")==k and x.get("rounds")==r and x.get("seed",42)==s for x in d) else 1)' \
    "$L/scalability.json" "$1" "$2" "$3"
}

emit () {         # emit <name> <command...>
  local name="$1"; shift
  echo "$* && echo OK $name >> $L/progress.log || echo FAILED $name >> $L/progress.log" >> "$JOBFILE"
}

emit_sim () {     # emit_sim <outfile> <tag> <extra args...>
  local out="$1" tag="$2"; shift 2
  if have "$L/$out"; then echo "  skip (exists): $out"; return; fi
  emit "$tag" "$PY -m src.fl.simulate --config configs/fl.yaml --model configs/model.yaml" \
       "--paths configs/paths.yaml --rounds $ROUNDS --tag $tag $*" \
       ">> $L/joblogs/$tag.log 2>&1"
}

# --------------------------------------------------------------------------- #
log "0/9  partitions for every heterogeneity level (sequential, must precede all)"
for a in 0.1 0.5 1.0; do
  t="dir_${a/./p}"
  if have "data/partitions/client_0_${t}.npz"; then
    echo "  skip (exists): client_0_${t}.npz"
  else
    $PY -m src.data.partitioner --config configs/fl.yaml --paths configs/paths.yaml \
        --alpha "$a" >> "$L/joblogs/partitioner.log" 2>&1 && echo "  built $t" || echo "  FAILED $t"
  fi
done

log "building job list"

# 1  centralized, 5 seeds, 20 epochs.
#    --epochs 20 --suffix _e20 are REQUIRED: without them repeated_runs.py writes
#    repeated_centralized.json under a 100-epoch early-stopping budget, while
#    ch4_final_analysis.py reads repeated_centralized_e20.json and the federated arm
#    is a matched 20-round budget. run_all_arms.sh omits both flags.
if seeds_done "$L/repeated_centralized_e20.json" 5; then echo "  skip (complete): repeated_centralized_e20.json"; else
  emit "centralized_e20" "$PY scripts/repeated_runs.py --arm centralized --seeds $SEEDS" \
       "--epochs 20 --suffix _e20 > $L/repeated_centralized.log 2>&1"
fi

# 2  federated IID, 5 seeds
if seeds_done "$L/repeated_federated_iid_r20.json" 5; then echo "  skip (complete): repeated_federated_iid_r20.json"; else
  emit "federated_iid_r20" "$PY scripts/repeated_runs.py --arm federated --seeds $SEEDS" \
       "--rounds $ROUNDS --suffix _r20 > $L/repeated_federated_r20.log 2>&1"
fi

# 3  federated non-IID at three concentrations, 5 seeds each
for a in 0.1 0.5 1.0; do
  for s in $SEEDS; do
    t="noniid_a${a/./p}_s${s}"
    emit_sim "fl_fedavg_dirichlet_${t}.json" "$t" --mode dirichlet --experiment fedavg --alpha "$a" --seed "$s"
  done
done

# 4  isolated local training, 5 seeds
for s in $SEEDS; do
  t="isolated_s${s}"
  emit_sim "fl_isolated_iid_${t}.json" "$t" --mode iid --experiment isolated --seed "$s"
done

# 5  poisoned without and with the norm filter, 5 seeds each
for s in $SEEDS; do
  t="poison_noclip_s${s}"
  emit_sim "fl_poison1_noclip_iid_${t}.json" "$t" --mode iid --experiment poison --poison-clients 1 --seed "$s"
  t="poison_clip_s${s}"
  emit_sim "fl_poison1_clip_iid_${t}.json" "$t" --mode iid --experiment poison --poison-clients 1 --clip --seed "$s"
done

# 6  clean federation with the filter on, 5 seeds (false-rejection cost)
for s in $SEEDS; do
  t="clean_clip_s${s}"
  emit_sim "fl_fedavg_iid_${t}.json" "$t" --mode iid --experiment fedavg --clip --seed "$s"
done

# 7  norm-filter multiplier sweep on the poisoned arm, 3 seeds
for kap in 1.5 2.0 2.5 3.0; do
  for s in 101 102 103; do
    t="kappa${kap/./p}_s${s}"
    emit_sim "fl_poison1_clip_iid_${t}.json" "$t" --mode iid --experiment poison \
             --poison-clients 1 --clip --clip-factor "$kap" --seed "$s"
  done
done

N=$(wc -l < "$JOBFILE" | tr -d ' ')
log "phase A: $N jobs, $JOBS concurrent, $THREADS threads each (of $CORES cores)"
if [ "${DRYRUN:-0}" = 1 ]; then echo "--- job list ($N) ---"; cat "$JOBFILE"; log "DRY RUN: stopping before execution"; exit 0; fi
if [ "$N" -gt 0 ]; then
  export TF_NUM_INTRAOP_THREADS=$THREADS
  export OMP_NUM_THREADS=$THREADS
  TF_NUM_INTEROP_THREADS=$(( THREADS / 2 )); [ "$TF_NUM_INTEROP_THREADS" -lt 1 ] && TF_NUM_INTEROP_THREADS=1
  export TF_NUM_INTEROP_THREADS
  tr '\n' '\0' < "$JOBFILE" | xargs -0 -P "$JOBS" -n 1 bash -c
fi
log "phase A finished"

# 8  scalability — ALONE. It records mean_round_seconds; contention would inflate it.
log "8/9  scalability at 5, 20 and 100 clients, 3 seeds (sequential, uncontended)"
unset TF_NUM_INTRAOP_THREADS TF_NUM_INTEROP_THREADS OMP_NUM_THREADS
for k in 5 20 100; do
  for s in 101 102 103; do
    if scal_done "$k" 5 "$s"; then echo "  skip (exists): K=$k seed=$s"; continue; fi
    log "  K=$k seed=$s"
    if $PY -m scripts.scalability --clients "$k" --rounds 5 --seed "$s" >> "$L/joblogs/scalability.log" 2>&1; then
      echo "OK scalability_k${k}_s${s}" >> "$L/progress.log"; echo "     ok"
    else
      echo "FAILED scalability_k${k}_s${s}" >> "$L/progress.log"; echo "     FAILED"
    fi
  done
done

log "9/9  all arms finished"
echo "  OK:     $(grep -c '^OK' "$L/progress.log" 2>/dev/null || true)"
echo "  FAILED: $(grep -c '^FAILED' "$L/progress.log" 2>/dev/null || true)"
grep '^FAILED' "$L/progress.log" 2>/dev/null || echo "  (no failures)"
echo "Copy results/metrics back to the Mac, then rebuild the documents."
