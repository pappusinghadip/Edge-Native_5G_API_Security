#!/usr/bin/env bash
# Round-5 tier 2: rerun the secondary experiments with the fixed focal loss, so every table in
# Chapter 4 uses the same loss as the exposure-matched principal arms. Poisoning runs now log
# per-client accept/reject decisions. 45 jobs, all 20 rounds x 3 local epochs:
#
#   non-IID FedAvg, alpha 0.1 / 0.5 / 1.0, 5 seeds                 15 jobs
#   poisoned without filter, with filter (kappa 2.5), clean+filter  15 jobs
#   kappa sweep 1.5 / 2.0 / 3.0 on the poisoned arm, 5 seeds       15 jobs (2.5 reused from above)
#
# About 12-13 h on the Asus at JOBS=3:
#   tmux new-session -d -s round5b "cd ~/5G_ML && JOBS=3 bash scripts/run_round5_tier2.sh 2>&1 | tee results/metrics/run_round5_tier2.log"
# Idempotent: finished jobs are skipped.
set -u
cd "$(dirname "$0")/.."

PY=.venv/bin/python
L=results/metrics
JOBS=${JOBS:-3}
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
THREADS=$(( CORES / JOBS )); [ "$THREADS" -lt 1 ] && THREADS=1
export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=""
export TF_NUM_INTRAOP_THREADS=$THREADS OMP_NUM_THREADS=$THREADS TF_NUM_INTEROP_THREADS=1
mkdir -p "$L/joblogs_r5b"

grep -q "focal_class_weighted: true" configs/model.yaml || { echo "fixed focal loss missing in configs/model.yaml"; exit 1; }
grep -q '"adversarial"' src/fl/simulate.py || { echo "per-client decision logging missing in src/fl/simulate.py"; exit 1; }
for a in 0p1 0p5 1p0; do [ -f data/partitions/client_0_dir_${a}.npz ] || { echo "missing partition dir_${a}"; exit 1; }; done

JOBFILE=$L/joblist_r5b.txt; : > "$JOBFILE"
add () { local out="$1"; shift; if [ -f "$L/$out" ]; then echo "skip (done): $out"; else echo "$out|$*" >> "$JOBFILE"; fi; }
SIM="$PY -m src.fl.simulate --config configs/fl.yaml --model configs/model.yaml --paths configs/paths.yaml --rounds 20 --local-epochs 3"

for s in 101 102 103 104 105; do
  for a in 0.1 0.5 1.0; do
    t="r5_noniid_a${a/./p}_s$s"
    add "fl_fedavg_dirichlet_$t.json" $SIM --mode dirichlet --experiment fedavg --alpha $a --seed $s --tag $t
  done
  add "fl_poison1_noclip_iid_r5_poison_noclip_s$s.json" $SIM --mode iid --experiment poison --poison-clients 1 --seed $s --tag r5_poison_noclip_s$s
  add "fl_poison1_clip_iid_r5_poison_clip_s$s.json"     $SIM --mode iid --experiment poison --poison-clients 1 --clip --clip-factor 2.5 --seed $s --tag r5_poison_clip_s$s
  add "fl_fedavg_iid_r5_clean_clip_s$s.json"            $SIM --mode iid --experiment fedavg --clip --clip-factor 2.5 --seed $s --tag r5_clean_clip_s$s
done
for s in 101 102 103 104 105; do
  for k in 1.5 2.0 3.0; do
    t="r5_kappa${k/./p}_s$s"
    add "fl_poison1_clip_iid_$t.json" $SIM --mode iid --experiment poison --poison-clients 1 --clip --clip-factor $k --seed $s --tag $t
  done
done

N=$(wc -l < "$JOBFILE" | tr -d ' ')
echo "[$(date +%H:%M:%S)] $N jobs, $JOBS in parallel, $THREADS threads each"
[ "$N" -eq 0 ] && { echo "nothing to do"; exit 0; }

run_one () {
  local out="${1%%|*}" cmd="${1#*|}"
  local log="results/metrics/joblogs_r5b/${out%.json}.log"
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
