#!/usr/bin/env bash
# Step 2+3 of the Chapter 4 backlog: a matched-budget federated rerun, then the ablation arms.
# Sequential on purpose — each stage already saturates the cores.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=10 TF_NUM_INTRAOP_THREADS=10 TF_NUM_INTEROP_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
L=results/metrics
step () { echo ""; echo "=== [$(date +%H:%M:%S)] $* ==="; }

step "1/2 federated repeated runs, 5 seeds x 20 rounds (matched budget)"
$PY scripts/repeated_runs.py --arm federated --seeds 101 102 103 104 105 --rounds 20 --suffix _r20 \
    > $L/repeated_federated_r20.log 2>&1 && echo "  ok" || echo "  FAILED"

step "2/2 ablation arms at 20 rounds"
for cfg in "isolated::" "poison:--poison-clients 1:noclip" "poison:--poison-clients 1 --clip:clip"; do
  exp="${cfg%%:*}"; rest="${cfg#*:}"; flags="${rest%%:*}"; tagsfx="${rest#*:}"
  tag="ablation${tagsfx:+_$tagsfx}"
  echo "  -> $exp $flags (tag=$tag)"
  $PY -m src.fl.simulate --config configs/fl.yaml --model configs/model.yaml \
      --paths configs/paths.yaml --mode iid --experiment "$exp" $flags \
      --rounds 20 --seed 101 --tag "$tag" >> $L/ablation.log 2>&1 \
      && echo "     ok" || echo "     FAILED"
done
echo ""; echo "=== [$(date +%H:%M:%S)] step 2+3 finished ==="
