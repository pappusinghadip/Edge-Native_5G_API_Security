#!/usr/bin/env bash
# Chapter 4 experiment pipeline (statistical validation, ablation, sensitivity).
#
# Runs strictly sequentially: each stage is TensorFlow-bound and already uses most
# cores, so running them in parallel only causes thrashing. Thread counts are pinned
# below the core count to leave the machine responsive.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=.
export OMP_NUM_THREADS=10 TF_NUM_INTRAOP_THREADS=10 TF_NUM_INTEROP_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=2
L=results/metrics

step () { echo ""; echo "=== [$(date +%H:%M:%S)] $* ==="; }

step "1/6 base detectors + stored probabilities"
$PY -m scripts.best_ensemble > $L/best_ensemble.log 2>&1 && echo "  ok" || echo "  FAILED"

step "2/6 repeated centralized runs (5 seeds)"
$PY scripts/repeated_runs.py --arm centralized --seeds 101 102 103 104 105 \
    > $L/repeated_centralized.log 2>&1 && echo "  ok" || echo "  FAILED"

step "3/6 repeated federated runs (5 seeds x 20 rounds)"
$PY scripts/repeated_runs.py --arm federated --seeds 101 102 103 104 105 --rounds 20 \
    > $L/repeated_federated.log 2>&1 && echo "  ok" || echo "  FAILED"

step "4/6 ablation arms at matched round budget (20 rounds)"
for cfg in "isolated::" "poison:--poison-clients 1:noclip" "poison:--poison-clients 1 --clip:clip"; do
  exp="${cfg%%:*}"; rest="${cfg#*:}"; flags="${rest%%:*}"; tagsfx="${rest#*:}"
  tag="ablation${tagsfx:+_$tagsfx}"
  echo "  -> $exp $flags (tag=$tag)"
  $PY -m src.fl.simulate --config configs/fl.yaml --model configs/model.yaml \
      --paths configs/paths.yaml --mode iid --experiment "$exp" $flags \
      --rounds 20 --seed 101 --tag "$tag" >> $L/ablation.log 2>&1 \
      && echo "     ok" || echo "     FAILED"
done

step "5/6 statistical analysis"
$PY scripts/statistical_analysis.py > $L/statistical_analysis.log 2>&1 && echo "  ok" || echo "  FAILED"

step "6/6 curves, extended metrics, k-of-n sensitivity"
$PY scripts/evaluation_curves.py > $L/evaluation_curves.log 2>&1 && echo "  ok" || echo "  FAILED"
$PY scripts/sensitivity_analysis.py --source federated > $L/sensitivity.log 2>&1 && echo "  ok" || echo "  FAILED"

echo ""; echo "=== [$(date +%H:%M:%S)] pipeline finished ==="
