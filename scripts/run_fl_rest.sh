#!/usr/bin/env bash
# Remaining Phase 3 experiments (IID FedAvg already done). Appends to fl_run.log.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CFG="--config configs/fl.yaml --model configs/model.yaml --paths configs/paths.yaml"
LOG=results/metrics/fl_run.log

run() { echo "=== $* ===" | tee -a "$LOG"; $PY -m src.fl.simulate $CFG "$@" 2>&1 | tee -a "$LOG"; }

run --mode dirichlet --experiment fedavg
run --mode iid       --experiment isolated
run --mode iid       --experiment poison --poison-clients 1
run --mode iid       --experiment poison --poison-clients 1 --clip

echo "REMAINING PHASE 3 EXPERIMENTS DONE" | tee -a "$LOG"
