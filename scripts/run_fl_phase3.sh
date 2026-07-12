#!/usr/bin/env bash
# Phase 3 experiment matrix. Run from repo root. Logs to results/metrics/fl_run.log.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CFG="--config configs/fl.yaml --model configs/model.yaml --paths configs/paths.yaml"
LOG=results/metrics/fl_run.log
: > "$LOG"

run() { echo "=== $* ===" | tee -a "$LOG"; $PY -m src.fl.simulate $CFG "$@" 2>&1 | tee -a "$LOG"; }

run --mode iid       --experiment fedavg
run --mode dirichlet --experiment fedavg
run --mode iid       --experiment isolated
run --mode iid       --experiment poison --poison-clients 1          # no clipping
run --mode iid       --experiment poison --poison-clients 1 --clip   # SafeFedAvg clipping

echo "ALL PHASE 3 EXPERIMENTS DONE" | tee -a "$LOG"
