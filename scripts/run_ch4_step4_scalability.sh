#!/usr/bin/env bash
# Step 3: extend the scalability sweep to 20, 50 and 100 simulated edge nodes.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=10 TF_NUM_INTRAOP_THREADS=10 TF_NUM_INTEROP_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
for k in 20 50 100; do
  echo "=== [$(date +%H:%M:%S)] clients=$k ==="
  .venv/bin/python -m scripts.scalability --clients "$k" --rounds 5 \
      >> results/metrics/scalability_extended.log 2>&1 && echo "  ok" || echo "  FAILED"
done
echo "=== [$(date +%H:%M:%S)] scalability sweep finished ==="
