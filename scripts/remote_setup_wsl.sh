#!/usr/bin/env bash
# Environment setup for training on the Windows laptop, inside WSL 2.
#
# Run this INSIDE WSL 2 Ubuntu 22.04 — not in PowerShell, and not in Ubuntu 26.04, whose Python is
# too new for TensorFlow 2.15 (which supports Python 3.9-3.11).
#
#   bash scripts/remote_setup_wsl.sh
#
# Training runs on the CPU. For this model the laptop's RTX 4060 was measured SLOWER than its CPU
# (172 vs 148 s per aggregation round), because a network this small is dominated by per-step
# framework overhead. scripts/run_all_arms_parallel.sh therefore sets CUDA_VISIBLE_DEVICES="".
# The CUDA wheels are still installed so the GPU can be tried, but nothing depends on them.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/6  GPU visible to the WSL kernel? (informational — training does not need it) =="
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "  nvidia-smi not available inside WSL — fine, training runs on the CPU."
fi

echo "== 2/6  system packages =="
sudo apt-get update -qq
sudo apt-get install -y -qq software-properties-common tmux rsync

echo "== 3/6  Python 3.10 =="
if ! command -v python3.10 >/dev/null 2>&1; then
  echo "  python3.10 not present, adding the deadsnakes PPA"
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -qq
fi
sudo apt-get install -y -qq python3.10 python3.10-venv python3.10-dev
python3.10 --version

echo "== 4/6  virtual environment and dependencies =="
[ -d .venv ] || python3.10 -m venv .venv
. .venv/bin/activate
pip install --quiet --upgrade pip setuptools wheel
# `tensorflow[and-cuda]==2.15.0` no longer resolves: tensorrt-libs==8.6.1 was withdrawn from PyPI.
# Install TensorFlow on its own, then the CUDA 12.2 wheels the 2026-09-07 PC run used — recorded in
# results/metrics_PC_run_2026-09-07/ENVIRONMENT_pip_freeze.txt — without TensorRT.
pip install --quiet "tensorflow==2.15.0"
pip install --quiet \
  nvidia-cublas-cu12==12.2.5.6 nvidia-cuda-cupti-cu12==12.2.142 nvidia-cuda-nvcc-cu12==12.2.140 \
  nvidia-cuda-nvrtc-cu12==12.2.140 nvidia-cuda-runtime-cu12==12.2.140 nvidia-cudnn-cu12==8.9.4.25 \
  nvidia-cufft-cu12==11.0.8.103 nvidia-curand-cu12==10.3.3.141 nvidia-cusolver-cu12==11.5.2.141 \
  nvidia-cusparse-cu12==12.1.2.141 nvidia-nccl-cu12==2.16.5 nvidia-nvjitlink-cu12==12.2.140
# tflite-runtime is only needed for edge benchmarking, not training
grep -vE '^(tensorflow|tflite-runtime)' requirements.txt > /tmp/req_train.txt
pip install --quiet -r /tmp/req_train.txt

echo "== 5/6  TensorFlow check =="
python - <<'PY'
import tensorflow as tf
print("TensorFlow", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs visible:", gpus or "none")
print("Training will use the CPU (run_all_arms_parallel.sh sets CUDA_VISIBLE_DEVICES='').")
PY

echo "== 6/6  data present? =="
missing=0
for d in data/processed data/partitions; do
  if [ -d "$d" ]; then echo "  ok  $d  $(du -sh "$d" | cut -f1)"
  else echo "  MISSING  $d"; missing=1; fi
done
[ "$missing" -eq 1 ] && echo "  Copy data/processed and data/partitions from the Mac (about 60 MB)."
echo ""
echo "setup complete — next:  tmux new -s thesis   then   JOBS=3 bash scripts/run_all_arms_parallel.sh 2>&1 | tee results/metrics/run_all_arms.log"
