#!/usr/bin/env bash
# Every training arm the fourth review asks to be repeated, at a matched budget over five seeds.
#
# Idempotent: an arm whose output file already exists is skipped, so the script can be
# interrupted and restarted. Run it under tmux — it takes hours.
#
#   tmux new -s thesis
#   bash scripts/run_all_arms.sh 2>&1 | tee results/metrics/run_all_arms.log
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2
SEEDS="101 102 103 104 105"
ROUNDS=20
L=results/metrics
mkdir -p "$L"

step () { echo ""; echo "=== [$(date +%H:%M:%S)] $* ==="; }
have () { [ -f "$1" ] && { echo "  skip (exists): $(basename "$1")"; return 0; } || return 1; }

sim () {   # sim <tag> <extra args...>
  local tag="$1"; shift
  $PY -m src.fl.simulate --config configs/fl.yaml --model configs/model.yaml \
      --paths configs/paths.yaml --rounds "$ROUNDS" --tag "$tag" "$@" \
      >> "$L/run_all_arms_detail.log" 2>&1 && echo "     ok" || echo "     FAILED"
}

step "0/8  partitions for every heterogeneity level"
for a in 0.1 0.5 1.0; do
  tag="dir_${a/./p}"
  have "data/partitions/client_0_${tag}.npz" || \
    $PY -m src.data.partitioner --config configs/fl.yaml --paths configs/paths.yaml --alpha "$a" \
      >> "$L/run_all_arms_detail.log" 2>&1
done

step "1/8  centralized, 5 seeds, 20 epochs"
have "$L/repeated_centralized_e20.json" || \
  $PY scripts/repeated_runs.py --arm centralized --seeds $SEEDS > "$L/repeated_centralized.log" 2>&1

step "2/8  federated IID, 5 seeds, $ROUNDS rounds"
have "$L/repeated_federated_iid_r20.json" || \
  $PY scripts/repeated_runs.py --arm federated --seeds $SEEDS --rounds $ROUNDS --suffix _r20 \
    > "$L/repeated_federated_r20.log" 2>&1

step "3/8  federated non-IID at three concentrations, 5 seeds each"
for a in 0.1 0.5 1.0; do
  for s in $SEEDS; do
    t="noniid_a${a/./p}_s${s}"
    have "$L/fl_fedavg_dirichlet_${t}.json" || { echo "  -> alpha=$a seed=$s"; \
      sim "$t" --mode dirichlet --experiment fedavg --alpha "$a" --seed "$s"; }
  done
done

step "4/8  isolated local training, 5 seeds"
for s in $SEEDS; do
  t="isolated_s${s}"
  have "$L/fl_isolated_iid_${t}.json" || { echo "  -> seed=$s"; \
    sim "$t" --mode iid --experiment isolated --seed "$s"; }
done

step "5/8  poisoned without and with the norm filter, 5 seeds each"
for s in $SEEDS; do
  t="poison_noclip_s${s}"
  have "$L/fl_poison1_noclip_iid_${t}.json" || { echo "  -> noclip seed=$s"; \
    sim "$t" --mode iid --experiment poison --poison-clients 1 --seed "$s"; }
  t="poison_clip_s${s}"
  have "$L/fl_poison1_clip_iid_${t}.json" || { echo "  -> clip seed=$s"; \
    sim "$t" --mode iid --experiment poison --poison-clients 1 --clip --seed "$s"; }
done

step "6/8  clean federation with the filter enabled, 5 seeds (false-rejection cost)"
for s in $SEEDS; do
  t="clean_clip_s${s}"
  have "$L/fl_fedavg_iid_${t}.json" || { echo "  -> seed=$s"; \
    sim "$t" --mode iid --experiment fedavg --clip --seed "$s"; }
done

step "7/8  norm-filter multiplier sweep on the poisoned arm, 3 seeds"
for kap in 1.5 2.0 2.5 3.0; do
  for s in 101 102 103; do
    t="kappa${kap/./p}_s${s}"
    have "$L/fl_poison1_clip_iid_${t}.json" || { echo "  -> kappa=$kap seed=$s"; \
      sim "$t" --mode iid --experiment poison --poison-clients 1 --clip \
          --clip-factor "$kap" --seed "$s"; }
  done
done

step "8/8  scalability at 5, 20 and 100 clients, 3 seeds"
for k in 5 20 100; do
  for s in 101 102 103; do
    echo "  -> K=$k seed=$s"
    $PY -m scripts.scalability --clients "$k" --rounds 5 --seed "$s" \
      >> "$L/run_all_arms_detail.log" 2>&1 && echo "     ok" || echo "     FAILED"
  done
done

echo ""; echo "=== [$(date +%H:%M:%S)] all arms finished ==="
echo "Copy results/metrics and results/figures back to the Mac, then rebuild the documents."
