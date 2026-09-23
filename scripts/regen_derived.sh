#!/usr/bin/env bash
# Regenerate every derived analysis from PC-trained artifacts, so Chapter 4 has one
# provenance instead of mixing PC-trained CNN arms with Mac-trained references.
#
# Dependency order matters: centralized_best.h5 is the linchpin — the ensemble,
# the mitigation agent and everything downstream load it.
#
# Idempotent: a step whose output exists is skipped. Launch detached:
#   setsid nohup bash scripts/regen_derived.sh > results/metrics/regen_derived.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=results/metrics
G=results/models
export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=""
mkdir -p "$L/joblogs" "$G" results/figures

log  () { echo "[$(date +%H:%M:%S)] $*"; }
have () { [ -f "$1" ] && { echo "   skip (exists): $(basename "$1")"; return 0; } || return 1; }
run  () {  # run <name> <logfile> <command...>
  local name="$1" lf="$2"; shift 2
  log "   running $name"
  if "$@" >> "$lf" 2>&1; then
    echo "OK $name" >> "$L/progress_regen.log"; log "   ok $name"
  else
    echo "FAILED $name" >> "$L/progress_regen.log"; log "   FAILED $name (see $lf)"
  fi
}

log "1/9  centralized 1D-CNN — produces centralized_best.h5 + centralized_results.json"
have "$G/centralized_best.h5" || \
  run centralized "$L/joblogs/regen_centralized.log" \
      $PY -m src.models.train --config configs/model.yaml --paths configs/paths.yaml

log "2/9  TFLite export + inference latency"
have "$L/latency_centralized.json" || \
  run export_tflite "$L/joblogs/regen_export.log" \
      $PY -m src.models.export --config configs/model.yaml --paths configs/paths.yaml

log "3/9  plain IID FedAvg at seed 101, tagged ablation"
have "$L/fl_fedavg_iid_ablation.json" || \
  run fedavg_ablation "$L/joblogs/regen_fedavg_ablation.log" \
      $PY -m src.fl.simulate --config configs/fl.yaml --model configs/model.yaml \
          --paths configs/paths.yaml --mode iid --experiment fedavg --rounds 20 \
          --seed 101 --tag ablation

log "4/9  ablation aliases from the matching seed-101 arms (identical configs, no retrain)"
for pair in \
  "fl_isolated_iid_isolated_s101.json:fl_isolated_iid_ablation.json" \
  "fl_poison1_noclip_iid_poison_noclip_s101.json:fl_poison1_noclip_iid_ablation.json" \
  "fl_poison1_clip_iid_poison_clip_s101.json:fl_poison1_clip_iid_ablation.json" ; do
  src="$L/${pair%%:*}"; dst="$L/${pair##*:}"
  if [ -f "$dst" ]; then echo "   skip (exists): $(basename "$dst")"
  elif [ -f "$src" ]; then cp "$src" "$dst"; echo "   aliased $(basename "$src") -> $(basename "$dst")"
  else echo "   MISSING SOURCE: $(basename "$src")"; fi
done

log "5/9  ensemble + detector_probs.npz (needs the centralized CNN)"
have "$L/detector_probs.npz" || \
  run best_ensemble "$L/joblogs/regen_ensemble.log" $PY scripts/best_ensemble.py

log "6/9  extra baselines"
have "$L/extra_baselines.json" || \
  run extra_baselines "$L/joblogs/regen_baselines.log" $PY scripts/extra_baselines.py

log "7/9  mitigation agent (k-of-n policy)"
have "$L/mitigation_results.json" || \
  run mitigation "$L/joblogs/regen_mitigation.log" \
      $PY -m src.mitigation.agent --model configs/model.yaml --paths configs/paths.yaml

log "8/9  sensitivity sweep + ablation summary + measured communication"
have "$L/sensitivity_analysis.json" || \
  run sensitivity "$L/joblogs/regen_sensitivity.log" $PY scripts/sensitivity_analysis.py
have "$L/ablation_analysis.json" || \
  run ablation "$L/joblogs/regen_ablation.log" $PY scripts/ablation_analysis.py
have "$L/measured_communication.json" || \
  run measured_comm "$L/joblogs/regen_comm.log" $PY scripts/measured_communication.py

log "9/9  evaluation curves + extended metrics"
have "$L/extended_metrics.json" || \
  run evaluation_curves "$L/joblogs/regen_curves.log" $PY scripts/evaluation_curves.py

log "finished"
echo "  OK:     $(grep -c '^OK' "$L/progress_regen.log" 2>/dev/null || true)"
echo "  FAILED: $(grep -c '^FAILED' "$L/progress_regen.log" 2>/dev/null || true)"
grep '^FAILED' "$L/progress_regen.log" 2>/dev/null || echo "  (no failures)"
