"""Record exactly what this training run produced, so downstream analysis never
needs to retrain. Writes RUN_MANIFEST.md next to the metrics."""
import hashlib, json, os, platform, subprocess, sys, time

M, MODELS, CFG = "results/metrics", "results/models", "configs"
SEEDS = [101, 102, 103, 104, 105]


def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()[:16]


def size(p):
    return f"{os.path.getsize(p)/1e6:.1f} MB"


out = ["# Run manifest — PC training run",
       "",
       "Everything below was produced by `scripts/run_all_arms_parallel.sh` in a single",
       "uninterrupted run. It exists so that later analysis (curves, intervals, thresholds,",
       "confusion matrices) can be done **without retraining**.", ""]

out += ["## Provenance", ""]
try:
    import tensorflow as tf
    tfv = tf.__version__
except Exception:
    tfv = "unavailable"
out += [f"- generated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- host: {platform.node()} — {platform.platform()}",
        f"- python: {platform.python_version()} · tensorflow: {tfv}",
        "- device: CPU (`CUDA_VISIBLE_DEVICES=\"\"`) — every arm, deliberately uniform",
        "- full package set: `ENVIRONMENT_pip_freeze.txt`",
        "- NOTE: `tensorflow[and-cuda]==2.15.0` no longer installs (`tensorrt-libs==8.6.1`",
        "  was withdrawn from PyPI). The env is TF 2.15.0 + the CUDA 12.2 wheels minus",
        "  TensorRT, so `requirements.txt` alone will not rebuild it.", ""]

out += ["## Inputs", ""]
for d in ("data/processed", "data/partitions"):
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if f.endswith((".npz", ".pkl", ".json")):
                out.append(f"- `{p}` — {size(p)} — sha256:{sha(p)}")
out.append("")
out += ["## Configs", ""]
for f in sorted(os.listdir(CFG)):
    if f.endswith(".yaml"):
        p = os.path.join(CFG, f)
        out.append(f"- `{p}` — sha256:{sha(p)}")
out.append("")

out += ["## Arms produced", "",
        "| arm | metrics JSON | test probs | model |", "|---|---|---|---|"]
gaps = []
for f in sorted(os.listdir(M)):
    if not f.endswith(".json") or f.startswith(("final_", "statistical_")):
        continue
    b = f[:-5]
    if b == "scalability":            # a sweep record, no per-run probs by design
        continue
    # repeated_runs.py stores probs per seed: <arm>_seed<N>_probs.npz
    per_seed = [s_ for s_ in SEEDS
                if os.path.exists(os.path.join(M, f"{b}_seed{s_}_probs.npz"))]
    # isolated arms save per-client scores/weights instead of a single global set
    client_probs = os.path.exists(os.path.join(M, b + "_client_probs.npz"))
    client_models = [f for f in os.listdir(MODELS)
                     if f.startswith(b + "_client") and f.endswith(".h5")]
    probs = (os.path.exists(os.path.join(M, b + "_probs.npz"))
             or len(per_seed) == len(SEEDS) or client_probs)
    model = os.path.exists(os.path.join(MODELS, b + "_global.h5")) or bool(client_models)
    if not probs:
        gaps.append(b)
    note = "yes" if probs else "**NO**"
    if per_seed and not os.path.exists(os.path.join(M, b + "_probs.npz")):
        note = f"yes ({len(per_seed)} per-seed)"
    if client_probs:
        note = "yes (per-client)"
    mnote = "yes"
    if client_models:
        mnote = f"yes ({len(client_models)} per-client)"
    elif not model:
        mnote = "—"
    out.append(f"| `{b}` | yes | {note} | {mnote} |")
out.append("")

out += ["## Recomputable without retraining", "",
        "Saved test-set predicted probabilities (`*_probs.npz`, aligned with",
        "`data/processed/test.npz` labels) support all of:", "",
        "- ROC and precision-recall curves, at any operating point",
        "- confusion matrices at any threshold; threshold re-tuning",
        "- bootstrap and DeLong confidence intervals on test-sample uncertainty",
        "- any re-derived metric (F1, recall, FPR, precision) at any threshold",
        "- paired per-seed comparisons and alternative non-inferiority margins", "",
        "Saved model weights (`*_global.h5`) additionally support:", "",
        "- inference-latency measurement, TFLite export, edge benchmarking",
        "- evaluation against a *new* test split without retraining", ""]

if gaps:
    out += ["## Gap — would require retraining", ""]
    for g in gaps:
        out.append(f"- `{g}` — no probs, no model")
    out += ["",
            "`run_isolated()` in `src/fl/simulate.py` trains one model per client and never",
            "sets `result._model`, so neither probabilities nor weights are saved. The",
            "per-client aggregate metrics in the JSON survive; curve- or interval-level",
            "analysis of the isolated baseline does not.", ""]

open(os.path.join(M, "RUN_MANIFEST.md"), "w", encoding="utf-8").write("\n".join(out))
print(f"wrote {M}/RUN_MANIFEST.md ({len(out)} lines); gaps: {len(gaps)}")
