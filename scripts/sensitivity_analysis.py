"""Sensitivity of the mitigation policy to the k-of-n window.

The values k = 3, n = 10 are used throughout the thesis but were never justified
empirically. This sweeps k and n over the same scenario generator used for the
headline mitigation result and reports the trade-off a reviewer would ask about:
detection rate and time-to-detect improve as k falls, false blocks rise.

Run: python scripts/sensitivity_analysis.py [--source federated|centralized]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.mitigation.agent import MitigationPolicy, run_real_scenarios
from src.models.io import load_processed_split
from src.utils.config import load_yaml, resolve_path

M = Path("results/metrics")


def load_scores(source: str) -> tuple[np.ndarray, float, dict]:
    """Per-test-flow malicious scores for the requested detector, with its own threshold.

    "federated" is the deployed model: the median-ROC-AUC Mac run among seeds 101-105 of the
    corrected-loss 60N arm, scored at that run's own validation-tuned threshold; "centralized" is the
    same choice from the centralized arm, kept as a reference. Averaging the seeds would build a score
    ensemble that is not the artefact any node would deploy.
    """
    fa = json.loads((M / "final_analysis.json").read_text())
    arm = fa["arms_mac"][source]
    seed = arm["median_auc_seed"]
    f = M / arm["probs_files"][str(seed)]
    if not f.exists():
        raise SystemExit(f"missing {f}")
    thr = float(arm["per_seed"][str(seed)]["threshold"])
    return np.load(f)["probs"], thr, {"seed": seed, "run": f.name, "roc_auc": arm["per_seed"][str(seed)]["roc_auc"],
                                      "focal_class_weighted": arm.get("focal_class_weighted")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="federated", choices=["federated", "centralized"])
    args = ap.parse_args()

    paths_cfg = load_yaml("configs/paths.yaml")
    X_te, y_te, _ = load_processed_split(resolve_path(paths_cfg["data"]["processed"]) / "test.npz")
    scores, threshold, prov = load_scores(args.source)
    if len(scores) != len(y_te):
        raise SystemExit(f"score/label length mismatch: {len(scores)} vs {len(y_te)}")

    benign, attack = scores[y_te == 0], scores[y_te == 1]
    lat = json.loads((M / "latency_centralized.json").read_text())
    per_flow_ms = float(lat.get("mean_latency_ms") or lat.get("mean_ms") or 0.0130)

    rows = []
    for n in (5, 10, 20):
        for k in (2, 3, 4, 5):
            if k > n:
                continue
            pol = MitigationPolicy(threshold=threshold, window=n, min_hits=k,
                                   action="block", cooldown_flows=100)
            r = run_real_scenarios(benign, attack, pol, per_flow_ms, n_scenarios=200)
            rows.append({"k": k, "n": n, **{key: r[key] for key in (
                "detection_rate", "conditional_detection_rate", "false_block_rate",
                "false_blocks", "detected_count", "clean_scenarios",
                "ttd_flows_mean", "ttd_flows_median", "ttd_flows_p90",
                "ttd_ms_mean", "ttd_ms_median")}})
            print(f"k={k} n={n}  det={r['detection_rate']:.3f} "
                  f"falseblock={r['false_block_rate']:.3f} "
                  f"ttd_mean={r['ttd_flows_mean']:.1f} ttd_med={r['ttd_flows_median']:.1f}", flush=True)

    out = {"source": args.source, "provenance": prov, "threshold": threshold,
           "per_flow_latency_ms": per_flow_ms, "scenarios": 200,
           "scenario_construction": {
               "benign_prefix_flows": 200, "attack_flows": 100, "sampling": "with replacement",
               "scenario_seed": 42, "cooldown_flows": 100,
               "shared_streams": "every k-n pair sees the identical scenario streams: the "
                                 "generator is re-seeded per call and draws before the policy runs",
               "detection_rate": "complete scenario success: no benign-prelude block and the attack blocked",
               "conditional_detection_rate": "attack blocked, among scenarios with no benign-prelude block",
               "false_block_rate": "benign prelude triggered a block, per scenario"},
           "grid": rows}
    name = "sensitivity_analysis.json" if args.source == "federated" else "sensitivity_analysis_centralized.json"
    (M / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", M / name)


if __name__ == "__main__":
    main()
