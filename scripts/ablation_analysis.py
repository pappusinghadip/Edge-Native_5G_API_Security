"""Ablation over the components of the framework, at a matched round budget.

Each stage adds one mechanism to the one above it, so the incremental contribution
of each is visible rather than only the end-to-end result:

  1. 1D-CNN, isolated      — local training only, no aggregation
  2. + Federated Averaging — collaborative training across the five nodes
  3. + SafeFedAvg          — norm-based rejection, evaluated with one poisoned node
  4. + Mitigation          — the k-of-n enforcement loop on top of the detector

Stages 1-3 come from the matched 20-round runs written by the experiment pipeline;
the poisoned arms show what the safeguard is worth, since FedAvg without it collapses.
Stage 4 is an operational layer, so it is reported with detection rate and
time-to-detect rather than AUC.

Run: python scripts/ablation_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

M = Path("results/metrics")


def load(name: str) -> dict | None:
    """Load a metrics file by name, or by prefix when the run carried an extra tag."""
    p = M / name
    if not p.exists():
        cands = sorted(M.glob(name.replace(".json", "*.json")), key=lambda f: f.stat().st_mtime)
        if not cands:
            return None
        p = cands[-1]
    return json.loads(p.read_text())


def fin(d: dict | None) -> dict:
    return (d or {}).get("final", {}) or {}


def main() -> None:
    stages = []

    iso = load("fl_isolated_iid_ablation.json")
    fed = load("fl_fedavg_iid_ablation.json") or load("fl_fedavg_iid.json")
    noclip = load("fl_poison1_noclip_iid_ablation.json")
    clip = load("fl_poison1_clip_iid_ablation.json")
    mit = load("mitigation_results.json")

    # Fall back to the repeated-run mean for the clean federated arm when available:
    # it is the same configuration measured over five seeds.
    rep = load("repeated_federated_iid.json")
    rep_mean = None
    if rep and rep.get("runs"):
        runs = rep["runs"]
        rep_mean = {k: sum(r[k] for r in runs) / len(runs)
                    for k in ("auc_roc", "f1_binary", "false_positive_rate", "recall_binary")}

    def row(label, d, note=""):
        f = fin(d) if isinstance(d, dict) and "final" in d else (d or {})
        stages.append({
            "stage": label,
            "auc_roc": f.get("auc_roc"),
            "f1_binary": f.get("f1_binary"),
            "recall_binary": f.get("recall_binary"),
            "false_positive_rate": f.get("false_positive_rate"),
            "rounds": (d or {}).get("rounds") if isinstance(d, dict) else None,
            "note": note,
        })

    row("1D-CNN, isolated (no aggregation)", iso, "each node trains alone")
    if rep_mean:
        stages.append({"stage": "+ Federated Averaging", **rep_mean,
                       "rounds": rep.get("rounds"), "note": "mean of 5 seeds"})
    else:
        row("+ Federated Averaging", fed, "single seed")
    row("+ 1 poisoned node, no safeguard", noclip, "sign-flipping attacker, FedAvg only")
    row("+ SafeFedAvg norm filter", clip, "same attacker, safeguard enabled")

    out = {"detector_stages": stages}
    if mit:
        out["mitigation_stage"] = {
            "stage": "+ Mitigation (k-of-n enforcement)",
            "detection_rate": mit.get("detection_rate"),
            "false_block_rate": mit.get("false_block_rate"),
            "ttd_flows_mean": mit.get("ttd_flows_mean"),
            "ttd_flows_median": mit.get("ttd_flows_median"),
            "note": "operational layer; measured as detection rate and time-to-detect, not AUC",
        }

    (M / "ablation_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'stage':40s} {'AUC':>8s} {'F1':>8s} {'FPR%':>8s} {'rounds':>7s}")
    for s in stages:
        a = s.get("auc_roc"); f1 = s.get("f1_binary"); fpr = s.get("false_positive_rate")
        astr = f"{a:8.4f}" if isinstance(a, float) and a == a else f"{'collapsed':>8s}"
        fstr = f"{f1:8.4f}" if isinstance(f1, float) else f"{'-':>8s}"
        pstr = f"{fpr*100:8.2f}" if isinstance(fpr, float) else f"{'-':>8s}"
        print(f"{s['stage']:40s} {astr} {fstr} {pstr} {str(s.get('rounds') or '-'):>7s}")
    if "mitigation_stage" in out:
        m = out["mitigation_stage"]
        print(f"\n{m['stage']}: detection={m['detection_rate']:.3f} "
              f"false-block={m['false_block_rate']:.3f} "
              f"TTD mean={m['ttd_flows_mean']:.1f} flows")
    print("\nsaved", M / "ablation_analysis.json")


if __name__ == "__main__":
    main()
