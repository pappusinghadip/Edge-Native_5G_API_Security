"""Analysis of the round-5 reruns: exposure-matched training with the class-dependent focal loss.

Reads the two Asus batches (tier 1: matched budgets; tier 2: non-IID, poisoning, kappa) and the Mac
repeat of the 60N pair, and writes results/metrics/round5_analysis.json. No training here.

Run: PYTHONPATH=. .venv/bin/python scripts/round5_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

M = Path("results/metrics")
T1 = Path("results/metrics_PC_run_2026-09-14_round5")
T2 = Path("results/metrics_PC_run_2026-09-16_round5b")
SEEDS = [101, 102, 103, 104, 105]
KEYS = ("auc_roc", "f1_binary", "recall_binary", "false_positive_rate")


def metrics_of(path: Path) -> dict:
    d = json.loads(path.read_text())
    return d["runs"][0] if "runs" in d else {**d["final"], **{k: v for k, v in d.items() if k != "final"}}


def arm(pattern: str, base: Path) -> dict:
    """Five seeds of one arm -> per-seed metrics plus mean, sd and 95% CI per metric."""
    per = {s: metrics_of(base / pattern.format(s=s)) for s in SEEDS}
    out = {"per_seed": {str(s): {k: per[s].get(k) for k in KEYS} for s in SEEDS}, "summary": {}}
    for k in KEYS:
        v = np.array([per[s].get(k) for s in SEEDS], dtype=float)
        if np.isnan(v).all():
            out["summary"][k] = {"mean": None, "std": None, "ci95_low": None, "ci95_high": None}
            continue
        m, sd = float(np.nanmean(v)), float(np.nanstd(v, ddof=1))
        h = float(stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)))
        out["summary"][k] = {"mean": m, "std": sd, "ci95_low": m - h, "ci95_high": m + h}
    ex = [per[s].get("sample_exposure") for s in SEEDS]
    out["sample_exposure"] = ex[0] if ex[0] else None
    out["epochs_completed"] = per[SEEDS[0]].get("epochs_completed")
    return out


def compare(a: dict, b: dict, label: str, margin: float = 0.02) -> dict:
    """Paired difference a - b on ROC-AUC, with the margins the review asks to be tested."""
    x = np.array([a["per_seed"][str(s)]["auc_roc"] for s in SEEDS], float)
    y = np.array([b["per_seed"][str(s)]["auc_roc"] for s in SEEDS], float)
    d = x - y
    m, sd, n = float(d.mean()), float(d.std(ddof=1)), len(d)
    se = sd / np.sqrt(n)
    lo, hi = m - stats.t.ppf(0.975, n - 1) * se, m + stats.t.ppf(0.975, n - 1) * se
    return {
        "label": label, "mean_difference": m, "diff_ci95": [float(lo), float(hi)],
        "one_sided_upper95": float(m + stats.t.ppf(0.95, n - 1) * se),
        "paired_t_p_two_sided": float(stats.ttest_rel(x, y).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(x, y).pvalue),
        "cohens_dz": float(m / sd) if sd else None,
        "margin": margin,
        "non_inferior": {str(dl): bool(hi < dl) for dl in (0.02, 0.01, 0.005)},
    }


def attribution(files: list[Path]) -> dict:
    """Malicious and legitimate rejections counted from the per-client decision log."""
    mal_rej = mal = leg_rej = leg = 0
    for f in files:
        for rnd in json.loads(f.read_text())["history"]:
            for c in rnd.get("clients", []):
                if c["adversarial"]:
                    mal += 1; mal_rej += not c["accepted"]
                else:
                    leg += 1; leg_rej += not c["accepted"]
    return {"malicious_rejected": mal_rej, "malicious_submitted": mal,
            "legitimate_rejected": leg_rej, "legitimate_submitted": leg,
            "malicious_accepted": mal - mal_rej, "measured": bool(mal or leg)}


res: dict = {"loss": "focal, class-dependent alpha (alpha on the malicious class, 1 - alpha on benign)",
             "alpha": 0.95, "gamma": 3.0, "seeds": SEEDS,
             "provenance": {"training arms": "AMD Ryzen 7 laptop (Asus), WSL 2, CPU only",
                            "cross-platform check": "Apple M4 Pro"}}

res["arms"] = {
    "centralized_20N": arm("repeated_centralized_r5_e20_s{s}.json", T1),
    "centralized_60N": arm("repeated_centralized_r5_e60_s{s}.json", T1),
    "federated_20N": arm("repeated_federated_iid_r5_le1_s{s}.json", T1),
    "federated_60N": arm("repeated_federated_iid_r5_le3_s{s}.json", T1),
    "isolated_60N": arm("fl_isolated_iid_r5_isolated_s{s}.json", T1),
    "noniid_0.1": arm("fl_fedavg_dirichlet_r5_noniid_a0p1_s{s}.json", T2),
    "noniid_0.5": arm("fl_fedavg_dirichlet_r5_noniid_a0p5_s{s}.json", T2),
    "noniid_1.0": arm("fl_fedavg_dirichlet_r5_noniid_a1p0_s{s}.json", T2),
    "poison_filter_off": arm("fl_poison1_noclip_iid_r5_poison_noclip_s{s}.json", T2),
    "poison_filter_on": arm("fl_poison1_clip_iid_r5_poison_clip_s{s}.json", T2),
    "clean_filter_on": arm("fl_fedavg_iid_r5_clean_clip_s{s}.json", T2),
}

# per-client spread of the isolated arm (the weakest-node question)
cl = [c["auc_roc"] for s in SEEDS
      for c in json.loads((T1 / f"fl_isolated_iid_r5_isolated_s{s}.json").read_text())["per_client"]]
res["isolated_per_client"] = {"clients_total": len(cl), "worst": min(cl), "best": max(cl),
                              "mean": float(np.mean(cl)), "std": float(np.std(cl, ddof=1)),
                              "below_0.75": int(sum(c < 0.75 for c in cl))}

A = res["arms"]
res["comparisons"] = {
    "matched_20N": compare(A["centralized_20N"], A["federated_20N"],
                           "centralized 20 epochs vs federated 20 rounds x 1 local epoch"),
    "matched_60N": compare(A["centralized_60N"], A["federated_60N"],
                           "centralized 60 epochs vs federated 20 rounds x 3 local epochs"),
    "federated_vs_isolated_60N": compare(A["federated_60N"], A["isolated_60N"],
                                         "federated vs isolated, both 60N"),
    "unmatched_thesis_schedule": compare(A["centralized_20N"], A["federated_60N"],
                                         "centralized 20 epochs vs federated 20 x 3 (unmatched, for reference)"),
}

res["poisoning"] = {
    "under_attack": attribution([T2 / f"fl_poison1_clip_iid_r5_poison_clip_s{s}.json" for s in SEEDS]),
    "no_attacker": attribution([T2 / f"fl_fedavg_iid_r5_clean_clip_s{s}.json" for s in SEEDS]),
    "filter_off_collapsed_seeds": [s for s in SEEDS
                                   if not (A["poison_filter_off"]["per_seed"][str(s)]["auc_roc"] or 0) > 0.6],
}
res["kappa_sweep"] = {}
for k, tag in ((1.5, "r5_kappa1p5"), (2.0, "r5_kappa2p0"), (2.5, "r5_poison_clip"), (3.0, "r5_kappa3p0")):
    a = arm(f"fl_poison1_clip_iid_{tag}_s{{s}}.json", T2)
    res["kappa_sweep"][str(k)] = {"auc_roc": a["summary"]["auc_roc"],
                                  "rejections": attribution([T2 / f"fl_poison1_clip_iid_{tag}_s{s}.json" for s in SEEDS])}

# Median-ROC-AUC run of each 60N arm, with its confusion matrix, for Figure 4.4
from src.models.io import load_processed_split          # noqa: E402
from src.utils.config import load_yaml, resolve_path    # noqa: E402

_, y_test, _ = load_processed_split(resolve_path(load_yaml("configs/paths.yaml")["data"]["processed"]) / "test.npz")
res["median_runs"] = {}
for key, pattern in (("centralized_60N", "repeated_centralized_r5_e60_s{s}"),
                     ("federated_60N", "repeated_federated_iid_r5_le3_s{s}")):
    per = res["arms"][key]["per_seed"]
    med = sorted(SEEDS, key=lambda s: per[str(s)]["auc_roc"])[len(SEEDS) // 2]
    run = json.loads((T1 / f"{pattern.format(s=med)}.json").read_text())["runs"][0]
    probs = np.load(T1 / f"{pattern.format(s=med)}_seed{med}_probs.npz")["probs"]
    pred = (probs >= run["threshold"]).astype(int)
    res["median_runs"][key] = {
        "seed": med, "threshold": run["threshold"],
        "confusion_matrix": [[int(((pred == j) & (y_test == i)).sum()) for j in (0, 1)] for i in (0, 1)]}

# first round after which the spread across the five IID seeds stays below 0.005
_h = np.array([[r["auc_roc"] for r in json.loads((T1 / f"repeated_federated_iid_r5_le3_s{s}.json").read_text())["runs"][0]["history"]]
               for s in SEEDS])
_sd = _h.std(axis=0, ddof=1)
res["settle_round"] = next((i + 1 for i in range(len(_sd)) if all(v < 0.005 for v in _sd[i:])), None)

# Mac repeat of the 60N pair, when it exists
if (M / f"repeated_centralized_r5mac_e60_s{SEEDS[-1]}.json").exists():
    res["arms"]["centralized_60N_mac"] = arm("repeated_centralized_r5mac_e60_s{s}.json", M)
    res["arms"]["federated_60N_mac"] = arm("repeated_federated_iid_r5mac_le3_s{s}.json", M)
    res["comparisons"]["matched_60N_mac"] = compare(res["arms"]["centralized_60N_mac"],
                                                    res["arms"]["federated_60N_mac"],
                                                    "the 60N pair repeated on the Mac")

(M / "round5_analysis.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
print("wrote", M / "round5_analysis.json")
for name, c in res["comparisons"].items():
    print(f"  {name:28} diff {c['mean_difference']:+.4f} CI [{c['diff_ci95'][0]:+.4f},{c['diff_ci95'][1]:+.4f}] "
          f"NI@0.02={c['non_inferior']['0.02']} @0.01={c['non_inferior']['0.01']} p={c['paired_t_p_two_sided']:.3f}")
print("  poisoning under attack:", res["poisoning"]["under_attack"])
print("  isolated per client:", {k: round(v, 4) if isinstance(v, float) else v for k, v in res["isolated_per_client"].items()})
