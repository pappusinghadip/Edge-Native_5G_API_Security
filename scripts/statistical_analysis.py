"""Statistical comparison of the centralized and federated arms.

Turns the per-seed results written by ``repeated_runs.py`` into the summary the
evaluation needs: mean, standard deviation and a 95% confidence interval for each
metric, plus a paired test of the centralized-versus-federated difference. Uses the
t-distribution rather than a normal approximation, which matters at n = 5.

Run: python scripts/statistical_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

METRICS = ["auc_roc", "f1_binary", "recall_binary", "false_positive_rate", "accuracy"]
M = Path("results/metrics")


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean, sd = float(a.mean()), float(a.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    else:
        half = 0.0
    return {"n": n, "mean": mean, "std": sd, "ci95_low": mean - half, "ci95_high": mean + half,
            "ci95_halfwidth": half}


def main() -> None:
    arms = {}
    # repeated_runs.py tags its output with the budget it ran at (…_e20, …_r10), so match
    # by prefix and take the newest rather than pinning an exact filename.
    for tag, pattern in (("centralized", "repeated_centralized*.json"),
                         ("federated", "repeated_federated_iid*.json")):
        cands = sorted(M.glob(pattern), key=lambda f: f.stat().st_mtime)
        if not cands:
            print(f"missing {M / pattern} — skipping {tag}")
            continue
        p = cands[-1]
        print(f"{tag}: {p.name}")
        arms[tag] = json.loads(p.read_text())["runs"]

    out: dict = {"per_arm": {}, "comparison": {}}
    for tag, runs in arms.items():
        out["per_arm"][tag] = {m: summarize([r[m] for r in runs]) for m in METRICS}
        out["per_arm"][tag]["seeds"] = [r["seed"] for r in runs]

    if len(arms) == 2:
        c, f = arms["centralized"], arms["federated"]
        # Pair by seed so the test removes seed-to-seed variation common to both arms.
        by_seed_c = {r["seed"]: r for r in c}
        by_seed_f = {r["seed"]: r for r in f}
        shared = sorted(set(by_seed_c) & set(by_seed_f))
        for m in METRICS:
            cv = np.array([by_seed_c[s][m] for s in shared], dtype=float)
            fv = np.array([by_seed_f[s][m] for s in shared], dtype=float)
            diff = cv - fv
            if len(shared) > 1 and np.any(diff != 0):
                t_stat, p_val = stats.ttest_rel(cv, fv)
                sd = diff.std(ddof=1)
                half = float(stats.t.ppf(0.975, len(shared) - 1) * sd / np.sqrt(len(shared)))
                cohen = float(diff.mean() / sd) if sd > 0 else float("nan")
            else:
                t_stat, p_val, half, cohen = float("nan"), float("nan"), 0.0, float("nan")
            out["comparison"][m] = {
                "paired_seeds": shared,
                "centralized_mean": float(cv.mean()),
                "federated_mean": float(fv.mean()),
                "mean_difference": float(diff.mean()),
                "diff_ci95_low": float(diff.mean() - half),
                "diff_ci95_high": float(diff.mean() + half),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": cohen,
                "significant_at_005": bool(p_val < 0.05) if p_val == p_val else False,
            }

    (M / "statistical_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for tag, d in out["per_arm"].items():
        print(f"\n=== {tag} (n={d['auc_roc']['n']}, seeds={d['seeds']}) ===")
        for m in METRICS:
            s = d[m]
            print(f"  {m:22s} {s['mean']:.4f} +/- {s['std']:.4f}   95% CI [{s['ci95_low']:.4f}, {s['ci95_high']:.4f}]")
    if out["comparison"]:
        print("\n=== paired comparison (centralized - federated) ===")
        for m, s in out["comparison"].items():
            star = "significant" if s["significant_at_005"] else "not significant"
            print(f"  {m:22s} diff={s['mean_difference']:+.4f} "
                  f"CI [{s['diff_ci95_low']:+.4f}, {s['diff_ci95_high']:+.4f}]  "
                  f"p={s['p_value']:.4f}  ({star})")
    print("\nsaved", M / "statistical_analysis.json")


if __name__ == "__main__":
    main()
