"""Indicative summary of the PC run. Authoritative numbers come from
ch4_final_analysis.py on the Mac; this is a sanity read only."""
import json, os, glob
import numpy as np

M = "results/metrics"
T975_4 = 2.776  # t(0.975, df=4)


def load(n):
    p = os.path.join(M, n)
    return json.load(open(p)) if os.path.exists(p) else None


def stat(vals):
    v = np.array(vals, float)
    m, sd = v.mean(), v.std(ddof=1)
    h = T975_4 * sd / np.sqrt(len(v))
    return m, sd, m - h, m + h


def line(label, vals):
    m, sd, lo, hi = stat(vals)
    print(f"  {label:<34} {m:.4f} +/- {sd:.4f}   CI [{lo:.4f}, {hi:.4f}]  n={len(vals)}")
    return m


print("=" * 74)
print(" HEADLINE ARMS (ROC-AUC over 5 seeds)")
print("=" * 74)
cen = [r["auc_roc"] for r in load("repeated_centralized_e20.json")["runs"]]
fed = [r["auc_roc"] for r in load("repeated_federated_iid_r20.json")["runs"]]
mc = line("centralized (PC)", cen)
mf = line("federated IID (PC)", fed)
print(f"  {'Mac reference: centralized':<34} 0.8306 +/- 0.0016   CI [0.8286, 0.8326]")
print(f"  {'Mac reference: federated':<34} 0.8281 +/- 0.0011   CI [0.8267, 0.8295]")

d = np.array(cen) - np.array(fed)
m, sd, lo, hi = stat(d)
print()
print(f"  difference (cen - fed)             {m:.4f}   CI [{lo:.4f}, {hi:.4f}]")
print(f"  Mac difference                     0.0025   CI [0.0017, 0.0033]")
print(f"  non-inferiority margin delta=0.02 -> upper bound {hi:.4f} "
      f"{'< 0.02  NON-INFERIOR' if hi < 0.02 else '>= 0.02  NOT SHOWN'}")

print()
print("=" * 74)
print(" NEWLY CLOSED ARMS (deferred experiments 1 and 2)")
print("=" * 74)
for a, lbl in (("0p1", "non-IID Dirichlet a=0.1"), ("0p5", "non-IID Dirichlet a=0.5"),
               ("1p0", "non-IID Dirichlet a=1.0")):
    v = [load(f"fl_fedavg_dirichlet_noniid_a{a}_s{s}.json")["final"]["auc_roc"]
         for s in (101, 102, 103, 104, 105)]
    line(lbl, v)

for pat, lbl in (("fl_isolated_iid_isolated_s%d.json", "isolated (no aggregation)"),
                 ("fl_poison1_noclip_iid_poison_noclip_s%d.json", "poisoned, filter OFF"),
                 ("fl_poison1_clip_iid_poison_clip_s%d.json", "poisoned, filter ON"),
                 ("fl_fedavg_iid_clean_clip_s%d.json", "clean, filter ON")):
    v = [load(pat % s)["final"]["auc_roc"] for s in (101, 102, 103, 104, 105)]
    line(lbl, v)

print()
print("  norm-filter multiplier sweep (3 seeds each, poisoned arm):")
for k, lab in (("1p5", "1.5"), ("2p0", "2.0"), ("2p5", "2.5"), ("3p0", "3.0")):
    v, rej = [], []
    for s in (101, 102, 103):
        d2 = load(f"fl_poison1_clip_iid_kappa{k}_s{s}.json")
        v.append(d2["final"]["auc_roc"])
        rej.append(sum(h.get("rejected_updates", 0) for h in d2["history"]))
    m, sd, lo, hi = stat(v)
    print(f"    kappa={lab:<4} auc {m:.4f} +/- {sd:.4f}   updates rejected (20 rounds): {rej}")

print()
print("=" * 74)
print(" SCALABILITY (uncontended timing)")
print("=" * 74)
sc = load("scalability.json")
for k in (5, 20, 100):
    rows = [x for x in sc if x["clients"] == k]
    rt = [x["mean_round_seconds"] for x in rows]
    au = [x["final_auc"] for x in rows]
    print(f"  K={k:<4} mean round {np.mean(rt):8.2f}s  (n={len(rows)})   "
          f"final auc {np.mean(au):.4f}   comm {rows[0]['total_comm_mb']} MB")
