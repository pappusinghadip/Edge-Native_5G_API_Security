"""Post-run integrity check: every arm run_all_arms_parallel.sh should have produced."""
import json, os, sys

M = "results/metrics"
SEEDS = [101, 102, 103, 104, 105]
missing, bad = [], []


def chk(name):
    p = os.path.join(M, name)
    if not os.path.exists(p):
        missing.append(name)
        return None
    try:
        return json.load(open(p))
    except Exception as e:
        bad.append((name, str(e)[:50]))
        return None


expected = []
for a in ("0p1", "0p5", "1p0"):
    expected += [f"fl_fedavg_dirichlet_noniid_a{a}_s{s}.json" for s in SEEDS]
for s in SEEDS:
    expected += [
        f"fl_isolated_iid_isolated_s{s}.json",
        f"fl_poison1_noclip_iid_poison_noclip_s{s}.json",
        f"fl_poison1_clip_iid_poison_clip_s{s}.json",
        f"fl_fedavg_iid_clean_clip_s{s}.json",
    ]
for k in ("1p5", "2p0", "2p5", "3p0"):
    expected += [f"fl_poison1_clip_iid_kappa{k}_s{s}.json" for s in (101, 102, 103)]

for f in expected:
    d = chk(f)
    if d is not None and "final" not in d:
        bad.append((f, "no 'final' block"))

print(f"tagged arms: {len(expected)} expected, {len(missing)} missing, {len(bad)} bad")
for f in missing:
    print("   MISSING  ", f)
for f, e in bad:
    print("   BAD      ", f, "-", e)

for f in ("repeated_centralized_e20.json", "repeated_federated_iid_r20.json"):
    d = chk(f)
    if d:
        seeds = sorted(r["seed"] for r in d["runs"])
        ok = "OK" if seeds == SEEDS else "INCOMPLETE"
        print(f"{f}: {len(seeds)}/5 seeds {seeds} {ok}")

d = chk("scalability.json")
if d:
    keys = sorted({(x["clients"], x["seed"]) for x in d})
    print(f"scalability.json: {len(d)} records, {len(keys)} distinct (clients,seed) {keys}")

# probs files ch4_final_analysis.py needs
for arm in ("repeated_centralized_e20", "repeated_federated_iid_r20"):
    have = [s for s in SEEDS if os.path.exists(os.path.join(M, f"{arm}_seed{s}_probs.npz"))]
    print(f"{arm} probs: {len(have)}/5 {have}")

sys.exit(1 if (missing or bad) else 0)
