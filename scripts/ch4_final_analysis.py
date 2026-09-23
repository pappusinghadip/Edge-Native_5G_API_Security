"""Chapter 4 analysis over both training platforms. No training happens here.

Provenance is kept per comparison, never mixed within one:
  * Mac (results/metrics): the detector comparison of Table 4.1, the mitigation and
    sensitivity analysis, latency, and the Mac replicate of the two headline arms.
  * PC (results/metrics_PC_run_2026-09-07): every training-arm comparison — centralized,
    federated IID, non-IID at three concentrations, isolated, poisoning with and without
    the filter, clean federation with the filter, the kappa sweep, and scalability.

Writes results/metrics/final_analysis.json.

Run: python scripts/ch4_final_analysis.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, matthews_corrcoef, roc_auc_score)

M = Path("results/metrics")
P = Path("results/metrics_PC_run_2026-09-07")
DELTA = 0.02            # predefined ROC-AUC non-inferiority margin
SEEDS = [101, 102, 103, 104, 105]
KAPPA_SEEDS = [101, 102, 103]
ROUNDS, CLIENTS = 20, 5
PC_TEST_SHA16 = "c16780b0a5922060"     # sha256 prefix of test.npz recorded in RUN_MANIFEST.md


def jload(p: Path):
    return json.loads(p.read_text())


def ci95(values) -> dict:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = len(a)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(a.mean()); sd = float(a.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def wilson(k: int, n: int, z: float = 1.959963985) -> dict:
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"point": p, "k": k, "n": n, "ci95_low": max(0.0, c - h), "ci95_high": min(1.0, c + h)}


def metrics_at(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "precision": float(prec), "recall": float(rec),
            "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
            "mcc": float(matthews_corrcoef(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]], "threshold": float(thr)}


def headline_arm(d: Path, prefix: str, runs_file: str, y) -> dict:
    """Full metric set for every seed of one arm, each at its own validation-tuned threshold."""
    runs = {r["seed"]: r for r in jload(d / runs_file)["runs"]}
    per = {s: metrics_at(y, np.load(d / f"{prefix}_seed{s}_probs.npz")["probs"], runs[s]["threshold"])
           for s in SEEDS}
    keys = [k for k in per[SEEDS[0]] if k != "confusion_matrix"]
    return {"per_seed": per, "summary": {k: ci95([per[s][k] for s in SEEDS]) for k in keys},
            "median_auc_seed": sorted(SEEDS, key=lambda s: per[s]["roc_auc"])[len(SEEDS) // 2],
            "histories": {s: runs[s].get("history") for s in SEEDS}}


def headline_arm_split(d: Path, stem: str, y) -> dict:
    """As headline_arm, for arms stored one file per seed: <stem>_s<seed>.json and its _seed<seed>_probs.npz."""
    runs = {s: jload(d / f"{stem}_s{s}.json")["runs"][0] for s in SEEDS}
    probs = {s: f"{stem}_s{s}_seed{s}_probs.npz" for s in SEEDS}
    per = {s: metrics_at(y, np.load(d / probs[s])["probs"], runs[s]["threshold"]) for s in SEEDS}
    keys = [k for k in per[SEEDS[0]] if k != "confusion_matrix"]
    return {"per_seed": per, "summary": {k: ci95([per[s][k] for s in SEEDS]) for k in keys},
            "median_auc_seed": sorted(SEEDS, key=lambda s: per[s]["roc_auc"])[len(SEEDS) // 2],
            "probs_files": probs, "focal_class_weighted": all(runs[s].get("focal_class_weighted") for s in SEEDS),
            "sample_exposure": runs[SEEDS[0]].get("sample_exposure"),
            "histories": {s: runs[s].get("history") for s in SEEDS}}


def paired(a: dict, b: dict, margin: float | None = None) -> dict:
    """Paired comparison a - b over shared seeds, with distribution-free checks."""
    seeds = sorted(set(a) & set(b))
    x = np.array([a[s] for s in seeds]); z = np.array([b[s] for s in seeds]); d = x - z
    n = len(d); sd = d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if sd > 0 else 0.0
    out = {"seeds": seeds, "mean_difference": float(d.mean()),
           "diff_ci95": [float(d.mean() - half), float(d.mean() + half)],
           "cohens_d_paired": float(d.mean() / sd) if sd > 0 else None}
    if sd > 0:
        out["paired_t_p_two_sided"] = float(stats.ttest_rel(x, z).pvalue)
        out["wilcoxon_p"] = float(stats.wilcoxon(x, z).pvalue)
        signs = np.random.default_rng(0).choice([-1.0, 1.0], size=(20000, n))
        out["permutation_p"] = float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(d.mean())))
    else:
        out.update({"paired_t_p_two_sided": None, "wilcoxon_p": None, "permutation_p": None,
                    "note": "differences are identically zero"})
    if margin is not None:
        out["margin"] = margin
        out["non_inferior"] = bool(out["diff_ci95"][1] < margin)
    return out


def arm_files(template: str, seeds) -> dict:
    return {s: jload(P / template.format(s=s)) for s in seeds if (P / template.format(s=s)).exists()}


def collapsed(final: dict) -> bool:
    """A run has collapsed when it emits a constant output: undefined ROC-AUC, or no positive
    prediction at the tuned threshold (F1 = 0 with FPR = 0)."""
    a = final.get("auc_roc")
    return (a is None or not np.isfinite(a)) or (final.get("f1_binary", 0) == 0 and final.get("false_positive_rate", 0) == 0)


def arm_summary(runs: dict) -> dict:
    per = {s: {k: r["final"].get(k) for k in ("auc_roc", "f1_binary", "recall_binary", "false_positive_rate")}
           for s, r in runs.items()}
    col = [s for s, r in runs.items() if collapsed(r["final"])]
    ok = [s for s in runs if s not in col]
    return {"seeds": sorted(runs), "collapsed_seeds": col, "per_seed": per,
            **{k: ci95([per[s][k] for s in runs]) for k in ("auc_roc", "f1_binary", "recall_binary", "false_positive_rate")},
            "auc_roc_converged": ci95([per[s]["auc_roc"] for s in ok])}


def rejection_breakdown(runs: dict, attacked: bool) -> dict:
    """Per-round rejection counts -> malicious rejected/accepted and legitimate rejected.

    The history records only how many updates were rejected in a round, not which. For attacked
    runs one client is malicious each round and its update is inflated five-fold, so a round with
    at least one rejection is taken to have rejected the malicious update and any further
    rejections to be legitimate. This inference is stated wherever the counts are reported.
    """
    per = {}
    for s, r in runs.items():
        rej = [h["rejected_updates"] for h in r["history"]]
        if attacked:
            per[s] = {"malicious_rejected": sum(1 for v in rej if v >= 1),
                      "malicious_accepted": sum(1 for v in rej if v == 0),
                      "legitimate_rejected": sum(v - 1 for v in rej if v > 1),
                      "rounds_with_extra_rejection": [i + 1 for i, v in enumerate(rej) if v > 1],
                      "rounds_malicious_accepted": [i + 1 for i, v in enumerate(rej) if v == 0]}
        else:
            per[s] = {"legitimate_rejected": sum(rej)}
    n = len(runs)
    tot = {k: sum(v[k] for v in per.values()) for k in per[next(iter(per))] if not k.startswith("rounds")}
    tot["malicious_submitted"] = n * ROUNDS if attacked else 0
    tot["legitimate_submitted"] = n * ROUNDS * ((CLIENTS - 1) if attacked else CLIENTS)
    return {"per_seed": per, "totals": tot,
            "inference": "malicious vs legitimate rejections inferred from per-round counts" if attacked else None}


PAYLOAD_BYTES = json.loads((M / "measured_communication.json").read_text())["model"]["serialized_float32_bytes"]


def main() -> None:
    test = Path("data/processed/test.npz")
    sha16 = hashlib.sha256(test.read_bytes()).hexdigest()[:16]
    assert sha16 == PC_TEST_SHA16, f"test set differs from the one the PC trained against ({sha16})"
    y = np.load(test)["y"]
    res: dict = {"margin_delta_roc_auc": DELTA, "seeds": SEEDS,
                 "provenance": {"mac": str(M), "pc": str(P), "test_sha256_16": sha16},
                 "test_set": {"n": int(len(y)), "malicious": int(y.sum()), "benign": int((y == 0).sum())}}

    # --- headline arms on both platforms ----------------------------------------------- #
    # round 6: the Mac detector rows and the deployed model use the corrected focal loss, i.e. the Mac 60N
    # pair (centralized 60 epochs, federated 20 rounds x 3 local epochs); the pre-correction e20/r20 Mac
    # runs are kept only as "arms_mac_pre_correction" for provenance
    res["arms_mac"] = {"centralized": headline_arm_split(M, "repeated_centralized_r5mac_e60", y),
                       "federated": headline_arm_split(M, "repeated_federated_iid_r5mac_le3", y)}
    res["arms_mac_pre_correction"] = {
        "centralized": headline_arm(M, "repeated_centralized_e20", "repeated_centralized_e20.json", y),
        "federated": headline_arm(M, "repeated_federated_iid_r20", "repeated_federated_iid_r20.json", y)}
    res["arms"] = {"centralized": headline_arm(P, "repeated_centralized_e20", "repeated_centralized_e20.json", y),
                   "federated": headline_arm(P, "repeated_federated_iid_r20", "repeated_federated_iid_r20.json", y)}
    auc = lambda arm: {s: arm["per_seed"][s]["roc_auc"] for s in SEEDS}
    res["non_inferiority"] = {**paired(auc(res["arms"]["centralized"]), auc(res["arms"]["federated"]), DELTA),
                              "platform": "pc"}
    res["non_inferiority_mac"] = {**paired(auc(res["arms_mac"]["centralized"]), auc(res["arms_mac"]["federated"]), DELTA),
                                  "platform": "mac"}
    res["cross_platform"] = {arm: {"mac": res["arms_mac"][arm]["summary"]["roc_auc"],
                                   "pc": res["arms"][arm]["summary"]["roc_auc"]} for arm in ("centralized", "federated")}

    # --- PC training arms --------------------------------------------------------------- #
    fed_auc = auc(res["arms"]["federated"])
    noniid = {}
    for tag, a in (("0p1", 0.1), ("0p5", 0.5), ("1p0", 1.0)):
        runs = arm_files(f"fl_fedavg_dirichlet_noniid_a{tag}_s{{s}}.json", SEEDS)
        sizes = []
        for i in range(CLIENTS):
            f = Path(f"data/partitions/client_{i}_dir_{tag}.npz")
            if f.exists():
                yy = np.load(f)["y"]; sizes.append({"client": i, "samples": int(len(yy)), "malicious": int(yy.sum())})
        noniid[str(a)] = {**arm_summary(runs),
                          "vs_iid": paired(fed_auc, {s: r["final"]["auc_roc"] for s, r in runs.items()}, DELTA),
                          "partition": sizes,
                          "curves": {s: [h["auc_roc"] for h in r["history"]] for s, r in runs.items()}}
    res["noniid"] = noniid

    iso = arm_files("fl_isolated_iid_isolated_s{s}.json", SEEDS)
    per_client = {s: [c["auc_roc"] for c in r["per_client"]] for s, r in iso.items()}
    allc = [v for vs in per_client.values() for v in vs]
    res["isolated"] = {**arm_summary(iso),
                       "vs_federated": paired(fed_auc, {s: r["final"]["auc_roc"] for s, r in iso.items()}),
                       "per_client": {"per_seed": per_client,
                                      "per_seed_worst": {s: min(v) for s, v in per_client.items()},
                                      "worst": float(min(allc)), "best": float(max(allc)),
                                      "clients_total": len(allc),
                                      "clients_below_0p75": int(sum(v < 0.75 for v in allc))}}

    noclip = arm_files("fl_poison1_noclip_iid_poison_noclip_s{s}.json", SEEDS)
    clip = arm_files("fl_poison1_clip_iid_poison_clip_s{s}.json", SEEDS)
    clean = arm_files("fl_fedavg_iid_clean_clip_s{s}.json", SEEDS)
    res["poisoning"] = {
        "lambda": 5.0, "kappa": 2.5, "attacker_fraction": "1 of 5",
        "filter_off": arm_summary(noclip),
        "filter_on": {**arm_summary(clip), "rejections": rejection_breakdown(clip, attacked=True),
                      "curves": {s: [h["auc_roc"] for h in r["history"]] for s, r in clip.items()}},
        "clean_filter_on": {**arm_summary(clean), "rejections": rejection_breakdown(clean, attacked=False),
                            "vs_federated": paired(fed_auc, {s: r["final"]["auc_roc"] for s, r in clean.items()}, DELTA)},
        "noclip_curves": {s: [h["auc_roc"] for h in r["history"]] for s, r in noclip.items()},
    }
    kappa = {}
    for k in (1.5, 2.0, 2.5, 3.0):
        runs = arm_files(f"fl_poison1_clip_iid_kappa{str(k).replace('.', 'p')}_s{{s}}.json", KAPPA_SEEDS)
        kappa[str(k)] = {**arm_summary(runs), "rejections": rejection_breakdown(runs, attacked=True)}
    res["kappa_sweep"] = kappa

    # determinism: the kappa = 2.5 sweep repeats the poison-clip configuration exactly
    res["determinism"] = {s: abs(kappa["2.5"]["per_seed"][s]["auc_roc"] - clip[s]["final"]["auc_roc"]) < 1e-12
                          for s in KAPPA_SEEDS}

    rows = jload(P / "scalability.json")
    res["scalability"] = {str(k): {"clients": k, "rounds": 5,
                                   "per_seed": {r["seed"]: r["final_auc"] for r in rows if r["clients"] == k},
                                   "final_auc": ci95([r["final_auc"] for r in rows if r["clients"] == k]),
                                   "mean_round_seconds": ci95([r["mean_round_seconds"] for r in rows if r["clients"] == k]),
                                   # one payload definition for every total (round-5 review, item 10):
                                   # the measured np.savez update of all model variables, down and up
                                   "total_comm_mb": 2 * PAYLOAD_BYTES * k * next(r["rounds"] for r in rows if r["clients"] == k) / 1e6}
                          for k in sorted({r["clients"] for r in rows})}
    res["scalability_mac_k100_r20"] = jload(M / "scalability_k100_r20.json")

    # --- Mac-derived: mitigation, sensitivity, time to enforcement ----------------------- #
    mit, sens = jload(M / "mitigation_results.json"), jload(M / "sensitivity_analysis.json")
    principal = next(g for g in sens["grid"] if g["k"] == 3 and g["n"] == 10)
    ns = sens["scenarios"]
    cen_sens = jload(M / "sensitivity_analysis_centralized.json") if (M / "sensitivity_analysis_centralized.json").exists() else None
    cen_principal = next((g for g in cen_sens["grid"] if g["k"] == 3 and g["n"] == 10), None) if cen_sens else None
    res["mitigation_proportions"] = {
        "scenarios": ns, "denominator": "independently generated attack scenarios, one outcome each",
        # the deployed model scores the streams; the centralized CNN is kept only as a reference
        "source": sens["source"], "provenance": sens.get("provenance"), "threshold": sens["threshold"],
        "scenario_construction": sens.get("scenario_construction"),
        "detection_rate": wilson(round(principal["detection_rate"] * ns), ns),
        "false_block_rate": wilson(round(principal["false_block_rate"] * ns), ns),
        "conditional_detection_rate": wilson(principal["detected_count"], principal["clean_scenarios"]),
        "centralized_reference": ({"source": cen_sens["source"], "threshold": cen_sens["threshold"],
                                   "detection_rate": cen_principal["detection_rate"],
                                   "false_block_rate": cen_principal["false_block_rate"],
                                   "ttd_flows_mean": cen_principal["ttd_flows_mean"],
                                   "ttd_flows_median": cen_principal["ttd_flows_median"]}
                                  if cen_principal else None),
        "superseded_100_scenario_run": {"source": "centralized 1D-CNN",
                                        **{k: mit[k] for k in ("scenarios", "detection_rate", "false_block_rate",
                                                               "ttd_flows_mean", "ttd_flows_median")}}}
    res["wall_clock_ttd"] = {
        "source": "Mac sensitivity grid on the deployed federated model, k=3 n=10, 200 scenarios",
        "ttd_flows_mean": principal["ttd_flows_mean"], "ttd_flows_median": principal["ttd_flows_median"],
        "ttd_flows_p90": principal["ttd_flows_p90"],
        "rates": [{"requests_per_second": r, "inter_arrival_s": 1 / r,
                   "ttd_seconds_mean": principal["ttd_flows_mean"] / r,
                   "ttd_seconds_median": principal["ttd_flows_median"] / r,
                   "ttd_seconds_p90": principal["ttd_flows_p90"] / r} for r in (1, 5, 10, 50, 100)]}

    # --- threshold sweep on the deployed model: the corrected-loss Mac federated median-AUC run --- #
    dep = res["arms_mac"]["federated"]
    s_med = dep["median_auc_seed"]
    probs = np.load(M / dep["probs_files"][s_med])["probs"]
    grid = []
    # a 0.01 grid, so the sweep cannot miss the validation-tuned threshold and under-report F1
    for t in np.round(np.arange(0.01, 1.00, 0.01), 2):
        m = metrics_at(y, probs, float(t))
        grid.append({k: m[k] for k in ("threshold", "precision", "recall", "f1", "false_positive_rate", "mcc")})
    res["threshold_sweep"] = {"arm": "federated (Mac, corrected loss, 60N)", "seed": s_med, "grid": grid,
                              "best_f1": max(grid, key=lambda r: r["f1"]),
                              "validation_tuned": {k: dep["per_seed"][s_med][k] for k in ("threshold", "f1")}}

    for arm in ("arms", "arms_mac", "arms_mac_pre_correction"):   # histories are bulky, figures only
        for a in res[arm].values():
            a.pop("histories", None)
    (M / "final_analysis.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")

    # --- digest --------------------------------------------------------------------------- #
    ni, nm = res["non_inferiority"], res["non_inferiority_mac"]
    for lab, n in (("PC ", ni), ("Mac", nm)):
        print(f"{lab} cen-fed {n['mean_difference']:.4f} CI [{n['diff_ci95'][0]:.5f}, {n['diff_ci95'][1]:.5f}] "
              f"t p={n['paired_t_p_two_sided']:.4f} wilcoxon p={n['wilcoxon_p']:.4f} non-inferior={n['non_inferior']}")
    for a, e in noniid.items():
        print(f"non-IID a={a}: auc {e['auc_roc']['mean']:.4f} ± {e['auc_roc']['std']:.4f} collapsed={e['collapsed_seeds']} "
              f"non-inferior={e['vs_iid']['non_inferior']}")
    i = res["isolated"]
    print(f"isolated: {i['auc_roc']['mean']:.4f} ± {i['auc_roc']['std']:.4f}; fed-iso {i['vs_federated']['mean_difference']:+.4f} "
          f"p={i['vs_federated']['paired_t_p_two_sided']:.3f}; worst client {i['per_client']['worst']:.4f}; "
          f"{i['per_client']['clients_below_0p75']}/{i['per_client']['clients_total']} clients below 0.75")
    pt = res["poisoning"]["filter_on"]["rejections"]["totals"]
    print(f"poison filter on: malicious rejected {pt['malicious_rejected']}/{pt['malicious_submitted']}, "
          f"legitimate rejected {pt['legitimate_rejected']}/{pt['legitimate_submitted']}; "
          f"filter off collapsed {len(res['poisoning']['filter_off']['collapsed_seeds'])}/5")
    ct = res["poisoning"]["clean_filter_on"]["rejections"]["totals"]
    print(f"clean filter on: legitimate rejected {ct['legitimate_rejected']}/{ct['legitimate_submitted']}")
    for k, e in kappa.items():
        t = e["rejections"]["totals"]
        print(f"kappa {k}: auc {e['auc_roc']['mean']:.4f} ± {e['auc_roc']['std']:.4f}  "
              f"legit rejected {t['legitimate_rejected']}/{t['legitimate_submitted']}  malicious accepted {t['malicious_accepted']}")
    print("determinism:", res["determinism"])
    print("saved", M / "final_analysis.json")


if __name__ == "__main__":
    main()
