"""Old (pre-patch) vs new isolated arms. The patch only added inference after
training, so every number must be identical."""
import json, os
import numpy as np

M = "results/metrics"
BAK = os.path.join(M, "_pre_patch_isolated")
allsame = True
print(f"{'seed':>5} | {'old mean auc':>12} | {'new mean auc':>12} | per-client max |diff|")
print("-" * 68)
for s in (101, 102, 103, 104, 105):
    f = f"fl_isolated_iid_isolated_s{s}.json"
    o = json.load(open(os.path.join(BAK, f)))
    n = json.load(open(os.path.join(M, f)))
    oa, na = o["final"]["auc_roc"], n["final"]["auc_roc"]
    oc = [c["auc_roc"] for c in o["per_client"]]
    nc = [c["auc_roc"] for c in n["per_client"]]
    d = max(abs(a - b) for a, b in zip(oc, nc))
    same = abs(oa - na) < 1e-12 and d < 1e-12
    allsame &= same
    print(f"{s:>5} | {oa:>12.6f} | {na:>12.6f} | {d:.2e}  {'MATCH' if same else 'DIFFER'}")

print()
print("VERDICT:", "identical - patch changed nothing but what is saved"
      if allsame else "DIFFERENCES FOUND - investigate before using")

p = os.path.join(M, "fl_isolated_iid_isolated_s101_client_probs.npz")
if os.path.exists(p):
    a = np.load(p)["probs"]
    print(f"\nnew artifact: client_probs shape {a.shape} dtype {a.dtype} "
          f"range [{a.min():.4f}, {a.max():.4f}]")
