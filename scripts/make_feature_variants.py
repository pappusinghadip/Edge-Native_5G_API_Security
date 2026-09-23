"""Round-6 review, Sections 3.4 and 7.2: data variants for the CNN feature study.

  perm    the ten features in a fixed random order (numpy seed 0), testing whether the convolution's
          assumption that neighbouring inputs are related matters for this arbitrary tabular order
  nottl   Time_To_Live removed (nine features); XGBoost ranks it second, and a TTL value may describe the
          capture set-up more than the attack, so this asks how much the CNN leans on it

Both are built from data/processed (same rows, split and scaling), with matching configs_<variant>/.
Run: PYTHONPATH=. .venv/bin/python scripts/make_feature_variants.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import yaml

SRC, FEATS = Path("data/processed"), yaml.safe_load(open("configs/model.yaml"))["dataset"]["features"]
perm = np.random.default_rng(0).permutation(len(FEATS))
assert (perm != np.arange(len(FEATS))).any()
VARIANTS = {"perm": list(perm), "nottl": [i for i, f in enumerate(FEATS) if f != "Time_To_Live"]}

for name, cols in VARIANTS.items():
    out = Path(f"data/processed_{name}"); out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        d = np.load(SRC / f"{split}.npz")
        np.savez_compressed(out / f"{split}.npz", X=d["X"][:, cols, :], y=d["y"], indices=d["indices"])
    cfg = Path(f"configs_{name}"); cfg.mkdir(exist_ok=True)
    shutil.copy("configs/fl.yaml", cfg / "fl.yaml")
    m = yaml.safe_load(open("configs/model.yaml"))
    m["dataset"]["features"] = [FEATS[i] for i in cols]
    m["model"]["input_shape"] = [len(cols), 1]
    yaml.safe_dump(m, open(cfg / "model.yaml", "w"), sort_keys=False)
    pth = yaml.safe_load(open("configs/paths.yaml"))
    pth["data"]["processed"] = str(out)
    pth["data"]["partitions"] = f"data/partitions_{name}"
    pth["results"]["figures"] = f"results/figures/{name}"
    yaml.safe_dump(pth, open(cfg / "paths.yaml", "w"), sort_keys=False)
    print(name, [FEATS[i] for i in cols])
