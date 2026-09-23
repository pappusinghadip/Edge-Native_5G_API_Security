"""Repeated-seed runs for centralized and federated training.

The single-run numbers reported in Chapter 4 do not say whether the gap between
centralized and federated training is larger than ordinary run-to-run variation.
This script trains both arms under the same set of seeds and records per-seed
test-set metrics and scores, so that means, standard deviations, confidence
intervals and a paired significance test can be computed afterwards.

Usage:
  python scripts/repeated_runs.py --arm centralized --seeds 101 102 103 104 105
  python scripts/repeated_runs.py --arm federated  --seeds 101 102 103 104 105 --rounds 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from src.fl.simulate import (
    best_threshold,
    evaluate_global,
    load_clients,
    load_global_splits,
    run_fedavg,
)
from src.models.cnn import build_model_from_config
from src.models.io import load_processed_split, one_hot_encode
from src.utils.config import load_yaml, resolve_path
from src.utils.logger import configure_logging
from src.utils.seed import set_global_seed

CFG = Path(os.environ.get("THESIS_CFG", "configs"))   # configs_clean for the round-6 sensitivity reruns


def _splits(paths_config: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for split in ("train", "val", "test"):
        p = resolve_path(paths_config["data"]["processed"]) / f"{split}.npz"
        X, y, _ = load_processed_split(p)
        out[split] = (X, y)
    return out


def run_centralized(model_config: dict, splits: dict, seed: int,
                    epochs: int | None = None, patience: int | None = None,
                    early_stopping: bool = True) -> tuple[dict, np.ndarray]:
    """Train the centralized 1D-CNN from scratch under one seed."""
    import tensorflow as tf

    set_global_seed(seed)
    tr = model_config["training"]
    num_classes = int(model_config["model"]["num_classes"])
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    model = build_model_from_config(model_config)
    es = tr["early_stopping"]
    callbacks = [] if not early_stopping else [
        tf.keras.callbacks.EarlyStopping(
            monitor=es["monitor"], mode=es["mode"],
            patience=int(patience if patience is not None else es["patience"]),
            restore_best_weights=bool(es["restore_best_weights"]),
        ),
    ]
    callbacks += [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=tr["reduce_lr"]["monitor"], mode=tr["reduce_lr"]["mode"],
            factor=float(tr["reduce_lr"]["factor"]), patience=int(tr["reduce_lr"]["patience"]),
            min_lr=float(tr["reduce_lr"]["min_lr"]),
        ),
    ]
    hist = model.fit(
        X_train, one_hot_encode(y_train, num_classes),
        validation_data=(X_val, one_hot_encode(y_val, num_classes)),
        epochs=int(epochs if epochs is not None else tr["epochs"]),
        batch_size=int(tr["batch_size"]),
        callbacks=callbacks, verbose=0,
    )
    t = best_threshold(model, X_val, y_val)
    metrics = evaluate_global(model, X_test, y_test, threshold=t)
    probs = model.predict(X_test, batch_size=1024, verbose=0)[:, 1]
    done = len(hist.history.get("loss", []))
    # exposure = training records seen, the quantity the round-5 review asks to be matched
    metrics.update({"epochs_completed": done, "early_stopping": early_stopping,
                    "sample_exposure": int(done * len(X_train))})
    return metrics, probs


def run_federated(fl_config: dict, model_config: dict, paths_config: dict,
                  splits: dict, seed: int, rounds: int, mode: str) -> tuple[dict, np.ndarray]:
    set_global_seed(seed)
    clients = load_clients(paths_config, mode, int(fl_config["federation"]["num_clients"]))
    gs = {"val": splits["val"], "test": splits["test"]}
    res = run_fedavg(clients, gs, fl_config, model_config, mode, rounds)
    model = getattr(res, "_model")
    probs = model.predict(splits["test"][0], batch_size=1024, verbose=0)[:, 1]
    out = dict(res.final)
    out["history"] = res.history
    le = int(fl_config["federation"]["local_epochs"])
    out.update({"local_epochs": le, "sample_exposure": int(rounds * le * sum(len(c.X) for c in clients))})
    return out, probs


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["centralized", "federated"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--mode", default="iid", choices=["iid", "dirichlet"])
    ap.add_argument("--epochs", type=int, default=None, help="Cap centralized epochs (compute budget)")
    ap.add_argument("--patience", type=int, default=None, help="Override early-stopping patience")
    ap.add_argument("--suffix", default="", help="Extra tag for the output filename")
    ap.add_argument("--local-epochs", type=int, default=None, help="Override federated local epochs per round")
    ap.add_argument("--no-early-stopping", action="store_true",
                    help="Train centralized for exactly --epochs, so its sample exposure is fixed")
    args = ap.parse_args()

    fl_config = load_yaml(CFG / "fl.yaml")
    model_config = load_yaml(CFG / "model.yaml")
    paths_config = load_yaml(CFG / "paths.yaml")
    if args.local_epochs is not None:
        fl_config["federation"]["local_epochs"] = args.local_epochs
    splits = _splits(paths_config)
    metrics_dir = resolve_path(paths_config["results"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    tag = (args.arm if args.arm == "centralized" else f"federated_{args.mode}") + args.suffix
    out_path = metrics_dir / f"repeated_{tag}.json"
    # Seeds are run one per invocation to stay inside the per-process time limit, so
    # existing results are carried forward and replaced only for the seeds re-run here.
    runs = []
    if out_path.exists():
        prior = json.loads(out_path.read_text()).get("runs", [])
        runs = [r for r in prior if r.get("seed") not in set(args.seeds)]
    for seed in args.seeds:
        t0 = time.time()
        if args.arm == "centralized":
            m, probs = run_centralized(model_config, splits, seed, args.epochs, args.patience,
                                       early_stopping=not args.no_early_stopping)
        else:
            m, probs = run_federated(fl_config, model_config, paths_config, splits,
                                     seed, args.rounds, args.mode)
        m["seed"] = seed
        m["wall_seconds"] = round(time.time() - t0, 1)
        m["epochs_cap"] = args.epochs
        m["focal_class_weighted"] = bool(model_config["model"].get("focal_class_weighted", True))
        runs.append(m)
        np.savez_compressed(metrics_dir / f"repeated_{tag}_seed{seed}_probs.npz",
                            probs=probs.astype(np.float32))
        print(f"[{tag}] seed={seed} auc={m['auc_roc']:.4f} f1={m['f1_binary']:.4f} "
              f"fpr={m['false_positive_rate']:.4f} ({m['wall_seconds']}s)", flush=True)
        # written after every seed so a long sweep is never lost to an interruption
        out_path.write_text(json.dumps({"arm": tag, "rounds": args.rounds,
                                        "runs": sorted(runs, key=lambda r: r["seed"])}, indent=2),
                            encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
