"""Phase 3, Task 3.3 — scalability sweep over the number of edge nodes.

Runs FedAvg with K clients (IID in-memory partitioning), timing each round and
recording final global AUC and total communication cost. Appends to
results/metrics/scalability.json so several K values accumulate across runs.

Run once per K:
  python -m scripts.scalability --clients 3 --rounds 5
  python -m scripts.scalability --clients 5 --rounds 5
  python -m scripts.scalability --clients 10 --rounds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.fl.simulate import (
    best_threshold, evaluate_global, make_model, weighted_average,
)
from src.models.io import load_processed_split, one_hot_encode
from src.utils.config import load_yaml, resolve_path
from src.utils.seed import set_global_seed

# The measured serialized update (np.savez of every model variable), not the raw tensor size
# (697,864 bytes): one payload definition throughout, as the round-5 review asks.
MODEL_BYTES = json.loads(Path("results/metrics/measured_communication.json").read_text())["model"]["serialized_float32_bytes"]


def iid_partition(n: int, k: int, seed: int) -> list[np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for partitioning and training; records are keyed on "
                         "(clients, rounds, seed) so repeats do not overwrite each other")
    args = ap.parse_args()

    fl_cfg = load_yaml("configs/fl.yaml")
    model_cfg = load_yaml("configs/model.yaml")
    paths_cfg = load_yaml("configs/paths.yaml")
    set_global_seed(args.seed)

    k, rounds = args.clients, args.rounds
    local_epochs = int(fl_cfg["federation"]["local_epochs"])
    batch = int(fl_cfg["federation"]["local_batch_size"])
    num_classes = int(model_cfg["model"]["num_classes"])

    processed = resolve_path(paths_cfg["data"]["processed"])
    X_train, y_train, _ = load_processed_split(processed / "train.npz")
    X_val, y_val, _ = load_processed_split(processed / "val.npz")
    X_test, y_test, _ = load_processed_split(processed / "test.npz")
    shards = iid_partition(len(X_train), k, seed=args.seed)

    global_model = make_model(fl_cfg, model_cfg)
    gw = global_model.get_weights()
    local = make_model(fl_cfg, model_cfg)

    round_times = []
    for rnd in range(1, rounds + 1):
        t0 = time.perf_counter()
        weights, sizes = [], []
        for sh in shards:
            local.set_weights(gw)
            local.fit(X_train[sh], one_hot_encode(y_train[sh], num_classes),
                      epochs=local_epochs, batch_size=batch, verbose=0)
            weights.append(local.get_weights()); sizes.append(len(sh))
        gw = weighted_average(weights, sizes)
        global_model.set_weights(gw)
        dt = time.perf_counter() - t0
        round_times.append(dt)
        m = evaluate_global(global_model, X_test, y_test, threshold=0.5)
        print(f"K={k} round {rnd}/{rounds}  {dt:.1f}s  auc={m['auc_roc']:.4f}")

    thr = best_threshold(global_model, X_val, y_val)
    final = evaluate_global(global_model, X_test, y_test, threshold=thr)
    comm_mb = 2.0 * (MODEL_BYTES / 1e6) * k * rounds

    record = {
        "clients": k,
        "seed": args.seed,
        "rounds": rounds,
        "mean_round_seconds": round(float(np.mean(round_times)), 2),
        "final_auc": round(float(final["auc_roc"]), 4),
        "final_f1": round(float(final["f1_binary"]), 4),
        "total_comm_mb": round(comm_mb, 1),
    }

    out = Path("results/metrics/scalability.json")
    data = json.loads(out.read_text()) if out.exists() else []
    key = (k, args.rounds, args.seed)
    data = [d for d in data
            if (d["clients"], d.get("rounds"), d.get("seed", 42)) != key] + [record]
    data.sort(key=lambda d: (d["clients"], d.get("rounds", 0), d.get("seed", 42)))
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("recorded:", json.dumps(record))


if __name__ == "__main__":
    main()
