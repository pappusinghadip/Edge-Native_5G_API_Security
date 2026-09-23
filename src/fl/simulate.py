"""Phase 3 — Federated Learning simulation (synchronous FedAvg).

This is a deterministic, single-process FedAvg simulator. It is mathematically
identical to Flower's ``FedAvg`` strategy (McMahan et al., 2017): each round every
client trains locally on its own partition for E epochs, then the server takes a
sample-weighted average of the client weights. Running it in one process (instead
of Flower's Ray-backed ``start_simulation``) keeps the experiment reproducible and
avoids Ray/TensorFlow instability on macOS. The Flower ``client.py`` / ``server.py``
entrypoints implement the same client for the Phase 4 Docker deployment, where
multi-process execution is the actual requirement.

Experiments (per revised-plan Phase 3):
  * fedavg  — global model via FedAvg over K clients (IID or non-IID partitions)
  * isolated — each client trains alone on its own data, no aggregation (baseline)
  * poison  — fedavg with M sign-flipping (Byzantine) clients, with/without the
              SafeFedAvg gradient-norm safeguard

Usage:
  python -m src.fl.simulate --config configs/fl.yaml --model configs/model.yaml \
      --paths configs/paths.yaml --mode iid --experiment fedavg --rounds 20
  python -m src.fl.simulate ... --smoke   # 1-round self-check on tiny subsets
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import false_positive_rate
from src.models.cnn import build_model, focal_loss
from src.models.io import load_processed_split, one_hot_encode
from src.utils.config import load_yaml, resolve_path
from src.utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClientData:
    client_id: int
    X: np.ndarray
    y: np.ndarray  # integer labels


def partition_files(paths_config: dict[str, Any], mode: str, num_clients: int,
                    alpha: float = 0.5) -> list[Path]:
    """Resolve the K partition file paths for the requested mode."""
    if mode == "iid":
        suffix = "iid"
    elif mode in ("dirichlet", "non-iid"):
        # alpha is carried in the filename, so several heterogeneity levels can coexist
        suffix = f"dir_{str(alpha).replace('.', 'p')}"
    else:
        raise ValueError(f"Unknown partition mode '{mode}' (use iid or dirichlet)")
    part_dir = resolve_path(paths_config["data"]["partitions"])
    return [part_dir / f"client_{i}_{suffix}.npz" for i in range(num_clients)]


def load_clients(paths_config: dict[str, Any], mode: str, num_clients: int,
                 alpha: float = 0.5) -> list[ClientData]:
    clients: list[ClientData] = []
    for i, path in enumerate(partition_files(paths_config, mode, num_clients, alpha)):
        X, y, _ = load_processed_split(path)
        clients.append(ClientData(client_id=i, X=X.astype(np.float32), y=y.astype(np.int64)))
    return clients


def load_global_splits(paths_config: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    processed = resolve_path(paths_config["data"]["processed"])
    X_val, y_val, _ = load_processed_split(processed / "val.npz")
    X_test, y_test, _ = load_processed_split(processed / "test.npz")
    return {"val": (X_val, y_val), "test": (X_test, y_test)}


# --------------------------------------------------------------------------- #
# Model factory (FL clients use SGD per FedAvg theory)
# --------------------------------------------------------------------------- #
def make_model(fl_config: dict[str, Any], model_config: dict[str, Any]) -> Any:
    """Build a compiled 1D-CNN identical in architecture to the centralized model,
    but optimized with SGD (as required by the FedAvg formulation)."""
    m = model_config["model"]
    fed = fl_config["federation"]
    return build_model(
        input_shape=tuple(m["input_shape"]),
        num_classes=int(m["num_classes"]),
        conv_filters=tuple(m["conv_filters"]),
        kernel_size=int(m["kernel_size"]),
        dense_units=tuple(m["dense_units"]),
        dropout_rates=tuple(m["dropout_rates"]),
        learning_rate=float(fed["learning_rate"]),
        momentum=float(m.get("momentum", 0.9)),
        optimizer="sgd",
        use_batch_norm=bool(m.get("use_batch_norm", True)),
        use_focal_loss=bool(m.get("use_focal_loss", True)),
        focal_alpha=float(m.get("focal_alpha", 0.95)),
        focal_gamma=float(m.get("focal_gamma", 3.0)),
        focal_class_weighted=bool(m.get("focal_class_weighted", True)),
        label_smoothing=float(m.get("label_smoothing", 0.1)),
    )


# --------------------------------------------------------------------------- #
# FedAvg mechanics
# --------------------------------------------------------------------------- #
def weighted_average(weight_list: list[list[np.ndarray]], sizes: list[int]) -> list[np.ndarray]:
    """Sample-weighted average of client weight tensors (the FedAvg update)."""
    total = float(sum(sizes))
    fractions = [s / total for s in sizes]
    return [
        sum(frac * client_w[layer] for frac, client_w in zip(fractions, weight_list))
        for layer in range(len(weight_list[0]))
    ]


def update_l2_norm(new_w: list[np.ndarray], base_w: list[np.ndarray]) -> float:
    """L2 norm of the client update delta (new - base) flattened across all layers."""
    return float(
        np.sqrt(sum(float(np.sum((n - b) ** 2)) for n, b in zip(new_w, base_w)))
    )


def sign_flip(new_w: list[np.ndarray], base_w: list[np.ndarray], scale: float = 5.0) -> list[np.ndarray]:
    """Byzantine sign-flip attack: push the update in the opposite direction, scaled up."""
    return [b - scale * (n - b) for n, b in zip(new_w, base_w)]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_global(model: Any, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Evaluate a global model on integer-labelled data at a fixed threshold."""
    from sklearn.metrics import f1_score, roc_auc_score

    probs = model.predict(X, batch_size=1024, verbose=0)[:, 1]
    preds = (probs >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float((tp + tn) / len(y)),
        "auc_roc": auc,
        "f1_binary": float(f1_score(y, preds, pos_label=1, zero_division=0)),
        "recall_binary": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "false_positive_rate": float(false_positive_rate(fp, tn)),
        "threshold": float(threshold),
    }


def best_threshold(model: Any, X_val: np.ndarray, y_val: np.ndarray, steps: int = 199) -> float:
    """Pick the malicious-class threshold maximizing F1 on the global validation split."""
    from sklearn.metrics import f1_score

    probs = model.predict(X_val, batch_size=1024, verbose=0)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, steps):
        f1 = f1_score(y_val, (probs >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t


# --------------------------------------------------------------------------- #
# Experiment runners
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    experiment: str
    mode: str
    rounds: int
    history: list[dict[str, float]] = field(default_factory=list)
    final: dict[str, Any] = field(default_factory=dict)
    per_client: list[dict[str, Any]] = field(default_factory=list)


def run_fedavg(
    clients: list[ClientData],
    global_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    fl_config: dict[str, Any],
    model_config: dict[str, Any],
    mode: str,
    rounds: int,
    poison_clients: int = 0,
    clip: bool = False,
    fedbn: bool = False,
) -> RunResult:
    """Synchronous FedAvg over K clients for R rounds.

    fedbn=True keeps every batch-normalisation variable local (FedBN, round-6 sensitivity): each client
    trains from the averaged non-BN weights plus its own BN weights, and result._client_weights holds each
    client's final model. The default (False) is plain FedAvg over all variables, as in every other run."""
    fed = fl_config["federation"]
    local_epochs = int(fed["local_epochs"])
    batch = int(fed["local_batch_size"])
    num_classes = int(model_config["model"]["num_classes"])
    max_norm = float(fed.get("max_grad_norm", 10.0))

    X_test, y_test = global_splits["test"]
    X_val, y_val = global_splits["val"]

    global_model = make_model(fl_config, model_config)
    global_weights = global_model.get_weights()

    experiment = "fedavg" if poison_clients == 0 else f"poison{poison_clients}_{'clip' if clip else 'noclip'}"
    result = RunResult(experiment=experiment, mode=mode, rounds=rounds)
    local_model = make_model(fl_config, model_config)  # reused scratch model
    bn_mask = ["batch_normalization" in v.name for v in global_model.weights]
    client_w: dict[int, list[np.ndarray]] = {}          # FedBN: each client's own last weights

    # SafeFedAvg cutoff: reject an update whose L2 delta-norm exceeds this multiple of
    # the cohort median. A relative cutoff adapts to model scale; a fixed absolute
    # threshold (the old max_grad_norm) rejected honest updates too.
    # A 2.5x-median cutoff drops a 5x sign-flip update while keeping honest peers;
    # retune if the assumed attack scale changes.
    clip_factor = float(fl_config.get("clip_factor", 2.5))

    for rnd in range(1, rounds + 1):
        updates: list[tuple[list[np.ndarray], int, float, int]] = []
        for c in clients:
            start = global_weights
            if fedbn and c.client_id in client_w:            # shared layers from the server, BN from the client
                start = [cw if m else g for g, cw, m in zip(global_weights, client_w[c.client_id], bn_mask)]
            local_model.set_weights(start)
            local_model.fit(
                c.X,
                one_hot_encode(c.y, num_classes),
                epochs=local_epochs,
                batch_size=batch,
                verbose=0,
            )
            w = local_model.get_weights()
            if fedbn:
                client_w[c.client_id] = w
            if c.client_id < poison_clients:
                w = sign_flip(w, global_weights)
            updates.append((w, len(c.X), update_l2_norm(w, global_weights), c.client_id))

        rejected, median, cutoff = 0, None, None
        if clip and len(updates) >= 3:
            norms = sorted(u[2] for u in updates)
            median = norms[len(norms) // 2]
            cutoff = clip_factor * max(median, 1e-12)
            kept = [(w, s) for (w, s, n, _cid) in updates if n <= cutoff]
            rejected = len(updates) - len(kept)
        else:
            kept = [(w, s) for (w, s, _n, _cid) in updates]
        # One record per client per round, so rejection attribution is measured, not inferred
        # from per-round counts (round-5 review).
        decisions = [{"client_id": int(cid), "update_norm": float(n),
                      "accepted": cutoff is None or n <= cutoff,
                      "adversarial": bool(cid < poison_clients)} for (_w, _s, n, cid) in updates]

        if kept:
            global_weights = weighted_average([k[0] for k in kept], [k[1] for k in kept])
            global_model.set_weights(global_weights)

        row = {"round": rnd, "rejected_updates": rejected,
               "median_norm": None if median is None else float(median),
               "norm_cutoff": None if cutoff is None else float(cutoff),
               "clients": decisions}
        row.update(evaluate_global(global_model, X_test, y_test, threshold=0.5))
        result.history.append(row)
        LOGGER.info(
            "[%s/%s] round %d/%d  acc=%.4f auc=%.4f f1=%.4f fpr=%.4f rejected=%d",
            experiment, mode, rnd, rounds, row["accuracy"], row["auc_roc"],
            row["f1_binary"], row["false_positive_rate"], rejected,
        )

    # Final metrics with threshold tuned on the global validation split (matches centralized eval)
    t = best_threshold(global_model, X_val, y_val)
    result.final = evaluate_global(global_model, X_test, y_test, threshold=t)
    result.final["experiment"] = experiment
    result.final["mode"] = mode
    result._model = global_model  # type: ignore[attr-defined]
    if fedbn:                                            # each client's deployable model: shared layers + own BN
        result._client_weights = {cid: [cw if m else g for g, cw, m in zip(global_weights, w, bn_mask)]  # type: ignore[attr-defined]
                                  for cid, w in client_w.items()}
    return result


def run_isolated(
    clients: list[ClientData],
    global_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    fl_config: dict[str, Any],
    model_config: dict[str, Any],
    mode: str,
    rounds: int,
) -> RunResult:
    """Isolated baseline: each client trains alone (no aggregation), evaluated on the
    global test set. Total local epochs = rounds * local_epochs for a fair budget."""
    fed = fl_config["federation"]
    total_epochs = int(fed["local_epochs"]) * rounds
    batch = int(fed["local_batch_size"])
    num_classes = int(model_config["model"]["num_classes"])
    X_test, y_test = global_splits["test"]
    X_val, y_val = global_splits["val"]

    result = RunResult(experiment="isolated", mode=mode, rounds=rounds)
    # The isolated arm has no global model, so save_result's _model path never fires and
    # nothing recoverable is written. Keep each client's test-set scores and weights, so
    # curves, thresholds and intervals for this baseline never need a retrain.
    client_probs: list[np.ndarray] = []
    client_models: list[Any] = []
    for c in clients:
        model = make_model(fl_config, model_config)
        model.fit(c.X, one_hot_encode(c.y, num_classes), epochs=total_epochs, batch_size=batch, verbose=0)
        t = best_threshold(model, X_val, y_val)
        metrics = evaluate_global(model, X_test, y_test, threshold=t)
        client_probs.append(model.predict(X_test, batch_size=1024, verbose=0)[:, 1])
        client_models.append(model)
        metrics["client_id"] = c.client_id
        metrics["train_samples"] = int(len(c.X))
        metrics["train_malicious"] = int(np.sum(c.y == 1))
        result.per_client.append(metrics)
        LOGGER.info("[isolated/%s] client %d  auc=%.4f f1=%.4f", mode, c.client_id, metrics["auc_roc"], metrics["f1_binary"])

    result._client_probs = np.stack(client_probs)   # type: ignore[attr-defined]
    result._client_models = client_models           # type: ignore[attr-defined]

    # Aggregate: mean of per-client global-test metrics
    keys = ["accuracy", "auc_roc", "f1_binary", "recall_binary", "false_positive_rate"]
    result.final = {k: float(np.mean([pc[k] for pc in result.per_client])) for k in keys}
    result.final["experiment"] = "isolated"
    result.final["mode"] = mode
    return result


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_result(
    result: RunResult,
    paths_config: dict[str, Any],
    suffix: str = "",
    seed: int | None = None,
    test_probs: np.ndarray | None = None,
) -> Path:
    metrics_dir = resolve_path(paths_config["results"]["metrics"])
    models_dir = resolve_path(paths_config["results"]["models"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    tag = f"fl_{result.experiment}_{result.mode}{suffix}"
    out = metrics_dir / f"{tag}.json"
    payload = {
        "experiment": result.experiment,
        "mode": result.mode,
        "rounds": result.rounds,
        "seed": seed,
        "history": result.history,
        "final": result.final,
        "per_client": result.per_client,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Test-set scores are kept so PR curves, confusion matrices, threshold sweeps and
    # time-to-detect can all be recomputed later without retraining.
    if test_probs is not None:
        np.savez_compressed(metrics_dir / f"{tag}_probs.npz", probs=test_probs.astype(np.float32))

    model = getattr(result, "_model", None)
    if model is not None:
        model.save(models_dir / f"{tag}_global.h5")

    # isolated arm: one set of scores and one model per client, no global model
    client_probs = getattr(result, "_client_probs", None)
    if client_probs is not None:
        np.savez_compressed(metrics_dir / f"{tag}_client_probs.npz",
                            probs=np.asarray(client_probs, dtype=np.float32))
    for i, m in enumerate(getattr(result, "_client_models", []) or []):
        m.save(models_dir / f"{tag}_client{i}.h5")
    LOGGER.info("Saved %s", out)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/fl.yaml")
    parser.add_argument("--model", required=True, help="Path to configs/model.yaml")
    parser.add_argument("--paths", required=True, help="Path to configs/paths.yaml")
    parser.add_argument("--mode", default="iid", choices=["iid", "dirichlet"])
    parser.add_argument("--experiment", default="fedavg", choices=["fedavg", "isolated", "poison"])
    parser.add_argument("--rounds", type=int, default=None, help="Override num_rounds")
    parser.add_argument("--local-epochs", type=int, default=None, help="Override local epochs per round")
    parser.add_argument("--batch", type=int, default=None, help="Override local batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override local learning rate")
    parser.add_argument("--poison-clients", type=int, default=1, help="Malicious clients (poison experiment)")
    parser.add_argument("--clip", action="store_true", help="Enable SafeFedAvg gradient-norm clipping")
    parser.add_argument("--clip-factor", type=float, default=None,
                        help="Norm-filter multiplier kappa (default 2.5)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Dirichlet concentration for --mode dirichlet (default 0.5)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed (for repeated runs)")
    parser.add_argument("--tag", default=None, help="Suffix appended to the output filename")
    parser.add_argument("--clients", type=int, default=None, help="Override number of clients (scalability study)")
    parser.add_argument("--smoke", action="store_true", help="1-round self-check on tiny subsets")
    return parser


def main() -> None:
    from src.utils.logger import configure_logging

    configure_logging()
    args = build_parser().parse_args()
    fl_config = load_yaml(args.config)
    model_config = load_yaml(args.model)
    paths_config = load_yaml(args.paths)
    seed = int(args.seed if args.seed is not None else fl_config.get("random_seed", 42))
    set_global_seed(seed)

    if args.local_epochs is not None:
        fl_config["federation"]["local_epochs"] = args.local_epochs
    if args.batch is not None:
        fl_config["federation"]["local_batch_size"] = args.batch
    if args.lr is not None:
        fl_config["federation"]["learning_rate"] = args.lr

    if args.clients is not None:
        fl_config["federation"]["num_clients"] = args.clients
    if args.clip_factor is not None:
        fl_config["clip_factor"] = args.clip_factor
    alpha = float(args.alpha if args.alpha is not None
                  else fl_config.get("partitioning", {}).get("dirichlet_alpha", 0.5))
    num_clients = int(fl_config["federation"]["num_clients"])
    rounds = int(args.rounds if args.rounds is not None else fl_config["federation"]["num_rounds"])

    clients = load_clients(paths_config, args.mode, num_clients, alpha)
    global_splits = load_global_splits(paths_config)

    if args.smoke:
        rounds = 1
        clients = [ClientData(c.client_id, c.X[:2000], c.y[:2000]) for c in clients]
        Xte, yte = global_splits["test"]
        Xva, yva = global_splits["val"]
        global_splits = {"test": (Xte[:2000], yte[:2000]), "val": (Xva[:2000], yva[:2000])}

    if args.experiment == "isolated":
        result = run_isolated(clients, global_splits, fl_config, model_config, args.mode, rounds)
    elif args.experiment == "poison":
        result = run_fedavg(clients, global_splits, fl_config, model_config, args.mode, rounds,
                            poison_clients=int(args.poison_clients), clip=bool(args.clip))
    else:
        result = run_fedavg(clients, global_splits, fl_config, model_config, args.mode, rounds)

    probs = None
    model = getattr(result, "_model", None)
    if model is not None:
        probs = model.predict(global_splits["test"][0], batch_size=1024, verbose=0)[:, 1]
    save_result(result, paths_config, suffix=(f"_{args.tag}" if args.tag else ""), seed=seed, test_probs=probs)

    if args.smoke:
        assert result.final, "final metrics missing"
        assert "auc_roc" in result.final, "auc missing from final metrics"
        LOGGER.info("SMOKE OK — final: %s", json.dumps(result.final))


if __name__ == "__main__":
    main()
