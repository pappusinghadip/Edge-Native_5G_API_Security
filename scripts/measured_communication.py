"""Measured, rather than assumed, per-round communication volume.

The communication figures in Chapter 4 were derived from the nominal model size. This
measures what a round actually moves: the serialized byte length of the weight arrays
a client uploads and the global model it downloads, using the same float32 encoding
the transport layer would carry. Reported both uncompressed and gzip-compressed,
since a real deployment would compress.

Run: python scripts/measured_communication.py
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import numpy as np

from src.fl.simulate import make_model
from src.utils.config import load_yaml, resolve_path

M = Path("results/metrics")


def serialized_bytes(weights: list[np.ndarray]) -> tuple[int, int]:
    """Bytes on the wire for one weight set: raw float32 payload and gzipped."""
    buf = io.BytesIO()
    np.savez(buf, **{f"w{i}": w.astype(np.float32) for i, w in enumerate(weights)})
    raw = buf.getvalue()
    return len(raw), len(gzip.compress(raw, compresslevel=6))


def main() -> None:
    fl_cfg = load_yaml("configs/fl.yaml")
    model_cfg = load_yaml("configs/model.yaml")
    paths_cfg = load_yaml("configs/paths.yaml")

    model = make_model(fl_cfg, model_cfg)
    w = model.get_weights()
    params = int(sum(x.size for x in w))
    up_raw, up_gz = serialized_bytes(w)

    K = int(fl_cfg["federation"]["num_clients"])
    R = 20  # matched to the reported federated configuration

    # A round moves the global model down and the client update up, per client.
    per_client_round_raw = 2 * up_raw
    per_client_round_gz = 2 * up_gz

    # Centralized comparison: the raw feature payload each node would ship instead.
    processed = resolve_path(paths_cfg["data"]["processed"])
    tr = np.load(processed / "train.npz")
    X = tr["X"]
    per_node_records = X.shape[0] / K
    bytes_per_record = int(X.dtype.itemsize * np.prod(X.shape[1:])) + 1  # +1 byte label
    centralized_per_node_raw = per_node_records * bytes_per_record

    out = {
        "model": {
            "total_parameters": params,  # all variables: 173,826 trainable + 640 batch-norm moving statistics
            "serialized_float32_bytes": up_raw,
            "serialized_float32_mb": round(up_raw / 1e6, 4),
            "gzip_bytes": up_gz,
            "gzip_mb": round(up_gz / 1e6, 4),
            "gzip_ratio": round(up_raw / up_gz, 2),
        },
        "per_client_per_round": {
            "uncompressed_mb": round(per_client_round_raw / 1e6, 4),
            "gzip_mb": round(per_client_round_gz / 1e6, 4),
            "note": "global model down + client update up",
        },
        "experimental_run": {
            "clients": K, "rounds": R,
            "total_uncompressed_mb": round(per_client_round_raw * K * R / 1e6, 2),
            "total_gzip_mb": round(per_client_round_gz * K * R / 1e6, 2),
            "centralized_raw_upload_mb": round(centralized_per_node_raw * K / 1e6, 2),
        },
        "per_node_dataset": {
            "records": int(per_node_records),
            "bytes_per_record": bytes_per_record,
            "raw_upload_mb": round(centralized_per_node_raw / 1e6, 2),
        },
        "method": "np.savez float32 payload; gzip level 6. Measured from the actual "
                  "compiled model, not from an assumed parameter count.",
    }
    (M / "measured_communication.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("saved", M / "measured_communication.json")


if __name__ == "__main__":
    main()
