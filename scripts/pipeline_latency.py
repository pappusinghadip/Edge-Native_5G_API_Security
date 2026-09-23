"""Round-5 review, critical item 6: time the enforcement path per flow, not only the model call.

Stages, each timed with perf_counter_ns on the Mac under the two-thread host configuration:
  parse    split one CICIoT2023 flow record (CSV line) and select the 10 model features
  scale    apply the saved min-max scaling to that one record
  infer    TensorFlow-Lite invocation (set input, invoke, read output)
  policy   update the k-of-n agent (k = 3, n = 10) with the score
  enforce  record a block in an in-memory blocklist when the policy fires (a stub, not a gateway call)
plus the end-to-end time of all five together. Not measured: packet capture and aggregation of packets
into flow records, network I/O, and a real gateway or firewall call.

Run: PYTHONPATH=. .venv/bin/python scripts/pipeline_latency.py
"""
from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

from src.mitigation.agent import MitigationAgent, MitigationPolicy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
M = Path("results/metrics")
FEATS = ["Header_Length", "Protocol Type", "Time_To_Live", "Rate", "ack_count", "syn_count",
         "Tot sum", "AVG", "IAT", "Number"]
WARMUP, RUNS, THREADS = 1_000, 10_000, 2


def records(path: Path, n: int) -> tuple[list[str], list[int]]:
    with path.open() as fh:
        header = fh.readline().strip().split(",")
        lines = [fh.readline() for _ in range(n)]
    return lines, [header.index(f) for f in FEATS]


atk, idx = records(ROOT / "CICIoT2023_CSV/DictionaryBruteForce/DictionaryBruteForce.pcap.csv", 6_000)
ben_file = next((ROOT / "CICIoT2023_CSV/Benign_Final").glob("*.csv"))
ben, idx_b = records(ben_file, 6_000)
assert idx == idx_b, "feature columns differ between files"
stream = [l for pair in zip(ben, atk) for l in pair][: WARMUP + RUNS]   # interleaved benign/attack

scaler = pickle.load(open(Path("data/processed/scaler.pkl"), "rb"))
smin, sscale = scaler.min_.astype(np.float32), scaler.scale_.astype(np.float32)
threshold = float(json.loads((M / "sensitivity_analysis.json").read_text())["threshold"])

interp = tf.lite.Interpreter(model_path="results/models/centralized_best.tflite", num_threads=THREADS)
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
agent = MitigationAgent(MitigationPolicy(threshold=threshold, window=10, min_hits=3, action="block",
                                         cooldown_flows=100))
blocklist: dict[str, int] = {}

stages = {k: [] for k in ("parse", "scale", "infer", "policy", "enforce", "end_to_end")}
ns = time.perf_counter_ns
for i, line in enumerate(stream):
    t0 = ns()
    cols = line.rstrip("\n").split(",")
    x = np.array([float(cols[j]) for j in idx], dtype=np.float32)
    t1 = ns()
    x = (x * sscale + smin).reshape(inp["shape"]).astype(inp["dtype"])
    t2 = ns()
    interp.set_tensor(inp["index"], x); interp.invoke()
    score = float(interp.get_tensor(out["index"])[0][-1])
    t3 = ns()
    action = agent.observe(score)
    t4 = ns()
    if action == "block":
        blocklist["source"] = i
    t5 = ns()
    if i >= WARMUP:
        for k, v in zip(("parse", "scale", "infer", "policy", "enforce", "end_to_end"),
                        (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0)):
            stages[k].append(v / 1e6)

def summ(v):
    a = np.array(v)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max())}

res = {"platform": "Apple M4 Pro, macOS, TensorFlow-Lite interpreter, two threads",
       "model": "centralized_best.tflite (same architecture and size as the deployed federated model)",
       "records": f"{RUNS} flow records interleaved benign/attack after {WARMUP} warm-up",
       "not_measured": ["packet capture and aggregation of packets into flow records", "network I/O",
                        "a real gateway or firewall call (enforcement is an in-memory stub)"],
       "stages": {k: summ(v) for k, v in stages.items()},
       "blocks_triggered": len(blocklist)}
(M / "pipeline_latency.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("wrote", M / "pipeline_latency.json")
for k, v in res["stages"].items():
    print(f"  {k:10} mean {v['mean_ms']:.4f} ms  median {v['median_ms']:.4f}  p99 {v['p99_ms']:.4f}")
