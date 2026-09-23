"""Local mitigation agent.

Closes the loop between detection and enforcement (the "mitigation" in the thesis
title). The detector emits a per-flow malicious probability; this agent applies a
policy and enforces an action — block or rate-limit — on the offending source.

Policy design choices:
  * consecutive_hits: require k consecutive above-threshold flows before acting, so a
    single false positive never blocks a legitimate source (precision guard).
  * cooldown_flows: auto-release the block after N quiet flows (avoids permanent
    lockout; a real deployment would use a wall-clock timer).

Time-to-Detect (TTD) is measured as the number of attack flows between attack onset
and the first enforced block. Multiplying by the measured per-flow inference latency
gives the compute-bound detection delay in milliseconds.

Run:  python -m src.mitigation.agent --model configs/model.yaml --paths configs/paths.yaml
Self-check only:  python -m src.mitigation.agent --demo
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MitigationPolicy:
    threshold: float = 0.5
    window: int = 10  # sliding window of recent flows
    min_hits: int = 3  # block when >= min_hits of the last `window` flows are flagged
    action: str = "block"  # "block" | "rate_limit"
    cooldown_flows: int = 100


class MitigationAgent:
    """Stateful per-source enforcement. One agent instance guards one source.

    Uses a k-of-n sliding window (min_hits of the last `window` flows above
    threshold) rather than strict consecutive hits — this tolerates the detector's
    per-flow misses while still requiring sustained evidence before blocking.
    """

    def __init__(self, policy: MitigationPolicy) -> None:
        from collections import deque

        self.policy = policy
        self.recent: deque[int] = deque(maxlen=policy.window)
        self.blocked = False
        self._since_block = 0

    def observe(self, score: float) -> str:
        """Feed one flow's malicious score; return the action for this flow."""
        if self.blocked:
            self._since_block += 1
            if self._since_block >= self.policy.cooldown_flows:
                self.blocked = False
                self.recent.clear()
                self._since_block = 0
                return "released"
            return self.policy.action  # still enforcing

        self.recent.append(1 if score >= self.policy.threshold else 0)
        if sum(self.recent) >= self.policy.min_hits:
            self.blocked = True
            self._since_block = 0
            return self.policy.action  # first enforcement
        return "allow"


def simulate_scenario(scores: np.ndarray, attack_start: int, policy: MitigationPolicy) -> dict:
    """Feed an ordered score stream (benign prefix then attack) through one agent.

    Returns time-to-detect (attack flows until first block) and whether the benign
    prefix caused a false block.
    """
    agent = MitigationAgent(policy)
    first_block_idx: int | None = None
    false_block = False
    enforcing_actions = {policy.action}
    for i, s in enumerate(scores):
        action = agent.observe(float(s))
        if action in enforcing_actions and first_block_idx is None:
            first_block_idx = i
            if i < attack_start:
                false_block = True
            break
    ttd_flows = None if first_block_idx is None else max(0, first_block_idx - attack_start)
    return {
        "detected": first_block_idx is not None and not false_block,
        "ttd_flows": ttd_flows,
        "false_block": false_block,
        "block_index": first_block_idx,
    }


def run_real_scenarios(
    model_scores_benign: np.ndarray,
    model_scores_attack: np.ndarray,
    policy: MitigationPolicy,
    per_flow_latency_ms: float,
    n_scenarios: int = 100,
    benign_prefix: int = 200,
    attack_len: int = 100,
    seed: int = 42,
) -> dict:
    """Build n scenarios (benign prefix + attack burst) from real model scores."""
    rng = np.random.default_rng(seed)
    ttds, detected, false_blocks = [], 0, 0
    for _ in range(n_scenarios):
        pre = rng.choice(model_scores_benign, size=benign_prefix, replace=True)
        atk = rng.choice(model_scores_attack, size=attack_len, replace=True)
        stream = np.concatenate([pre, atk])
        r = simulate_scenario(stream, attack_start=benign_prefix, policy=policy)
        if r["false_block"]:
            false_blocks += 1
        if r["detected"] and r["ttd_flows"] is not None:
            detected += 1
            ttds.append(r["ttd_flows"])
    ttd_mean = float(np.mean(ttds)) if ttds else float("nan")
    # The TTD distribution is right-skewed (a few scenarios need many flows), so the
    # median is reported alongside the mean rather than the mean alone.
    ttd_median = float(np.median(ttds)) if ttds else float("nan")
    # The reviewer asks for three distinct proportions rather than one: a scenario can fail
    # because the benign prelude triggered a block, or because the attack was never blocked.
    clean = n_scenarios - false_blocks
    return {
        "policy": {
            "threshold": policy.threshold,
            "window": policy.window,
            "min_hits": policy.min_hits,
            "action": policy.action,
            "cooldown_flows": policy.cooldown_flows,
        },
        "scenarios": n_scenarios,
        "benign_prefix": benign_prefix,
        "attack_len": attack_len,
        "scenario_seed": seed,
        "false_blocks": false_blocks,
        "detected_count": detected,
        "clean_scenarios": clean,
        # detection among scenarios that survived the benign prelude
        "conditional_detection_rate": (detected / clean) if clean else float("nan"),
        "detection_rate": detected / n_scenarios,
        "false_block_rate": false_blocks / n_scenarios,
        "ttd_flows_mean": ttd_mean,
        "ttd_flows_median": ttd_median,
        "ttd_flows_p90": float(np.percentile(ttds, 90)) if ttds else float("nan"),
        "ttd_ms_mean": ttd_mean * per_flow_latency_ms if ttds else float("nan"),
        "ttd_ms_median": ttd_median * per_flow_latency_ms if ttds else float("nan"),
        "per_flow_latency_ms": per_flow_latency_ms,
    }


def demo() -> None:
    """Self-check: benign traffic must not block; a sustained attack must block."""
    policy = MitigationPolicy(threshold=0.5, window=10, min_hits=3, cooldown_flows=50)

    # All-benign stream (low scores) -> never blocks
    benign = np.full(500, 0.05)
    r = simulate_scenario(benign, attack_start=10_000, policy=policy)
    assert not r["detected"] and r["block_index"] is None, f"blocked benign traffic: {r}"

    # Benign prefix then attack burst (high scores) -> blocks shortly after onset
    stream = np.concatenate([np.full(200, 0.05), np.full(50, 0.95)])
    r = simulate_scenario(stream, attack_start=200, policy=policy)
    assert r["detected"] and not r["false_block"], f"missed attack: {r}"
    assert r["ttd_flows"] == policy.min_hits - 1, f"unexpected TTD: {r}"  # 3rd hit triggers

    # Single false positive inside benign must not block (needs 3 consecutive)
    spike = np.full(300, 0.05)
    spike[100] = 0.99
    r = simulate_scenario(spike, attack_start=10_000, policy=policy)
    assert not r["detected"], f"single FP blocked: {r}"

    print("MITIGATION DEMO OK — benign safe, attack blocked, TTD/precision-guard verified")


def main() -> None:
    from src.utils.config import load_yaml, resolve_path
    from src.utils.logger import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--paths")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo or not (args.model and args.paths):
        demo()
        return

    import tensorflow as tf

    from src.models.cnn import focal_loss
    from src.models.io import load_processed_split

    model_config = load_yaml(args.model)
    paths_config = load_yaml(args.paths)
    processed = resolve_path(paths_config["data"]["processed"])
    models_dir = resolve_path(paths_config["results"]["metrics"]).parent / "models"

    X_test, y_test, _ = load_processed_split(processed / "test.npz")
    cnn = tf.keras.models.load_model(
        models_dir / model_config["training"]["checkpoint_name"],
        custom_objects={"focal_loss": focal_loss()},
    )
    scores = cnn.predict(X_test, batch_size=1024, verbose=0)[:, 1]

    # Per-flow latency from the Phase 2 benchmark, if available
    latency_path = resolve_path(paths_config["results"]["metrics"]) / "latency_centralized.json"
    per_flow_ms = 0.013
    if latency_path.exists():
        per_flow_ms = float(json.loads(latency_path.read_text())["mean_ms"])

    # Threshold from the CNN's evaluation decision threshold, if available
    threshold = 0.5
    results_path = resolve_path(paths_config["results"]["metrics"]) / "centralized_results.json"
    if results_path.exists():
        threshold = float(json.loads(results_path.read_text()).get("decision_threshold", 0.5))

    policy = MitigationPolicy(threshold=threshold, window=10, min_hits=3, action="block", cooldown_flows=100)
    stats = run_real_scenarios(
        model_scores_benign=scores[y_test == 0],
        model_scores_attack=scores[y_test == 1],
        policy=policy,
        per_flow_latency_ms=per_flow_ms,
    )
    out = resolve_path(paths_config["results"]["metrics"]) / "mitigation_results.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    LOGGER.info("Mitigation: detect_rate=%.2f false_block_rate=%.2f TTD=%.1f flows (~%.3f ms) -> %s",
                stats["detection_rate"], stats["false_block_rate"],
                stats["ttd_flows_mean"], stats["ttd_ms_mean"], out)


if __name__ == "__main__":
    main()
