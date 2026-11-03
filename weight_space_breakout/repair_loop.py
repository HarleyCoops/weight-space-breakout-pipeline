"""Data-only policy repair and promotion checks shared by the hosted verifier.

No model call or publishing happens here. The verifier, not the editor, decides
whether an edit passes the predeclared promotion rule.
"""
import hashlib
import json
import numpy as np
from . import __version__
from .evaluation import evaluate, rollout
from .policy import Policy, SHAPES, apply_edits, faulty_policy, parse_submission


def fingerprint(policy):
    digest = hashlib.sha256()
    for name in SHAPES:
        digest.update(name.encode())
        digest.update(policy.weights[name].astype("<f4").tobytes())
    return digest.hexdigest()


def from_sparse(text):
    if not isinstance(text, str) or len(text) > 2_000_000:
        raise ValueError("Invalid policy snapshot")
    zero = Policy({name: np.zeros(shape, np.float32) for name, shape in SHAPES.items()})
    return apply_edits(zero, {"edits": json.loads(text)}, max_edits=17253)


def comparison(before, after, seeds):
    base, candidate = evaluate(before, seeds), evaluate(after, seeds)
    delta = np.array(candidate["returns"]) - base["returns"]
    se = float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0
    return {"episodes": len(delta), "baseline_return": base["mean_return"],
            "candidate_return": candidate["mean_return"], "gain": float(delta.mean()),
            "paired_standard_error": se, "baseline_returns": base["returns"],
            "candidate_returns": candidate["returns"], "seed_start": int(seeds[0]),
            "seed_end": int(seeds[-1]), "candidate_clear_rate": candidate["clear_rate"]}


def passes_gate(report, changed):
    # A conservative engineering gate, not a claim of general statistical proof.
    return bool(changed > 0 and report["gain"] > 0.10
                and report["gain"] - 2.0 * report["paired_standard_error"] > 0)


def verify_round(text, baseline, round_index, final_round=False):
    if type(round_index) is not int or not 0 <= round_index < 20:
        raise ValueError("Round index must be in 0..19")
    report = {"round_index": round_index, "baseline_hash": fingerprint(baseline),
              "valid": False, "accepted": False, "reward": 0.0,
              "gate_rule": "changed>0 and gain>0.10 and gain-2*paired_SE>0",
              "verifier": f"weight-space-breakout@{__version__}"}
    candidate = baseline
    try:
        payload = parse_submission(text)
        candidate = apply_edits(baseline, payload)
        changed = sum(int(np.count_nonzero(candidate.weights[k] != baseline.weights[k])) for k in SHAPES)
        gate = comparison(baseline, candidate, tuple(range(60000+512*round_index, 60512+512*round_index)))
        accepted = passes_gate(gate, changed)
        report.update(valid=True, accepted=accepted, reward=float(accepted),
                      candidate_hash=fingerprint(candidate), changed_parameters=changed,
                      edits=payload, gate=gate)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        report["error"] = str(exc)
    selected = candidate if report["accepted"] else baseline
    report["selected_hash"] = fingerprint(selected)
    # This final audit is never consulted for promotion or sent to a later editor.
    if final_round:
        report["final_audit"] = comparison(faulty_policy("reversed_controls"), selected, tuple(range(90000, 91024)))
    report["selected_episode"] = rollout(selected, 314159, record=True)
    return report


def practice_feedback(policy, round_index):
    seeds = tuple(range(40000+32*round_index, 40032+32*round_index))
    stats = evaluate(policy, seeds)
    # Describe observed failure trajectories; do not supply a reference solution.
    episodes = []
    for seed in seeds[:3]:
        episode = rollout(policy, seed, record=True)
        frames = []
        for f in episode["frames"][-8:]:
            grid = np.asarray(f["observation"]).reshape(10, 10, 5)
            frames.append({"step": f["step"], "action": f["action"],
                           "ball": np.argwhere(grid[:, :, 1]).tolist(),
                           "paddle": np.argwhere(grid[:, :, 0]).tolist(), "reward": f["reward"]})
        episodes.append({"seed": seed, "return": episode["return"], "reason": episode["reason"], "last_steps": frames})
    return {"mean_return": stats["mean_return"], "episodes": 32, "examples": episodes}
