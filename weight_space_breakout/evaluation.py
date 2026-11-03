import hashlib
import json
import numpy as np
from .game import Breakout
from .policy import apply_edits, faulty_policy, parse_submission, tracking_policy

PRACTICE_SEEDS = tuple(range(32))
TEST_SEEDS = tuple(range(10000, 10128))


def rollout(policy, seed=0, mode="sample", record=False, max_steps=200):
    game = Breakout(max_steps)
    obs, _ = game.reset(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 73021]))
    frames = []
    if record:
        frames.append({"observation": obs.astype(int).tolist(), "reward": 0,
                       "action": None, "probabilities": None, **game.info()})
    while not (game.terminated or game.truncated):
        probs = np.full(5, .2) if policy is None else policy.probabilities(obs)
        action = int(rng.choice(5, p=probs)) if mode == "sample" or policy is None else int(probs.argmax())
        obs, reward, _, _, info = game.step(action)
        if record:
            frames.append({"observation": obs.astype(int).tolist(), "reward": reward,
                           "action": action, "probabilities": probs.tolist(), **info})
    return {**game.info(), "seed": seed, "frames": frames}


def evaluate(policy, seeds=TEST_SEEDS, mode="sample"):
    if mode not in ("sample", "greedy"):
        raise ValueError("mode must be sample or greedy")
    episodes = [rollout(policy, int(s), mode) for s in seeds]
    if not episodes:
        raise ValueError("At least one episode required")
    returns = np.array([e["return"] for e in episodes])
    return {"episodes": len(episodes), "mode": mode,
            "mean_return": float(returns.mean()),
            "standard_error": float(returns.std(ddof=1) / np.sqrt(len(returns))) if len(returns) > 1 else 0.0,
            "mean_bricks": float(np.mean([24-e["bricks_remaining"] for e in episodes])),
            "clear_rate": float(np.mean([e["reason"] == "cleared" for e in episodes])),
            "mean_steps": float(np.mean([e["step"] for e in episodes])),
            "returns": returns.tolist(), "seeds": list(seeds)}


def score_submission(text, fault, seeds=TEST_SEEDS):
    base = faulty_policy(fault)
    try:
        edited = apply_edits(base, parse_submission(text))
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        return {"valid": False, "reward": 0.0, "error": str(exc)}
    before = evaluate(base, seeds)
    after = evaluate(edited, seeds)
    reference = evaluate(tracking_policy(), seeds)
    gain = after["mean_return"] - before["mean_return"]
    scale = max(1.0, reference["mean_return"] - before["mean_return"])
    delta = np.array(after["returns"]) - before["returns"]
    return {"valid": True, "reward": float(np.clip(gain / scale, 0, 1)),
            "gain": gain, "baseline_return": before["mean_return"],
            "edited_return": after["mean_return"], "reference_return": reference["mean_return"],
            "paired_gain_standard_error": float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else 0.,
            "changed_parameters": sum(int(np.count_nonzero(edited.weights[k] != base.weights[k])) for k in base.weights),
            "episodes": len(seeds), "seed_sha256": hashlib.sha256(json.dumps(list(seeds)).encode()).hexdigest()}
