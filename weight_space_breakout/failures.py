"""Observed failures and controlled replays; no assumed fault label or gold edits."""
import json
import numpy as np
from .game import Breakout, ACTIONS
from .evaluation import rollout
from .policy import parse_submission


def observation_summary(observation):
    x = np.asarray(observation).reshape(10, 10, 5)
    return {"ball": np.argwhere(x[:, :, 1]).tolist(),
            "paddle": np.argwhere(x[:, :, 0]).tolist(),
            "bricks_remaining": int(x[:, :, 2].sum())}


def failure_case(policy, seed):
    """Record facts up to a miss, with pre-action inputs correctly aligned."""
    from .repair_loop import fingerprint
    episode = rollout(policy, seed, record=True)
    if episode["reason"] != "miss":
        return None
    frames = episode["frames"]
    # Rewind to just after the last successful paddle encounter (or initial state).
    # From there the ball path is independent of paddle actions until this miss.
    intervention_step = 0
    for f in frames[:-1]:
        if observation_summary(f["observation"])["ball"][0][0] == 8:
            intervention_step = f["step"]
    transitions = []
    for before, after in zip(frames, frames[1:]):
        transition = {"step": before["step"],
                      "before": observation_summary(before["observation"]),
                      "action": after["action"], "action_name": ACTIONS[after["action"]],
                      "action_probabilities": after["probabilities"],
                      "after": observation_summary(after["observation"]),
                      "reward": after["reward"], "outcome": after["reason"]}
        if before["step"] >= max(0, episode["step"]-12):
            # Exact input and activations for the final decisions; full trajectory above.
            x = np.asarray(before["observation"], dtype=np.float32)
            w = policy.weights
            h1 = np.tanh(x @ w["W1"] + w["b1"])
            h2 = np.tanh(h1 @ w["W2"] + w["b2"])
            transition.update(input_ones=np.flatnonzero(x).tolist(),
                              h1=h1.tolist(),h2=h2.tolist(),logits=policy.logits(x).tolist())
        transitions.append(transition)
    case = {"schema":"observed-miss-v1", "policy_hash":fingerprint(policy),
            "seed":seed, "failure_step":episode["step"], "return":episode["return"],
            "reason":"miss", "intervention_step":intervention_step,
            "prefix_actions":[f["action"] for f in frames[1:intervention_step+1]],
            "trajectory":transitions,
            "input_encoding":"input_ones lists the 1-valued indices of the exact 500-element pre-action observation; all other inputs are 0",
            "interpretation":"A miss is observed. Its cause is unknown; no particular weight is labelled erroneous."}
    case["case_id"] = case_digest(case)
    return case


def case_digest(case):
    import hashlib
    data = {k:v for k,v in case.items() if k != "case_id"}
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def find_failure(policy, round_index, preferred_seed=None):
    """Prefer the just-deployed episode; otherwise inspect at most 32 practice games."""
    seeds = [] if preferred_seed is None else [preferred_seed]
    seeds += list(range(1040000+32*round_index,1040032+32*round_index))
    for seed in dict.fromkeys(seeds):
        case = failure_case(policy,seed)
        if case is not None:
            return case
    raise ValueError("No observed miss in the bounded practice search; do not invent a fault")


def parse_failure_submission(text):
    value = parse_submission(text)
    if not isinstance(value,dict) or set(value) != {"diagnosis","expected_prevention","edits"}:
        raise ValueError("Expected diagnosis, expected_prevention, and edits")
    for key in ("diagnosis","expected_prevention"):
        if not isinstance(value[key],str) or not 1 <= len(value[key].strip()) <= 2000:
            raise ValueError(f"{key} must be a brief explanation (1..2000 characters)")
    return {"edits":value["edits"]}, {k:value[k] for k in ("diagnosis","expected_prevention")}


def replay_failure(policy, case):
    """Freeze the history and randomness, change only policy after the intervention."""
    if case.get("case_id") != case_digest(case):
        raise ValueError("Failure case checksum mismatch")
    if case.get("reason") != "miss" or not 0 <= case["intervention_step"] < case["failure_step"] <= 200:
        raise ValueError("Invalid failure replay boundaries")
    if len(case["prefix_actions"]) != case["intervention_step"]:
        raise ValueError("Incomplete frozen history")
    game = Breakout()
    obs, _ = game.reset(case["seed"])
    rng = np.random.default_rng(np.random.SeedSequence([case["seed"],73021]))
    # NumPy's categorical sampler draws one uniform variate per action. Burn the
    # original draws using the recorded distributions while forcing the prefix.
    for step,action in enumerate(case["prefix_actions"]):
        rng.choice(5,p=np.asarray(case["trajectory"][step]["action_probabilities"],dtype=np.float32))
        obs, *_ = game.step(action)
    actions = []
    while game.steps < case["failure_step"] and not (game.terminated or game.truncated):
        action = int(rng.choice(5,p=policy.probabilities(obs)))
        actions.append(action)
        obs, *_ = game.step(action)
    last = case["trajectory"][-1]
    same_contact = (game.steps == case["failure_step"] and [game.row,game.col] == last["after"]["ball"][0])
    caught = bool(same_contact and game.row == 8 and game.reason != "miss" and game.dy == -1)
    return {"case_id":case["case_id"],"seed":case["seed"],
            "intervention_step":case["intervention_step"],"failure_step":case["failure_step"],
            "prevented_recorded_miss":caught,"same_ball_contact":same_contact,
            "actions_after_intervention":actions,"ball":[game.row,game.col],
            "paddle_center":game.paddle,**game.info()}


def verify_failure_round(text, baseline, round_index, case, final_round=False):
    from . import __version__
    from .policy import apply_edits, SHAPES, faulty_policy
    from .repair_loop import fingerprint, comparison, passes_gate
    if type(round_index) is not int or not 0 <= round_index < 20:
        raise ValueError("Round index must be in 0..19")
    if case["policy_hash"] != fingerprint(baseline):
        raise ValueError("Failure belongs to different weights")
    before = replay_failure(baseline,case)
    if before["prevented_recorded_miss"] or before["reason"] != "miss" or not before["same_ball_contact"]:
        raise ValueError("Baseline did not reproduce the recorded miss")
    report = {"round_index":round_index,"protocol":"failure-driven-v1",
              "baseline_hash":fingerprint(baseline),"valid":False,"accepted":False,"reward":0.,
              "verifier":f"weight-space-breakout@{__version__}","failure_case":case,
              "baseline_replay":before,
              "gate_rule":"prevent recorded miss AND changed>0 AND gain>0.10 AND gain-2*paired_SE>0"}
    candidate = baseline
    try:
        edits, explanation = parse_failure_submission(text)
        candidate = apply_edits(baseline,edits)
        changed = sum(int(np.count_nonzero(candidate.weights[k] != baseline.weights[k])) for k in SHAPES)
        replay = replay_failure(candidate,case)
        gate = comparison(baseline,candidate,tuple(range(1060000+512*round_index,1060512+512*round_index)))
        accepted = bool(replay["prevented_recorded_miss"] and passes_gate(gate,changed))
        report.update(valid=True,accepted=accepted,reward=float(accepted),edits=edits,
                      explanation=explanation,candidate_hash=fingerprint(candidate),
                      changed_parameters=changed,candidate_replay=replay,gate=gate)
    except (ValueError,TypeError,OverflowError,RecursionError) as exc:
        report["error"] = str(exc)
    selected = candidate if report["accepted"] else baseline
    report["selected_hash"] = fingerprint(selected)
    report["selected_episode"] = rollout(selected,case["seed"],record=True)
    if final_round:
        report["final_audit"] = comparison(faulty_policy("reversed_controls"),selected,tuple(range(1090000,1091024)))
    return report
