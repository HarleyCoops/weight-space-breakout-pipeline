"""Inference for a 17,253-parameter tanh policy. No Transformers or GPU needed."""
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

SHAPES = {"W1": (500, 32), "b1": (32,), "W2": (32, 32),
          "b2": (32,), "W3": (32, 5), "b3": (5,)}
ACTIONS = ["no-op", "up", "down", "left", "right"]


def load_weights(path):
    w = load_file(str(path))
    if set(w) != set(SHAPES):
        raise ValueError("Unexpected tensors")
    for name, shape in SHAPES.items():
        if w[name].shape != shape or not np.isfinite(w[name]).all():
            raise ValueError(f"Invalid {name}")
    return w


def predict(w, observation, sample_seed=0):
    x = np.asarray(observation, dtype=np.float32)
    if x.shape == (10, 10, 5):
        x = x.reshape(500)
    if x.shape != (500,) or not np.isfinite(x).all() or not np.isin(x, [0, 1]).all():
        raise ValueError("Provide 500 binary values, or an array of shape [10,10,5]")
    h1 = np.tanh(x @ w["W1"] + w["b1"])
    h2 = np.tanh(h1 @ w["W2"] + w["b2"])
    logits = h2 @ w["W3"] + w["b3"]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    if type(sample_seed) is not int or not 0 <= sample_seed <= 2**32-1:
        raise ValueError("sample_seed must be an integer in [0, 2^32-1]")
    action = int(np.random.default_rng(sample_seed).choice(5, p=p))
    return {"logits": logits.tolist(), "probabilities": p.tolist(),
            "sampled_action": action, "sampled_action_name": ACTIONS[action],
            "greedy_action": int(p.argmax()), "greedy_action_name": ACTIONS[int(p.argmax())]}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="model.safetensors")
    parser.add_argument("--observation", default="example_observation.json")
    args = parser.parse_args()
    print(json.dumps(predict(load_weights(args.weights), json.loads(Path(args.observation).read_text())), indent=2))
