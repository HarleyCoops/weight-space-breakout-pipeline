"""Export the preselected successful repair, with its complete experiment record."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weight_space_breakout.game import Breakout
from weight_space_breakout.policy import Policy, faulty_policy

RUN = ROOT / "artifacts/llm-repair-run"
OUT = ROOT / "artifacts/hf-model"
FAULT = "reversed_controls"


def main():
    results = json.loads((RUN / "results.json").read_text())
    plan = json.loads((RUN / "experiment-plan.json").read_text())
    r = results[FAULT]
    for split in ("published_test", "fresh_holdout"):
        if not r[split]["valid"] or r[split]["gain"] <= 0:
            raise RuntimeError(f"Export gate failed on {split}; do not call this a successful repair")
    source = RUN / f"{FAULT}-llm-policy.npz"
    if hashlib.sha256(source.read_bytes()).hexdigest() != r["policy_sha256"]:
        raise RuntimeError("Candidate changed after freezing")
    OUT.mkdir(exist_ok=True)
    policy = Policy.load(source)
    save_file(policy.weights, OUT / "model.safetensors", metadata={"format":"numpy", "source":"LLM scalar edits"})
    save_file(faulty_policy(FAULT).weights, OUT / "baseline.safetensors", metadata={"format":"numpy"})
    policy.save(OUT / "policy.npz")
    obs, _ = Breakout().reset(0)
    (OUT / "example_observation.json").write_text(json.dumps(obs.astype(int).tolist()))
    config = {"architecture":"tanh_mlp", "dimensions":[500,32,32,5], "parameters":17253,
              "dtype":"float32", "weight_orientation":"input_by_output", "observation_shape":[10,10,5],
              "flatten_order":"row_column_channel", "channels":["paddle","ball","bricks","walls","previous_paddle_or_ball"],
              "actions":["no-op","up","down","left","right"], "action_selection":"softmax sampling",
              "environment":"harleycooper/weight-space-breakout@0.1.0", "starting_policy":"handwritten reversed-controls teaching policy",
              "editor_model":plan["model"], "protocol":"one-shot, no tools, no feedback", "fault":FAULT}
    (OUT / "config.json").write_text(json.dumps(config,indent=2))
    for name in ("inference.py",):
        shutil.copy2(ROOT / "deployment" / name, OUT / name)
    shutil.copy2(ROOT / "LICENSE", OUT / "LICENSE")
    audit = OUT / "experiment"
    audit.mkdir(exist_ok=True)
    for name in ["experiment-plan.json","results.json","frozen-policy-hashes.json"]:
        shutil.copy2(RUN/name, audit/name)
    for fault in results:
        for suffix in ["prompt.txt","request.json","response.json","answer.txt"]:
            shutil.copy2(RUN/f"{fault}-{suffix}",audit/f"{fault}-{suffix}")
    spec = importlib.util.spec_from_file_location("export_inference", OUT / "inference.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    w = mod.load_weights(OUT / "model.safetensors")
    for seed in range(16):
        x,_=Breakout().reset(seed)
        np.testing.assert_allclose(mod.predict(w,x)["logits"],policy.logits(x),atol=1e-6)
    files = ["model.safetensors","baseline.safetensors","policy.npz","config.json","inference.py"]
    hashes = {n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in files}
    (OUT / "checksums.json").write_text(json.dumps(hashes,indent=2))
    card = (ROOT / "deployment/model_card.md").read_text(encoding="utf-8")
    (OUT / "README.md").write_text(card,encoding="utf-8")
    print(json.dumps({"folder":str(OUT),"hashes":hashes,"export_validation":"16 observations matched"},indent=2))


if __name__ == "__main__": main()
