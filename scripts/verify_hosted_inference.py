"""Test the actual public Space against the locally frozen model."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from gradio_client import Client
from huggingface_hub import HfApi

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from weight_space_breakout.game import Breakout
from weight_space_breakout.policy import Policy
from weight_space_breakout.evaluation import rollout

state=json.loads((ROOT/"artifacts/hf-publication.json").read_text())
modeldir=ROOT/"artifacts/hf-model"
spec=importlib.util.spec_from_file_location("export_inference",modeldir/"inference.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
weights=module.load_weights(modeldir/"model.safetensors")
client=Client(state["space_id"],verbose=False,download_files=False)
report={"space_id":state["space_id"],"model_id":state["model_id"],"revision":state["model_revision"],
        "prediction_checks":[],"episode_checks":[]}
for seed in [0,7,1234]:
    obs,_=Breakout().reset(seed)
    expected=module.predict(weights,obs,seed)
    actual=client.predict(json.dumps(obs.astype(int).tolist()),seed,api_name="/predict")
    assert actual["revision"]==state["model_revision"]
    for key in ("logits","probabilities"):
        np.testing.assert_allclose(actual[key],expected[key],atol=1e-5,rtol=1e-5)
    assert actual["sampled_action"]==expected["sampled_action"]
    report["prediction_checks"].append({"seed":seed,"match":True,**actual})
    print(f"Hosted prediction seed {seed}: matched",flush=True)
for seed in [0,7,1234]:
    _,actual=client.predict(seed,"LLM-repaired",api_name="/episode")
    local=rollout(Policy(weights),seed,record=True)
    assert actual["return"]==local["return"]
    assert actual["step"]==local["step"]
    assert actual["actions"]==[f["action"] for f in local["frames"][1:]]
    report["episode_checks"].append({"seed":seed,"match":True,**actual})
    print(f"Hosted episode seed {seed}: matched, return {actual['return']}",flush=True)
try:
    client.predict("[0]",0,api_name="/predict")
except Exception as exc:
    if "500 binary values" not in str(exc):
        raise
    report["invalid_observation_rejected"]=True
else:
    raise AssertionError("Malformed input was accepted")
runtime=HfApi().get_space_runtime(state["space_id"])
report.update(space_stage=runtime.stage,hardware=runtime.hardware,status="passed")
(ROOT/"artifacts/hosted-inference-verification.json").write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ("prediction_checks","episode_checks")},indent=2))
