"""Publish only the validated export and stage a CPU-only Gradio Space."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "HarleyCooper/weight-space-breakout-llm-repaired"
SPACE_ID = "HarleyCooper/weight-space-breakout-inference"
STATE = ROOT / "artifacts/hf-publication.json"


def main():
    model = ROOT / "artifacts/hf-model"
    checksums = json.loads((model/"checksums.json").read_text())
    for name, digest in checksums.items():
        assert hashlib.sha256((model/name).read_bytes()).hexdigest() == digest
    api = HfApi()
    assert api.whoami()["name"] == "HarleyCooper"
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if "model_revision" not in state:
        # Do not overwrite an unrelated preexisting repository.
        if not state.get("model_created"):
            api.create_repo(MODEL_ID, repo_type="model", private=False)
            state["model_created"] = True
            STATE.write_text(json.dumps(state,indent=2))
        commit = api.upload_folder(repo_id=MODEL_ID, folder_path=model, ignore_patterns=["__pycache__/*","*.pyc"],
                                   commit_message="Publish one-shot LLM repair with frozen holdout evaluation")
        state.update(model_id=MODEL_ID, model_revision=commit.oid, model_url=f"https://huggingface.co/{MODEL_ID}")
        STATE.write_text(json.dumps(state,indent=2))
    path = hf_hub_download(MODEL_ID,"model.safetensors",revision=state["model_revision"],force_download=True,token=False)
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == checksums["model.safetensors"]
    space = ROOT / "artifacts/hf-space"
    space.mkdir(exist_ok=True)
    for name in ("app.py","inference.py"):
        shutil.copy2(ROOT/"deployment"/name,space/name)
    shutil.copy2(ROOT/"weight_space_breakout/game.py",space/"game.py")
    shutil.copy2(model/"example_observation.json",space/"example_observation.json")
    shutil.copy2(ROOT/"LICENSE",space/"LICENSE")
    manifest={"model_id":MODEL_ID,"revision":state["model_revision"],
              "sha256":{k:checksums[k] for k in ("model.safetensors","baseline.safetensors")}}
    (space/"deployment.json").write_text(json.dumps(manifest,indent=2))
    (space/"requirements.txt").write_text("numpy==2.4.1\nsafetensors==0.8.0\nhuggingface_hub==1.27.0\npillow==11.3.0\n")
    card = (ROOT / "deployment/space_card.md").read_text(encoding="utf-8")
    (space / "README.md").write_text(card.format(model_revision=state["model_revision"]), encoding="utf-8")
    if "--space" in sys.argv:
        if not state.get("space_created"):
            api.create_repo(SPACE_ID,repo_type="space",space_sdk="gradio",space_hardware="cpu-basic",private=False)
            state["space_created"]=True
            STATE.write_text(json.dumps(state,indent=2))
        commit=api.upload_folder(repo_id=SPACE_ID,repo_type="space",folder_path=space,ignore_patterns=["__pycache__/*","*.pyc"],
                                 commit_message="Deploy pinned LLM-repaired policy on CPU with prediction and episode APIs")
        state.update(space_id=SPACE_ID,space_revision=commit.oid,space_url=f"https://huggingface.co/spaces/{SPACE_ID}")
        STATE.write_text(json.dumps(state,indent=2))
    print(json.dumps(state,indent=2))


if __name__=="__main__": main()
