"""Public simulator for the versioned, verifier-gated policy repair loop."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import secrets
import tempfile
import gradio as gr
from huggingface_hub import HfApi, hf_hub_download
import numpy as np
from PIL import Image, ImageDraw
from game import Breakout
from inference import load_weights, predict

MODEL_ID = "HarleyCooper/weight-space-breakout-iterative"


def resolve_revision(revision):
    if revision in (None, "", "latest"):
        return HfApi(token=False).model_info(MODEL_ID).sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise gr.Error("Use latest or an exact 40-character model revision")
    return revision


@lru_cache(maxsize=12)
def checkpoint(revision):
    manifest = json.loads(Path(hf_hub_download(MODEL_ID, "manifest.json", revision=revision, token=False)).read_text())
    path = hf_hub_download(MODEL_ID, "model.safetensors", revision=revision, token=False)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != manifest["weights_file_sha256"]:
        raise RuntimeError("Published weight checksum mismatch")
    weights = load_weights(path)
    h = hashlib.sha256()
    for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
        h.update(name.encode())
        h.update(weights[name].astype("<f4").tobytes())
    if h.hexdigest() != manifest["policy_hash"]:
        raise RuntimeError("Policy fingerprint mismatch")
    return weights, manifest


def status_text():
    revision = resolve_revision("latest")
    _, manifest = checkpoint(revision)
    rows = ["| Round | Before | Candidate | Decision |", "| --- | ---: | ---: | --- |"]
    for item in manifest["history"]:
        feedback = item["feedback"]
        if "gain" in feedback:
            rows.append(f"| {item['round']+1} | {feedback['baseline_return']:.3f} | {feedback['candidate_return']:.3f} | {'Accepted' if item['accepted'] else 'Rejected; incumbent retained'} |")
        else:
            reason = {"turn_limit_without_submission":"No submission: turn limit", "token_limit_without_submission":"No submission: token limit"}.get(feedback.get("outcome"),"Invalid submission")
            rows.append(f"| {item['round']+1} | — | {reason} | Rejected; incumbent retained |")
    text = f"### Current published policy\n\n**{manifest['label']}** · {len(manifest['history'])} completed repair rounds.\n\n"
    text += "\n".join(rows) if manifest["history"] else "The deliberately faulty starting policy is published. No repair has been promoted yet."
    text += f"\n\n[View model and full audit](https://huggingface.co/{MODEL_ID}) · [Pipeline source](https://github.com/HarleyCoops/weight-space-breakout-pipeline)\n\nCurrent model revision: `{revision}`"
    if manifest.get("run_plan",{}).get("protocol_id") == "failure-driven-v1":
        text += "\n\nProtocol: failure-driven diagnosis with tools. The editor targets a specific recorded miss; acceptance requires both a successful controlled replay and broader improvement."
    else:
        text += "\n\nProtocol: initial experiment, one response per round with no tool calls."
    if manifest.get("final_audit"):
        audit = manifest["final_audit"]
        text += f"\n\nFinal independent audit ({audit['episodes']} games): initial faulty policy **{audit['baseline_return']:.3f}**, final selected policy **{audit['candidate_return']:.3f}**. This audit was not used to choose edits."
        if not any(item["accepted"] for item in manifest["history"]):
            text += " No edit was promoted in this run; the live weights remain the initial faulty policy."
    return text


def frame(obs, info, label):
    image = Image.new("RGB", (400, 450), "#0b1220")
    draw = ImageDraw.Draw(image)
    grid = obs.reshape(10, 10, 5)
    for r in range(10):
        for c in range(10):
            x, y = 40*c, 40*r
            v = grid[r,c]
            draw.rectangle((x,y,x+39,y+39), outline="#24344b")
            if v[3]: draw.rectangle((x+2,y+2,x+37,y+37), fill="#34465e")
            if v[2]: draw.rectangle((x+3,y+6,x+36,y+33), fill="#ffc66d")
            if v[4]: draw.rectangle((x+12,y+12,x+27,y+27), outline="#8c71db")
            if v[0]: draw.rectangle((x+2,y+26,x+37,y+35), fill="#76aaff")
            if v[1]: draw.ellipse((x+14,y+12,x+26,y+24), fill="white")
    draw.text((10,408), f"{label} | step {info['step']} | return {info['return']:.0f}", fill="white")
    draw.text((10,428), f"{info['bricks_remaining']} bricks | {info['reason']}", fill="#6de5bd")
    return image


def episode(seed=0, revision="latest"):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise gr.Error("Seed must be an integer in [0, 2^32-1]")
    revision = resolve_revision(revision)
    weights, manifest = checkpoint(revision)
    game = Breakout()
    obs, info = game.reset(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed,73021]))
    images, actions = [frame(obs, info, "Published policy")], []
    while not (game.terminated or game.truncated):
        result = predict(weights, obs)
        action = int(rng.choice(5, p=np.asarray(result["probabilities"], dtype=np.float32)))
        actions.append(action)
        obs, _, _, _, info = game.step(action)
        images.append(frame(obs, info, "Published policy"))
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        path = f.name
    images[0].save(path, save_all=True, append_images=images[1:], duration=130, loop=0)
    return path, {"model_id": MODEL_ID, "revision": revision, "policy_hash": manifest["policy_hash"],
                  "policy_label": manifest["label"], "seed": seed, **info, "actions": actions}


def fresh():
    seed = secrets.randbelow(2**32)
    path, report = episode(seed, "latest")
    return path, report, seed, report["revision"], status_text()


with gr.Blocks(title="Breakout · Hosted Weight Repair Loop", delete_cache=(3600,3600)) as demo:
    gr.Markdown("""# Breakout: Hosted Weight Repair Loop

**Faulty weights → Prime-hosted games and verifier → Qwen edit → fresh evaluation →
accepted weights on Hugging Face → next simulation.**

This is a bounded experiment, with three editor attempts in its first run. Every attempt
is recorded, and edits that fail the improvement gate are rejected. The next attempt starts
from the most recently accepted policy. The editor receives failure examples and prior-round
feedback; this is a different protocol from our earlier one-shot experiment.
The protocol and budget are recorded with the published checkpoint. The initial run
used one response per round with no tool calls. The failure-driven extension allows
tools and asks the editor to diagnose and prevent a specific recorded miss.

**Playing here does not rewrite weights or call the editor.** This page runs the policy
published by the pipeline. **Play latest weights** fetches the newest published checkpoint
and chooses a new seed. **Replay pinned version** uses the exact revision and seed shown.
The GIF loops one computed game. The game and small network run on Hugging Face CPU hardware.

The policies are deliberately faulty handwritten teaching controllers, not language models.
An edit that helps this game does not establish that direct editing improves Cree or Dakota.

Independent conceptual replication attempt inspired by
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab.
[Original X post](https://x.com/rikiyatakehi/status/2117313742891327994) ·
[Prime environment](https://app.primeintellect.ai/dashboard/environments/harleycooper/weight-space-breakout).
Our simulator, faults, and results are independent of the original research setup.
""")
    status = gr.Markdown("Loading published run status…")
    refresh = gr.Button("Refresh pipeline results")
    with gr.Row():
        seed = gr.Number(value=0, precision=0, label="Game seed")
        revision = gr.Textbox(value="latest", label="Pinned model revision (or latest)")
    with gr.Row():
        play = gr.Button("Play latest weights", variant="primary")
        replay = gr.Button("Replay pinned version")
    animation = gr.Image(type="filepath", format="gif", label="Looping replay of this episode")
    report = gr.JSON(label="Episode and checkpoint provenance")
    play.click(fresh, [], [animation,report,seed,revision,status], api_name="fresh")
    replay.click(episode, [seed,revision], [animation,report], api_name="episode")
    refresh.click(status_text, [], status, api_name="status")
    demo.load(status_text, [], status, api_name=False)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, max_size=16).launch()
