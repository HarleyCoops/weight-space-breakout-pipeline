"""Public CPU inference service: actual forward passes against pinned Hub weights."""
import hashlib
import json
from pathlib import Path
import secrets
import tempfile

import gradio as gr
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image, ImageDraw

from game import Breakout
from inference import load_weights, predict

ROOT = Path(__file__).parent
DEPLOYMENT = json.loads((ROOT / "deployment.json").read_text())
MODEL_ID = DEPLOYMENT["model_id"]
REVISION = DEPLOYMENT["revision"]


def download(name):
    path = hf_hub_download(MODEL_ID, name, revision=REVISION, token=False)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != DEPLOYMENT["sha256"][name]:
        raise RuntimeError("Checkpoint checksum mismatch")
    return load_weights(path)


WEIGHTS = {"LLM-repaired": download("model.safetensors"),
           "Faulty baseline": download("baseline.safetensors")}
EXAMPLE = json.loads((ROOT / "example_observation.json").read_text())


def infer(observation_json, sample_seed=0):
    if not isinstance(observation_json, str) or len(observation_json) > 12000:
        raise gr.Error("Observation must be JSON text, at most 12,000 characters")
    try:
        obs = json.loads(observation_json)
        result = predict(WEIGHTS["LLM-repaired"], obs, sample_seed)
    except (ValueError, TypeError, OverflowError) as exc:
        raise gr.Error(str(exc)) from exc
    return {"model_id": MODEL_ID, "revision": REVISION,
            "weights_sha256": DEPLOYMENT["sha256"]["model.safetensors"], **result}


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


def play_episode(seed=0, policy="LLM-repaired"):
    if type(seed) is not int or not 0 <= seed <= 2**32-1:
        raise gr.Error("Seed must be an integer in [0, 2^32-1]")
    if policy not in WEIGHTS:
        raise gr.Error("Unknown policy")
    game = Breakout()
    obs, info = game.reset(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed,73021]))
    images = [frame(obs,info,policy)]
    actions = []
    while not (game.terminated or game.truncated):
        result = predict(WEIGHTS[policy], obs)
        action = int(rng.choice(5, p=np.asarray(result["probabilities"], dtype=np.float32)))
        actions.append(action)
        obs, _, _, _, info = game.step(action)
        images.append(frame(obs,info,policy))
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        path = f.name
    images[0].save(path, save_all=True, append_images=images[1:], duration=130, loop=0)
    return path, {"model_id": MODEL_ID, "revision": REVISION, "policy": policy,
                  "seed": seed, **info, "actions": actions}


def play_random_episode(policy="LLM-repaired"):
    seed = secrets.randbelow(2**32)
    path, result = play_episode(seed, policy)
    return path, result, seed


with gr.Blocks(title="Weight Space Breakout · LLM-repaired policy", delete_cache=(3600,3600)) as demo:
    gr.Markdown(f"""# Weight Space Breakout
Can a language model improve a game-playing network simply by reading and editing its weights?
This teaching experiment explores that question with a small Breakout policy. **One of three
LLM-generated repairs improved performance; the other two made it worse.** This Space runs the
successful candidate on a hosted CPU, so you do not need a local GPU to try it. This is an
**independent conceptual replication attempt and teaching adaptation** of the weight-space
programming idea; the original work is credited below.

## Three pieces, three different jobs

| Piece | What it does | Where it lives |
| --- | --- | --- |
| **Breakout environment and verifier** | Defines the game rules, supplies faulty policies, accepts proposed weight edits, and measures their effect on game performance. The verifier is scoring code, not another LLM. | [Published on Prime Intellect](https://app.primeintellect.ai/dashboard/environments/harleycooper/weight-space-breakout) as an installable package with source, tests, and documentation. |
| **Editor LLM: Qwen** | Reads a faulty policy and proposes numerical changes in one response. It does not control the paddle during play. | `qwen/qwen3.6-35b-a3b`, run through **Prime-hosted inference** for the experiment. |
| **Edited game policy** | Turns the current grid observation into probabilities for five actions. This is the small network that actually plays the game. | [Weights and experiment records on Hugging Face](https://huggingface.co/{MODEL_ID}), with live inference in **this Space**. |

The game policy has **17,253 parameters**: 500 inputs, two hidden layers of 32 units with tanh
activations, and five action outputs. Its inputs describe the paddle, ball, bricks, walls, and
previous frame. Its outputs correspond to no-op, up, down, left, and right; up and down do
nothing in this Breakout environment. Actions are sampled from the policy's probabilities.

## Where the computation happened

The large Qwen model ran on **Prime Intellect's hosted infrastructure**. The evaluation games
ran **locally during our experiment**, using the tiny NumPy policy and simulator rather than
the large LLM. Those evaluations require little computing power and could also run on hosted
compute in a future experiment.

**This Space now runs the small policy and simulator on Hugging Face's CPU Basic hardware.**
Qwen is not running here. Publishing the environment to Prime created a reusable package;
it did not automatically launch training or a persistent game server.

## How we tested the edits

We began with a **handwritten teaching controller**, then deliberately damaged it in three
different ways. These starting policies were not PPO-trained. Qwen received the network
description, its nonzero weights, the game rules, and a practice performance summary.
It made one independent attempt per fault, with **no tools, simulation access, reference edits,
or gameplay feedback**. It returned scalar weight assignments as JSON.

We froze all submitted policies before scoring them. In addition to our environment's
128 evaluation seeds, we tested each candidate on **1,024 additional game seeds** specified
before generation and not included in its prompt. Each baseline and candidate used matched
environment and action-sampling seeds. We did not revise the candidates after seeing scores.

## What the experiment showed

The table reports **average episode return** on the 1,024 additional seeds. Return is the sum
of game rewards: +1 for each brick removed and −1 for missing the ball. It is not a win rate
or percentage. A game ends on a miss, after clearing all 24 bricks, or at the 200-step limit.

| Deliberate fault | Original average return | After LLM edits | Outcome |
| --- | ---: | ---: | --- |
| Reversed controls | 1.30 | **2.43** | Improved |
| Strong left bias | 0.66 | 0.37 | Worse |
| Missing right connection | 1.81 | 0.38 | Worse |

The successful candidate changed **five of 17,253 parameters**. It corrected the left/right
output signs, increased their strength, and strongly suppressed the three no-op actions.
We exported this policy from the **reversed-controls task, which was selected for export before
generation**, rather than choosing the best task after seeing the results.

The repaired policy still fell short of the intact handwritten reference controller, which
averaged **2.95** on the same additional seeds. Individual episodes can still end quickly
with a miss. All three submissions and their results, including the failures, are preserved
in the [model repository](https://huggingface.co/{MODEL_ID}/tree/main/experiment).

## What this does—and does not—establish

This is evidence that **an LLM-generated weight edit can measurably improve this particular
faulty policy without a training loop or gradient updates**. The two unsuccessful attempts
also show why proposed edits need to be evaluated rather than accepted on appearance alone.

Three simple tasks and one attempt per task do **not** establish reliable general-purpose
weight repair, improved PPO training, or a new ability learned by Qwen. We did not fine-tune
Qwen. Our simulator and deliberately damaged teaching policies differ from the original
research setup, so these scores do not reproduce or directly validate the authors' results.

## Credit and scope of this replication attempt

Credit for **Sparks of Weight-Space Programming: LLMs can natively read & write neural network
weights** goes to **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab** (MIT CSAIL,
November 2, 2026). See the [original article](https://rikiyat.github.io/blog/weight-space-programming/),
[Rikiya Takehi's original X post](https://x.com/rikiyatakehi/status/2117313742891327994), and
[released Experiment 1 logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game).

This project began as an attempt to replicate that work. The released **Experiment 1** asks an
LLM to identify a game from a trained network's weights; it is not a weight-repair task.
Our current experiment instead tests the broader idea of direct weight editing, using Qwen,
our own Breakout simulator, and handwritten policies with known faults. We reuse the published
500 → 32 → 32 → 5 interface, but **do not claim an exact reproduction of the original environment,
PPO training, or reported results**. The original research weights are not served in this demo
or included in our edited model repository. Our results above belong to this separate setup.

## Try the deployed policy

Choose **Play a fresh game** to draw a new random seed and compute a new episode on the server.
The chosen seed appears in the **Game seed** field and in the result. Choose **Replay this seed**
to reproduce it, or enter a seed yourself. Use the same seed with **Faulty baseline** and
**LLM-repaired** for a matched comparison. The same seed and policy deliberately produce the
same action sequence; a different seed can change the starting state and sampled actions,
although this simple policy can still behave similarly in different games.

**The animation loops a replay of one completed episode.** Each loop is not a new model run.
Use **Play a fresh game** for another game. It is not a prerecorded example from the research paper.

Choose **Inspect one decision** to send a grid observation and receive logits, action
probabilities, a sampled action, and a greedy action. The live API matched local predictions
on three observations and reproduced three complete game trajectories. That checks deployment
consistency; the larger evaluation above is the evidence for performance improvement.
""")
    with gr.Tab("Watch the policy play"):
        gr.Markdown("**Play a fresh game** picks a random seed. **Replay this seed** reproduces the selected game. The GIF loops one completed episode; it does not start a new game on each loop.")
        with gr.Row():
            seed = gr.Number(value=0, precision=0, label="Game seed (repeatable)")
            policy = gr.Dropdown(list(WEIGHTS), value="LLM-repaired", label="Policy")
        with gr.Row():
            fresh = gr.Button("Play a fresh game", variant="primary")
            play = gr.Button("Replay this seed")
        animation = gr.Image(label="Looping replay of this episode", type="filepath", format="gif")
        episode_result = gr.JSON(label="Episode result")
        play.click(play_episode, [seed,policy], [animation,episode_result], api_name="episode")
        fresh.click(play_random_episode, [policy], [animation,episode_result,seed], api_name="random_episode")
    with gr.Tab("Inspect one decision"):
        gr.Markdown("500 binary values → two tanh layers → five action probabilities. The same observation always produces the same probabilities. The action-sampling seed controls the sampled action; changing it may still select the same action when one has high probability. Greedy selection always takes the highest probability.")
        observation = gr.Textbox(value=json.dumps(EXAMPLE), lines=6, label="Observation JSON")
        sample_seed = gr.Number(value=0,precision=0,label="Action-sampling seed")
        button = gr.Button("Run inference",variant="primary")
        output = gr.JSON(label="Hosted model prediction")
        button.click(infer,[observation,sample_seed],output,api_name="predict")
    gr.Markdown(f"[Prime environment](https://app.primeintellect.ai/dashboard/environments/harleycooper/weight-space-breakout) · Pinned model revision: `{REVISION}`")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, max_size=16).launch()
