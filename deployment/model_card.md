---
license: mit
tags:
- reinforcement-learning
- breakout
- weight-space-programming
- numpy
- safetensors
model-index: []
---
# LLM-repaired Weight Space Breakout policy

This is the **separate model repository for our edited game policy**. `model.safetensors` and `policy.npz` contain the five-weight repair; `baseline.safetensors` preserves the faulty starting policy. It is a 17,253-parameter game controller, not a language model or a fine-tuned Qwen model.

Can a language model improve a game-playing network simply by reading and editing its weights?
This teaching experiment explores that question with a small Breakout policy. **One of three
LLM-generated repairs improved performance; the other two made it worse.** The [live HF Space](https://huggingface.co/spaces/HarleyCooper/weight-space-breakout-inference) runs the
successful candidate on a hosted CPU, so you do not need a local GPU to try it. This is an
**independent conceptual replication attempt and teaching adaptation** of the weight-space
programming idea; the original work is credited below.

## Three pieces, three different jobs

| Piece | What it does | Where it lives |
| --- | --- | --- |
| **Breakout environment and verifier** | Defines the game rules, supplies faulty policies, accepts proposed weight edits, and measures their effect on game performance. The verifier is scoring code, not another LLM. | [Published on Prime Intellect](https://app.primeintellect.ai/dashboard/environments/harleycooper/weight-space-breakout) as an installable package with source, tests, and documentation. |
| **Editor LLM: Qwen** | Reads a faulty policy and proposes numerical changes in one response. It does not control the paddle during play. | `qwen/qwen3.6-35b-a3b`, run through **Prime-hosted inference** for the experiment. |
| **Edited game policy** | Turns the current grid observation into probabilities for five actions. This is the small network that actually plays the game. | [Weights and experiment records on Hugging Face](https://huggingface.co/HarleyCooper/weight-space-breakout-llm-repaired), with live inference in the [live HF Space](https://huggingface.co/spaces/HarleyCooper/weight-space-breakout-inference). |

The game policy has **17,253 parameters**: 500 inputs, two hidden layers of 32 units with tanh
activations, and five action outputs. Its inputs describe the paddle, ball, bricks, walls, and
previous frame. Its outputs correspond to no-op, up, down, left, and right; up and down do
nothing in this Breakout environment. Actions are sampled from the policy's probabilities.

## Where the computation happened

The large Qwen model ran on **Prime Intellect's hosted infrastructure**. The evaluation games
ran **locally during our experiment**, using the tiny NumPy policy and simulator rather than
the large LLM. Those evaluations require little computing power and could also run on hosted
compute in a future experiment.

**The [live HF Space](https://huggingface.co/spaces/HarleyCooper/weight-space-breakout-inference) now runs the small policy and simulator on Hugging Face's CPU Basic hardware.**
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
in the [model repository](https://huggingface.co/HarleyCooper/weight-space-breakout-llm-repaired/tree/main/experiment).

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

## Detailed results for the exported policy


| Evaluation | Episodes | Faulty baseline | LLM-edited | Mean gain | Paired standard error of gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published benchmark seeds | 128 | 1.1641 | 2.7969 | 1.6328 | 0.2843 |
| Fresh holdout seeds | 1024 | 1.2969 | 2.4268 | 1.1299 | 0.0982 |

Changed parameters: **5 / 17,253**.
Holdout seeds were specified before generation; all candidates were frozen before scoring. Policies use paired environment and action-sampling seeds.
One attempt per task is not a statistical evaluation of an LLM's general repair ability.
The reference tracker is a teaching baseline, not an optimal policy. Improvement does not imply every game is won.

## Use locally (optional)


```bash
pip install numpy safetensors huggingface_hub
hf download HarleyCooper/weight-space-breakout-llm-repaired --local-dir policy
cd policy
python inference.py
```

`inference.py` accepts `--observation` with a JSON array of 500 binary values (or shape 10×10×5).
It returns logits, probabilities, a sampled action, and the greedy action. Review downloaded Python before running it.
`model.safetensors` and `policy.npz` contain the revised weights; `baseline.safetensors` contains the faulty starting weights.
No Transformers custom code or `trust_remote_code` is required. This custom policy is served by a dedicated CPU Space rather than a generic text-generation provider.

## Reproducibility and files

The full network specification is in `config.json`, exact prompts, submissions, frozen-candidate hashes, and results are in [`experiment/`](https://huggingface.co/HarleyCooper/weight-space-breakout-llm-repaired/tree/main/experiment), and export hashes are in `checksums.json`. Infrastructure errors before any completion are recorded in the experiment plan. The experiment used environment version **0.1.0**; **0.1.1** updates its documentation without changing the game or scoring.

The export matched the original NumPy candidate on 16 seeded observations. Three remote predictions and three complete episodes matched the frozen local policy; see [`hosted-inference-verification.json`](https://huggingface.co/HarleyCooper/weight-space-breakout-llm-repaired/blob/main/hosted-inference-verification.json). The Space pins the weight-bearing model revision and checks the weight file hashes. Documentation updates do not change those weights.

The MIT license covers our implementation and teaching weights. No original research weights are redistributed in this repository.
