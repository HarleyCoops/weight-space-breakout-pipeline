# Weight Space Breakout

**New in 0.2.0:** opt-in hosted iterative repair, with candidate verification, fresh
evaluation seeds, versioned Hugging Face publication, and deployment checks.
See [the pipeline](PIPELINE.md). The original three one-shot tasks remain the default.

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

## Documentation release

Version **0.1.1** adds the experiment report, deployment links, and original-work attribution. Game rules, task prompts, and verifier scoring are unchanged from the **0.1.0** package used for the reported experiment.

## Two loops

1. **Game loop:** observation → small neural policy → sampled action → next observation and reward.
2. **Repair loop:** language model reads a faulty policy → submits JSON weight edits → verifier runs the edited policy → reward measures improvement.

The simulator is the world. The policy is the player. An episode is one game. Return is the sum of rewards in that episode. The verifier is the scoring code, not a second language model. Publishing an environment does not itself train a model.

## Quick start

```bash
uv sync
uv run weight-space-breakout demo
uv run weight-space-breakout evaluate --policy tracker
uv run weight-space-breakout evaluate --policy reversed_controls
uv run python -m unittest discover -s tests -v
```

Open `artifacts/demo.html` in a browser. The standalone viewer has policy/seed selection, play/pause, stepping, a timeline, observation memory, rewards and action probabilities. It replays frames from the same Python simulator used for scoring.

## Exact rules, version 1

- 10×10 grid, outer border walls, playable coordinates 1–8. Three-cell paddle on row 8; center starts uniformly in columns 2–7.
- 24 bricks fill rows 1–3, columns 1–8. Ball starts on row 6, column uniformly in 1–8, vertical velocity −1, horizontal velocity uniformly −1 or +1. Reset draws paddle, ball column, then velocity using NumPy's seeded default generator.
- Actions: 0 no-op, 1 up, 2 down, 3 left, 4 right. Up/down are no-ops. Paddle moves first by one cell, center clamped to 2–7.
- Ball advances diagonally one cell per tick. Side and top walls reflect velocity. Vertical and horizontal neighboring bricks reflect the corresponding component and are removed. If neither face hits, a diagonal brick reflects vertical velocity and is removed. Walls are rechecked after brick reflections. If the new destination is still occupied by a brick, the ball stays in place for this tick with its reflected velocity.
- At row 8, a paddle cell reflects the ball upward. Missing the paddle ends the episode. Clearing every brick also ends the episode. At 200 steps an unfinished episode is truncated. Bricks never respawn. There is one life and no sticky-action randomness.
- Reward: +1 per brick removed, −1 for missing, no extra completion bonus. Maximum return 24. Termination and time-limit truncation are reported separately.
- Observation: float32 binary array of shape `(500,)`, reshaped `(10,10,5)`. Channels: paddle, ball, bricks, obstacles, previous-frame paddle-or-ball. The previous-frame channel starts at zero. Flatten index `(row*10+column)*5+channel`.

`Breakout.reset(seed)` returns `(observation, info)`; `step(action)` returns `(observation, reward, terminated, truncated, info)`. This follows the familiar Gym-style tuple convention but does not claim Gymnasium registration or subclass compatibility.

## Policies and repair task

All policies use 500 → 32 → 32 → 5, tanh hidden activations, 17,253 scalar parameters. Softmax sampling is the primary evaluation; greedy mode is a separate diagnostic.

The v0.1 teaching baseline is **handwritten, not PPO-trained**: a hidden path compares the current ball column with the paddle's center and favors left/right. It does not predict landings, and is neither optimal nor guaranteed to clear a board. Three deliberate faults form three tasks: reversed action connections, a strong left bias, and a missing right connection. The prompt contains all nonzero weights; every unlisted weight is exactly zero.

The model returns only a JSON object, optionally in a `json` code fence:

```json
{"edits":[{"tensor":"W3","index":[0,3],"value":-8.0}]}
```

This is an example of the format, not a complete answer for every task. At most 64 unique scalar assignments, valid tensor indices, finite values in [−100,100], and at most 20,000 response characters. Invalid submissions score zero. The evaluator executes no submitted code and does not use `eval`, `exec`, or pickle.

Practice seeds: 0–31. Evaluation seeds: 10000–10127. Baseline, candidate and reference use the same environment seeds and seeded action sampling. The score is:

```text
clip((candidate_mean_return - faulty_baseline_mean_return)
     / max(1, intact_reference_mean_return - faulty_baseline_mean_return), 0, 1)
```

An unchanged policy scores zero; restoring the reference scores one on the shipped evaluation set. The report also retains raw gains, changed-parameter counts, and paired standard error. A score of one means recovering the reference improvement, not winning every game. The reference is not an upper bound on attainable game performance.

**Limitations:** three simple faults are a teaching smoke benchmark, not a broad measure of weight understanding. Test seeds are separate from the practice prompt but public in this open package, not a secret leaderboard set. A serious study should freeze a candidate before evaluating on a newly chosen external seed set. The one-shot/no-tool restriction requires an appropriately restricted harness; the task stops after one turn but cannot audit external computation by itself. The included reference checks are separate from the LLM repair experiment described above; no PPO training was performed.

## Prime Intellect / Verifiers

Use Linux or WSL for this integration. The core simulator and viewer also work on Windows.

```bash
uv sync --extra prime
uv run vf-validate weight-space-breakout --runtime.type subprocess -n 3
```

The package exports `BreakoutTaskset`, with a single-turn final-answer task and a deterministic reward function. Verifiers 0.4.0 is pinned for the tested v1 API. The reward runs on the evaluator and sends only the prompt to the solver; no training GPU is required for game simulation. Use a harness without shell/tools for the intended one-shot protocol. Hosted model evaluations may have provider costs.

Publish after checks using an authenticated Prime CLI:

```bash
prime env push --path . --visibility PUBLIC --runtime v1
```

## Research reference and Hugging Face

The supplied paper checkpoint is kept separate from the teaching tasks. Its rounded attachment can be reconstructed and measured under our rules, but our scores cannot be compared directly with the paper's scores. Do not claim compatibility with the missing original environment.

The MIT license applies to this implementation. Any downloaded research weights require attribution under the source repository's CC BY 4.0 license. They are not bundled with this environment. Hugging Face can host exported weights and a separate inference demo; it is distinct from Prime's environment registry.

## Initial verification

On 128 fixed evaluation seeds, sampled actions:

| Policy | Mean return |
| --- | ---: |
| Random actions | 0.633 |
| Reversed controls | 1.164 |
| Strong left bias | 0.648 |
| Missing right connection | 1.875 |
| Intact handwritten tracker / reference repairs | 3.383 |

Nine core tests pass, covering collisions, reward accounting, observations, deterministic seeds, invalid edits, export/reload and reference repairs. Verifiers gold and no-op validation passes for all three tasks. These checks use known reference edits; they are not an evaluation of an LLM's repair ability.
