# Hosted iterative weight repair

**Current extension:** version 0.4.3 implements [prepared tool-assisted repair](PREPARED_EDITOR.md)
with no editor-turn cap and an explicit final submission tool. It extends the
[failure-driven repair](FAILURE_DRIVEN.md) replay and broad-performance checks. Fresh
controller plans use that protocol. The description below records the completed
initial 0.2.1 experiment; its scores and no-tool protocol are unchanged.

This extends the original one-shot experiment into a bounded feedback loop. It starts
from the deliberately bad reversed-controls teaching policy, not a language model.
The completed demonstration used environment 0.2.1. The subsequent 0.2.2 release
adds complete source/setup documentation and corrects the prompt's claim about tool
availability; it does not change the recorded run or its checkpoints.

1. Freeze a run plan (three rounds by default, one Qwen response per round).
2. Send the current policy snapshot and its hash to `harleycooper/weight-space-breakout@0.2.1`.
3. Prime's hosted environment runs 32 practice games and provides short trajectories.
4. Prime-hosted `qwen/qwen3.6-35b-a3b` proposes up to 64 scalar assignments, with no tool use.
5. The hosted verifier validates the edits and compares the candidate and incumbent
   on 512 matched, fresh evaluation seeds. It accepts only a real change with mean
   gain greater than 0.10 and `gain - 2 * paired_standard_error > 0`.
6. The controller checks the verifier's hashes, publishes the accepted policy to a
   versioned Hugging Face repository, and requests a fresh hosted simulation using
   exactly that published revision. A rejected candidate is recorded, not promoted.
7. The next round starts from the accepted weights and receives a summary of the
   previous attempt. After the final round, the verifier reports a separate 1,024-game
   audit against the original faulty policy. That audit is not used for promotion.

## Fixed boundaries

Practice seeds are `40000 + 32*round` through `40031 + 32*round`. Promotion seeds
are `60000 + 512*round` through `60511 + 512*round`. Final audit seeds are
`90000..91023`. Rounds are zero-based and limited to 20 in the environment.
The first published run is limited to three rounds and 4,096 output tokens per call.
No GPU is needed for the game. LLM inference and hosted evaluation can incur Prime charges.

**Harness disclosure:** Prime's default harness advertised `bash` and `edit` tools.
The task prompt prohibited their use (and inaccurately said they were unavailable).
The controller requires exactly one completed model call and zero actual tool calls
before accepting a trace. The audit preserves the advertised tool names and usage.
Thus this run demonstrates edits made without tool use, not a technically tool-free harness.

The promotion rule is an engineering screen, not proof of general statistical
significance. Adaptive search still needs independent replication. An unaccepted edit
does not change the live policy. Results may plateau or all candidates may fail.

## What runs where

- **Prime:** editor inference, practice games, verifier comparisons, final audit.
- **Controller:** lightweight orchestration, JSON validation, weight serialization,
  publication, and deployment checks. It runs no large model or evaluation games.
- **Hugging Face:** versioned policy weights and CPU inference/simulation.

The earlier one-shot model and its demo remain available as a separate experiment.
Each round records its starting checkpoint, exact response, candidate hash, verifier
report, accept/reject decision, published revision, and deployed episode result.

## Tool-assisted failure diagnosis: now implemented

The first run used one response per round. Version 0.3.0 adds the separate
`failure-driven-v1` configuration: tools and practice simulations, up to eight editor
turns, an explicit diagnosis, and both targeted replay and broad evaluation gates.
See [the failure protocol](FAILURE_DRIVEN.md) for exact budgets and validation status.
Clicking Play in the existing demonstration does not launch an editor experiment.
The controller still enforces the original protocol when resuming its frozen plan.

Keep the final evaluation and promotion logic outside the editor's writable runtime.
Tool-assisted editors should receive practice feedback while the trusted verifier
loads the submitted data-only patch in a separate process. Compare tool-assisted and
single-response runs using separate result records rather than pooling their scores.

## Relationship to Cree and Dakota

The same orchestration can be adapted, but the game verifier cannot measure language
quality. See `LANGUAGE_EXTENSION.md` for the inspected repositories, existing scoring
code, and the work needed before a language-model editing experiment.

## Original-work credit

Independent conceptual replication attempt and teaching adaptation of
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (MIT CSAIL, November 2, 2026).
[Original X post](https://x.com/rikiyatakehi/status/2117313742891327994) ·
[Experiment 1 logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game).
Experiment 1 identifies a game from weights. This project instead tests editing in
an independently reconstructed game with handwritten faults. These are our results,
not a reproduction of the original PPO training, simulator, or reported scores.
