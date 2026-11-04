# Weight Space Breakout: hosted repair pipeline

**Version 0.4.4 adds a separately specified [basic tool-assisted repair trial](BASIC_REPAIR.md).**
It tests one broad improvement and the complete deployment path. The example miss is
diagnostic only in this new protocol; prior failure-driven trials keep their stricter
rules. The new workflow passes 44 tests, including both protocols through the real
agent, tool transport, final submission, and verifier with a scripted test provider.
A passing software test is not a real hosted LLM result.
See the [independent design review](reviews/2026-11-03-independent-review.md).

The subsequent 0.4.3 attempt completed 141 valid practice tests but never submitted
an edit in 144 turns. It stopped on a provider HTTP 429 concurrency-limit error.
No independent promotion check ran and no weights were promoted. This separates
working practice tools from the still-unproven hosted-editor submission milestone.

**Version 0.4.3:** [prepared practice tools and a final submission action](PREPARED_EDITOR.md),
with no editor-turn limit. Startup prepares the numerical tools, and the last action
commits the proposed weight edit for independent verification. Tool calling is enabled.

The 0.4.1 hosted attempt made 123 model turns but never submitted an edit. It
repeated an oversized patch and eventually ended with a provider connection error.
No weights were promoted. Version 0.4.2 improves tool constraints and repeated-patch
feedback and preserves failed-attempt audits; its effect on editor behavior still
requires a new hosted trial. See [the full recorded outcome](PREPARED_EDITOR.md).
Version 0.4.3 updates the verifier dependency for the current hosted runtime;
the 39 tests also pass against that dependency.

**Previous 0.3.0 trial:** [failure-driven diagnosis and replay](FAILURE_DRIVEN.md). The next
editor receives a specific failed episode without a supplied diagnosis, can use tools,
and must propose an explanation plus weight edits. Promotion requires preventing that
miss under controlled replay and improving broader play. Its first three-attempt trial
is now complete: all attempts exhausted a budget before submitting valid edits, so
the faulty weights remained unchanged (final audit **1.3809 → 1.3809**).
See [the failure-driven results](FAILURE_DRIVEN.md) and
[its separate demo](https://huggingface.co/spaces/HarleyCooper/weight-space-breakout-failure-driven-demo-01).
The successful earlier-protocol run described below remains a separate experiment.

A working experiment that starts with known-bad policy weights, asks a hosted LLM for
an edit, scores the candidate in Prime's environment, publishes accepted weights to
Hugging Face, and runs the next simulation from that exact checkpoint.

**The first bounded run completed: one repair accepted, two regressions rejected.**
On a separate 1,024-game final audit, average reward improved from **1.3018 to 2.9180**.
The accepted edit changed two of 17,253 parameters and exactly restored the intact
handwritten reference policy. The other two candidates made play worse.

[Live repair-loop simulation](https://huggingface.co/spaces/HarleyCooper/weight-space-breakout-repair-loop) ·
[Versioned weights and full audit](https://huggingface.co/HarleyCooper/weight-space-breakout-iterative) ·
[Prime environment](https://app.primeintellect.ai/dashboard/environments/harleycooper/weight-space-breakout) ·
[Pipeline source](https://github.com/HarleyCoops/weight-space-breakout-pipeline)

## What runs where

| Piece | Job | Location |
| --- | --- | --- |
| Editor: `qwen/qwen3.6-35b-a3b` | Reads current weights, practice trajectories, and prior-round feedback; proposes scalar edits | Prime-hosted inference |
| Game and verifier | Runs practice games, compares candidate to incumbent, decides acceptance, runs final audit | Prime-hosted evaluation |
| Controller | Checks traces and hashes, serializes weights, publishes versions, verifies deployment | Lightweight Python orchestration; no local model or evaluation games |
| Game policy | Maps a 10x10 observation to five action probabilities | 17,253-parameter NumPy network on Hugging Face CPU Basic |

The controller was run from a workstation for this demonstration. It can also run on
ordinary hosted CPU compute. Prime and Hugging Face handled the experiment's model
inference and gameplay. Local development unit tests exercised the small game separately.
The three-round run has finished; no editor job or ongoing repair schedule remains active.

## First iterative run: November 3, 2026

| Round | Incumbent return | Candidate return | Mean gain | Decision |
| --- | ---: | ---: | ---: | --- |
| 1 | 1.4512 | 2.7773 | +1.3262 | Accepted: reverse the two output signs |
| 2 | 2.8535 | 2.4395 | -0.4141 | Rejected: stronger controls and fewer no-ops |
| 3 | 2.8535 | 2.3789 | -0.4746 | Rejected: extreme control strengths |

Each row uses 512 fresh matched seeds. Row means are not directly comparable across
rounds. The next round uses the last **accepted** policy and the previous attempt's
feedback. A rejected candidate remains in the audit but never replaces live weights.

Acceptance requires a real parameter change, mean gain above 0.10, and
`gain - 2 * paired_standard_error > 0`. This is an engineering screen, not a general
statistical guarantee. The final audit used 1,024 additional seeds, never fed to the
editor or used for promotion: baseline **1.3017578125**, selected **2.91796875**, paired
gain **1.6162109375**, standard error **0.1107816995**. Neither policy cleared a board
on that audit. Individual games still often fail quickly.

For every round, the deployed Hugging Face episode matched the Prime verifier's
policy hash, seed, complete action sequence, reward, and termination. Immutable
Hub revisions preserve the initial bad policy and every published state.

This establishes the end-to-end feedback, rejection, publication, and inference
mechanism. It does not establish reliable recursive improvement or language-model
repair. This is a handwritten teaching controller with a deliberately simple fault,
not a trained Qwen policy. Qwen is the editor, not the exported game network.

## Use and reproduce

- **Play:** open the live Space and choose **Play latest weights** for a fresh seed.
  **Replay pinned version** repeats the displayed seed and exact model revision.
  The animation loops one computed episode; playing does not trigger weight editing.
- **Run the pipeline:** see [setup and commands](pipeline/README.md).
- **Inspect the protocol:** [PIPELINE.md](PIPELINE.md) specifies seeds, gates, compute,
  and the disclosed tool-harness limitation.
- **Inspect all attempts:** [public experiment records](https://huggingface.co/HarleyCooper/weight-space-breakout-iterative/tree/main/experiment).
- **Extend to languages:** [Cree and Dakota assessment](LANGUAGE_EXTENSION.md).
- **Compare the earlier one-shot experiment:** [ONE_SHOT.md](ONE_SHOT.md). That earlier
  experiment used a different protocol and local game evaluation; its results remain separate.

The new failure-driven protocol allows tools and multiple editor turns under an explicit budget.
This initial run used exactly one response and zero tool calls per round. Prime's
harness advertised tools despite the prompt saying they were unavailable; actual
non-use is verified from each trace. See the protocol disclosure for details.

## Code and verification

`weight_space_breakout/` contains the game, policy, data-only edit parser, Prime taskset,
and verifier. `pipeline/control.py` implements the resumable bounded loop.
`pipeline/space_app.py` serves immutable policy revisions. `tests/` covers game rules,
invalid edits, the acceptance gate, trace tampering, tool-use checks, and interrupted
state recovery. The original build passed 18 tests; the current suite passes 39 on Linux, including
the complete agent/tool/submission lifecycle with a scripted provider, failure replay,
practice-tool execution, and rejecting a targeted rescue that harms
broader play. All three original real hosted deployment
comparisons passed. The run used environment **0.2.1**; subsequent documentation or
packaging releases do not change those recorded results.

## Credit and scope

Independent conceptual replication attempt and teaching adaptation inspired by
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (MIT CSAIL, November 2, 2026).
[Original X post](https://x.com/rikiyatakehi/status/2117313742891327994) ·
[Experiment 1 logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game).

The original experiment identifies a game from its weights. We reconstructed our own
game and tested repairs to intentionally faulty handwritten weights. This is not an
exact reproduction of their simulator, PPO training, or reported scores.
