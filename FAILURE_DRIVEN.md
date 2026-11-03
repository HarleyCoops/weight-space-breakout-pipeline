# Repairing an observed failure without a known diagnosis

Version 0.3.0 adds `failure-driven-v1`. The input is a failed game and its policy,
not a label naming the faulty weight. The editor must infer a plausible cause,
propose concrete assignments, and predict the behavioral change they should cause.
Tools and practice simulations are allowed. This is a new protocol; the published
three-round 0.2.1 experiment has not been rerun or relabelled.

**Live trial: November 3, 2026.** The first bounded run completed all three attempts
on Prime environment **0.3.0**, with `qwen/qwen3.6-35b-a3b` as editor. None returned a
valid final edit. The selected weights remain the initial faulty reversed-controls
policy. This run demonstrates recorded rejection and deployment continuity, not an
improved policy or successful failure repair.

| Attempt | Model calls | Tool calls | Outcome |
| --- | ---: | ---: | --- |
| 1 | 8 | 8 | Turn limit reached while exploring; no final JSON |
| 2 | 1 | 0 | 4,096-token response limit reached during analysis; no final JSON |
| 3 | 8 | 8 | Turn limit reached while exploring; no final JSON |

The final independent 1,024-game audit returned **1.380859375 before and after**,
with zero gain and no cleared boards. No candidate reached the 512-game promotion
test. All three published checkpoints reproduced Prime's selected episode exactly
on Hugging Face, including the policy hash, actions, reward, and termination.

[Live demo](https://huggingface.co/spaces/HarleyCooper/weight-space-breakout-failure-driven-demo-01) ·
[Weights and complete attempt audit](https://huggingface.co/HarleyCooper/weight-space-breakout-failure-driven-01) ·
[Compact result record](results/2026-11-03-failure-driven-01.json)

The original controller halted when it saw a truncated trace, even though Prime's
verifier had already rejected the attempt. Collection now recognizes these specific
budget outcomes and records rejection only when the verifier reports no candidate,
zero reward, and unchanged selected weights. The saved evaluations were collected
without retrying them. Environment, budgets, seeds, and promotion rules stayed fixed;
later prompts received budget and tool-usage feedback. Prime and HF login worked.

The editor runtimes initially lacked NumPy; installing it and inspecting helper
interfaces consumed part of the turn budget. A future separately versioned protocol
should prepare dependencies before the editor starts, expose a simpler practice
interface, and reserve a final submission step. These changes were not applied to
this completed trial. Its result does not establish whether a properly submitted
repair would have succeeded. The controller and game suite passes 31 tests.

## Handoff

After publication, the controller runs and verifies a full HF episode. If it ends in
a miss, its seed and exact policy fingerprint are passed to the next Prime task.
Prime independently reproduces the failure from the submitted weights. If that
episode did not miss, Prime searches at most 32 designated practice seeds. If none
fails, it stops without inventing an error. The initial target seed is 314159.

The editor receives:

- The current weight snapshot and its fingerprint.
- A complete compact trajectory with each **pre-action** state, sampled action,
  action probabilities, subsequent state, reward, and termination.
- Exact binary input indices, logits, and hidden activations for the last 12 decisions.
- The previous candidate's explanation, edits, replay result, and broader scores.
- A checksum identifying this particular failure case.

The request is: **What change to these weights could have prevented this observed
miss, and what behavior do you expect that change to produce?**

The editor's final answer contains `diagnosis`, `expected_prevention`, and `edits`.
Explanations are short hypotheses, not executable code or automatically verified facts.
Up to 64 bounded scalar assignments are accepted by the edit parser.

## Two separate checks

1. **Specific failure:** freeze the original action history through the last successful
   paddle contact (or reset if this was the first contact). Reuse the original random
   draws and give control to the edited policy from that intervention point. Check
   whether it catches the **same ball at the same contact step** that was originally
   missed. A no-edit replay must reproduce the original miss exactly.
2. **Broader play:** independently compare the candidate and incumbent on 512 matched
   selection seeds. Require a real weight change, gain above 0.10, and
   `gain - 2 * paired_standard_error > 0`.

Both checks must pass to publish changed weights. This tests a counterfactual claim
about a particular replay, not whether the editor's entire causal explanation is
correct, whether its patch is unique, or whether it is globally optimal.
Candidate full-episode simulation starts at reset; conditional failure replay starts
from the frozen intervention state. Those are explicitly different measurements.

A useful test case uses the existing selected policy at seed 314159. On the last
decision, the ball is at row 7, column 3, heading toward column 2. The paddle moves
left from center 5 to center 4, covering columns 3..5, and misses. It already chose
the right direction; an earlier decision needs to change. A deliberately huge left
bias catches that replay but reduces broader reward, so the verifier rejects it.
This is a development test, not a new LLM-generated result.

## Tools, budget, and trusted scoring

The separate editor runtime receives a practice bundle at `/tmp/repair_lab.zip`,
the policy and failure JSON files, and `/tmp/repair_practice.py`. With NumPy installed,
the helper can test a candidate JSON file against the failure and 32 practice games.
The bundle contains game mechanics and inference/edit code. It excludes gold policies,
reference repairs, selection seeds, final-test seeds, and promotion logic.

The harness permits tools and up to eight model turns per round, with 4,096 output
tokens per call and 16,384 output tokens total. The controller also refuses to accept
a trace with more than 24 tool calls. The controller's existing 15-minute wait limit
remains. Practice files are writable by the editor; their output is not trusted for
promotion. The verifier reconstructs candidates from JSON and scores them in its
own runtime. Submitted Python is never executed by the verifier or publisher.

Practice helper seeds: `1040000..1040031`. Bounded failure search:
`1040000+32*r .. 1040031+32*r`. Selection seeds:
`1060000+512*r .. 1060511+512*r`. Final audit: `1090000..1091023`.
These differ from the completed initial experiment. The final audit is never
used to choose an edit or passed into a subsequent editor prompt.

## Running it

Fresh runs initialized by `pipeline/control.py` now use this protocol. Existing
saved plans keep their original behavior. Use a fresh run directory and new HF
repository names as described in [the setup guide](pipeline/README.md).

For direct Prime evaluation, use `taskset.iterative=true`,
`taskset.failure_driven=true`, a policy snapshot/hash, and optionally
`taskset.failure_seed`. Set `agent.max_turns=8` and
`agent.max_output_tokens=16384`. Without the failure-driven flag, the Hub taskset
retains its original behavior; existing one-shot tasks remain the default.

Future work should compare multiple failure families and baseline editors. A single
successful replay is insufficient evidence of general repair ability, including for
Cree or Dakota language models.
