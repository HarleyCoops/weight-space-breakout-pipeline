# Independent review: failure-driven repair and KimiBrainSurgery

The user requested an independent subagent review. This assessment inspected
KimiBrainSurgery at commit `230091604f067c669439dcabfee513eb07694aa8` and this
pipeline while failure-driven run 03 was still in progress. No results from that
unfinished run were assumed. The reviewer made no code or cloud changes.

## Findings from KimiBrainSurgery

1. Its programs use numerical solvers to edit the small Breakout MLP. They contain
   no LLM inference calls and do not edit a Qwen language-model checkpoint. Kimi's
   authorship of the programs is distinct from a hosted LLM choosing edits during a
   rollout. The script entrypoints write result reports, not HF checkpoints.
   [Source](https://github.com/HarleyCoops/KimiBrainSurgery/blob/230091604f067c669439dcabfee513eb07694aa8/failure_gym.py).
2. The three loops keep `new_weights` unconditionally after measuring the gate.
   The gate reports regressions but does not reject them. At lambda=100, measured
   mean return declines from 8.15625 to 7.34375 after eight edits. The first edit
   changes 7,474 of 21,402 probe decisions (34.9%); a small median across edits is
   not a safety bound on every edit.
   [Loop](https://github.com/HarleyCoops/KimiBrainSurgery/blob/230091604f067c669439dcabfee513eb07694aa8/nullspace_gym.py#L73),
   [results](https://github.com/HarleyCoops/KimiBrainSurgery/blob/230091604f067c669439dcabfee513eb07694aa8/results_nullspace_100.json).
3. `fixed` checks the action at one stored observation. It does not establish
   episode rescue or retention of previous repairs. The same seed-12 failures at
   steps 53 and 63 recur. A locally feasible action may still require different
   subsequent actions to catch the ball.
   [Mining and gate](https://github.com/HarleyCoops/KimiBrainSurgery/blob/230091604f067c669439dcabfee513eb07694aa8/failure_gym.py#L82).
4. The landing predictor omits collision cases present in the actual engine.
   The pairwise W3 update can outrank the old wrong action while leaving a third
   unsafe action on top. W2 refinement continues comparing against the original
   wrong action. These are concrete targets for validation before adopting the
   solver as a repair tool.
   [Predictor](https://github.com/HarleyCoops/KimiBrainSurgery/blob/230091604f067c669439dcabfee513eb07694aa8/failure_gym.py#L64),
   [engine](https://github.com/HarleyCoops/KimiBrainSurgery/blob/230091604f067c669439dcabfee513eb07694aa8/assets/game.py#L49),
   [W2 solver](https://github.com/HarleyCoops/KimiBrainSurgery/blob/230091604f067c669439dcabfee513eb07694aa8/w2_gym.py#L63).

These are findings about that exact revision, not claims about subsequent changes.

## Findings from our pipeline

Our verifier already rejects harmful or insufficient edits and retains the incumbent.
The previous 123-turn attempt did not submit any final edit. It repeated an invalid
oversized patch before a provider disconnect; that is distinct from an impossible
repair task or demonstrated credential failure.

The current failure-driven task combines four challenges: repair a planted fault,
prevent a particular observed miss, preserve/improve broader performance, and manage
a tool conversation through a committed final submission. The simplest reference
repair need not satisfy all four.

A separate handwritten engineering control tested only exposed practice data:

| Reference direction repair plus right bias | Practice mean | Recorded miss prevented |
|---|---:|---|
| 0 | 2.65625 | No |
| 0.5 | 2.0625 | No |
| 1 | 2.46875 | Yes |
| 2 | 4.1875 | Yes |
| 4 | 4.90625 | Yes |
| 8 | 0.96875 | Yes |

Baseline practice mean was 1.5. These six handwritten controls were not LLM results,
were not submitted or promoted, and did not access the independent gate or audit.
They establish feasibility on the exposed cases only. They also show why rescuing
one miss cannot replace the broad performance check.

## Recommended sequence

Keep the live run frozen. First establish one tool-assisted basic repair with broad
held-out improvement, actual submission, publication, and verified deployed inference.
Record the target replay diagnostically. Then attempt unknown-failure repair with
retained regression cases and fresh broad checks. Compare raw LLM proposals against
a transparent numerical-solver tool and a solver-only control.

The one-active-path teaching policy has limited room for localized readout edits;
null-space results from a denser trained policy cannot be assumed to transfer.
No Breakout result here demonstrates improvement of Cree or Dakota language models.

See [the separately specified basic trial](../BASIC_REPAIR.md).

## Follow-up after the review

Run 03 subsequently ended with a provider HTTP 429 concurrency-limit error after
144 model turns and 150 model calls including retries. It produced 141 valid
practice measurements, one input-index error, no target rescues, and no final
submission. Its best practice mean was 2.59375; that was never independently
verified or promoted. The full trace and failure audit remain attached to the run.

The reviewer also checked the basic-repair implementation at commit `632415f` and
found no blocking acceptance, seed, receipt, or publication mismatch. The follow-up
change explicitly labels target replay as diagnostic in every basic practice reply.
