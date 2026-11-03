# Prepared tools and an explicit final weight edit

Environment 0.4.2 provides `failure-driven-v2`. Tool calling is enabled. The editor
has **no model-turn limit, tool-call limit, or aggregate output-token limit**. The
earlier no-tool proof of concept and the unsuccessful eight-turn trial retain their
original records and checkpoints.

The editor receives a failed episode, the current weights, and prior-round feedback.
It can call three prepared tools:

1. `repair_inspect_policy`: inspect current weights and a measured baseline.
2. `repair_test_edits`: test a complete proposed patch against the recorded miss and
   32 practice games. Repeat to investigate and revise the proposal.
3. `repair_submit_edits`: the **final action**, committing the diagnosis, predicted
   behavioral change, and scalar weight assignments. It ends the attempt.

The tool runtime already has the package and NumPy. Startup reproduces the baseline
miss before editing begins. The editor does not need to install packages or discover
Python object attributes. The model's runtime exposes these task tools directly;
the `null` harness name means no built-in shell, **not no tools**.

Tool calling is required and sequential. A successful submission stores an immutable
receipt in the task state. The controller checks that the last model action, stored
receipt, submitted patch, candidate fingerprint, and verifier report agree. Ordinary
analysis, a practice call, an empty patch, and a partial response cannot stand in for
the final edit. Tools refuse further changes after submission.

The independent verifier then asks whether the edit catches the same ball contact
under controlled replay and improves mean reward across 512 separate seeds. Only
accepted weights become the next live policy. Hugging Face runs a complete game
from that exact published revision. A miss becomes the next repair task; a rejected
proposal retains the incumbent and passes its failure feedback to the next attempt.
If the deployed game does not miss, a bounded search checks 32 practice seeds and
stops without inventing a failure when none is found. Game episodes retain the
simulator's 200-tick horizon.

No golden replacement weights are given to the editor. Practice tools expose no
reference solution, selection returns, or final audit returns. The final verifier
reconstructs the data-only patch independently from the current policy.

## Remaining operational boundaries

Removing a turn cap does not remove the provider's context window or service
timeouts. Each response allows 16,384 output tokens. Hosted jobs and the controller
retain a 60-minute timeout. Neither a timeout nor context exhaustion is accepted as
a repair. Such failures are preserved for diagnosis.

The default validation experiment runs **three repair rounds** so it can complete a
separate final audit. This is distinct from the unlimited number of editor turns
inside each round. Playing the demo does not start a paid editing job.

Practice seeds: `2040000..2040031`; fallback failure search:
`2040000+32*r .. 2040031+32*r`; selection:
`2060000+512*r .. 2060511+512*r`; final audit: `2090000..2091023`.

## Validation

All 39 tests pass on Linux, including real-server startup, preparation, practice execution,
explicit submission, rejection after submission, and controller validation of a
30-call trace. A scripted local provider also drives the real agent harness through
inspection, practice, and submission: the session stops with `edit_submitted`, makes
no fourth model call, and runs the verifier. This is an integration test using a known
fixture, not an LLM repair result. Tests also cover invalid/repeated patch feedback,
the advertised 64-item tool schema, and archiving an execution failure without
advancing the round or promoting a practice candidate.

## Hosted 0.4.1 trial: November 3, 2026

The prepared tools worked in the hosted environment. Qwen made 123 model turns:
one inspection call and 122 practice-test calls. There were 21 valid practice
results, 100 input errors (99 oversized patches and one invalid index), and one
test call without a recorded result. The same 69-assignment patch was requested
100 times. There was no final `repair_submit_edits` call.

None of the valid practice tests prevented the recorded miss. The best practice
mean was 1.65625 against a 1.5 baseline on the 32 practice seeds; these are search
results, not an independent performance estimate. No 512-game promotion test or
final audit ran. A provider connection closed with an incomplete response (HTTP
502), ending the attempt after approximately 46 minutes of rollout time. This was
not a login error, turn-limit stop, or the declared 60-minute timeout.

The controller halted. No weights were promoted, and rounds two and three were
not launched. The initial faulty policy remains published. The raw trace is saved
locally under `artifacts/failure-driven-run-02/round-00/trace.json`; a compact public
failure audit accompanies the model. A failed execution is not relabelled as a
scored rejection or silently retried.

- [Prime evaluation](https://app.primeintellect.ai/dashboard/evaluations/a4hrpun4gvsa9kwnjekri7od)
- [Model and failure audit](https://huggingface.co/HarleyCooper/weight-space-breakout-failure-driven-02)
- [Live simulation of the unchanged policy](https://huggingface.co/spaces/HarleyCooper/weight-space-breakout-failure-driven-demo-02)

Version 0.4.2 makes the assignment limit and valid scalar types explicit in the tool
schema, gives tensor-specific index errors, identifies repeated practice patches,
and retains the best practice patch for inspection. Its prompt explains that a
patch omits unchanged weights and that splitting a patch across calls does not
accumulate changes. No turn cap or golden repair was added. The service connection
failure is separate; these changes address the observed unproductive tool loop,
not the upstream network. The controller now archives failed traces and publishes
the failure status with the unchanged weights. A new hosted trial is needed to
measure whether these feedback changes improve Qwen's behavior.

The first 0.4.0 hosted startup was cancelled during tool-server setup: its Python
module lacked the executable entry point. No editor trace was produced. Version
0.4.1 fixes that entry point; the additional test starts the actual server and checks
its advertised tools. The cancelled evaluation is retained as a setup failure,
separate from scored editor attempts
([cancelled setup evaluation](https://app.primeintellect.ai/dashboard/evaluations/tub63az7xebkonisq5hr6nca)).

Run a fresh trial using [the controller setup guide](pipeline/README.md). Direct Prime
users set `taskset.iterative=true`, `taskset.failure_driven=true`,
`taskset.task.prepared_tools=true`, `agent.harness.id="null"`,
`agent.sampling.tool_choice="required"`, and
`agent.sampling.parallel_tool_calls=false`. Leave agent `max_turns` and aggregate
token limits unset. The controller writes these settings automatically.
