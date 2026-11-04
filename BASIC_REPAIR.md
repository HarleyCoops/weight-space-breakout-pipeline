# Basic tool-assisted repair

`basic-repair-v1`, introduced in environment 0.4.4, separates the first deployment
milestone from the harder unknown-failure task. This is a new experiment. It does
not reinterpret or replace the recorded failure-driven attempts.

The question is: can a hosted editor inspect a deliberately damaged, 17,253-parameter
teaching policy, use practice tools, commit an improving weight edit, and deploy the
accepted checkpoint so that the live simulation reproduces the verifier's game?

## Frozen trial

- One editor attempt, Qwen `qwen/qwen3.6-35b-a3b`, temperature 0.6.
- Prepared inspect/test/submit tools; sequential tool calling is required.
- No editor-turn, tool-call, or aggregate output-token limit. Each response allows
  16,384 output tokens; the hosted job and controller retain 60-minute timeouts.
- At most 64 scalar assignments, each in [-100,100]. No golden patch, numerical
  candidate solver, or gradient training is supplied to the editor.
- Practice games use seeds 3040000..3040031. The recorded miss at seed 314159 is
  diagnostic evidence. The editor is explicitly told it need not rescue that miss.
- After the final submission, promotion uses 512 separate games, seeds
  3060000..3060511. Require an actual weight change, paired mean gain >0.10, and
  gain minus twice the paired standard error >0.
- The final 1,024-game audit uses seeds 3090000..3091023. It is never used to select
  or revise the edit. Rejected candidates and failed executions remain recorded.

The final action remains `repair_submit_edits`. Its `expected_prevention` field is
retained for receipt compatibility; in this protocol it describes the expected
behavioral improvement, without claiming guaranteed prevention of the example miss.

The collector preserves failed HTTP request attempts. A recovered transient request
with no assistant output does not invalidate a later completed submission. Partial
generations, terminal trace errors, changed sampling, and missing final receipts
remain failures; this does not weaken the policy promotion rule.

The temperature differs from the earlier trial. Qwen's published coding guidance
uses nonzero sampling; this trial uses temperature 0.6 but does not claim to reproduce
every setting in that preset. This is an engineering milestone, not an isolated
causal comparison of temperature or task difficulty.
[Official model guidance](https://huggingface.co/Qwen/Qwen3.6-35B-A3B#best-practices).

## Run from PowerShell

Publish environment 0.4.4 only after any live trial that requires 0.4.3 has finished:
Prime uses the latest published environment, and this controller checks its version.
Use fresh repository names for a new trial. Existing plans are immutable on resume.

```powershell
$env:PRIME_CLI = '/home/chris/.local/bin/prime'
$env:REPAIR_PROTOCOL = 'basic-repair-v1'
$env:REPAIR_RUN_DIR = Join-Path $PWD 'artifacts/basic-repair-run-01'
$env:REPAIR_MODEL_ID = 'HarleyCooper/weight-space-breakout-basic-repair-01'
$env:REPAIR_SPACE_ID = 'HarleyCooper/weight-space-breakout-basic-repair-demo-01'
uv run --no-project --with-requirements pipeline/requirements.txt python pipeline/control.py run
```

The controller publishes the faulty baseline, starts the hosted editor, validates
the committed proposal and independent measurements, publishes the selected weights,
and checks inference from the exact HF revision. Playing the Space itself never
launches an editor or rewrites weights.

## What success would and would not establish

Success establishes one tool-assisted repair of a planted fault and a verified
deployment. It does not establish sustained recursive improvement, prevention of
arbitrary failures, or improvement of a language model. If the edited policy exactly
matches the handwritten intact reference, report that restoration explicitly.

The next distinct experiment can require unknown-failure prevention, keep regression
tests for earlier repairs, and compare raw LLM edits with a generic constrained
numerical solver. If the desired objective is prevention without meaningful broader
degradation, define a non-inferiority tolerance beforehand. Requiring significant
broader improvement on every round is a stronger condition.

A rollback gate makes accepted performance monotonic only under its measured rule.
It is not a guarantee over all games. Repeated selection against one test set can
overfit it, and a stalled search does not prove that the network has exhausted its
representational capacity.
