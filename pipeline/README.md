# Running the hosted repair loop

The checked-in controller implements a bounded three-round experiment. It uses
Prime-hosted Qwen and Prime-hosted games, then publishes to Hugging Face CPU Basic.
The process running the controller needs no GPU. It does need network access,
authenticated Prime CLI, and Hugging Face repository write permission.

Fresh plans use `failure-driven-v1` from environment 0.3.0. The editor diagnoses an
observed failure and can use tools and practice simulations for up to eight model
turns per round. Acceptance requires both preventing that miss and broader improvement.
See [the exact protocol and validation status](../FAILURE_DRIVEN.md). Previously saved
plans retain their original single-response/no-tool behavior.

## Setup

1. Install Python 3.11+ and `uv`, clone the source repository, and open its root.
2. Install and authenticate the official Prime CLI with `prime login`.
3. Authenticate Hugging Face using `hf auth login` or a locally managed `HF_TOKEN`.
   Never put tokens into this repository or command arguments.
4. Choose **new** model and Space repository names under your own HF account.
   Existing untracked repositories are refused. Both new repositories are public.

On Linux, run these in your shell (replace `YOUR_ACCOUNT`):

```bash
export REPAIR_RUN_DIR="$PWD/artifacts/my-repair-run"
export REPAIR_MODEL_ID="YOUR_ACCOUNT/breakout-repair-policy"
export REPAIR_SPACE_ID="YOUR_ACCOUNT/breakout-repair-demo"
export PRIME_CLI="$(command -v prime)"
uv run --no-project --with-requirements pipeline/requirements.txt python pipeline/control.py run
```

On Windows, the tested controller uses the authenticated Prime installation in
Ubuntu/WSL. Set the Linux executable path explicitly if it differs from the demo:

```powershell
$env:PRIME_CLI = '/home/YOUR_USER/.local/bin/prime'
$env:REPAIR_RUN_DIR = Join-Path $PWD 'artifacts/my-repair-run'
$env:REPAIR_MODEL_ID = 'YOUR_ACCOUNT/breakout-repair-policy'
$env:REPAIR_SPACE_ID = 'YOUR_ACCOUNT/breakout-repair-demo'
uv run --no-project --with-requirements pipeline/requirements.txt python pipeline/control.py run
```

HF repository IDs and run directory must remain unchanged when resuming. The source
Space's links to the original experiment and source repository remain as attribution.
Compute and token charges apply on Prime. There is no automatic periodic schedule.
The default frozen plan allows three rounds, with up to eight model calls and 16,384
output tokens per round in the new protocol;
it stops a still-running evaluation after a 15-minute wait. The platform's own
evaluation timeout is 60 minutes. Infrastructure failure stops the pipeline for
inspection rather than silently choosing a different candidate.

A hosted job marked `COMPLETED` can still contain an unsuccessful editor attempt.
If the editor reaches its declared turn limit without submitting final JSON, the
controller records `turn_limit_without_submission`, keeps the incumbent, and counts
that attempt toward the round limit. It passes setup/tool errors to the next round
as feedback. It never extracts a candidate from unfinished tool-call arguments.
Unexpected errors, other truncation reasons, and contradictory verifier reports still
stop publication. This distinction fixes the first live tool-assisted run's collection
failure without replaying or discarding its first attempt.

## What happens

`run` creates the faulty baseline, publishes it and the demo, submits round one,
waits for Prime, validates the trace and verifier arithmetic, publishes the selected
checkpoint, and calls the Space using that immutable revision. It repeats for three
rounds, then publishes the independent final audit. The controller checks publication
hashes, the recorded model/tool budget, model identity, evaluation seeds, failure-case
identity, targeted replay outcomes, candidate edits, and exact deployment episode equivalence.
Legacy plans continue to require exactly one editor call and zero tool calls.

The verifier, not the LLM, computes scores. The LLM returns JSON assignments to
existing tensors; its text is never executed by the controller. The experiment
does not modify the editor LLM's weights. Rejected policies remain in the report.

Commands can also be run individually, with the same `uv run ... python` prefix:

```text
pipeline/control.py init
pipeline/control.py publish-model
pipeline/control.py publish-space
pipeline/control.py submit --round 0
pipeline/control.py collect --round 0
pipeline/control.py publish-model
pipeline/control.py verify-deployment --round 0
```

Rounds are zero-based. `collect` requires the hosted evaluation to be completed.
Use `prime eval get EVALUATION_ID --output json` to inspect it. Evaluation submissions
and outcomes are saved under the run directory; rerunning `run` resumes those IDs.
Never delete a submission record to retry a completed model response. If submission
output is ambiguous, inspect the Prime dashboard before launching another evaluation.
HF startup can take a few minutes; if the Space is still building when the deployment
check runs, wait for it to become Running and resume the same command.

The hosted platform currently selects the latest published environment even when a
version is requested. The controller checks the actual trace version against the
frozen plan and stops if they differ. The published demonstration used 0.2.1;
check the environment version before beginning another run.

## Artifacts and tests

`plan.json` freezes the experiment settings. `state.json` records the current accepted
policy and progress. Every round preserves its baseline, response, candidate, verifier
report, publication revision, and deployed episode. Only explicitly selected audit
files are uploaded; raw service traces and account configuration remain local.

Run the small development suite with `python -m unittest discover -s tests -v`
after installing NumPy and safetensors. Those tests run local toy games; they are
separate from the experiment's hosted evaluations. The controller tests use synthetic
traces and catch tampered metrics, unexpected tools, and incorrect promotion records.

The current taskset supplies the tool-assisted configuration through its failure-driven
mode; the controller writes the matching harness limits into the hosted request.
Practice scores are untrusted diagnostics. Only the independent verifier promotes edits.
