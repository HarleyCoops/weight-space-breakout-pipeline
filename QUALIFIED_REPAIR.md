# Automatic submission of the first qualifying edit

`qualified-repair-v1`, introduced in environment 0.4.5, tests one tool-assisted
repair of a deliberately damaged 17,253-parameter Breakout policy. The LLM writes
the actual weight assignments. A predefined practice check decides when to end
editing automatically; a separate verifier decides whether to publish those weights.
Earlier experiments retain their original rules and results.

## Frozen procedure

1. Publish the known-bad reversed-controls baseline to a fresh model repository.
2. Prime hosts Qwen `qwen/qwen3.6-35b-a3b` with temperature 0.6 and 16,384 output
   tokens per response. The editor has `repair_inspect_policy` and
   `repair_propose_edits`. Each proposal contains a diagnosis, expected behavioral
   improvement, and a complete patch of at most 64 scalar assignments in [-100,100].
   Proposals always start from the baseline; changes do not accumulate across calls.
3. Test every valid proposal on the same 32 practice seeds, 4040000..4040031.
   The **first** proposal with an actual weight change, mean paired return gain
   strictly greater than 0.10, and gain strictly greater than twice its paired
   standard error is frozen automatically. The proposal is the final editor action.
   There is no separate submit tool and no opportunity to keep searching after
   qualification. Nonqualifying proposals receive practice feedback.
4. The trusted verifier reproduces practice qualification, then tests the frozen
   candidate on 512 separate matched seeds, 4060000..4060511. Promotion requires
   an actual weight change, gain >0.10, and gain >2 paired standard errors on this
   independent set. A practice pass can still be rejected here. There is one
   submitted candidate in this trial; rejection ends it with the baseline retained.
5. Audit the selected checkpoint on 1,024 separate seeds, 4090000..4091023.
   This final audit is never fed back to the editor or used for selection. Publish
   the selected checkpoint and records to Hugging Face, then reproduce the pinned
   verifier episode using the deployed Space and compare its actions and outcome.

For paired differences `d = candidate_return - baseline_return`, the standard
error is `sample_std(d) / sqrt(number_of_games)`. Practice qualification is a
stopping heuristic, not independent statistical evidence after repeated searches.
The promotion set is evaluated only after the candidate has been frozen.

The recorded miss at seed 314159 is diagnostic only. The editor need not prevent
that particular miss. No golden patch, numerical candidate solver, or gradient
training is supplied. There is no editor-turn, tool-call, or aggregate output-token
cap. Operational safeguards remain: a 60-minute hosted service/controller timeout
and the per-response output allowance. A provider error or timeout cannot promote
an uncommitted practice patch; failed attempts remain in the audit.

The collector checks the ordered proposal history, exact submitted arguments,
trusted receipt hash and proposal number, reproduced practice qualification, and
independent gate arithmetic. The framework can stop on the committed state before
recording the final tool response; that case requires the trusted state receipt,
not an inference from unexecuted tool arguments.

This measures one planted-fault repair and deployment. It does not demonstrate
unknown-failure recovery, sustained recursive improvement, or language-model repair.
The edited network is the small game policy, not Qwen's language-model weights.
If the edit restores the intact handwritten reference, report that explicitly.

## Run from PowerShell

Publish environment 0.4.5 only after live trials requiring older versions finish.
Prime currently runs the latest published environment; the collector checks the
actual version against the frozen plan. Use new repository and run-directory names
for a new trial. Resuming an existing directory preserves its plan and hosted ID.

```powershell
$env:PRIME_CLI = '/home/chris/.local/bin/prime'
$env:REPAIR_PROTOCOL = 'qualified-repair-v1'
$env:REPAIR_RUN_DIR = Join-Path $PWD 'artifacts/qualified-repair-run-01'
$env:REPAIR_MODEL_ID = 'HarleyCooper/weight-space-breakout-qualified-repair-01'
$env:REPAIR_SPACE_ID = 'HarleyCooper/weight-space-breakout-qualified-repair-demo-01'
uv run --no-project --with-requirements pipeline/requirements.txt python pipeline/control.py run
```

The Space plays the currently published checkpoint. Pressing Play never starts
another repair job or rewrites weights. Rejected edits and failed executions are
retained for inspection; they do not replace the incumbent policy.

## Attribution

This is an independent conceptual replication attempt inspired by Rikiya Takehi's
[weight-space programming article](https://rikiyat.github.io/blog/weight-space-programming/),
[published experiment logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game),
and [original post](https://x.com/rikiyatakehi/status/2117313742891327994).
Our independently implemented game, handwritten policy, tool access, and stopping
rule differ from the original experiment. This is not an exact reproduction.
