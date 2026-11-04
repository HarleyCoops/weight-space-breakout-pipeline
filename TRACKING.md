# Weights & Biases evidence tracking

[Live evidence dashboard](https://wandb.ai/christian-cooper-us/weight-space-breakout?nw=r5pbu5udmj0)
contains 10 imported hosted attempts from the saved experiments, including failed
executions. It separates independent gate comparisons, final audits, and practice
diagnostics. Imported records are historical; they are not new inference runs.

Tracking is optional and cannot select weights or change acceptance. One W&B run
corresponds to one hosted attempt, identified by its Prime evaluation ID and HF
model repository. Re-sync uses the same W&B ID and does not log the same metrics
again. Distinct protocols remain tagged separately; Kimi's separate numerical
repair experiment is not silently merged with these hosted LLM experiments.

- `practice/*`: exploratory scores visible to the editor, plotted by proposal.
- `qualification/*`: reproduced stopping check, not held-out evidence.
- `gate/*`: independent baseline/candidate returns, paired gain and standard error.
  Paired win/loss/tie counts and worst score change expose regressions that a
  positive mean can hide; these diagnostics do not silently change the frozen gate.
- `selection/accepted`: whether the gate promoted the candidate.
- `audit/*`: final independent audit, never used to choose weights.
- `execution/failed`: service/provider failure. Missing game scores stay absent.
- `deployment/verified`: the deployed revision reproduced the expected episode.

Evaluation artifacts preserve the plan, verifier report, publication revision,
deployment check, and available baseline/candidate snapshots. Model artifacts
contain the selected policy's exact sparse weights and hash. Raw prompts, raw
traces, credentials, source-directory contents, and machine telemetry are not
uploaded. Numeric practice fields are extracted from traces locally.

## Publish saved results

Use your existing W&B login, or run `wandb login` yourself. Never paste API keys
into chat. Set the actual saved experiment directory, then:

```powershell
$env:WANDB_ENTITY = 'christian-cooper-us'
$env:WANDB_PROJECT = 'weight-space-breakout'
$env:REPAIR_RUN_DIR = Join-Path $PWD 'artifacts/iterative-repair'
uv run --no-project --with-requirements pipeline/requirements.txt --with-requirements pipeline/tracking-requirements.txt python pipeline/control.py sync-wandb
```

To synchronize future controller runs on exit, also set `REPAIR_WANDB=1` and use
the same two requirements files with `control.py run`. W&B outages produce a
tracking warning and leave the saved experiment and policy decisions intact.
Retry `sync-wandb` later without rerunning inference. The W&B project inherits
your account's visibility settings; HF policy and evidence links remain available.

Gate means from different rounds use different matched seeds. Do not join those
means into a purported guaranteed-monotonic performance curve. Compare baseline
and candidate within a round, and report the final independent audit separately.
