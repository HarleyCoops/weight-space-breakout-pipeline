# First LLM repair experiment

Editor: `qwen/qwen3.6-35b-a3b`, served by Prime Intellect. One fresh, no-tool request per fault; temperature 0; no reference edits or evaluation feedback supplied.

## All results

Fresh holdout: 1,024 seeded games per policy, candidates frozen before evaluation.

| Fault | Baseline return | LLM-edited return | Gain |
| --- | ---: | ---: | ---: |
| reversed_controls | 1.297 | 2.427 | +1.130 |
| left_bias | 0.656 | 0.369 | -0.287 |
| missing_right | 1.809 | 0.382 | -1.427 |

One of three edits improved performance. The export task (reversed controls) was fixed before generation, not selected after comparing outcomes. Its five scalar edits reversed the action signs and strongly suppressed no-op/up/down. It improved the faulty baseline but did not reach the intact reference tracker (2.952 mean return on this split). These are deliberate faults in a handwritten teaching policy, not edits to the paper's PPO policy.

## Deployment

- Model: https://huggingface.co/HarleyCooper/weight-space-breakout-llm-repaired
- Live CPU inference: https://huggingface.co/spaces/HarleyCooper/weight-space-breakout-inference
- Environment: https://app.primeintellect.ai/dashboard/environments/harleycooper/weight-space-breakout
- Serving model revision: `67c7a2b5a0a8ad99f72d1247f5d7d6926cd875fd`
- Weight SHA-256: `b308ac473bbabf2287b3e62a67e0d562a0ac553a9424e3db5bbbec888aa2b7f0`

Actual hosted prediction checks passed for seeds 0, 7, 1234. Complete remote episodes matched local action sequences, returns, and lengths for the same three seeds. All three of those demonstration episodes scored 0; the aggregate improvement is the 1,024-episode result above. Invalid observation input was rejected. Space hardware was CPU Basic, status RUNNING.

## Provider diagnosis

The local Prime configuration has no provider or routing restriction fields, and the CLI request includes only model, messages, temperature and max_tokens. A minimal direct request to the external Qwen-235B route still returned HTTP 404 citing an upstream provider.only allowlist. This is not a local setting we changed. Prime-hosted Qwen works; the external gateway route remains unresolved. The redacted diagnostic is in `artifacts/prime-routing-diagnostic.json`. No support message was sent.

## Artifacts

- `artifacts/llm-repair-run/`: frozen protocol, exact prompts, requests, responses, candidates, results.
- `artifacts/hf-model/`: public model package, checksums and experiment record.
- `artifacts/hosted-inference-verification.json`: live remote checks.
- `scripts/run_llm_repairs.py`: model-call and evaluation workflow (requires authenticated Prime CLI in WSL).
- `scripts/verify_hosted_inference.py`: rerun the deployed API checks.

## Original-work credit and replication scope

Independent conceptual replication attempt and teaching adaptation of **Sparks of Weight-Space Programming** by **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab** (MIT CSAIL, November 2, 2026). Sources: [article](https://rikiyat.github.io/blog/weight-space-programming/), [original X post](https://x.com/rikiyatakehi/status/2117313742891327994), and [Experiment 1 logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game). The original Experiment 1 identifies games from trained weights. Our repair task uses a new simulator and deliberately damaged handwritten policies; it does not reproduce the original simulator, PPO training, or reported results.

## Demo seed behavior

The original demo defaulted to seed 0 for every request and looped the resulting GIF. Live checks confirmed that two requests with seed 0 returned exactly the same six actions and return 0, while seed 1 produced 22 actions and return 2, and seed 2 produced a different six-action sequence. The updated interface separates fresh random games from replaying a selected seed. The model weights and physics are unchanged.
