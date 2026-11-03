# Extending the loop to Cree and Dakota

Read-only inspection on November 3, 2026. No language-model weights were edited or
training jobs launched during this Breakout build.

## What already exists

| Project | Verified building blocks | Immediate gap for direct weight editing |
| --- | --- | --- |
| [Cree1865](https://github.com/HarleyCoops/Cree1865) | Cree dictionary QA verifier, exact/containment/orthography/character-F1/length channels, Tinker training and inference workflow. Git revision `7d0a7ac2a98307a4f184e84395aa113f00e77308`. | At inspected HF revision `6091549795224ccfeafb0c2e72c11f798b0a3e9c`, `HarleyCooper/Cree1865` contains a model card and images, not downloadable weights. Retrieve/export the actual Tinker checkpoint before attempting parameter edits. |
| [Dakota1890](https://github.com/HarleyCoops/Dakota1890) | Grammar environment, training rubric, held-out evaluation and reward-hacking probes. Git revision `332bbb441de3544d3aedc0c9c35f728634dbd117`. | The published `HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO` has a real rank-32 LoRA adapter (`all-linear`) for `Qwen/Qwen3.6-35B-A3B`, at revision `f2b63463219c69250cc28b5fc008ee051ed2417e`. A compatible hosted base-model runtime is still required. |

Sources inspected: [Cree verifier](https://github.com/HarleyCoops/Cree1865/blob/7d0a7ac2a98307a4f184e84395aa113f00e77308/environments/cree1865_dictionary_qa/cree1865_dictionary_qa/environment.py),
[Dakota held-out evaluator](https://github.com/HarleyCoops/Dakota1890/blob/332bbb441de3544d3aedc0c9c35f728634dbd117/dakota_rl_training/eval_heldout.py),
[Cree model card](https://huggingface.co/HarleyCooper/Cree1865),
[Dakota adapter configuration](https://huggingface.co/HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO/blob/f2b63463219c69250cc28b5fc008ee051ed2417e/adapter_config.json).

## What can transfer

The orchestration pattern transfers: pinned incumbent → failure examples → candidate
edit → independent evaluation → accept/reject → versioned checkpoint → hosted serving.
The model-specific editor and verifier do not transfer unchanged. These are much larger
language models, with tokenization, multiple layers, and distributed capabilities.
Success repairing a simple paddle controller gives no assurance of improving them.

Start with a narrow measurable failure (such as a repeated orthography error or an
unwanted generation loop). Compare direct edits with a no-edit control and the existing
LoRA/GRPO approach. Direct editing and gradient-based training must be reported separately.
Use a separate candidate repository; retain the original adapter and base revision.
An accepted adapter must improve the target test while passing regression checks.

## Evaluation requirements

- Freeze training/development/selection/final-test splits by dictionary entry or grammar
  rule, not just by row. Keep reverse-direction questions and paraphrases of the same
  entry together so they cannot leak across splits.
- Pin the full dataset. Cree's environment can fall back to a packaged smoke dataset
  when the full local data is absent; that fallback is not a broad language benchmark.
- Retain component scores. Cree's character-F1 uses character counts, so it does not
  fully test character order or meaning. Containment and mark-recall can reward extra
  text. Include adversarial probes for these shortcuts.
- Dakota's inspected held-out script labels exact match as its `semantic` channel;
  this is a useful answer check, not a comprehensive measure of meaning. Its optional
  judge is separate. Neither a judge nor string similarity replaces fluent-speaker review.
- Evaluate orthography, translation direction, morphology, unwanted repetition, general
  instruction following, and regressions on unaffected items.
- Keep the final test out of iterative feedback. Use fresh selection sets or a clearly
  documented fixed development set, and limit the number of adaptive attempts.
- Ask relevant speakers to assess language usefulness, variants, and contemporary usage.
  A historical dictionary's spelling conventions and synthetic answers are a bounded
  source of supervision, not an authority over the living language.

## Next concrete experiment

First establish reproducible baseline outputs from a pinned hosted checkpoint. Then
choose one failure family and a bounded edit budget. For Cree, locate the checkpoint
behind the Tinker sampler and export it if available. For Dakota, select and pin the
base plus adapter and provision compatible hosted inference. Larger-model compute
requirements and costs should be scoped before launching the language experiment.
There is no guarantee that recursive edits improve language ability; reject regressions
and report unsuccessful attempts just as in the game experiment.
