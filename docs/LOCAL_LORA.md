# Phase 2: optional local causal-LM weight adaptation

The current primary study is memory/structure/frontier harness adaptation. The user has assigned model-parameter adaptation to a later phase when compute is available. This document describes prepared code; no pretrained-model experiment has been run.

The updated parameters belong to the **search policy** that selects experiments. Candidate language models still undergo their separately budgeted approximately ten-minute training runs. Replaying a measured table lets the later policy study reuse those observations while charging their recorded candidate-training cost.

## Prepared phase launcher

```bash
python scripts/golf/run_weight_phase.py \
  --table results/h200/table.json \
  --model-path /path/to/local/causal-model \
  --output results/h200_weight_phase2 \
  --device cuda:0 --dry-run
```

The dry run only prints commands. When the model and compute are ready, run the same command without `--dry-run`. It executes the adaptive/frozen pair under both query and measured-cost budgets, with fresh per-run adapter/optimizer/sampler state. It does not launch candidate-model training or download pretrained weights. Controller GPU time and token usage are reported separately from lookup-table training charges.

The first weight-adaptation study uses the causal model to select candidate IDs. Having it propose edits to the harness graph is a subsequent extension, not a claimed behavior of this adapter.

## Direct replay interface

The default weight pair uses a small NumPy policy. To evaluate actual LLM weights, provide an existing local HuggingFace causal-model directory containing its tokenizer and model files. Loading is local-only and does not execute remote custom code.

```bash
python -m pip install -e '.[llm]'
python scripts/golf/replay.py --table results/h200/table.json \
  --output results/h200/local_lm_weights \
  --methods weight-adaptive weight-frozen \
  --weight-model-path /path/to/local/model \
  --weight-device cuda --weight-max-input-tokens 4096
```

Use a dedicated output directory. These two methods' results belong to a separate pretrained-model cohort; do not relabel the committed NumPy results.

## Policy and updates

The model sees public configuration settings, available actions and previously measured validation scores. Shared fields appear once in the prompt. Each configuration is assigned a verified distinct one-token action label. Sampling is restricted to available configurations, so invalid free-form generation is not a source of artificial failure.

LoRA adapters update using an on-policy validation advantage, a reference-policy KL penalty, and entropy regularization. The reference distribution comes from the same frozen base model with adapters disabled. The baseline uses history available before the sampled action. Common initialization examples are off-policy and never generate gradients. An unaffordable unevaluated proposal is cancelled without updating.

Every run resets adapters, optimizer and sampling RNG. Adaptive and frozen conditions start identically under the same seed. Base tensors remain frozen. The policy records exact model/config/tokenizer/action-map fingerprints, trainable parameter count, actual update norms, proposal/update time and forward-token accounting.

## Scope

This implements constrained choice-policy adaptation over a known finite design space. It does not claim the full TTT-Discover algorithm, free-form code generation, a learned improvement operator, or pretrained-model gains without measured experiments.

The software tests use randomly initialized tiny GPT-2 models and explicitly synthetic test rewards. They prove that adapters change, base weights remain fixed, frozen controls stay frozen, and checkpoints resume optimizer/sampler state. They do not measure pretrained-model search quality.

## Programmatic API

```python
from golfbench.model_policy import HuggingFaceLoRAPolicy
from golfbench.search import run_search

policy = HuggingFaceLoRAPolicy.from_local(local_path, oracle.candidates(), device="cpu")
result = run_search(oracle.candidates(), oracle.evaluate,
                    method="weight-adaptive", seed=0, budget=12, warm_start=4,
                    weight_policy=policy)
```

The policy exposes `identity`, `summary()`, `save_checkpoint(path)` and `load_checkpoint(path)`. Checkpoints contain adapters, optimizer, sampler and a pending proposal, and reject mismatched model/tokenizer/design identities. The batch replay CLI resumes completed runs; it does not promise automatic recovery halfway through one LM search run. A custom orchestrator can use the policy checkpoint API for that purpose.
