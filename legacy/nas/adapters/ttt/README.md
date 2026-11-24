# Online policy-weight adaptation

`WeightEvolutionController` implements two weight backends behind the same NAS
controller interface. The table evaluator remains separate; neither backend can
read a benchmark table or a held-out test result.

| Backend | What changes | Requirements |
| --- | --- | --- |
| `numpy` | 405 active autoregressive policy parameters | NumPy, CPU |
| `hf_lora` | Low-rank adapters inside a local causal language model | PyTorch, Transformers, local HF weights; CPU or adapt code for GPU |

The NumPy policy is a small NAS policy, not a language model. The HF backend uses
real LoRA parameter updates in the generative language model; it does not change
candidate CNN weights. Its default targets are `q_proj` and `v_proj`, suitable
for Qwen2.5. Other model families may need different `target_modules`.

## Objective and the relationship to TTT

The closest reference is **Learning to Discover at Test Time** (TTT-Discover),
which adapts an LLM through reinforcement learning on the current discovery
problem: [paper](https://arxiv.org/abs/2601.16175),
[official code](https://github.com/test-time-training/discover).

We implement a small NAS adaptation of its adaptive entropic objective. A batch
contains only proposals sampled by the current policy, with **weights held
constant throughout the batch**. Validation accuracy supplies the reward.
After centering/scaling rewards, bisection chooses beta so that the exponential
reward tilt has at most `entropic_kl` divergence from the uniform batch measure.
With `q_i = softmax(beta * reward)_i`, the stop-gradient advantage is
`A_i = N*q_i - 1` and the minimized loss is

```text
L = -mean_i A_i * log pi_theta(architecture_i)
    + kl_coefficient * mean_i sum_edges KL(pi_theta || pi_initial)
    - entropy_coefficient * mean_i sum_edges H(pi_theta).
```

Conditional KL and entropy are evaluated on the sampled action prefixes. We
take one Adam step on each batch. There is no off-policy replay or imitation of
warm-start examples. The complete replayable log/checkpoint retains each sampled
action and its behavior likelihood. Training rejects stale behavior likelihoods.

Differences from TTT-Discover: a finite-batch mean baseline instead of its
leave-one-out estimator; no PUCT state reuse; six constrained operation choices
instead of free-form reasoning/code; smaller batches and models. This is not a
reproduction of the paper. It also differs from the self-supervised image
adaptation in [Sun et al., ICML 2020](https://proceedings.mlr.press/v119/sun20b.html)
and the learned recurrent TTT layers in
[Sun et al., 2024/ICML 2025](https://arxiv.org/abs/2407.04620).

## Install and configure

```bash
python -m pip install -r adapters/ttt/requirements.txt
```

Use the repository evolution runner with a weight-controller JSON config:

```json
{
  "backend": "hf_lora",
  "model_path": "/absolute/path/to/Qwen2.5-0.5B-Instruct",
  "adaptive": true,
  "batch_size": 4,
  "learning_rate": 0.0001,
  "lora_rank": 4,
  "lora_alpha": 8,
  "target_modules": ["q_proj", "v_proj"],
  "kl_coefficient": 0.02,
  "entropy_coefficient": 0.001,
  "entropic_kl": 0.6931471805599453,
  "temperature": 1.0,
  "num_threads": 2
}
```

The backend loads local files only and runs float32 on CPU. Loading a 0.5B model
needs several GB of RAM, and backward passes take longer than table lookups.
No API key or hosted training service is needed. Keep model files outside Git.

For the cheap NumPy backend use `{"backend":"numpy"}`. Default learning rate is
0.03 for NumPy and 0.0001 for HF. Set `adaptive` to `false` for the exact same
policy and prompt with all updates disabled. Both conditions use the same
proposal budget and share the runner's initial candidates.

## Action likelihood and checkpoints

HF sampling restricts each of six output steps to digit tokens 0 through 4,
mapped to the repository's operation order. Training normalizes over those same
five tokens and uses exactly the six generated token positions. Prompt tokens
receive no policy-gradient loss. The prompt is fixed and model dropout is
disabled, so differences between frozen and adaptive policies come from weights.

All updates use current-run validation rewards. Warm starts are not on-policy
samples and do not train the model. Duplicates still consume attempts and may
supply their cached reward; no candidate is resampled for free. A final partial
batch with at least two actions is trained on `finalize()`; a singleton is logged
and discarded because it has no within-batch reward contrast.

Checkpoints include LoRA parameters, Adam moments/step, local RNG state, the
pending action, partial batch, configuration, and frozen-model content hashes.
Frozen base weights must remain available at the same model path for resuming.
Loading rejects a changed base model, tokenizer, or adapter configuration.
Weight hashes, parameter counts, per-step delta norm, gradient norm, and update
counts are recorded for audit. Each run starts from a fresh policy/adapters;
there is no cross-seed weight reuse.

Run meaningful implementation checks with:

```bash
python -m unittest discover -s tests -p 'test_weight_evolution.py' -v
```

Tests verify analytic gradients, actual weight changes, frozen controls,
autoregressive conditioning, exact checkpoint continuation, held-out exclusion,
and optional real LoRA gradient/backward behavior on a tiny causal LM. A tiny
model software test establishes update correctness, not NAS search quality.
