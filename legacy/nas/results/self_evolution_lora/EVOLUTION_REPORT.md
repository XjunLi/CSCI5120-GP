# Self-evolution experiment results

NATS-Bench TSS v1_0-3ffb9, cifar10-valid, 200 epochs. Each run uses 26 attempts including 10 shared initial candidates.

Scores below are mean ± sample SD across search seeds. Test values are added only by the separate finalization command.

| Method | Backend | Seeds | Validation (%) | CIFAR-10 test (%) |
| --- | --- | ---: | ---: | ---: |
| weight-frozen | hf_lora-policy | 3 | 90.9869 ± 0.5621 | 93.7583 ± 0.6350 |
| weight-ttt | hf_lora-policy | 3 | 90.9869 ± 0.5621 | 93.7583 ± 0.6350 |

## Matched adaptation effect

Each difference is adaptive minus frozen for the same initialization seed. Positive values favor adaptation. These descriptive comparisons are not significance tests.

| Adaptive method | Validation difference (pp) | Wins / ties / losses |
| --- | ---: | ---: |
| weight-ttt | +0.0000 ± 0.0000 | 0 / 3 / 0 |

## Mechanism and cost evidence

- **weight-frozen / hf_lora-policy** — total_updates: mean 0; weight_delta_l2: mean 0; model_call_count: mean 16; invalid_excluding_warm: mean 0; duplicates_excluding_warm: mean 3.6667; search_elapsed_seconds: mean 31.782; adaptation_seconds: mean 0.012374.
- **weight-ttt / hf_lora-policy** — total_updates: mean 4; weight_delta_l2: mean 0.1142; model_call_count: mean 16; invalid_excluding_warm: mean 0; duplicates_excluding_warm: mean 4; search_elapsed_seconds: mean 78.275; adaptation_seconds: mean 36.868.

See each run's trajectory and events for every candidate, promotion, replacement and gradient update. The `local-policy` and `numpy-policy` backends are explicit small CPU policies, not LLM results. `hf_lora-policy` uses actual language-model adapters. No candidate CNN is trained during these searches.
