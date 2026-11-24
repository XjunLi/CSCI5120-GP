# Self-evolution experiment results

NATS-Bench TSS v1_0-3ffb9, cifar10-valid, 200 epochs. Each run uses 13 attempts including 10 shared initial candidates.

Scores below are mean ± sample SD across search seeds. Test values are added only by the separate finalization command.

| Method | Backend | Seeds | Validation (%) | CIFAR-10 test (%) |
| --- | --- | ---: | ---: | ---: |
| harness | Qwen2.5-0.5B-Instruct | 1 | 90.3380 ± 0.0000 | 93.0400 ± 0.0000 |
| multiagent | Qwen2.5-0.5B-Instruct | 1 | 90.3380 ± 0.0000 | 93.0400 ± 0.0000 |

## Matched adaptation effect

Each difference is adaptive minus frozen for the same initialization seed. Positive values favor adaptation. These descriptive comparisons are not significance tests.

| Adaptive method | Validation difference (pp) | Wins / ties / losses |
| --- | ---: | ---: |

## Mechanism and cost evidence

- **harness / Qwen2.5-0.5B-Instruct** — promotions: mean 0; rejections: mean 1; program_versions: mean 2; model_call_count: mean 4; invalid_excluding_warm: mean 3; duplicates_excluding_warm: mean 0; search_elapsed_seconds: mean 101.15; adaptation_seconds: mean 0.00030533.
- **multiagent / Qwen2.5-0.5B-Instruct** — generation: mean 1; births: mean 1; model_call_count: mean 6; invalid_excluding_warm: mean 1; duplicates_excluding_warm: mean 0; search_elapsed_seconds: mean 127.63; adaptation_seconds: mean 0.00093892.

See each run's trajectory and events for every candidate, promotion, replacement and gradient update. The `local-policy` and `numpy-policy` backends are explicit small CPU policies, not LLM results. `hf_lora-policy` uses actual language-model adapters. No candidate CNN is trained during these searches.
