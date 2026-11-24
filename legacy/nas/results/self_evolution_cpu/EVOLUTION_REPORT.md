# Self-evolution experiment results

NATS-Bench TSS v1_0-3ffb9, cifar10-valid, 200 epochs. Each run uses 50 attempts including 10 shared initial candidates.

Scores below are mean ± sample SD across search seeds. Test values are added only by the separate finalization command.

| Method | Backend | Seeds | Validation (%) | CIFAR-10 test (%) |
| --- | --- | ---: | ---: | ---: |
| harness | local-policy | 20 | 91.1475 ± 0.2727 | 93.8445 ± 0.3471 |
| harness-frozen | local-policy | 20 | 91.3125 ± 0.2149 | 94.0203 ± 0.3066 |
| multiagent | local-policy | 20 | 91.2539 ± 0.2928 | 93.9849 ± 0.2659 |
| multiagent-frozen | local-policy | 20 | 91.1571 ± 0.3933 | 93.9633 ± 0.3302 |
| weight-frozen | numpy-policy | 20 | 90.7083 ± 0.5114 | 93.5277 ± 0.5020 |
| weight-ttt | numpy-policy | 20 | 90.7327 ± 0.5515 | 93.6287 ± 0.4416 |

## Matched adaptation effect

Each difference is adaptive minus frozen for the same initialization seed. Positive values favor adaptation. These descriptive comparisons are not significance tests.

| Adaptive method | Validation difference (pp) | Wins / ties / losses |
| --- | ---: | ---: |
| harness | -0.1650 ± 0.2696 | 5 / 1 / 14 |
| multiagent | +0.0968 ± 0.3257 | 8 / 2 / 10 |
| weight-ttt | +0.0245 ± 0.4266 | 3 / 14 / 3 |

## Mechanism and cost evidence

- **harness / local-policy** — promotions: mean 2.35; rejections: mean 2.65; program_versions: mean 6; model_call_count: mean 0; invalid_excluding_warm: mean 0; duplicates_excluding_warm: mean 0.25; search_elapsed_seconds: mean 0.11061; adaptation_seconds: mean 0.00089552.
- **harness-frozen / local-policy** — promotions: mean 0; rejections: mean 0; program_versions: mean 1; model_call_count: mean 0; invalid_excluding_warm: mean 0; duplicates_excluding_warm: mean 0; search_elapsed_seconds: mean 0.10295; adaptation_seconds: mean 0.00060711.
- **multiagent / local-policy** — generation: mean 3; births: mean 3; model_call_count: mean 0; invalid_excluding_warm: mean 0; duplicates_excluding_warm: mean 0; search_elapsed_seconds: mean 0.13368; adaptation_seconds: mean 0.0051112.
- **multiagent-frozen / local-policy** — generation: mean 3; births: mean 0; model_call_count: mean 0; invalid_excluding_warm: mean 0; duplicates_excluding_warm: mean 0; search_elapsed_seconds: mean 0.13323; adaptation_seconds: mean 0.004942.
- **weight-frozen / numpy-policy** — total_updates: mean 0; weight_delta_l2: mean 0; model_call_count: mean 0; invalid_excluding_warm: mean 0; duplicates_excluding_warm: mean 0.05; search_elapsed_seconds: mean 0.1356; adaptation_seconds: mean 0.0070458.
- **weight-ttt / numpy-policy** — total_updates: mean 10; weight_delta_l2: mean 2.2543; model_call_count: mean 0; invalid_excluding_warm: mean 0; duplicates_excluding_warm: mean 0; search_elapsed_seconds: mean 0.22031; adaptation_seconds: mean 0.016243.

See each run's trajectory and events for every candidate, promotion, replacement and gradient update. The `local-policy` and `numpy-policy` backends are explicit small CPU policies, not LLM results. `hf_lora-policy` uses actual language-model adapters. No candidate CNN is trained during these searches.
