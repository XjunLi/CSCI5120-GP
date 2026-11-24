# Initial search experiment protocol

This protocol fixes the first pilot before inspecting method comparisons. It is a search benchmark over existing NAS-Bench-201 topology evaluations, not a network-training experiment.

## Search objective and candidates

- A cell has four ordered nodes and six edges: `0->1, 0->2, 1->2, 0->3, 1->3, 2->3`.
- Each edge selects one operation from `none`, `skip_connect`, `nor_conv_1x1`, `nor_conv_3x3`, `avg_pool_3x3`.
- The search domain contains 15,625 encodings. Uniqueness is by canonical encoding, without extra graph-isomorphism reduction.
- The objective is maximum CIFAR-10 validation accuracy under the benchmark's `cifar10-valid` training split, at the final epoch of the 200-epoch configuration. Dataset provenance must explicitly establish that the supplied score is `x-valid`, not the original test set.
- Evaluation metrics and their aggregation over training trials come from a documented, versioned real benchmark. The runner must reject a table whose validation semantics have not been established.
- Candidate FLOPs and parameter counts are diagnostic observations, not additional optimization objectives in this pilot.

## Budget and paired initialization

- Each run has 50 proposal attempts, including ten shared random initialization candidates.
- Search seeds control the initialization and conventional search randomness. The same seed produces the same ten initial encodings for every method.
- Every submitted duplicate or malformed proposal consumes an attempt. Duplicate encodings use cached scores.
- Random search samples encodings uniformly without replacement.
- Regularized evolution uses population size 10, tournament size 3, single-edge mutation to a different operation, and removal of the oldest population entry. No candidate is rejected merely for containing disconnected operations.
- The initial baseline experiment uses 20 search seeds, 0 through 19. The LLM pilot targets three paired initializations, seeds 0, 1, and 2.

## LLM conditions

The two LLM conditions receive the same task description, structural rules, initial candidates, remaining budget, and previous candidate encodings.

1. **Feedback:** previous validation accuracy, FLOPs, and parameter counts are visible.
2. **No feedback:** all evaluation metrics, including initialization scores and best-so-far values, are hidden. Validity and duplicate status remain visible.

The measured intervention is access to evaluation feedback as a whole, including cost metrics. This is not a validation-score-only ablation.

Each condition and seed uses a separate fresh model context. An in-session ChatGPT proposer must make one architecture submission per interaction and use only the allowlisted session command; it must not inspect dataset files, private state, other runs, literature results, or the evaluator source. Its brief proposal rationale is logged, but success is assessed from observed scores.

Interactive ChatGPT sessions do not expose an API model revision, temperature, token usage, or reproducible model sampling seed. These fields must be recorded as unavailable, not inferred. The shared search seed does not control the service's internal sampling. A separately identified local small-model adapter may provide a reproducible comparison, with its model revision and generation settings recorded.

The session interface enforces response masking and budgets. It does not provide an operating-system security boundary against a proposer that intentionally reads unrelated files. The intended evaluation uses only the allowed tool surface; a hardened multi-tool agent benchmark would additionally separate filesystem and evaluator permissions.

## Held-out evaluation

All search winners are selected solely by validation accuracy, with first-visited tie breaking. No test score is sent to any proposer. Final test lookup is a separate root-controlled operation after the search conditions finish and the winners are locked.

The final CIFAR-10 score uses the benchmark's full-training (`cifar10`) test result for the selected encoding. Optional CIFAR-100 and ImageNet16-120 scores describe the same encoding trained separately on those datasets. They are architecture-transfer measurements, not zero-shot transfer of CIFAR-trained weights.

## Reported outputs

- Validation best-so-far versus proposal attempts, with mean and sample standard deviation across runs.
- Final selected validation and held-out test scores.
- Number of unique queried encodings, invalid and duplicate proposal counts, and proposal time.
- Actual model token usage when provided; otherwise unavailable.
- Complete candidate trajectories and compact provenance: source revision, download checksum, metric fields, aggregation rule, code revision, and model configuration.

For the paired LLM comparison, show all three individual pairs as well as the aggregate. Three runs support a pilot diagnosis; standard-deviation bands are not confidence intervals or evidence of statistical significance. Do not select the best seed for the main result.

## Scope

This initial experiment covers conventional NAS baselines and frozen-LLM iterative architecture search. Changing a candidate architecture is not itself harness self-evolution. The follow-up implementation now modifies and evaluates search programs, multi-agent populations and online policy weights in separate matched experiments; see [SELF_EVOLUTION.md](SELF_EVOLUTION.md). Results from those controllers are kept separate from the frozen-LLM pilot described here.
