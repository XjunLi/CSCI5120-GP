# Memory, workflow and frontier evolution for Parameter Golf

CSCI5120 project: measure whether experiment feedback should change a search harness's **memory organization, executable decision graph and exploration frontier**, and whether those changes improve compact-language-model search. Pretrained-model parameter adaptation is a prepared second phase, to be run when the user allocates compute.

**Default experiment: one H200 per candidate, approximately 10 minutes of training.** The first design contains 24 configurations and three training seeds: **72 real training jobs, approximately 12 training GPU hours**. Compilation, data checks, evaluation and export are recorded separately. No GPU experiment was run during this implementation; GPU execution is left to the user.

The complete earlier NAS project, including its original results and adapters, is preserved in [`legacy/nas/`](legacy/nas/README.md).

## What is ready

- Pinned Parameter Golf trainer, adapted for a single GPU and a time-fraction learning-rate schedule.
- Immutable candidate/job manifests, complete measured-table validation, execution fingerprints and resume guards.
- A typed memory/structure/frontier harness with independently switchable components, evidence-linked edits, actual node execution and same-history graph diagnostics.
- Random/evolution baselines, three single-component freezes and all eight factorial combinations; earlier harness/multiagent/weight baselines remain available.
- Genuine optional HuggingFace causal-LM LoRA updates for the weight-policy pair, using an explicitly provided local model.
- Separate query-budget and measured-training-cost comparisons; complete trajectories and winner locks.
- Retraining of locked configurations with fresh seeds on a separate confirmation text split.
- A fully executed CPU plumbing experiment: **72 training runs, 480 baseline search replays, and 18 fresh-seed confirmation runs**.
- A separate exploratory harness study: **600 additional replays** of the same measured table, including full-factorial effects and anytime regret. It adds no training jobs.
- **55 new tests and all 50 preserved NAS tests pass** (105 tests; additional parametrized subtests also pass). The new tests include real Transformer training, actual LoRA gradients, oracle isolation, independent component switches and analytical metric checks.

The CPU experiment trains tiny models on an authored text fixture. Its scores are software/experimental-protocol checks, not FineWeb, H200 or pretrained-LLM performance. The optional pretrained LM path is implemented and tested with local test weights; no pretrained-model search result is claimed.

## Start here

| Need | File |
| --- | --- |
| Research question, scope and claims | [Project scope](docs/PROJECT_SCOPE.md) |
| New harness research direction and implementation | [Memory, workflow and frontier evolution](docs/HARNESS_EVOLUTION.md) |
| Exact controls, budgets and statistics | [Evaluation protocol](docs/EVALUATION_PROTOCOL.md) |
| Run the H200 experiment | [H200 runbook](docs/H200_RUNBOOK.md) |
| Current harness results | [Harness factorial report](results/harness_memory_frontier_20261124/REPORT.md) |
| Earlier measurements and confirmation | [CPU pipeline report](results/cpu_smoke_20261124/REPORT.md) |
| Resume this project in another conversation | [Progress and recovery record](docs/PROGRESS.md) |
| Next phase: prepared LM weight adaptation | [Local LoRA policy and launcher](docs/LOCAL_LORA.md) |

## H200 workflow

Use a Python environment with a working CUDA build of PyTorch. From this repository:

```bash
python3 -m pip install -e '.[test]'
python3 -m pip install sentencepiece huggingface-hub

bash scripts/golf/run_h200.sh data
CUDA_VISIBLE_DEVICES=0 bash scripts/golf/run_h200.sh preflight
CUDA_VISIBLE_DEVICES=0 bash scripts/golf/run_h200.sh train --job-limit 1
# Inspect the first result and its raw log; the next command resumes it.
CUDA_VISIBLE_DEVICES=0 bash scripts/golf/run_h200.sh train
bash scripts/golf/run_h200.sh analyze
CUDA_VISIBLE_DEVICES=0 bash scripts/golf/run_h200.sh confirm
```

Set `GOLFBENCH_PYTHON` to an environment's Python executable if needed. The data stage downloads eight fixed training shards, then partitions official validation documents into an 80% search split and a 20% confirmation split. It does not change the tokenizer. Read the runbook before selecting a different training-data prefix or batch size; make that choice before building the table.

## Design space

| Factor | Levels |
| --- | --- |
| Transformer depth | 4, 6, 8 |
| Model width | 256, 384 |
| Tied input/output embeddings | true, false |
| Learning-rate multiplier | 0.7, 1.3 |

The 24 combinations use fixed sequence length 1024, vocabulary 1024, MLP expansion 2, and the same optimizer/export recipe. The global training batch is 524,288 tokens. The pinned upstream single-GPU runner accumulates eight microbatches, giving microbatch size 64 at sequence length 1024. These are registered starting settings, not measured H200 throughput tuning.

The score is the mean **post-compression, reloaded-model BPB** across three training seeds. A feasible model must fit within 16,000,000 bytes including its standalone code and compressed model. Different hardware, datasets, evaluation protocols or execution versions cannot be mixed into one table.

## Methods

| Method | Persistent object | Matched control |
| --- | --- | --- |
| `memory-frontier-adaptive` | Evidence grouping/retrieval, typed workflow nodes/edges/routes, frontier priority channels and focus regions | `memory-frontier-frozen` retains all feedback with fixed update rules |
| `memory-frontier-no-memory` / `no-structure` / `no-frontier` | The other two components may adapt | One component's adaptation is frozen, without deleting observations |
| `harness-adaptive` | Bounded executable search program: parent choice, memory, exploration and mutation | `harness-frozen` executes the same challenger trials but discards promotions |
| `multiagent-adaptive` | Three roles, separate memories, lineage and communication routes | `multiagent-frozen` collaborates and observes feedback with fixed population/routes |
| `weight-adaptive` | NumPy policy weights, or optional causal-LM LoRA adapters | `weight-frozen` uses the same initialized policy without parameter updates |

The default H200 search study runs **seven methods**: random, evolution, full memory/structure/frontier adaptation, fully frozen, and the three component freezes. Memory-only, structure-only and frontier-only arms complete the eight-cell factorial when requested. Older harness/multiagent methods and the weight pair are explicit optional conditions.

Default controllers are local algorithms. Their results must not be described as pretrained LLM agent performance. An architecture becoming better is also distinct from the search procedure becoming better; the adaptive/frozen comparisons target the latter.

## Current finding

The new CPU harness study finds no stable benefit from retaining all three adaptations. Under both budgets, the full harness achieves mean best BPB **3.560348**, versus **3.553285** for its frozen counterpart: **0 wins / 29 ties / 1 loss** across paired seeds. Query-prefix and cost-integrated regret intervals also include zero. Actual workflow edits changed the first subsequent proposal in 16 of 54 same-history comparisons, so state evolution was exercised rather than inferred from a name.

The frozen harness reaches the fixed table optimum in 30/30 runs. This small authored fixture saturates quickly; the results establish implementation, controls and an interpretable negative finding. They do not establish H200 effectiveness. The new study was developed after the earlier CPU confirmation was inspected, so it has no new untouched held-out claim.

## Run the complete harness factorial

Once a measured table exists:

```bash
python scripts/golf/run_harness_study.py --table results/h200/table.json \
  --output results/h200_factorial --full-factorial
```

This runs ten conditions (eight component combinations plus two baselines) × 30 seeds × two budgets, without retraining the candidate models. One configuration query represents all three training seeds, approximately **30 training GPU minutes**; offline table construction is paid once.

## Reproduce the CPU check

Install a CPU or CUDA PyTorch build appropriate to the machine, then:

```bash
python -m pip install -e '.[training,test,llm]'
python -m pytest tests -q
python scripts/golf/prepare_data.py --output-dir data/cpu_fixture
python scripts/golf/run_batch.py --manifest configs/golf/cpu_smoke_manifest.json \
  --output results/my_cpu/raw --data-dir data/cpu_fixture
python scripts/golf/build_table.py --manifest configs/golf/cpu_smoke_manifest.json \
  --runs results/my_cpu/raw --regime smoke --output results/my_cpu/table.json
python scripts/golf/replay.py --table results/my_cpu/table.json --output results/my_cpu/attempts
python scripts/golf/replay.py --table results/my_cpu/table.json --output results/my_cpu/cost \
  --budget 24 --cost-fraction 0.5
python scripts/golf/confirm.py --table results/my_cpu/table.json \
  --locks results/my_cpu/attempts/locked_winners.json results/my_cpu/cost/locked_winners.json \
  --data-dir data/cpu_fixture --output results/my_cpu/confirmation
```

These reproduction commands use the current primary harness methods. The earlier eight-method experiment is preserved as its own immutable cohort. Use a new output directory for a changed implementation or environment. Both committed cohorts include source snapshots matching their recorded fingerprints.

Complete per-seed replay records are stored losslessly in `trajectories.zip` alongside readable summaries, locks and SHA-256 indices. The report generator reads these archives directly. To restore individual records before a compatible resume or inspection:

```bash
python scripts/golf/archive_replays.py unpack \
  results/harness_memory_frontier_20261124/replay_attempts \
  results/harness_memory_frontier_20261124/replay_cost
```

## Relationship to the official task

[Official Parameter Golf](https://github.com/openai/parameter-golf) currently specifies 8×H100 SXM and a 10-minute training limit. This project uses the user's separate **1×H200, approximately 10-minute** task. It uses a development/confirmation partition and therefore does not report canonical leaderboard scores or claim a sixfold speedup over an official one-hour baseline.

Upstream commit: `f5c079314c4877fbb0af378c0abade5a8ca33d3a`. The vendored code and its MIT license are retained in [`vendor/parameter_golf/`](vendor/parameter_golf/UPSTREAM.json).
