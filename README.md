# Benchmarking LLM Agents for Neural Architecture Search

CSCI5120 group project: evaluate whether an LLM can use search feedback to find good neural-network architectures more efficiently than random search and regularized evolution.

The first experiment uses **NATS-Bench TSS**, which shares the NAS-Bench-201 topology space. A candidate selects one of five operations on each of six edges in a four-node cell. All 15,625 candidate encodings already have real training evaluations in the official benchmark. Searching this table requires a CPU; no candidate CNN is trained during these experiments.

## Implemented

- Official NATS-Bench evaluator and a provenance-aware JSON/CSV table evaluator.
- Random search and regularized evolution (population 10, tournament 3, single-edge mutation, oldest-member removal).
- Single-LLM iterative search and a matched condition with evaluation feedback hidden.
- An interactive, one-proposal-at-a-time interface for ChatGPT and a generic JSON model-adapter interface.
- Paired initial candidates, strict attempt budgets, duplicate/invalid accounting, complete trajectories, and delayed held-out test evaluation.
- Protocol tests, summary tables, and publication-friendly PNG/PDF search curves.

This is the initial **frozen-LLM search pilot**. Harness self-evolution, multi-agent search, and online weight adaptation are follow-up experiments. Evolving a candidate architecture is not itself evidence that the search harness has improved.

## Experiment settings

| Setting | Value |
| --- | --- |
| Search space | NATS-Bench TSS, 15,625 encodings |
| Search metric | CIFAR-10 `x-valid`, accessed through `cifar10-valid` |
| Training recipe queried | 200 epochs, final epoch |
| Training trials | Mean of available official runs (`is_random=False`) |
| Per-run budget | 50 total proposal attempts |
| Shared initialization | 10 random candidates, included in the budget |
| Baseline repetitions | 20 seeds, 0 through 19 |
| LLM pilot repetitions | 3 paired initializations, seeds 0, 1, 2 |
| Final winner | Highest validation accuracy; first visited wins ties |
| Held-out evaluation | Separate command after winners are locked |

See [the full protocol](docs/EXPERIMENT_PROTOCOL.md) for the information boundary, cost reporting, and interpretation of the feedback ablation.

## Install

Use Python 3.12 or later for archive preparation; the search package itself supports Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[nats,plots]'
python -m unittest discover -s tests -v
```

The interactive session implementation uses Linux/macOS file locking. On Windows, use WSL2.

## Prepare official data

Download **`NATS-tss-v1_0-3ffb9-simple.tar`** from the [author's download table](https://github.com/D-X-Y/NATS-Bench#preparation-and-download) or its [official Google Drive link](https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view). This metrics-only archive is about 1.15 GB. Raw images and the much larger trained-weight archives are unnecessary.

```bash
python scripts/prepare_data.py \
  --archive /path/to/NATS-tss-v1_0-3ffb9-simple.tar \
  --destination data
```

The preparation command verifies the archive SHA256, extracts it, and records provenance. Keep the data and model weights outside Git; `.gitignore` excludes them.

## Run the two baselines

```bash
python scripts/run_benchmarks.py \
  --backend nats \
  --data data/NATS-tss-v1_0-3ffb9-simple \
  --output results/my_baselines \
  --methods random evolution \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --budget 50 --warm-start 10
```

Every submitted candidate consumes one attempt. Invalid outputs and duplicates count; duplicate scores are cached. Random search avoids duplicates in its proposer by sampling without replacement. An LLM is equally allowed to use the supplied candidate history to avoid repeating itself.

`summary.json` initially contains validation results only. `locked_winners.json` fixes the chosen encodings before any final test lookup.

## Run an LLM through an adapter

The adapter reads one JSON observation from stdin and writes one JSON response to stdout. Only the allowlisted observation is sent to it. The minimum response is:

```json
{"ops":["nor_conv_3x3","nor_conv_1x1","skip_connect","nor_conv_3x3","avg_pool_3x3","skip_connect"]}
```

Optional fields are `model`, `usage`, and a short `rationale`. The outer runner saves the complete raw response. Logs belong on stderr, not stdout.

```bash
python scripts/run_benchmarks.py \
  --backend nats --data data/NATS-tss-v1_0-3ffb9-simple \
  --output results/my_llm \
  --methods llm llm-no-feedback --seeds 0 1 2 \
  --budget 50 --warm-start 10 \
  --model-label your-model-and-version \
  --adapter-command 'python your_adapter.py'
```

In the no-feedback condition, the model still sees candidate encodings, validity, duplicates, and remaining budget, but no accuracy or cost metrics. In both conditions the evaluator selects the final winner by validation after the search ends.

## Interactive ChatGPT proposer

Create a separate session for each condition/seed. The evaluator initializes ten shared candidates:

```bash
python scripts/session.py init \
  --session private_sessions/feedback_seed0 \
  --backend nats --data data/NATS-tss-v1_0-3ffb9-simple \
  --condition feedback --seed 0 --budget 50 --warm-start 10
```

Use `--condition no-feedback` in a separate session for the ablation. A fresh model context may use **only** `status` and `propose`:

```bash
python scripts/session.py status --session private_sessions/feedback_seed0

python scripts/session.py propose \
  --session private_sessions/feedback_seed0 --expected-step 11 \
  --ops '["nor_conv_3x3","nor_conv_1x1","skip_connect","nor_conv_3x3","avg_pool_3x3","skip_connect"]' \
  --rationale 'A brief explanation of this candidate.'
```

Make one proposal at a time until `complete=true`. The model must not inspect private state, the benchmark table, evaluator code, other runs, or external resources. The interface masks scores and enforces the budget, but it is not an OS sandbox. For a hardened coding-agent benchmark, run the proposer without access to evaluator files.

After completion, the evaluator exports the measured trajectory:

```bash
python scripts/session.py export \
  --session private_sessions/feedback_seed0 \
  --output results/my_chatgpt_feedback_seed0
```

Interactive ChatGPT runs do not expose an API revision, temperature, token count, or controllable model sampling seed. These are recorded as unavailable. Search seeds refer to paired initialization, not hidden service randomness.

## Final held-out scoring and plots

Only after all search conditions have finished:

```bash
python scripts/finalize_results.py \
  results/my_baselines results/my_chatgpt_feedback_seed0 \
  --backend nats --data data/NATS-tss-v1_0-3ffb9-simple

python scripts/plot_results.py \
  results/my_baselines results/my_chatgpt_feedback_seed0 \
  --output results/my_comparison.png
```

Pass all completed experiment directories to these commands. The figure reports mean and sample standard deviation, with the actual number of runs shown in the legend. The standard-deviation bands are not confidence intervals.

CIFAR-100 and ImageNet16-120 scores describe the selected architecture trained separately on those target datasets. They do not measure zero-shot transfer of CIFAR-trained weights.

## Research references

- [NATS-Bench paper](https://arxiv.org/abs/2009.00437) and [official API](https://github.com/D-X-Y/NATS-Bench).
- [NAS-Bench-201](https://arxiv.org/abs/2001.00326).
- [Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/abs/1802.01548).
- [CoLLM-NAS](https://arxiv.org/abs/2509.26037).
