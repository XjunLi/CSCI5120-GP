# H200 execution runbook

The user runs GPU experiments. Each candidate uses **one H200 for approximately 10 minutes of training**. The default plan contains 24 configurations × three training seeds = 72 jobs. It does not require an A100/H100 comparison or one-hour runs.

## 1. Environment

Use a server image with a working **CUDA build of PyTorch**. Do not copy the CPU-only virtual environment from the development workspace.

```bash
git clone --branch feat/parameter-golf https://github.com/XjunLi/CSCI5120-GP.git
cd CSCI5120-GP
python3 -m venv --system-site-packages .venv-h200
source .venv-h200/bin/activate
python -m pip install -e '.[test]'
python -m pip install sentencepiece huggingface-hub
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
```

The self-contained vendored trainer imports PyTorch, NumPy and SentencePiece; the cached data downloader additionally uses Hugging Face Hub. The baseline uses CUDA bfloat16, scaled-dot-product attention and `torch.compile`. Keep the environment fixed across the table and confirmation. Actual Python, PyTorch, CUDA and hardware details are saved.

## 2. Freeze and prepare data

```bash
python vendor/parameter_golf/data/cached_challenge_fineweb.py \
  --variant sp1024 --train-shards 8
python scripts/golf/prepare_data.py --mode fineweb \
  --source-dir vendor/parameter_golf/data/datasets/fineweb10B_sp1024 \
  --tokenizer-path vendor/parameter_golf/data/tokenizers/fineweb_1024_bpe.model \
  --output-dir data/fineweb_sp1024_partition
```

Eight shards are the registered training-data prefix for this first runbook. At the pinned export's 100 million tokens per shard, that is 800 million tokens, about 1.6 GB of uint16 payload. The downloader also obtains validation and tokenizer files. More shards may be chosen before preparing a new manifest; do not add them partway through the same experiment. Absolute source paths and symlinks must remain valid while jobs run.

The preparation command creates separate `search/` and `confirm/` validation shards at a BOS document boundary. It retains the official training shards as links and hashes all inputs. Search scores do not represent the official full-validation score.

## 3. Preflight and first candidate

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/golf/run_h200.sh preflight
CUDA_VISIBLE_DEVICES=0 bash scripts/golf/run_h200.sh train --job-limit 1
```

Exactly one H200 must be visible. Inspect the first run's `result.json`, `launch.json` and `stdout.log`: the result should have status `completed`, finite `quantized_bpb`, a real artifact byte count, sensible step/token counts, and the expected H200 hardware. The run is part of the 72-job plan and will be reused.

The registered global training batch is 524,288 tokens: eight accumulation steps × microbatch 64 × context 1024. The validation batch also uses 524,288 tokens. These settings follow the baseline's batch definition and have not been benchmarked on H200 here. If the first job exposes a hardware/runtime issue, fix the recipe and create a new plan/output directory before completing the table. Do not silently lower one candidate's batch.

Training stops after the current step when its measured training timer reaches 600 seconds. A 60-second tolerance flags abnormal overrun at aggregation. Compilation/warmup, data verification, final evaluation and export are additional elapsed time and are all recorded; the complete process may take longer than ten minutes.

## 4. Complete the measured table

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/golf/run_h200.sh train
```

The launcher resumes matching completed jobs. For a remote terminal, run it inside the server's usual persistent terminal/session mechanism. Do not run two launchers against the same output directory. Use independent output directories for concurrent studies; each candidate must still receive one dedicated H200.

Expected training cost is about **12 H200-hours**, plus measured overhead. A failure is retained with its cost. The table builder requires every registered job and will not invent missing scores. Resolve infrastructure failures in a new, clearly identified cohort or preregister a retry policy; do not discard unfavorable observations.

## 5. Search and analyze on CPU

```bash
bash scripts/golf/run_h200.sh analyze
```

This builds `results/h200/table.json`, then runs:

- 7 methods × 30 paired search seeds × 12 attempts, including four shared initial candidates: random, evolution, full memory/structure/frontier adaptation, fully frozen, and three single-component freezes.
- The same methods/seeds with a common ceiling of half the measured table-training cost and at most 24 attempts.

Each query charges all three underlying training replicates. The replay does not spend another 12 GPU hours. Summaries and immutable winner locks are under `results/h200/replay_attempts/` and `results/h200/replay_cost/`.

One candidate configuration query therefore represents approximately **30 training GPU minutes** (three ten-minute seed jobs). A 12-query replay represents approximately six GPU hours of hypothetical online candidate training. The offline table itself is built once for approximately 12 GPU hours and reused by all search methods/seeds.

To measure averaged component effects and interactions, use a separate full-factorial output directory:

```bash
python scripts/golf/run_harness_study.py --table results/h200/table.json \
  --output results/h200_factorial --full-factorial
```

This runs the eight on/off combinations plus random and evolution, under both budgets. It starts no additional candidate training. Compare its own locked winners with `confirm.py` if this is the declared primary study; use all intended locks before looking at confirmation data. The older harness, multiagent and feature-policy methods remain explicitly selectable through `replay.py --methods`. The pretrained-policy weight study is prepared separately in [LOCAL_LORA.md](LOCAL_LORA.md).

## 6. Confirm locked choices

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/golf/run_h200.sh confirm
```

Both replay cohorts' locks are validated first. Unique winning configurations are retrained with seeds 100/101/102 and evaluated on confirmation text. No search choice is changed after confirmation. If K configurations are selected, allow approximately **K/2 additional training H200-hours** plus overhead.

## Alternative direct commands

```bash
python scripts/golf/build_table.py --manifest configs/golf/h200_manifest.json \
  --runs results/h200/raw --regime short --output results/h200/table.json
python scripts/golf/replay.py --table results/h200/table.json \
  --output results/h200/replay_attempts
python scripts/golf/replay.py --table results/h200/table.json \
  --output results/h200/replay_cost --budget 24 --cost-fraction 0.5
python scripts/golf/confirm.py --table results/h200/table.json \
  --locks results/h200/replay_attempts/locked_winners.json results/h200/replay_cost/locked_winners.json \
  --data-dir data/fineweb_sp1024_partition --output results/h200/confirmation
```

Set `GOLFBENCH_PYTHON=/path/to/python` for the shell wrapper if Python is not on the active environment's PATH. Changing the environment after training intentionally causes a resume/confirmation identity check to fail; restore that environment or begin a separately identified cohort.
