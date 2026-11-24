#!/usr/bin/env bash
set -euo pipefail
GOLFBENCH_PYTHON="${GOLFBENCH_PYTHON:-python3}"
GOLFBENCH_STAGE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
cd "$(dirname "$0")/../.."
case "$GOLFBENCH_STAGE" in
  data)
    "$GOLFBENCH_PYTHON" vendor/parameter_golf/data/cached_challenge_fineweb.py --variant sp1024 --train-shards 8
    "$GOLFBENCH_PYTHON" scripts/golf/validate_h200_data.py \
      --source-dir vendor/parameter_golf/data/datasets/fineweb10B_sp1024
    if [[ ! -f data/fineweb_sp1024_partition/manifest.json ]]; then
      "$GOLFBENCH_PYTHON" scripts/golf/prepare_data.py --mode fineweb \
        --source-dir vendor/parameter_golf/data/datasets/fineweb10B_sp1024 \
        --tokenizer-path vendor/parameter_golf/data/tokenizers/fineweb_1024_bpe.model \
        --output-dir data/fineweb_sp1024_partition
    fi
    "$GOLFBENCH_PYTHON" scripts/golf/validate_h200_data.py --data-dir data/fineweb_sp1024_partition
    ;;
  preflight)
    "$GOLFBENCH_PYTHON" scripts/golf/validate_h200_data.py --data-dir data/fineweb_sp1024_partition
    "$GOLFBENCH_PYTHON" - <<'PY'
import sys
sys.path.insert(0,"src")
import torch
from golfbench.training import execution_context
assert torch.cuda.is_available(), "CUDA build of PyTorch required"
assert torch.cuda.device_count()==1, "Expose exactly one GPU with CUDA_VISIBLE_DEVICES"
assert "H200" in torch.cuda.get_device_name(0), "This plan requires H200"
context=execution_context("parameter_golf","data/fineweb_sp1024_partition")
print("Hardware:",torch.cuda.get_device_name(0))
print("PyTorch:",torch.__version__)
print("Verified dataset:",context["dataset_fingerprint"])
PY
    "$GOLFBENCH_PYTHON" scripts/golf/run_batch.py --manifest configs/golf/h200_manifest.json \
      --output results/h200/raw --data-dir data/fineweb_sp1024_partition --dry-run
    ;;
  train)
    GOLFBENCH_VALIDATE_DATA=1
    for GOLFBENCH_ARG in "$@"; do
      case "$GOLFBENCH_ARG" in --dry-run|--help|-h) GOLFBENCH_VALIDATE_DATA=0 ;; esac
    done
    if [[ "$GOLFBENCH_VALIDATE_DATA" == 1 ]]; then
      "$GOLFBENCH_PYTHON" scripts/golf/validate_h200_data.py --data-dir data/fineweb_sp1024_partition
    fi
    "$GOLFBENCH_PYTHON" scripts/golf/run_batch.py --manifest configs/golf/h200_manifest.json \
      --output results/h200/raw --data-dir data/fineweb_sp1024_partition "$@"
    ;;
  analyze)
    if [[ ! -f results/h200/table.json ]]; then
      "$GOLFBENCH_PYTHON" scripts/golf/build_table.py --manifest configs/golf/h200_manifest.json \
        --runs results/h200/raw --regime short --output results/h200/table.json
    fi
    "$GOLFBENCH_PYTHON" scripts/golf/replay.py --table results/h200/table.json --output results/h200/replay_attempts
    "$GOLFBENCH_PYTHON" scripts/golf/replay.py --table results/h200/table.json --output results/h200/replay_cost --budget 24 --cost-fraction 0.5
    "$GOLFBENCH_PYTHON" scripts/golf/report.py --root results/h200
    ;;
  confirm)
    "$GOLFBENCH_PYTHON" scripts/golf/confirm.py --table results/h200/table.json \
      --locks results/h200/replay_attempts/locked_winners.json results/h200/replay_cost/locked_winners.json \
      --data-dir data/fineweb_sp1024_partition --output results/h200/confirmation
    "$GOLFBENCH_PYTHON" scripts/golf/report.py --root results/h200
    ;;
  *)
    printf '%s\n' 'Usage: bash scripts/golf/run_h200.sh {data|preflight|train|analyze|confirm}' \
      'Example: CUDA_VISIBLE_DEVICES=0 bash scripts/golf/run_h200.sh train --job-limit 1'
    ;;
esac
