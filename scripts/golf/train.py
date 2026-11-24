#!/usr/bin/env python3
"""Train one measured candidate. See --help for backend and split choices."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from golfbench.training import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON config file, or a file containing a config field")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["cpu_tiny_lm", "parameter_golf", "upstream_single_gpu"], default="cpu_tiny_lm")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-seconds", type=float, default=10)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--split", choices=["search", "confirm"], default="search")
    parser.add_argument("--data-dir")
    parser.add_argument("--upstream-dir")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    result = run_training(config.get("config", config), args.candidate_id, args.seed, args.output_dir,
                          backend=args.backend, train_budget_seconds=args.budget_seconds, max_steps=args.max_steps,
                          split=args.split, data_dir=args.data_dir, upstream_dir=args.upstream_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
