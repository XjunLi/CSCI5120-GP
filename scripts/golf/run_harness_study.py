#!/usr/bin/env python3
"""Run both harness-search budgets on an existing measured table, without training."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import _bootstrap
from golfbench.analysis import FACTORIAL_CONDITIONS
from golfbench.io import read_json, write_json
from golfbench.search import DEFAULT_METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-factorial", action="store_true", help="all eight memory/structure/frontier settings plus random/evolution")
    parser.add_argument("--prior-confirmation-inspected", action="store_true", help="label reused data as exploratory after earlier confirmation was inspected")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(30)))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    methods = ["random", "evolution", *FACTORIAL_CONDITIONS.values()] if args.full_factorial else list(DEFAULT_METHODS)
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        table = read_json(args.table)
        destination = args.output / "table.json"
        if destination.exists():
            if read_json(destination) != table:
                raise ValueError("Output directory contains a different measured table")
        else:
            write_json(destination, table)
        source = {"table_path": os.path.relpath(args.table.resolve(), args.output.resolve()),
                  "table_id": table["table_id"], "new_model_training_runs": 0,
                  "prior_confirmation_inspected": args.prior_confirmation_inspected}
        source_path = args.output / "measurement_source.json"
        if source_path.exists():
            if read_json(source_path) != source:
                raise ValueError("Cannot change the measurement-origin declaration")
        else:
            write_json(source_path, source)
    common = [sys.executable, str(_bootstrap.ROOT / "scripts/golf/replay.py"),
              "--table", str((args.output / "table.json").resolve()),
              "--methods", *methods, "--seeds", *map(str, args.seeds)]
    commands = [common + ["--output", str((args.output / "replay_attempts").resolve())],
                common + ["--output", str((args.output / "replay_cost").resolve()), "--budget", "24", "--cost-fraction", "0.5"],
                [sys.executable, str(_bootstrap.ROOT / "scripts/golf/report.py"), "--root", str(args.output.resolve())]]
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=_bootstrap.ROOT, check=True)


if __name__ == "__main__":
    main()
