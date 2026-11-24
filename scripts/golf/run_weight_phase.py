#!/usr/bin/env python3
"""Prepare or run the optional pretrained-policy LoRA study after harness work."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import _bootstrap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", required=True, help="Completed measured table; never starts candidate training")
    p.add_argument("--model-path", required=True, help="Explicit local HuggingFace causal LM directory")
    p.add_argument("--output", required=True, type=Path, help="New output root for the weight-adaptation cohort")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    p.add_argument("--dry-run", action="store_true", help="Print both commands without loading a model or running experiments")
    a = p.parse_args()
    common = [sys.executable, str(_bootstrap.ROOT / "scripts/golf/replay.py"),
              "--table", str(Path(a.table).resolve()), "--methods", "weight-adaptive", "weight-frozen",
              "--weight-model-path", str(Path(a.model_path).resolve()), "--weight-device", a.device,
              "--weight-max-input-tokens", str(a.max_input_tokens), "--seeds", *map(str, a.seeds)]
    commands = [common + ["--output", str((a.output / "replay_attempts").resolve())],
                common + ["--output", str((a.output / "replay_cost").resolve()),
                          "--budget", "24", "--cost-fraction", "0.5"]]
    for command in commands:
        print(shlex.join(command), flush=True)
        if not a.dry_run:
            subprocess.run(command, cwd=_bootstrap.ROOT, check=True)


if __name__ == "__main__":
    main()
