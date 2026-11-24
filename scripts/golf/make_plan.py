#!/usr/bin/env python3
"""Write all immutable designs before looking at any training score."""
import argparse
from pathlib import Path
import _bootstrap
from golfbench.io import write_json
from golfbench.space import make_space, make_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/golf"))
    parser.add_argument("--include-long-calibration", action="store_true", help="Optional 1-hour reference; excluded from default H200 work")
    args = parser.parse_args()
    for profile in ["gpu", "cpu_smoke"]:
        space = make_space(profile)
        write_json(args.output / f"{profile}_space.json", space)
        for hardware in (["CPU"] if profile == "cpu_smoke" else ["H200"]):
            manifest = make_manifest(space, hardware, include_long_calibration=args.include_long_calibration)
            name = "cpu_smoke" if hardware == "CPU" else hardware.lower()
            write_json(args.output / f"{name}_manifest.json", manifest)
            print(name, len(manifest["jobs"]), "jobs;", manifest["training_budget_gpu_hours"], "training GPU hours")


if __name__ == "__main__":
    main()
