#!/usr/bin/env python3
"""Check the fixed eight-shard training prefix used by run_h200.sh.

This is a lightweight handoff check. The trainer separately verifies the data
manifest, file hashes and prepared links before starting any training.
"""
import argparse
import json
from pathlib import Path


EXPECTED = {f"fineweb_train_{index:06d}.bin" for index in range(8)}


def check_names(names, label):
    actual = set(names)
    if len(names) != 8 or actual != EXPECTED:
        missing = sorted(EXPECTED - actual)
        extra = sorted(actual - EXPECTED)
        duplicates = len(names) - len(actual)
        raise ValueError(
            f"{label} must contain exactly training shards 000000-000007; "
            f"found {len(names)} entries, missing={missing}, extra={extra}, "
            f"duplicates={duplicates}. Use an independent data prefix containing "
            "only those eight shards and a new prepared-data directory; existing "
            "files are not removed. The wrapper's paths must point to that prefix."
        )


def validate_source(source_dir):
    paths = sorted(source_dir.glob("fineweb_train_*.bin"))
    check_names([path.name for path in paths], str(source_dir))
    if any(not path.is_file() for path in paths):
        raise ValueError(f"Every training shard must be a readable file: {source_dir}")


def validate_prepared(data_dir):
    path = data_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    entries = manifest.get("splits", {}).get("train", {}).get("files")
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) or not isinstance(entry.get("name"), str)
        for entry in entries
    ):
        raise ValueError(f"Prepared manifest lacks a valid training-file list: {path}")
    check_names([entry["name"] for entry in entries], str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.source_dir is None and args.data_dir is None:
        parser.error("provide --source-dir and/or --data-dir")
    try:
        if args.source_dir is not None:
            validate_source(args.source_dir)
        if args.data_dir is not None:
            validate_prepared(args.data_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"H200 data-prefix check failed: {exc}\n")
    print("Verified fixed H200 training prefix: eight shards 000000-000007")


if __name__ == "__main__":
    main()
