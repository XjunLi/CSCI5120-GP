#!/usr/bin/env python3
import argparse
import _bootstrap
from golfbench.io import read_json, write_json
from golfbench.table import build_table, load_records


def main():
    p = argparse.ArgumentParser(description="Validate and aggregate measured results, never impute missing scores")
    p.add_argument("--manifest", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--regime", required=True, choices=["smoke", "short", "long"])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    manifest = read_json(a.manifest)
    table = build_table(manifest, load_records(manifest, a.runs, a.regime), a.regime)
    write_json(a.output, table)
    print(f"{table['raw_run_count']} measured runs -> {len(table['rows'])} candidates; table={table['table_id']}")


if __name__ == "__main__":
    main()
