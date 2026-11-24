#!/usr/bin/env python3
import argparse
import _bootstrap
from golfbench.io import read_json, write_json
from golfbench.analysis import proxy_calibration
from golfbench.table import TableOracle
from golfbench.space import validate_manifest


def main():
    p=argparse.ArgumentParser(description="Compare measured short/long ranking on preregistered anchors")
    p.add_argument("--short-table",required=True)
    p.add_argument("--long-table",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    short,long,manifest=read_json(a.short_table),read_json(a.long_table),read_json(a.manifest)
    validate_manifest(manifest)
    TableOracle(short);TableOracle(long)
    if short["manifest_id"]!=manifest["manifest_id"] or long["manifest_id"]!=manifest["manifest_id"]:
        raise ValueError("manifest mismatch")
    result=proxy_calibration(short,long,manifest["anchor_candidate_ids"])
    write_json(a.output,result)
    print(result)


if __name__=="__main__":
    main()
