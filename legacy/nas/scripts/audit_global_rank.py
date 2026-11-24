#!/usr/bin/env python3
"""Post-search full-space validation ranking. Never expose this output to proposers."""
import _bootstrap
import argparse
import json
from pathlib import Path

from nas_agent_bench.architecture import parse_architecture
from nas_agent_bench.oracles import load_oracle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dirs", nargs="+")
    p.add_argument("--backend", choices=["nats", "table"], default="nats")
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="results/global_rank_audit.json")
    a = p.parse_args()
    selected = []
    for directory in a.result_dirs:
        locked = json.loads((Path(directory) / "locked_winners.json").read_text())
        if any(not r["complete"] or r["completed_attempts"] != r["budget_attempts"] for r in locked):
            raise ValueError("Full-space audit is allowed only after all specified searches finish")
        selected.extend(locked)
    protocols = {(r["dataset_for_search"], r["search_hp"], r["data_sha256"]) for r in selected}
    if len(protocols) != 1:
        raise ValueError("Mismatched search protocols")
    dataset, hp, expected_hash = next(iter(protocols))
    oracle = load_oracle(a.backend, a.data, dataset, hp)
    actual_hash = oracle.metadata.get("data_sha256") or oracle.metadata.get("source_archive_sha256")
    if actual_hash != expected_hash:
        raise ValueError("Global audit does not use the same source archive as search")
    candidates = []
    if a.backend == "nats":
        for index in range(len(oracle.api)):
            ops = parse_architecture(oracle.api.arch(index))
            candidates.append({"benchmark_index": index, "ops": list(ops), **oracle.validation(ops)})
            # Learning curves dominate memory; only scalar validation rows are needed.
            oracle.api.arch2infos_dict.pop(index, None)
    else:
        for ops, row in oracle.rows.items():
            candidates.append({"benchmark_index": row.get("benchmark_index"), "ops": list(ops), **oracle.validation(ops)})
    if len(candidates) != 15625:
        raise ValueError("Expected the complete 15,625-encoding TSS space")
    lookup = {tuple(r["ops"]): r for r in candidates}
    maximum = max(r["val_accuracy"] for r in candidates)
    winners = []
    for r in selected:
        observed = lookup[tuple(r["winner_ops"])]["val_accuracy"]
        if observed != r["best_val_accuracy"]:
            raise ValueError("Full-space value differs from locked search result")
        winners.append({"method": r["method"], "seed": r["seed"], "val_accuracy": observed,
                        "rank": 1 + sum(c["val_accuracy"] > observed for c in candidates),
                        "regret_percentage_points": maximum - observed, "ops": r["winner_ops"]})
    result = {
        "purpose": "Post-hoc evaluator-only analysis after all search winners were locked; not proposal feedback",
        "dataset": dataset, "hp": hp, "training_seed_policy": oracle.training_seed_policy,
        "source_archive_sha256": expected_hash,
        "normalized_table_sha256": oracle.metadata.get("normalized_data_sha256"),
        "candidate_count": len(candidates), "rank_rule": "1 + number of encodings with strictly higher validation accuracy",
        "maximum_validation_accuracy": maximum,
        "global_validation_maximizers": [r for r in candidates if r["val_accuracy"] == maximum],
        "test_ranking_performed": False, "selected_winners": winners,
    }
    destination = Path(a.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {destination}; {len(candidates)} encodings; maximum validation={maximum:.6f}")


if __name__ == "__main__":
    main()
