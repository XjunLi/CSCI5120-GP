#!/usr/bin/env python3
"""Evaluate held-out tests for locked winners, after every search is finished."""
import _bootstrap
import argparse
import json
from pathlib import Path
from nas_agent_bench.engine import write_summaries
from nas_agent_bench.oracles import load_oracle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dirs", nargs="+")
    p.add_argument("--backend", choices=["table", "nats"], default="table")
    p.add_argument("--data", required=True)
    p.add_argument("--dataset", default="cifar10-valid")
    p.add_argument("--hp", default="200")
    a = p.parse_args()
    loaded = []
    for directory in a.result_dirs:
        folder = Path(directory)
        locked = json.loads((folder / "locked_winners.json").read_text())
        if not locked or any(not row["complete"] or row["completed_attempts"] != row["budget_attempts"] for row in locked):
            raise ValueError("Every requested search must be complete before test access")
        identities = [(row["method"], row["seed"]) for row in locked]
        if len(set(identities)) != len(identities):
            raise ValueError("Locked winners contain duplicate method/seed entries")
        config_path = folder / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if "methods" in config and "seeds" in config:
                expected = {(method, seed) for method in config["methods"] for seed in config["seeds"]}
            elif "method" in config and "seed" in config:
                expected = {(config["method"], config["seed"])}
            else:
                expected = None
            if expected is not None and set(identities) != expected:
                raise ValueError("Every declared condition and seed must finish before test access")
        loaded.append((folder, locked))
    oracle = load_oracle(a.backend, a.data, a.dataset, a.hp)
    fingerprint = oracle.metadata.get("normalized_data_sha256", oracle.metadata.get("data_sha256"))
    for _, rows in loaded:
        if any(row["data_sha256"] != fingerprint or row["search_hp"] != oracle.hp or row["dataset_for_search"] != oracle.dataset for row in rows):
            raise ValueError("Final evaluator does not match the search oracle")
    # No proposal or policy update occurs after this boundary.
    for folder, rows in loaded:
        for row in rows:
            row.update(oracle.final_test(tuple(row["winner_ops"])))
            row["test_evaluated"] = True
            row["transfer_interpretation"] = "same selected architecture, separately trained per target dataset; not zero-shot weights"
        write_summaries(folder, rows)
    print("Held-out scoring completed for all locked winners.")


if __name__ == "__main__":
    main()
