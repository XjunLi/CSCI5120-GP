#!/usr/bin/env python3
"""Run validation-only searches. Final test scoring is deliberately separate."""
import _bootstrap
import argparse
import json
from pathlib import Path
from nas_agent_bench.engine import run_search, write_summaries
from nas_agent_bench.oracles import load_oracle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["table", "nats"], default="table")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--methods", nargs="+", choices=["random", "evolution", "llm", "llm-no-feedback"], default=["random", "evolution"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--warm-start", type=int, default=10)
    p.add_argument("--population", type=int, default=10)
    p.add_argument("--tournament", type=int, default=3)
    p.add_argument("--dataset", default="cifar10-valid")
    p.add_argument("--hp", default="200")
    p.add_argument("--adapter-command")
    p.add_argument("--adapter-timeout", type=float, default=180)
    p.add_argument("--model-label", help="Model identifier recorded separately from the search method")
    a = p.parse_args()
    if not 1 <= a.warm_start <= a.budget <= 15625 or a.population < a.warm_start or not 1 <= a.tournament <= a.population:
        p.error("Invalid budget, warm-start, population or tournament")
    if len(set(a.methods)) != len(a.methods) or len(set(a.seeds)) != len(a.seeds):
        p.error("Methods and seeds must not contain duplicates")
    if any(m.startswith("llm") for m in a.methods) and not a.adapter_command:
        p.error("LLM methods require --adapter-command")
    output = Path(a.output)
    if output.exists() and any(output.iterdir()):
        p.error("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    oracle = load_oracle(a.backend, a.data, a.dataset, a.hp)
    (output / "config.json").write_text(json.dumps(dict(vars(a), provenance=oracle.metadata), indent=2))
    summaries = []
    for seed in a.seeds:
        for method in a.methods:
            print(f"Searching {method}, seed {seed}, budget {a.budget}", flush=True)
            result = run_search(oracle, method, seed, a.budget, a.warm_start, output,
                                a.population, a.tournament, a.adapter_command, a.adapter_timeout, a.model_label)
            summaries.append(result)
            write_summaries(output, summaries)
    (output / "locked_winners.json").write_text(json.dumps(summaries, indent=2))
    print(f"Searches complete. Winners locked; test scoring deferred. Results: {output}")


if __name__ == "__main__":
    main()
