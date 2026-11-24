"""Search baselines and model adapters; final test access is a separate phase."""
import csv
import json
import random
import shlex
import subprocess
import time
from collections import deque
from pathlib import Path

from .architecture import OPS, random_unseen, warm_start
from .protocol import configuration, evaluate_proposal, observation, summarize


def write_summaries(output, summaries):
    output = Path(output)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8")
    fields = list(dict.fromkeys(key for row in summaries for key in row))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in summaries)


def run_search(oracle, method, seed, budget, warm_count, output,
               population_size=10, tournament_size=3, adapter_command=None, adapter_timeout=180, model_label=None):
    config = configuration(oracle, method, seed, budget, warm_count, population_size, tournament_size)
    if method not in ("random", "evolution") and model_label:
        config["model_label"] = model_label
    history, population = [], deque(maxlen=population_size)
    warm, rng = warm_start(seed, warm_count)
    output = Path(output)
    name = f"{method}_seed{seed}"
    trajectory = name + ".jsonl"
    (output / (name + "_config.json")).write_text(json.dumps(config, indent=2), encoding="utf-8")
    start = time.perf_counter()
    with (output / trajectory).open("x", encoding="utf-8") as stream:
        for step in range(1, budget + 1):
            phase, metadata, raw = "warm_start" if step <= warm_count else "search", {}, None
            if phase == "warm_start":
                ops = warm[step - 1]
            elif method == "random" or (method == "evolution" and len(population) < population_size):
                ops = random_unseen(rng, {tuple(r["ops"]) for r in history if r["ops"] is not None})
            elif method == "evolution":
                candidates = rng.sample(list(population), min(tournament_size, len(population)))
                parent, _ = max(candidates, key=lambda item: item[1])
                ops = list(parent)
                edge = rng.randrange(6)
                ops[edge] = rng.choice([op for op in OPS if op != ops[edge]])
            else:
                call_start = time.perf_counter()
                response = subprocess.run(shlex.split(adapter_command), input=json.dumps(observation(config, history)),
                                          text=True, encoding="utf-8", capture_output=True,
                                          timeout=adapter_timeout, check=True)
                raw = response.stdout
                metadata["proposal_seconds"] = time.perf_counter() - call_start
                try:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError("Expected a JSON object")
                    ops = parsed.get("ops")
                    metadata.update({k: parsed[k] for k in ("model", "usage", "rationale") if k in parsed})
                except (json.JSONDecodeError, ValueError):
                    ops = None
            row = evaluate_proposal(oracle, history, ops, phase, raw, metadata)
            row.update({"method": method, "seed": seed, "data_label": oracle.label,
                        "elapsed_seconds": time.perf_counter() - start})
            history.append(row)
            if method == "evolution" and row["ops"] is not None:
                population.append((row["ops"], row["val_accuracy"]))
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    result = summarize(config, history, trajectory)
    result["search_elapsed_seconds"] = time.perf_counter() - start
    return result
