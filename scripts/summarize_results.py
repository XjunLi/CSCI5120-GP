#!/usr/bin/env python3
"""Aggregate completed runs; report paired comparisons separately from all runs."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "best_val_accuracy", "cifar10_test_accuracy_hp200",
    "cifar100_test_accuracy_hp200", "ImageNet16-120_test_accuracy_hp200",
    "unique_evaluations_including_warm", "duplicates_excluding_warm",
    "postwarm_mean_incumbent_val_accuracy",
)


def describe(values):
    return {"mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dirs", nargs="+")
    p.add_argument("--paired-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--output", default="results/aggregate_summary.json")
    a = p.parse_args()
    runs, identities, specs, warms = [], set(), set(), {}
    for directory in a.result_dirs:
        folder = Path(directory)
        for saved in json.loads((folder / "summary.json").read_text()):
            r = dict(saved)
            identity = (r["method"], r.get("model_label"), r["seed"])
            if identity in identities:
                raise ValueError(f"Duplicate run: {identity}")
            identities.add(identity)
            if not r["complete"] or not r["test_evaluated"]:
                raise ValueError("Aggregate only completed, finalized runs")
            specs.add(tuple(r[k] for k in (
                "dataset_for_search", "search_hp", "training_seed_policy",
                "data_sha256", "budget_attempts", "warm_start_attempts")))
            rows = [json.loads(line) for line in (folder / r["trajectory_file"]).read_text().splitlines()]
            if [t["step"] for t in rows] != list(range(1, r["budget_attempts"] + 1)):
                raise ValueError("Incomplete or misordered trajectory")
            warm = [t["ops"] for t in rows if t["phase"] == "warm_start"]
            if len(warm) != r["warm_start_attempts"]:
                raise ValueError("Wrong initialization size")
            previous = warms.setdefault(r["seed"], warm)
            if previous != warm:
                raise ValueError("Paired initialization differs across methods")
            best = max(t["val_accuracy"] for t in rows if t.get("val_accuracy") is not None)
            if best != r["best_val_accuracy"]:
                raise ValueError("Summary winner differs from trajectory")
            r["winner_step"] = next(t["step"] for t in rows if t["best_so_far_val_accuracy"] == best)
            r["first_step_at_least_91_5"] = next((t["step"] for t in rows if t["best_so_far_val_accuracy"] >= 91.5), None)
            adaptive = [t["best_so_far_val_accuracy"] for t in rows if t["phase"] != "warm_start"]
            r["postwarm_mean_incumbent_val_accuracy"] = statistics.mean(adaptive)
            r["result_directory"] = folder.as_posix()
            runs.append(r)
    if len(specs) != 1:
        raise ValueError("Runs do not share one evaluation protocol")
    cohorts = {}
    for name, subset in (("all_runs", runs), ("paired_seeds", [r for r in runs if r["seed"] in a.paired_seeds])):
        grouped = defaultdict(list)
        for r in subset:
            grouped[r["method"]].append(r)
        cohorts[name] = {}
        for method, group in grouped.items():
            if name == "paired_seeds" and sorted(r["seed"] for r in group) != sorted(a.paired_seeds):
                raise ValueError(f"Incomplete paired cohort for {method}")
            cohorts[name][method] = {
                "n": len(group), "seeds": sorted(r["seed"] for r in group),
                "metrics": {metric: describe([r[metric] for r in group]) for metric in METRICS},
            }
    paired = {(r["method"], r["seed"]): r for r in runs}
    differences = {}
    if all((method, seed) in paired for method in ("chatgpt-feedback", "chatgpt-no-feedback") for seed in a.paired_seeds):
        for metric in METRICS:
            values = [paired["chatgpt-feedback", seed][metric] - paired["chatgpt-no-feedback", seed][metric] for seed in a.paired_seeds]
            differences[metric] = {"seeds": a.paired_seeds, "paired_differences": values, **describe(values)}
    result = {
        "protocol": dict(zip(("dataset", "hp", "training_seed_policy", "source_data_sha256", "budget", "warm_start"), next(iter(specs)))),
        "cohorts": cohorts, "feedback_minus_no_feedback": differences,
        "total_runs": len(runs), "total_attempts": sum(r["completed_attempts"] for r in runs),
        "chatgpt_postwarm_proposals": sum(r["proposal_attempts_excluding_warm"] for r in runs if r["method"].startswith("chatgpt")),
        "accuracy_unit": "percent; differences in percentage points",
        "uncertainty": "sample SD across search runs; not confidence intervals",
        "exploratory_metrics": ["postwarm_mean_incumbent_val_accuracy", "first_step_at_least_91_5"],
        "runs": sorted(runs, key=lambda r: (r["method"], r["seed"])),
    }
    destination = Path(a.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with destination.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cohort", "method", "n", "seeds", "metric", "mean", "sample_sd"])
        writer.writeheader()
        for cohort, methods in cohorts.items():
            for method, entry in methods.items():
                for metric, stats in entry["metrics"].items():
                    writer.writerow({"cohort": cohort, "method": method, "n": entry["n"], "seeds": ",".join(map(str, entry["seeds"])), "metric": metric, **stats})
    print(f"Saved {destination}: {len(runs)} complete runs")


if __name__ == "__main__":
    main()
