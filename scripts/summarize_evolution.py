#!/usr/bin/env python3
"""Aggregate complete adaptive/frozen pairs without pooling different protocols."""
import _bootstrap
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


PAIRS = (("harness", "harness-frozen"), ("multiagent", "multiagent-frozen"),
         ("weight-ttt", "weight-frozen"))


def describe(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values}


def analyze(folder):
    summaries = json.loads((folder / "summary.json").read_text())
    if not summaries or any(not s["complete"] for s in summaries):
        raise ValueError(f"{folder}: incomplete runs")
    protocol_keys = ("data_label", "dataset_for_search", "search_hp", "budget_attempts",
                     "warm_start_attempts", "training_seed_policy", "data_sha256")
    protocols = {tuple(s[k] for k in protocol_keys) for s in summaries}
    if len(protocols) != 1:
        raise ValueError("A cohort must use the same data, split, recipe and budget")
    groups = defaultdict(list)
    identities = set()
    for row in summaries:
        key = (row["method"], row.get("model_label"))
        identity = key + (row["seed"],)
        if identity in identities:
            raise ValueError(f"Duplicate run: {identity}")
        identities.add(identity)
        groups[key].append(row)
    aggregate = []
    numeric = ("best_val_accuracy", "unique_evaluations_including_warm", "invalid_excluding_warm",
               "duplicates_excluding_warm", "model_call_count", "search_elapsed_seconds", "adaptation_seconds")
    for (method, model), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["seed"])
        metrics = {key: describe([r[key] for r in rows]) for key in numeric
                   if all(isinstance(r.get(key), (int, float)) for r in rows)}
        for key in sorted({k for r in rows for k in r if "_test_accuracy_hp" in k}):
            if all(key in r for r in rows):
                metrics[key] = describe([r[key] for r in rows])
        evidence = {}
        for key in ("promotions", "rejections", "program_versions", "generation", "births",
                    "total_updates", "weight_delta_l2"):
            values = [r.get("adaptation_summary", {}).get(key) for r in rows]
            if all(isinstance(v, (int, float)) for v in values):
                evidence[key] = describe(values)
        aggregate.append({"method": method, "model_label": model, "seeds": [r["seed"] for r in rows],
                          "metrics": metrics, "mechanism_evidence": evidence})
    paired = []
    for adaptive, frozen in PAIRS:
        models = {model for method, model in groups if method == adaptive}
        for model in models:
            if (frozen, model) not in groups:
                continue
            active = {r["seed"]: r for r in groups[(adaptive, model)]}
            fixed = {r["seed"]: r for r in groups[(frozen, model)]}
            if set(active) != set(fixed):
                raise ValueError("Frozen and adaptive runs must use identical seed sets")
            seeds = sorted(active)
            metrics = {}
            for key in ("best_val_accuracy", "cifar10_test_accuracy_hp200"):
                if all(key in active[s] and key in fixed[s] for s in seeds):
                    differences = [active[s][key] - fixed[s][key] for s in seeds]
                    metrics[key] = dict(describe(differences),
                                        wins=sum(d > 1e-10 for d in differences),
                                        ties=sum(abs(d) <= 1e-10 for d in differences),
                                        losses=sum(d < -1e-10 for d in differences))
            paired.append({"adaptive": adaptive, "frozen": frozen, "model_label": model,
                           "seeds": seeds, "adaptive_minus_frozen": metrics})
    return {"protocol": dict(zip(protocol_keys, next(iter(protocols)))),
            "runs": len(summaries), "groups": aggregate, "paired_comparisons": paired}


def write_report(folder, report):
    p = report["protocol"]
    lines = ["# Self-evolution experiment results", "",
             f"{p['data_label']}, {p['dataset_for_search']}, {p['search_hp']} epochs. "
             f"Each run uses {p['budget_attempts']} attempts including {p['warm_start_attempts']} shared initial candidates.", "",
             "Scores below are mean ± sample SD across search seeds. Test values are added only by the separate finalization command.", "",
             "| Method | Backend | Seeds | Validation (%) | CIFAR-10 test (%) |", "| --- | --- | ---: | ---: | ---: |"]
    for group in report["groups"]:
        metrics = group["metrics"]
        val = metrics["best_val_accuracy"]
        test = metrics.get("cifar10_test_accuracy_hp200")
        test_text = f"{test['mean']:.4f} ± {test['sd']:.4f}" if test else "Deferred"
        lines.append(f"| {group['method']} | {group['model_label']} | {val['n']} | {val['mean']:.4f} ± {val['sd']:.4f} | {test_text} |")
    lines += ["", "## Matched adaptation effect", "",
              "Each difference is adaptive minus frozen for the same initialization seed. Positive values favor adaptation. These descriptive comparisons are not significance tests.", "",
              "| Adaptive method | Validation difference (pp) | Wins / ties / losses |", "| --- | ---: | ---: |"]
    for pair in report["paired_comparisons"]:
        delta = pair["adaptive_minus_frozen"]["best_val_accuracy"]
        lines.append(f"| {pair['adaptive']} | {delta['mean']:+.4f} ± {delta['sd']:.4f} | {delta['wins']} / {delta['ties']} / {delta['losses']} |")
    lines += ["", "## Mechanism and cost evidence", ""]
    for group in report["groups"]:
        evidence = [f"{name}: mean {value['mean']:.5g}" for name, value in group["mechanism_evidence"].items()]
        for key in ("model_call_count", "invalid_excluding_warm", "duplicates_excluding_warm", "search_elapsed_seconds", "adaptation_seconds"):
            if key in group["metrics"]:
                evidence.append(f"{key}: mean {group['metrics'][key]['mean']:.5g}")
        lines.append(f"- **{group['method']} / {group['model_label']}** — " + "; ".join(evidence) + ".")
    lines += ["", "See each run's trajectory and events for every candidate, promotion, replacement and gradient update. The `local-policy` and `numpy-policy` backends are explicit small CPU policies, not LLM results. `hf_lora-policy` uses actual language-model adapters. No candidate CNN is trained during these searches.", ""]
    (folder / "EVOLUTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (folder / "evolution_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def plot(folder, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    summaries = json.loads((folder / "summary.json").read_text())
    present = [pair for pair in PAIRS if all(any(s["method"] == m for s in summaries) for m in pair)]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(5.0 * len(present), 4.1), squeeze=False, constrained_layout=True)
    for ax, (adaptive, frozen) in zip(axes.flat, present):
        for method, color, style in ((frozen, "#8A9298", "--"), (adaptive, "#147D69", "-")):
            rows = [s for s in summaries if s["method"] == method]
            curves = [[json.loads(line)["best_so_far_val_accuracy"]
                       for line in (folder / s["trajectory_file"]).read_text().splitlines()] for s in rows]
            values = np.asarray(curves)
            x = np.arange(1, values.shape[1] + 1)
            mean = values.mean(axis=0)
            start = report["protocol"]["warm_start_attempts"] - 1
            ax.plot(x[start:], mean[start:], color=color, ls=style, lw=2, label=f"{method} (n={len(rows)})")
            if len(rows) > 1:
                sd = values.std(axis=0, ddof=1)
                ax.fill_between(x[start:], (mean - sd)[start:], (mean + sd)[start:], color=color, alpha=.10)
        ax.set(title=adaptive, xlabel="Budgeted proposal attempts", ylabel="Best validation accuracy (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.16)
        ax.legend(fontsize=8)
    fig.savefig(folder / "adaptation_curves.png", dpi=180)
    fig.savefig(folder / "adaptation_curves.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    for directory in args.result_dirs:
        folder = Path(directory)
        report = analyze(folder)
        write_report(folder, report)
        if args.plot:
            plot(folder, report)
        print(f"{folder}: {report['runs']} runs summarized")


if __name__ == "__main__":
    main()
