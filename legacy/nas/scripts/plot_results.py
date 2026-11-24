#!/usr/bin/env python3
"""Compare completed validation trajectories with search-run SD bands."""
import _bootstrap
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dirs", nargs="+")
    p.add_argument("--output", default="results/search_curves.png")
    p.add_argument("--seeds", type=int, nargs="+", help="Optional common seed subset for a paired comparison")
    p.add_argument("--from-step", type=int, default=1, help="First displayed attempt, without changing data or metrics")
    a = p.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves, specs, identities = defaultdict(list), set(), set()
    for directory in a.result_dirs:
        folder = Path(directory)
        for summary in json.loads((folder / "summary.json").read_text()):
            if a.seeds is not None and summary["seed"] not in a.seeds:
                continue
            if not summary["complete"]:
                raise ValueError("Refusing to plot an incomplete run")
            if "TOY" in summary["data_label"]:
                raise ValueError("Synthetic checks are not benchmark results")
            model_label = summary.get("model_label")
            identity = summary["method"], model_label, summary["seed"]
            if identity in identities:
                raise ValueError(f"Duplicate run: {identity}")
            identities.add(identity)
            specs.add(tuple(summary[k] for k in ("data_label", "dataset_for_search", "search_hp",
                                                  "budget_attempts", "warm_start_attempts", "training_seed_policy", "data_sha256")))
            rows = [json.loads(line) for line in (folder / summary["trajectory_file"]).read_text().splitlines()]
            if len(rows) != summary["budget_attempts"] or [r["step"] for r in rows] != list(range(1, len(rows) + 1)):
                raise ValueError("Incomplete trajectory")
            curves[(summary["method"], model_label)].append([r["best_so_far_val_accuracy"] for r in rows])
    if len(specs) != 1:
        raise ValueError("All plotted runs must share data/split/hp/budget/initialization/seed policy")
    label, dataset, hp, budget, warm, _, _ = next(iter(specs))
    if not 1 <= a.from_step <= budget:
        raise ValueError("from-step must lie within the completed budget")
    fig, ax = plt.subplots(figsize=(9.2, 5.5), constrained_layout=True)
    palette = {"random": "#687078", "evolution": "#d78a26", "llm": "#16826c", "llm-no-feedback": "#8266aa",
               "chatgpt-feedback": "#16826c", "chatgpt-no-feedback": "#8266aa"}
    x = np.arange(1, budget + 1)
    visible = slice(a.from_step - 1, None)
    pretty = {"random": "Random search", "evolution": "Regularized evolution",
              "chatgpt-feedback": "ChatGPT + feedback", "chatgpt-no-feedback": "ChatGPT, no feedback"}
    for (method, model_label), values in curves.items():
        array = np.asarray(values, dtype=float)
        mean, color = array.mean(0), palette.get(method)
        display = pretty.get(method, method)
        if model_label and not method.startswith("chatgpt"):
            display += " / " + model_label
        ax.plot(x[visible], mean[visible], color=color, linewidth=2, label=f"{display} (n={len(array)})")
        if len(array) > 1:
            sd = array.std(0, ddof=1)
            ax.fill_between(x[visible], (mean - sd)[visible], (mean + sd)[visible], color=color, alpha=.13)
    if a.from_step <= warm:
        ax.axvspan(a.from_step, warm, color="#aaaaaa", alpha=.13, label=f"Shared initialization ({warm})")
    ax.set(xlabel="Proposal attempts (duplicates and invalids included)", ylabel="Best validation accuracy (%)",
           title=f"{label} | {dataset} | {hp} epochs\nMean with search-run SD; frozen table evaluation")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    destination = Path(a.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    if destination.suffix.lower() != ".pdf":
        fig.savefig(destination.with_suffix(".pdf"))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
