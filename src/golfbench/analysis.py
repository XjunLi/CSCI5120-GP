"""Post-search statistics. Full-landscape scores are analyst-only information."""
from __future__ import annotations

import math
import statistics
from itertools import combinations
import numpy as np


PAIRS = [("harness-adaptive", "harness-frozen"),
         ("multiagent-adaptive", "multiagent-frozen"),
         ("weight-adaptive", "weight-frozen"),
         ("memory-frontier-adaptive", "memory-frontier-frozen"),
         ("memory-frontier-adaptive", "memory-frontier-no-memory"),
         ("memory-frontier-adaptive", "memory-frontier-no-structure"),
         ("memory-frontier-adaptive", "memory-frontier-no-frontier")]

FACTORIAL_CONDITIONS = {
    (0, 0, 0): "memory-frontier-frozen",
    (1, 0, 0): "memory-frontier-memory-only",
    (0, 1, 0): "memory-frontier-structure-only",
    (0, 0, 1): "memory-frontier-frontier-only",
    (1, 1, 0): "memory-frontier-no-frontier",
    (1, 0, 1): "memory-frontier-no-structure",
    (0, 1, 1): "memory-frontier-no-memory",
    (1, 1, 1): "memory-frontier-adaptive",
}


def bootstrap_mean(values: list[float], seed: int = 5120, samples: int = 10000) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    x = np.array(values, dtype=float)
    draws = x[rng.integers(len(x), size=(samples, len(x)))].mean(axis=1)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def anytime_regret(run: dict, optimum: float | None) -> float | None:
    """Post-warm-start incumbent quality; unqueried scores enter only via analyst optimum.

    Query budgets use the mean regret after each remaining attempt. Cost budgets
    integrate the incumbent available *before* the next result arrives, from
    the shared warm-start cost to the common cap, divided by that interval.
    Missing feasible incumbents return None rather than an invented penalty.
    """
    trajectory = run.get("trajectory", [])
    if not trajectory or optimum is None:
        return None
    warm = len(run.get("initial_candidate_ids", []))
    cap = run.get("budget_charge_seconds")
    if cap is None:
        end = run.get("budget_attempts", len(trajectory))
        if end <= warm:
            return None
        values = []
        current = None
        index = 0
        for attempt in range(1, end + 1):
            while index < len(trajectory) and trajectory[index]["attempt"] <= attempt:
                current = trajectory[index].get("best_val_bpb")
                index += 1
            if attempt > warm:
                if current is None:
                    return None
                values.append(max(0.0, current - optimum))
        return statistics.mean(values)
    prefix = [row for row in trajectory if row["attempt"] <= warm]
    if not prefix or prefix[-1]["attempt"] != warm:
        return None
    anchor = prefix[-1]["cumulative_charge_seconds"]
    current = prefix[-1].get("best_val_bpb")
    if current is None or cap <= anchor:
        return None
    area = 0.0
    left = anchor
    for row in trajectory:
        if row["attempt"] <= warm:
            continue
        right = min(cap, row["cumulative_charge_seconds"])
        area += max(0.0, right - left) * max(0.0, current - optimum)
        left = right
        if row.get("best_val_bpb") is not None:
            current = row["best_val_bpb"]
    area += max(0.0, cap - left) * max(0.0, current - optimum)
    return area / (cap - anchor)


def factorial_effects(runs: list[dict], optimum: float | None) -> dict | None:
    """Paired full-factorial contrasts, conditional on complete feasible blocks."""
    expected = set(FACTORIAL_CONDITIONS.values())
    if not expected <= {r["method"] for r in runs}:
        return None
    by_key = {(r["method"], r["search_seed"]): r for r in runs}
    seeds = sorted({r["search_seed"] for r in runs if r["method"] in expected})
    labels = ("memory", "structure", "frontier")
    output = {"factor_order": list(labels), "conditions": {
        "".join(map(str, bits)): method for bits, method in FACTORIAL_CONDITIONS.items()},
        "interpretation": "main effects average over other switches; interactions are averaged differences in differences; negative quality effects are favorable",
        "uncertainty_unit": "paired search-seed block on this fixed measured table", "metrics": {}}
    for metric in ("terminal_bpb", "anytime_regret"):
        blocks = []
        for seed in seeds:
            values = {}
            for bits, method in FACTORIAL_CONDITIONS.items():
                run = by_key.get((method, seed))
                if run is None:
                    break
                value = ((run.get("winner") or {}).get("val_bpb") if metric == "terminal_bpb"
                         else anytime_regret(run, optimum))
                if value is None:
                    break
                values[bits] = value
            if len(values) == 8:
                blocks.append(values)
        effects = []
        for order in (1, 2, 3):
            for axes in combinations(range(3), order):
                deltas = [sum((1 if sum(bits[a] == 0 for a in axes) % 2 == 0 else -1) * value
                              for bits, value in block.items()) / (2 ** (3 - order)) for block in blocks]
                effects.append({"factors": [labels[a] for a in axes], "order": order,
                                "mean_effect": statistics.mean(deltas) if deltas else None,
                                "paired_bootstrap_95ci": bootstrap_mean(deltas)})
        output["metrics"][metric] = {"complete_seed_blocks": len(blocks),
                                     "incomplete_seed_blocks": len(seeds) - len(blocks), "effects": effects}
    return output


def summarize_runs(runs: list[dict], table: dict) -> dict:
    optimum = min((r["val_bpb"] for r in table["rows"] if r["feasible"]), default=None)
    methods = sorted({run["method"] for run in runs})
    summary = {}
    for method in methods:
        group = [r for r in runs if r["method"] == method]
        metrics = [r["winner"]["val_bpb"] for r in group if r["winner"] is not None]
        anytime = [v for r in group for v in [anytime_regret(r, optimum)] if v is not None]
        successful = len(metrics)
        summary[method] = {"runs":len(group), "feasible_winner_runs":successful,
                           "mean_best_bpb":statistics.mean(metrics) if metrics else None,
                           "std_best_bpb":statistics.stdev(metrics) if len(metrics)>1 else None,
                           "mean_gap_to_table_optimum":statistics.mean(metrics)-optimum if metrics and optimum else None,
                           "optimum_hit_runs":sum(abs(x-optimum)<1e-12 for x in metrics) if optimum else 0,
                           "mean_attempts":statistics.mean(r["attempts"] for r in group),
                           "mean_charge_seconds":statistics.mean(r["charged_seconds"] for r in group),
                           "anytime_feasible_runs":len(anytime),
                           "mean_anytime_regret":statistics.mean(anytime) if anytime else None}
    comparisons = []
    by_key = {(r["method"], r["search_seed"]):r for r in runs}
    for adaptive, frozen in PAIRS:
        paired_seeds = sorted({r["search_seed"] for r in runs if r["method"]==adaptive}
                              & {r["search_seed"] for r in runs if r["method"]==frozen})
        if not paired_seeds:
            continue
        differences = []
        anytime_differences = []
        adaptive_only=frozen_only=both_failed=0
        for seed in paired_seeds:
            av = anytime_regret(by_key[(adaptive,seed)], optimum)
            fv = anytime_regret(by_key[(frozen,seed)], optimum)
            if av is not None and fv is not None:
                anytime_differences.append(av-fv)
            a, f = by_key[(adaptive,seed)]["winner"], by_key[(frozen,seed)]["winner"]
            if a is not None and f is not None:
                differences.append(a["val_bpb"]-f["val_bpb"])
            elif a is not None:
                adaptive_only+=1
            elif f is not None:
                frozen_only+=1
            else:
                both_failed+=1
        comparisons.append({"adaptive":adaptive,"frozen":frozen,"pairs":len(paired_seeds),
                            "both_feasible_pairs":len(differences),
                            "adaptive_only_feasible_pairs":adaptive_only,"frozen_only_feasible_pairs":frozen_only,
                            "neither_feasible_pairs":both_failed,
                            "difference_condition":"both methods found a feasible winner; also inspect failure counts",
                            "mean_adaptive_minus_frozen_bpb":statistics.mean(differences) if differences else None,
                            "paired_bootstrap_95ci":bootstrap_mean(differences),
                            "paired_anytime_regret":{"both_feasible_pairs":len(anytime_differences),
                                "mean_difference":statistics.mean(anytime_differences) if anytime_differences else None,
                                "paired_bootstrap_95ci":bootstrap_mean(anytime_differences)},
                            "wins":sum(x < -1e-12 for x in differences),
                            "ties":sum(abs(x)<=1e-12 for x in differences),
                            "losses":sum(x > 1e-12 for x in differences)})
    return {"table_id":table["table_id"],"runs":len(runs),"methods":summary,"paired_comparisons":comparisons,
            "factorial_effects":factorial_effects(runs,optimum),
            "anytime_metric_definition":"query: mean incumbent regret after attempts warm+1..budget; cost: left-continuous regret integral from common warm-start cost to cap, divided by that interval; missing incumbents excluded and counted",
            "table_optimum_bpb":optimum,
            "uncertainty_unit":"paired search seed, conditional on this fixed measured table",
            "interpretation":"negative adaptive-minus-frozen difference is better; CPU smoke is not a GPU or LLM result"}


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0]*len(values)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:
            j+=1
        for k in range(i,j):
            ranks[order[k]]=(i+j-1)/2
        i=j
    return ranks


def _pearson(x, y) -> float | None:
    a=np.asarray(x,dtype=float); b=np.asarray(y,dtype=float)
    a=a-a.mean(); b=b-b.mean()
    d=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/d) if d>0 else None


def proxy_calibration(short: dict, long: dict, anchor_ids: list[str]) -> dict:
    for field in ("space_id", "backend", "device_class", "hardware_fingerprint", "evaluation_fingerprint", "execution_fingerprint", "training_seeds"):
        if short[field] != long[field]:
            raise ValueError(f"proxy calibration must hold {field} fixed")
    if short["train_budget_seconds"] >= long["train_budget_seconds"]:
        raise ValueError("reference must have a longer training budget")
    s={r["candidate_id"]:r for r in short["rows"]}; l={r["candidate_id"]:r for r in long["rows"]}
    if set(l) != set(anchor_ids) or not set(anchor_ids)<=set(s):
        raise ValueError("long-run table must match all preregistered anchors")
    if any(not s[c]["feasible"] or not l[c]["feasible"] for c in anchor_ids):
        raise ValueError("calibration requires complete feasible paired anchors; report failures separately")
    x=[s[c]["val_bpb"] for c in anchor_ids]; y=[l[c]["val_bpb"] for c in anchor_ids]
    rho=_pearson(_ranks(x),_ranks(y))
    concordant=discordant=tiex=tiey=0
    for i in range(len(x)):
        for j in range(i):
            a=np.sign(x[i]-x[j]); b=np.sign(y[i]-y[j])
            concordant+=int(a*b>0); discordant+=int(a*b<0)
            tiex+=int(a==0 and b!=0); tiey+=int(b==0 and a!=0)
    denominator=math.sqrt((concordant+discordant+tiex)*(concordant+discordant+tiey))
    tau=(concordant-discordant)/denominator if denominator else None
    rng=np.random.default_rng(5120)
    boots=[]
    for _ in range(5000):
        ids=rng.integers(len(x),size=len(x))
        value=_pearson(_ranks([x[i] for i in ids]),_ranks([y[i] for i in ids]))
        if value is not None:
            boots.append(value)
    k=min(3,len(x))
    short_top=set(sorted(range(len(x)),key=lambda i:(x[i],anchor_ids[i]))[:k])
    long_top=set(sorted(range(len(y)),key=lambda i:(y[i],anchor_ids[i]))[:k])
    choice=min(range(len(x)),key=lambda i:(x[i],anchor_ids[i]))
    return {"short_table_id":short["table_id"],"long_table_id":long["table_id"],
            "anchors":len(anchor_ids),"anchor_candidate_ids":anchor_ids,"spearman":rho,"kendall_tau_b":tau,
            "spearman_bootstrap_95ci":[float(v) for v in np.quantile(boots,[.025,.975])] if boots else None,
            "top_k":k,"top_k_recall":len(short_top&long_top)/k,
            "short_winner_long_regret_on_anchors":y[choice]-min(y),
            "bootstrap_unit":"configuration, after averaging training seeds",
            "scope":"ranking and regret on preregistered anchors, not the unmeasured complete long-run landscape"}
