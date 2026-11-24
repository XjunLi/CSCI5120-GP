"""Post-search statistics. Full-landscape scores are analyst-only information."""
from __future__ import annotations

import math
import statistics
import numpy as np


PAIRS = [("harness-adaptive", "harness-frozen"),
         ("multiagent-adaptive", "multiagent-frozen"),
         ("weight-adaptive", "weight-frozen")]


def bootstrap_mean(values: list[float], seed: int = 5120, samples: int = 10000) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    x = np.array(values, dtype=float)
    draws = x[rng.integers(len(x), size=(samples, len(x)))].mean(axis=1)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def summarize_runs(runs: list[dict], table: dict) -> dict:
    optimum = min((r["val_bpb"] for r in table["rows"] if r["feasible"]), default=None)
    methods = sorted({run["method"] for run in runs})
    summary = {}
    for method in methods:
        group = [r for r in runs if r["method"] == method]
        metrics = [r["winner"]["val_bpb"] for r in group if r["winner"] is not None]
        successful = len(metrics)
        summary[method] = {"runs":len(group), "feasible_winner_runs":successful,
                           "mean_best_bpb":statistics.mean(metrics) if metrics else None,
                           "std_best_bpb":statistics.stdev(metrics) if len(metrics)>1 else None,
                           "mean_gap_to_table_optimum":statistics.mean(metrics)-optimum if metrics and optimum else None,
                           "optimum_hit_runs":sum(abs(x-optimum)<1e-12 for x in metrics) if optimum else 0,
                           "mean_attempts":statistics.mean(r["attempts"] for r in group),
                           "mean_charge_seconds":statistics.mean(r["charged_seconds"] for r in group)}
    comparisons = []
    by_key = {(r["method"], r["search_seed"]):r for r in runs}
    for adaptive, frozen in PAIRS:
        paired_seeds = sorted({r["search_seed"] for r in runs if r["method"]==adaptive}
                              & {r["search_seed"] for r in runs if r["method"]==frozen})
        if not paired_seeds:
            continue
        differences = []
        adaptive_only=frozen_only=both_failed=0
        for seed in paired_seeds:
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
                            "wins":sum(x < -1e-12 for x in differences),
                            "ties":sum(abs(x)<=1e-12 for x in differences),
                            "losses":sum(x > 1e-12 for x in differences)})
    return {"table_id":table["table_id"],"runs":len(runs),"methods":summary,"paired_comparisons":comparisons,
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
    for field in ("space_id", "backend", "device_class", "hardware_fingerprint", "evaluation_fingerprint", "training_seeds"):
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
