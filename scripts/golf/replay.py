#!/usr/bin/env python3
"""Replay measured tables with paired seeds, lock all winners before confirmation."""
import argparse
import hashlib
from pathlib import Path
import _bootstrap
from golfbench.io import read_json, write_json, digest
from golfbench.table import TableOracle
from golfbench.search import METHODS, DEFAULT_METHODS, run_search
from golfbench.analysis import summarize_runs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table",required=True)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--methods",nargs="+",choices=METHODS,default=list(DEFAULT_METHODS))
    p.add_argument("--seeds",nargs="+",type=int,default=list(range(30)))
    p.add_argument("--budget",type=int,default=12)
    p.add_argument("--warm-start",type=int,default=4)
    costs=p.add_mutually_exclusive_group()
    costs.add_argument("--cost-budget-seconds",type=float)
    costs.add_argument("--cost-fraction",type=float,help="Common budget as fraction of entire measured table cost")
    p.add_argument("--cost-basis",choices=["training","end_to_end"],default="training")
    p.add_argument("--weight-model-path",help="Explicit local HF model for weight conditions; never downloaded automatically")
    p.add_argument("--weight-device",default="cpu",help="Device used by the optional local LM policy, independently charged")
    p.add_argument("--weight-max-input-tokens",type=int,default=2048)
    a=p.parse_args()
    table=read_json(a.table)
    oracle=TableOracle(table,cost=a.cost_basis)
    cap=a.cost_budget_seconds
    if a.cost_fraction is not None:
        if not 0<a.cost_fraction<=1:
            p.error("--cost-fraction must be in (0,1]")
        cap=a.cost_fraction*sum(c["charge_seconds"] for c in oracle.candidates())
    policy=None
    if a.weight_model_path:
        from golfbench.model_policy import HuggingFaceLoRAPolicy
        policy=HuggingFaceLoRAPolicy.from_local(a.weight_model_path,oracle.candidates(),device=a.weight_device,
                                              max_input_tokens=a.weight_max_input_tokens)
    spec={"table_id":table["table_id"],"methods":a.methods,"seeds":a.seeds,
          "budget":a.budget,"warm_start":a.warm_start,"cost_budget_seconds":cap,
          "cost_basis":a.cost_basis,"weight_model_path":a.weight_model_path,
          "weight_model_identity":None if policy is None else policy.identity}
    sources = ["search.py", "analysis.py", "table.py"]
    if any(method.startswith("memory-frontier-") for method in a.methods):
        sources.append("harness_graph.py")
    if policy:
        sources.append("model_policy.py")
    spec["search_implementation_sha256"]={name:hashlib.sha256((_bootstrap.ROOT/"src/golfbench"/name).read_bytes()).hexdigest()
                                            for name in sources}
    a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/"experiment.json"
    if path.exists():
        if read_json(path)!=spec:
            raise ValueError("cannot resume a different experiment in this directory")
    else:
        write_json(path,spec)
    runs=[]
    for seed in a.seeds:
        for method in a.methods:
            path=a.output/f"{method}_seed{seed}.json"
            if path.exists():
                result=read_json(path)
                if result.get("experiment_id")!=digest(spec):
                    raise ValueError("run belongs to a different experiment")
            else:
                oracle=TableOracle(table,cost=a.cost_basis)
                kwargs={}
                if policy is not None and method.startswith("weight-"):
                    kwargs["weight_policy"]=policy
                result=run_search(oracle.candidates(),oracle.evaluate,method,seed,
                                  budget=a.budget,warm_start=a.warm_start,cost_budget_seconds=cap,**kwargs)
                result.update({"table_id":table["table_id"],"experiment_id":digest(spec)})
                write_json(path,result)
            runs.append(result)
    summary=summarize_runs(runs,table)
    summary["experiment"]=spec
    write_json(a.output/"summary.json",summary,overwrite=True)
    lock={"schema_version":1,"table_id":table["table_id"],"experiment_id":digest(spec),
          "training_seeds_used_for_search":table["training_seeds"],
          "winners":[{"method":r["method"],"search_seed":r["search_seed"],"winner":r["winner"],"winner_lock":r["winner_lock"]} for r in runs]}
    lock["lock_id"]=digest(lock)
    existing=a.output/"locked_winners.json"
    if existing.exists():
        if read_json(existing)!=lock:
            raise ValueError("winner lock changed")
    else:
        write_json(existing,lock)
    print(f"{len(runs)} completed search runs; all winners locked; {a.output/'summary.json'}")
    for pair in summary["paired_comparisons"]:
        print(pair["adaptive"], "mean paired difference", pair["mean_adaptive_minus_frozen_bpb"],
              "W/T/L",pair["wins"],pair["ties"],pair["losses"])


if __name__=="__main__":
    main()
