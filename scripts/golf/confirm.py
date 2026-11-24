#!/usr/bin/env python3
import argparse
from pathlib import Path
import statistics
import _bootstrap
from golfbench.io import read_json,write_json,digest
from golfbench.table import TableOracle
from golfbench.confirmation import confirmation_plan


def main():
    p=argparse.ArgumentParser(description="Retrain locked winners with new seeds on hidden confirmation text")
    p.add_argument("--table",required=True)
    p.add_argument("--locks",required=True,nargs="+")
    p.add_argument("--seeds",nargs="+",type=int,default=[100,101,102])
    p.add_argument("--data-dir",required=True)
    p.add_argument("--upstream-dir")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    table=read_json(a.table)
    TableOracle(table)
    locks=[read_json(path) for path in a.locks]
    plan=confirmation_plan(locks,table,a.seeds)
    if a.dry_run:
        print(f"Locked {len(plan['jobs'])//len(a.seeds)} unique configurations; {len(plan['jobs'])} fresh training runs")
        return
    # Import and open prepared data only after validating every winner lock.
    from golfbench.training import run_training,execution_context
    if table["device_class"]!="CPU":
        import torch
        if not torch.cuda.is_available() or table["device_class"] not in torch.cuda.get_device_name(0) or torch.cuda.device_count()!=1:
            raise RuntimeError("confirmation requires exactly one GPU matching the search table")
    context=execution_context(table["backend"],a.data_dir,a.upstream_dir)
    if digest(context)!=table["execution_fingerprint"]:
        raise ValueError("confirmation training implementation/data/software differs from search table")
    a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/"confirmation_plan.json"
    if path.exists():
        if read_json(path)!=plan:
            raise ValueError("confirmation plan cannot be changed after observing hidden scores")
    else:
        write_json(path,plan)
    records=[]
    for i,job in enumerate(plan["jobs"]):
        directory=a.output/job["job_id"]
        path=directory/"result.json"
        if path.exists():
            record=read_json(path)
        else:
            record=run_training(config=job["config"],candidate_id=job["candidate_id"],training_seed=job["training_seed"],
                                output_dir=directory,backend=job["backend"],train_budget_seconds=job["train_budget_seconds"],
                                max_steps=job["max_steps"],split="confirm",data_dir=a.data_dir,upstream_dir=a.upstream_dir)
            if not path.exists():
                write_json(path,record)
        for key in ("candidate_id","config","training_seed","backend","split","train_budget_seconds","max_steps"):
            if record.get(key)!=job[key]:
                raise ValueError(f"confirmation record mismatch: {key}")
        if record.get("execution_fingerprint")!=digest(context):
            raise ValueError("confirmation execution identity changed")
        if record["evaluation_fingerprint"]==table["evaluation_fingerprint"]:
            raise ValueError("confirmation must use a distinct evaluation split")
        records.append(record)
        print(f"[{i+1}/{len(plan['jobs'])}] {job['candidate_id']} seed={job['training_seed']} {record['status']}",flush=True)
    grouped={}
    for cid in sorted({j["candidate_id"] for j in plan["jobs"]}):
        group=[r for r in records if r["candidate_id"]==cid]
        okay=all(r["status"]=="completed" and r["artifact_bytes"]<=16_000_000 and
                 (table["device_class"]=="CPU" or r["train_seconds"]<=table["train_budget_seconds"]+table["budget_tolerance_seconds"])
                 for r in group)
        grouped[cid]={"training_seeds":a.seeds,"all_feasible":okay,
                      "mean_quantized_bpb":statistics.mean(r["quantized_bpb"] for r in group) if okay else None,
                      "std_quantized_bpb":statistics.stdev(r["quantized_bpb"] for r in group) if okay and len(group)>1 else None,
                      "per_seed_quantized_bpb":[r.get("quantized_bpb") for r in group]}
    mapped=[]
    for lock in locks:
        for selection in lock["winners"]:
            cid=selection["winner"]["candidate_id"] if selection["winner"] else None
            mapped.append({"lock_id":lock["lock_id"],"method":selection["method"],"search_seed":selection["search_seed"],
                           "candidate_id":cid,"confirmation":grouped.get(cid)})
    report={"plan_id":plan["plan_id"],"raw_training_runs":len(records),"unique_configurations":len(grouped),
            "training_seconds":sum(r["train_seconds"] for r in records),
            "end_to_end_seconds":sum(r["end_to_end_seconds"] for r in records),
            "by_candidate":grouped,"locked_selections":mapped,
            "selection_after_confirmation":False,
            "independence_note":"Repeated search seeds that select the same configuration share the same confirmation measurements."}
    write_json(a.output/"summary.json",report,overwrite=True)
    print(f"Confirmed {len(grouped)} unique configurations; no winner was reselected")


if __name__=="__main__":
    main()
