"""Freeze search winners before making confirmation data available."""
from __future__ import annotations
from .io import digest


def confirmation_plan(locks: list[dict], table: dict, seeds: list[int]) -> dict:
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError("confirmation seeds must be distinct nonnegative integers")
    if set(seeds)&set(table["training_seeds"]):
        raise ValueError("confirmation requires fresh training seeds")
    if not locks:
        raise ValueError("at least one complete winner lock is required")
    candidates={r["candidate_id"]:r for r in table["rows"]}
    chosen=set()
    for lock in locks:
        if lock.get("lock_id")!=digest({k:v for k,v in lock.items() if k!="lock_id"}):
            raise ValueError("winner lock checksum mismatch")
        if lock["table_id"]!=table["table_id"] or lock["training_seeds_used_for_search"]!=table["training_seeds"]:
            raise ValueError("winner lock belongs to another training table")
        for selection in lock["winners"]:
            winner=selection["winner"]
            if winner is None:
                continue
            cid=winner["candidate_id"]
            if cid not in candidates or candidates[cid]["config"]!=winner["config"]:
                raise ValueError("locked winner config differs from the measured table")
            if candidates[cid]["val_bpb"]!=winner["val_bpb"]:
                raise ValueError("locked score differs from the measured table")
            inner=selection["winner_lock"]
            if inner.get("sha256")!=digest({k:v for k,v in inner.items() if k!="sha256"}):
                raise ValueError("inner winner lock checksum mismatch")
            if inner["candidate_id"]!=cid or inner["method"]!=selection["method"] or inner["search_seed"]!=selection["search_seed"]:
                raise ValueError("winner selection and inner lock differ")
            chosen.add(cid)
    jobs=[]
    for cid in sorted(chosen):
        for seed in seeds:
            job={"candidate_id":cid,"config":candidates[cid]["config"],"training_seed":seed,
                 "backend":table["backend"],"device_class":table["device_class"],"split":"confirm",
                 "train_budget_seconds":table["train_budget_seconds"],"max_steps":table["max_steps"]}
            job["job_id"]=digest(job)[:20]
            jobs.append(job)
    plan={"schema_version":1,"table_id":table["table_id"],"lock_ids":[l["lock_id"] for l in locks],
          "training_seeds":seeds,"jobs":jobs,
          "interpretation":"retrain locked configurations with fresh seeds; evaluate hidden text; never reselect using these scores"}
    plan["plan_id"]=digest(plan)
    return plan
