#!/usr/bin/env python3
"""Execute an immutable job plan. Existing matching measurements are resumed."""
import argparse
import json
from pathlib import Path
import time
import _bootstrap
from golfbench.io import read_json, write_json, digest
from golfbench.space import validate_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--upstream-dir")
    p.add_argument("--regime", choices=["smoke", "short", "long"])
    p.add_argument("--job-limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    manifest = read_json(a.manifest)
    validate_manifest(manifest)
    jobs = [j for j in manifest["jobs"] if a.regime is None or j["regime"] == a.regime]
    if a.job_limit is not None:
        if a.job_limit < 1:
            p.error("--job-limit must be positive")
        jobs = jobs[:a.job_limit]
    if a.dry_run:
        print(json.dumps({"manifest_id":manifest["manifest_id"], "jobs":len(jobs),
                          "device_class":manifest["device_class"],
                          "training_budget_seconds":sum(j["train_budget_seconds"] for j in jobs),
                          "sample_job":jobs[0] if jobs else None}, indent=2))
        return
    from golfbench.training import run_training, execution_context
    if manifest["device_class"] != "CPU":
        import torch
        if not torch.cuda.is_available() or manifest["device_class"] not in torch.cuda.get_device_name(0):
            raise RuntimeError(f"This plan requires one {manifest['device_class']}; no jobs launched")
        if torch.cuda.device_count()!=1:
            raise RuntimeError("Expose exactly one GPU, for example CUDA_VISIBLE_DEVICES=0")
    context = execution_context(jobs[0]["backend"], a.data_dir, a.upstream_dir)
    a.output.mkdir(parents=True, exist_ok=True)
    context_path=a.output/"execution_context.json"
    if context_path.exists():
        if read_json(context_path)!=context:
            raise ValueError("code, software or data changed; use a new output directory")
    else:
        write_json(context_path,context)
    plan_path = a.output / "manifest.json"
    if plan_path.exists():
        if read_json(plan_path) != manifest:
            raise ValueError("output directory belongs to a different manifest")
    else:
        write_json(plan_path, manifest)
    start = time.monotonic()
    completed = resumed = 0
    for i, job in enumerate(jobs):
        folder = a.output / job["job_id"]
        result_path = folder / "result.json"
        job_path = folder / "job.json"
        if job_path.exists() and read_json(job_path) != job:
            raise ValueError("job identity changed on resume")
        if not job_path.exists():
            write_json(job_path, job)
        if result_path.exists():
            result = read_json(result_path)
            if result.get("execution_fingerprint")!=digest(context):
                raise ValueError("record execution identity changed")
            if result.get("status")=="unavailable":
                raise ValueError("pending hardware record; use a new output directory on the requested hardware")
            for key in ("candidate_id", "config", "training_seed", "backend", "split", "train_budget_seconds"):
                if result.get(key) != job[key]:
                    raise ValueError(f"resume mismatch for {key}")
            print(f"[{i+1}/{len(jobs)}] resume {job['job_id']} ({result['status']})", flush=True)
            resumed += 1
            continue
        result = run_training(config=job["config"], candidate_id=job["candidate_id"],
                              training_seed=job["training_seed"], output_dir=folder,
                              backend=job["backend"], train_budget_seconds=job["train_budget_seconds"],
                              max_steps=job["max_steps"], split=job["split"],
                              data_dir=a.data_dir, upstream_dir=a.upstream_dir)
        if not result_path.exists():
            write_json(result_path, result)
        completed += 1
        print(f"[{i+1}/{len(jobs)}] {job['job_id']} {result['status']} bpb={result.get('quantized_bpb')} "
              f"train={result.get('train_seconds')}", flush=True)
        if result["status"] == "unavailable":
            raise RuntimeError("required hardware unavailable; no further jobs launched")
        write_json(a.output / "progress.json", {"manifest_id": manifest["manifest_id"],
                   "new_runs": completed, "resumed_runs":resumed, "selected_jobs":len(jobs),
                   "elapsed_seconds":time.monotonic()-start, "last_job_id":job["job_id"]}, overwrite=True)
    print(f"Finished: {completed} new, {resumed} resumed, {time.monotonic()-start:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
