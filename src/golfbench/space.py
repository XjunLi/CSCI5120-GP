"""Preregistered factorial designs. Candidate identity includes the full recipe."""
from __future__ import annotations

import itertools
import random
from .io import digest

UPSTREAM_COMMIT = "f5c079314c4877fbb0af378c0abade5a8ca33d3a"
FACTORS = ("num_layers", "model_dim", "tied_embeddings", "lr_scale")


def make_space(profile: str = "gpu") -> dict:
    if profile not in {"gpu", "cpu_smoke"}:
        raise ValueError("profile must be gpu or cpu_smoke")
    smoke = profile == "cpu_smoke"
    levels = {
        "num_layers": [1, 2, 3] if smoke else [4, 6, 8],
        "model_dim": [32, 48] if smoke else [256, 384],
        "tied_embeddings": [True, False],
        "lr_scale": [0.7, 1.3],
    }
    candidates = []
    for values in itertools.product(*(levels[key] for key in FACTORS)):
        config = dict(zip(FACTORS, values))
        config.update({
            "train_seq_len": 64 if smoke else 1024,
            "eval_seq_len": 64 if smoke else 1024,
            "vocab_size": 256 if smoke else 1024,
            "num_heads": 4 if smoke else config["model_dim"] // 64,
            "num_kv_heads": 4 if smoke else config["model_dim"] // 128,
            "mlp_mult": 2,
            "train_batch_tokens": 256 if smoke else 524288,
            "warmdown_fraction": 0.3,
        })
        if smoke:
            config["base_lr"] = 0.003
        candidates.append({"candidate_id": ("cpu-" if smoke else "gpu-") + digest(config)[:12], "config": config})
    result = {"schema_version": 1, "profile": profile, "factors": levels,
              "backend": "cpu_tiny_lm" if smoke else "parameter_golf",
              "upstream_commit": None if smoke else UPSTREAM_COMMIT,
              "candidates": candidates}
    result["space_id"] = digest(result)
    return result


def choose_anchors(space: dict, count: int = 8) -> list[str]:
    """Score-independent maximin design; tie breaking is fixed by the space."""
    candidates = space["candidates"]
    if not 1 <= count <= len(candidates):
        raise ValueError("invalid anchor count")
    levels = space["factors"]
    features = {c["candidate_id"]: [levels[k].index(c["config"][k]) / max(1, len(levels[k]) - 1)
                                     for k in FACTORS] for c in candidates}
    ids = sorted(features)
    chosen = [ids[0]]
    while len(chosen) < count:
        def distance(cid: str) -> tuple:
            score = min(sum((a-b)**2 for a,b in zip(features[cid], features[seen])) for seen in chosen)
            return (-score, cid)
        chosen.append(min((i for i in ids if i not in chosen), key=distance))
    return chosen


def make_manifest(space: dict, hardware: str = "H200", *, include_long_calibration: bool = False) -> dict:
    smoke = space["profile"] == "cpu_smoke"
    if hardware not in {"H200", "CPU"}:
        raise ValueError("hardware must be H200 or CPU")
    if smoke != (hardware == "CPU"):
        raise ValueError("CPU and GPU spaces are separate tasks")
    anchors = choose_anchors(space)
    jobs = []
    for candidate in space["candidates"]:
        regimes = [("smoke", 10.0, 40)] if smoke else [("short", 600.0, None)]
        if not smoke and include_long_calibration and candidate["candidate_id"] in anchors:
            regimes.append(("long", 3600.0, None))
        for regime, seconds, steps in regimes:
            for seed in [0, 1, 2]:
                job = {**candidate, "training_seed": seed, "backend": space["backend"],
                       "device_class": hardware, "regime": regime, "split": "search",
                       "train_budget_seconds": seconds, "max_steps": steps,
                       "time_policy": "fixed_steps_with_cap" if smoke else "approximate_target",
                       "budget_tolerance_seconds": 0.0 if smoke else 60.0}
                job["job_id"] = digest(job)[:20]
                jobs.append(job)
    random.Random(5120).shuffle(jobs)
    result = {"schema_version": 1, "space_id": space["space_id"], "profile": space["profile"],
              "device_class": hardware, "training_seeds": [0, 1, 2],
              "anchor_selection": "maximin public design; no observed scores; frozen before training",
              "anchor_candidate_ids": anchors, "jobs": jobs,
              "training_budget_gpu_hours": 0 if smoke else sum(j["train_budget_seconds"] for j in jobs) / 3600,
              "status": "planned", "overhead_budget_included": False,
              "long_calibration_enabled": bool(include_long_calibration and not smoke)}
    result["manifest_id"] = digest(result)
    return result


def validate_manifest(manifest: dict) -> None:
    claimed = manifest.get("manifest_id")
    actual = digest({k: v for k, v in manifest.items() if k != "manifest_id"})
    if claimed != actual:
        raise ValueError("manifest content does not match its identity")
    jobs = manifest["jobs"]
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate job identities")
    for job in jobs:
        if digest({k: v for k, v in job.items() if k != "job_id"})[:20] != job["job_id"]:
            raise ValueError("modified job identity")
