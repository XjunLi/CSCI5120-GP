"""Build a complete, provenance-consistent table from measured training runs.

Search receives an allowlisted observation. Confirmation records are a separate
dataset and are never imported into a search table. A missing run is an error,
not an invitation to fill a score with a surrogate prediction.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import math
from pathlib import Path
import statistics

from .io import digest, read_json
from .space import validate_manifest


def _finite(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (positive and value <= 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def validate_record(record: dict, job: dict) -> None:
    for key in ("candidate_id", "training_seed", "backend", "split", "config", "train_budget_seconds"):
        if record.get(key) != job.get(key):
            raise ValueError(f"record/job mismatch: {key} for {job['job_id']}")
    if record["split"] != "search":
        raise ValueError("confirmation data must not enter a search table")
    if record.get("max_steps") != job.get("max_steps"):
        raise ValueError("step budget differs from preregistration")
    status = record.get("status")
    if status not in {"completed", "failed", "oom", "timeout", "unavailable"}:
        raise ValueError(f"unknown training status: {status}")
    if status == "unavailable":
        raise ValueError("unavailable hardware is pending work, not an evaluated table row")
    if not record.get("evaluation_fingerprint"):
        raise ValueError("missing evaluation fingerprint")
    if not record.get("hardware_fingerprint"):
        raise ValueError("missing hardware fingerprint")
    if not record.get("execution_fingerprint"):
        raise ValueError("missing execution fingerprint")
    if job["device_class"] != "CPU" and job["device_class"] not in str(record.get("device_class", "")):
        raise ValueError("GPU class differs from manifest")
    if job["device_class"] == "CPU" and str(record.get("device_class", "")).upper() != "CPU":
        raise ValueError("smoke records must identify CPU")
    elapsed = _finite(record.get("train_seconds"), "train_seconds")
    total = _finite(record.get("end_to_end_seconds"), "end_to_end_seconds")
    if total + 1e-6 < elapsed:
        raise ValueError("end-to-end time is less than training time")
    if status == "completed":
        _finite(record.get("quantized_bpb"), "quantized_bpb", positive=True)
        _finite(record.get("val_bpb"), "val_bpb", positive=True)
        size = record.get("artifact_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("invalid artifact byte count")
        if not record.get("provenance"):
            raise ValueError("missing training provenance")
    elif record.get("quantized_bpb") is not None:
        raise ValueError("failed training must not supply a score")


def build_table(manifest: dict, records: list[dict], regime: str) -> dict:
    validate_manifest(manifest)
    jobs = [j for j in manifest["jobs"] if j["regime"] == regime]
    if not jobs:
        raise ValueError("no jobs for requested regime")
    expected = {(j["candidate_id"], j["training_seed"]): j for j in jobs}
    by_key = {}
    for record in records:
        key = (record.get("candidate_id"), record.get("training_seed"))
        if key not in expected:
            raise ValueError(f"unexpected record {key}")
        if key in by_key:
            raise ValueError(f"duplicate training record {key}")
        validate_record(record, expected[key])
        by_key[key] = record
    missing = sorted(set(expected) - set(by_key))
    if missing:
        raise ValueError(f"incomplete table: {len(missing)} missing runs, first={missing[0]}")
    for field in ("backend", "evaluation_fingerprint", "hardware_fingerprint", "execution_fingerprint", "train_budget_seconds", "max_steps"):
        if len({digest(r.get(field)) for r in records}) != 1:
            raise ValueError(f"mixed {field}; use separate tables")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["candidate_id"]].append(record)
    rows = []
    for cid in sorted(grouped):
        group = sorted(grouped[cid], key=lambda r: r["training_seed"])
        if [r["training_seed"] for r in group] != manifest["training_seeds"]:
            raise ValueError("training seed cohort mismatch")
        completed = all(r["status"] == "completed" for r in group)
        feasible = completed and all(r["artifact_bytes"] <= 16_000_000 for r in group)
        # The H200 protocol targets ~600s, not a leaderboard-style exact cutoff.
        # Recompute from actual times; never trust a trainer's boolean alone.
        if manifest["device_class"] != "CPU":
            feasible = feasible and all(r["train_seconds"] <= expected[(cid,r["training_seed"])]["train_budget_seconds"]
                                        + expected[(cid,r["training_seed"])]["budget_tolerance_seconds"] for r in group)
        metric = statistics.mean(r["quantized_bpb"] for r in group) if feasible else None
        train_seconds = sum(r["train_seconds"] for r in group)
        total_seconds = sum(r["end_to_end_seconds"] for r in group)
        rows.append({"candidate_id": cid, "config": group[0]["config"], "val_bpb": metric,
                     "metric": "mean_quantized_roundtrip_bpb", "feasible": feasible,
                     "status": "completed" if feasible else "infeasible",
                     "training_seeds": [r["training_seed"] for r in group],
                     "per_seed_quantized_bpb": [r.get("quantized_bpb") for r in group],
                     "std_bpb": statistics.stdev(r["quantized_bpb"] for r in group) if feasible and len(group)>1 else None,
                     "artifact_bytes_max": max((r.get("artifact_bytes") or 0) for r in group),
                     "train_seconds": train_seconds, "end_to_end_seconds": total_seconds,
                     "charge_seconds": train_seconds, "source_record_hashes": [digest(r) for r in group]})
    first = records[0]
    result = {"schema_version": 1, "manifest_id": manifest["manifest_id"], "space_id": manifest["space_id"],
              "profile": manifest["profile"], "device_class": manifest["device_class"], "regime": regime,
              "backend": first["backend"], "evaluation_fingerprint": first["evaluation_fingerprint"],
              "hardware_fingerprint": first["hardware_fingerprint"],
              "execution_fingerprint": first["execution_fingerprint"],
              "train_budget_seconds": first["train_budget_seconds"], "max_steps": first["max_steps"],
              "training_seeds": manifest["training_seeds"], "rows": rows, "raw_run_count": len(records),
              "offline_training_seconds": sum(r["train_seconds"] for r in records),
              "offline_end_to_end_seconds": sum(r["end_to_end_seconds"] for r in records),
              "cost_definition": "sum of measured training seconds across the complete training-seed cohort",
              "time_policy": jobs[0]["time_policy"], "budget_tolerance_seconds":jobs[0]["budget_tolerance_seconds"],
              "complete": True, "score_source": "measured_training_only"}
    result["table_id"] = digest(result)
    return result


def load_records(manifest: dict, output: str | Path, regime: str) -> list[dict]:
    records = []
    for job in manifest["jobs"]:
        if job["regime"] == regime:
            path = Path(output) / job["job_id"] / "result.json"
            if not path.exists():
                raise ValueError(f"missing measured result: {path}")
            records.append(read_json(path))
    return records


class TableOracle:
    """Whitelisted trusted lookup. This is not an operating-system sandbox."""

    def __init__(self, table: dict, *, cost: str = "training"):
        if table.get("table_id") != digest({k:v for k,v in table.items() if k != "table_id"}):
            raise ValueError("table checksum mismatch")
        if not table.get("complete") or table.get("score_source") != "measured_training_only":
            raise ValueError("only complete measured tables are accepted")
        if cost not in {"training", "end_to_end"}:
            raise ValueError("unknown cost basis")
        self.table_id = table["table_id"]
        self._rows = {row["candidate_id"]: copy.deepcopy(row) for row in table["rows"]}
        self._cost_key = "train_seconds" if cost == "training" else "end_to_end_seconds"
        self.calls = 0

    def candidates(self) -> list[dict]:
        return [{"candidate_id": row["candidate_id"], "config": copy.deepcopy(row["config"]),
                 "charge_seconds": row[self._cost_key]} for row in self._rows.values()]

    def evaluate(self, candidate_id: str) -> dict:
        row = self._rows[candidate_id]
        self.calls += 1
        return {"candidate_id": candidate_id, "val_bpb": row["val_bpb"], "feasible": row["feasible"],
                "status": row["status"], "train_seconds": row["train_seconds"],
                "charge_seconds": row[self._cost_key]}
