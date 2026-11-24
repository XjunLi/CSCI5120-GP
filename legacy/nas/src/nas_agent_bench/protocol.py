"""Budget accounting, validation-only selection and matched agent observations."""
import json
import time

from .architecture import OPS, EDGES, canonical, validate_ops


def configuration(oracle, method, seed, budget, warm_count, population=10, tournament=3):
    return {"schema_version": 1, "method": method, "seed": seed, "budget": budget,
            "warm_start": warm_count, "population": population, "tournament": tournament,
            "benchmark": oracle.label, "dataset": oracle.dataset, "hp": oracle.hp,
            "training_seed_policy": oracle.training_seed_policy,
            "provenance": oracle.metadata,
            "budget_policy": "Every proposal costs one, including invalids and duplicates; warm starts included",
            "uniqueness": "canonical six-operation encoding; no graph-isomorphism deduplication",
            "random_sampling": "uniform without replacement over encodings",
            "accuracy_unit": "percentage points",
            "objective": "maximize validation accuracy; costs are descriptive only"}


def evaluate_proposal(oracle, history, value, phase="search", raw_response=None, metadata=None):
    start = time.perf_counter()
    row = {"step": len(history) + 1, "phase": phase, "status": "invalid", "ops": None}
    if raw_response is not None:
        row["raw_response"] = raw_response
    if metadata:
        row.update({k: metadata[k] for k in ("model", "usage", "rationale", "proposal_seconds") if k in metadata})
    try:
        ops = validate_ops(value)
    except (ValueError, TypeError) as exc:
        row["error"] = str(exc)
    else:
        previous = next((item for item in history if item.get("ops") == list(ops)), None)
        row.update({"ops": list(ops), "architecture": canonical(ops),
                    "status": "duplicate" if previous else "valid_new"})
        feedback = {k: previous[k] for k in ("val_accuracy", "flops", "params")} if previous else oracle.validation(ops)
        row.update(feedback)
    valid_rows = [item for item in history + [row] if item.get("ops") is not None]
    row["best_so_far_val_accuracy"] = max((r["val_accuracy"] for r in valid_rows), default=None)
    row["unique_evaluations"] = len({tuple(r["ops"]) for r in valid_rows})
    row["evaluation_seconds"] = time.perf_counter() - start
    return row


def public_row(row, feedback):
    keys = ["step", "phase", "status", "ops", "error"]
    if feedback:
        keys += ["val_accuracy", "flops", "params"]
    return {k: row[k] for k in keys if k in row}


def observation(config, history):
    feedback = config["method"] in ("llm", "chatgpt-feedback", "local-llm-feedback")
    return {
        "task": "Propose one image-classification cell to maximize validation accuracy.",
        "architecture_context": {
            "nodes": "Four nodes 0,1,2,3. Node 0 is input; node 3 is output.",
            "aggregation": "Each node sums transformed incoming features; operations lie on edges.",
            "macro_architecture": "Fixed surrounding network and fixed training recipe.",
            "none": "Removes the edge. Disconnected encodings remain valid."},
        "search_context": {k: config[k] for k in ("dataset", "hp", "training_seed_policy", "accuracy_unit", "objective")},
        "allowed_operations": list(OPS), "edge_order": list(EDGES),
        "response_format": {"ops": ["six operation tokens"], "rationale": "optional short explanation"},
        "rules": ["Return exactly one candidate.",
                  "Every proposal consumes one attempt, including invalid and repeated candidates.",
                  "Use only supplied history. Do not read benchmark tables, source files, external resources, or other run results.",
                  "No future candidates may be submitted or precomputed in the same step."],
        "feedback_enabled": feedback, "run_seed": config["seed"],
        "next_attempt": len(history) + 1 if len(history) < config["budget"] else None,
        "remaining_attempts": config["budget"] - len(history),
        "complete": len(history) == config["budget"],
        "history": [public_row(row, feedback) for row in history],
    }


def summarize(config, history, trajectory_file="trajectory.jsonl"):
    valid = [r for r in history if r.get("ops") is not None]
    if not valid:
        raise ValueError("No valid architecture was evaluated")
    winner = max(valid, key=lambda r: r["val_accuracy"])
    searched = history[config["warm_start"]:]
    is_model = config["method"] not in ("random", "evolution")
    model_timing_complete = all("proposal_seconds" in r for r in searched) if is_model else True
    return {
        "data_label": config["benchmark"], "method": config["method"], "seed": config["seed"],
        "model_label": config.get("model_label", config.get("model")),
        "dataset_for_search": config["dataset"], "search_hp": config["hp"],
        "training_seed_policy": config["training_seed_policy"],
        "data_sha256": config["provenance"].get("normalized_data_sha256", config["provenance"].get("data_sha256")),
        "budget_attempts": config["budget"], "warm_start_attempts": config["warm_start"],
        "completed_attempts": len(history), "complete": len(history) == config["budget"],
        "proposal_attempts_excluding_warm": len(searched),
        "unique_evaluations_including_warm": len({tuple(r["ops"]) for r in valid}),
        "validity_rate_excluding_warm": sum(r["ops"] is not None for r in searched) / len(searched) if searched else None,
        "invalid_excluding_warm": sum(r["status"] == "invalid" for r in searched),
        "duplicates_excluding_warm": sum(r["status"] == "duplicate" for r in searched),
        "best_val_accuracy": winner["val_accuracy"], "winner_ops": winner["ops"],
        "winner_architecture": winner["architecture"], "winner_flops": winner["flops"], "winner_params": winner["params"],
        "winner_rule": "highest validation accuracy; earliest visited wins exact ties",
        "model_seconds": sum(r.get("proposal_seconds", 0.0) for r in history) if model_timing_complete else None,
        "model_timing_recorded": model_timing_complete,
        "trajectory_file": trajectory_file, "test_evaluated": False,
    }
