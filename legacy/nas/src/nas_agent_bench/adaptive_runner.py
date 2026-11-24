"""A common, resumable evaluator for three distinct kinds of search adaptation.

Controllers see validation-only dictionaries, never an oracle. The evaluator
alone accounts for attempts and chooses the winner. Final test access remains
in scripts/finalize_results.py, after every requested run has been locked.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

from .architecture import SPACE_SIZE, warm_start
from .protocol import configuration, evaluate_proposal, observation, public_row, summarize


METHODS = ("harness", "harness-frozen", "multiagent", "multiagent-frozen",
           "weight-ttt", "weight-frozen")


def make_controller(method, seed, settings=None, model=None):
    """Lazily import optional learning dependencies only for weight methods."""
    if method not in METHODS:
        raise ValueError(f"Unknown adaptive method: {method}")
    config = copy.deepcopy(settings or {})
    checked_model = None
    if model is not None:
        def checked_model(payload):
            result = model(payload)
            if not isinstance(result, dict):
                raise ValueError("A model must return one JSON object")
            _json(result)  # Reject non-finite/unserializable metadata before credit is assigned.
            return result
    if method.startswith("harness"):
        from .harness_evolution import HarnessEvolutionController
        config["evolve"] = method != "harness-frozen"
        return HarnessEvolutionController(seed, config, model=checked_model)
    if method.startswith("multiagent"):
        from .multiagent_evolution import MultiAgentEvolutionController
        config["evolve"] = method != "multiagent-frozen"
        return MultiAgentEvolutionController(seed, config, model=checked_model)
    if model is not None:
        raise ValueError("Weight controllers own their trainable model; do not supply a frozen proposer")
    from .weight_evolution import WeightEvolutionController
    config["adaptive"] = method != "weight-frozen"
    return WeightEvolutionController(seed, config, model=None)


def adaptive_observation(config, history):
    """Reuse the original allowlist; do not expose provenance or private state."""
    public_config = dict(config, method="llm")
    payload = observation(public_config, history)
    payload["rules"] = [
        "Submit exactly one final architecture for this evaluation attempt.",
        "Every submitted invalid or repeated architecture consumes an attempt.",
        "Use only the supplied observations; no benchmark files, other runs, or external scores.",
        "Internal draft/critique and program changes may use no additional evaluator queries.",
        "Report internal model calls and adaptation changes separately from evaluation attempts.",
    ]
    return payload


def _json(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))


def _atomic_json(path, value):
    """A checkpoint, not an append log, is the authoritative transaction."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_json(row) + "\n")
    temporary.replace(path)


def _identity(config):
    # The table checksum and all supplied algorithm/model settings are included.
    return hashlib.sha256(_json(config).encode()).hexdigest()


def _controller_summary(state):
    keys = ("promotions", "rejections", "incumbent_version", "generation", "births",
            "proposal_count", "message_count", "total_updates", "total_proposals",
            "trainable_parameters", "initial_weight_sha256", "current_weight_sha256",
            "weight_delta_l2", "base_model", "model_calls")
    result = {k: state[k] for k in keys if k in state}
    if "versions" in state:
        result["program_versions"] = len(state["versions"])
    if "archive" in state:
        result["occupied_niches"] = len(state["archive"])
    return result


def _model_call_count(state, events, history, model):
    audited = [e for e in events if e["event_source"] == "model_adapter" and e.get("kind") == "model_call"]
    if audited or (model is not None and hasattr(model, "drain_events")):
        return len(audited)
    if state.get("controller") == "weight_evolution":
        return sum(r.get("usage", {}).get("model_calls", 0) for r in history)
    if "model_calls" in state:
        calls = state["model_calls"]
        return sum(calls.values()) if isinstance(calls, dict) else calls
    return 0 if model is None else None


def _check_history(history, budget):
    if len(history) > budget or [r.get("step") for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("Checkpoint has inconsistent attempt accounting")
    for row in history:
        if row.get("ops") is not None and not math.isfinite(row["val_accuracy"]):
            raise ValueError("Checkpoint contains a non-finite validation score")


def run_adaptive_search(oracle, method, seed, budget, warm_count, output,
                        controller_config=None, model=None, model_label=None,
                        resume=False, stop_after=None, controller=None):
    """Run or resume one isolated search, writing an atomic checkpoint each step.

    ``stop_after`` is an absolute attempt number for reproducible interruption
    tests and short checkpoints. A resumed run must keep the exact original
    budget, seed, dataset, model identity and controller configuration.
    ``controller`` is an injection point for protocol tests.
    """
    if method not in METHODS or not 1 <= warm_count <= budget <= SPACE_SIZE:
        raise ValueError("Invalid method, budget or warm start")
    if stop_after is not None and not 0 <= stop_after <= budget:
        raise ValueError("stop_after must be an absolute attempt number inside the budget")
    config = configuration(oracle, method, seed, budget, warm_count)
    config.update({"schema_version": 2, "controller_config": copy.deepcopy(controller_config or {}),
                   "model_label": model_label, "feedback_enabled": True,
                   "external_model_enabled": model is not None,
                   "adaptation_boundary": "validation only; reset controller and weights for each run",
                   "model_budget_policy": "record all internal calls separately; one final evaluation per attempt"})
    if model is not None and hasattr(model, "configuration"):
        config["proposer_configuration"] = model.configuration()
    config_id = _identity(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = f"{method}_seed{seed}"
    trajectory_path = output / f"{name}.jsonl"
    events_path = output / f"{name}_events.jsonl"
    checkpoint_path = output / "checkpoints" / f"{name}.json"
    config_path = output / f"{name}_config.json"
    checkpoint_path.parent.mkdir(exist_ok=True)
    if not resume and (trajectory_path.exists() or checkpoint_path.exists() or config_path.exists()):
        raise FileExistsError(f"Run already exists: {name}; use --resume with identical settings")
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint for {name}")
    if controller is None:
        controller = make_controller(method, seed, controller_config, model)
    warm, _ = warm_start(seed, warm_count)
    history, events, elapsed_before, finalized = [], [], 0.0, False
    if resume:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("config_sha256") != config_id or checkpoint.get("configuration") != config:
            raise ValueError("Checkpoint does not match dataset, budget, seed, model or controller settings")
        history, events = checkpoint["history"], checkpoint["events"]
        _check_history(history, budget)
        controller.load_state_dict(checkpoint["controller_state"])
        if model is not None and checkpoint.get("model_state") is not None:
            if not hasattr(model, "load_state_dict"):
                raise ValueError("The adapter cannot restore its checkpointed state")
            model.load_state_dict(checkpoint["model_state"])
        elapsed_before = checkpoint.get("elapsed_seconds", 0.0)
        finalized = checkpoint.get("finalized", False)
    started = time.perf_counter()
    _atomic_json(config_path, config)

    def collect(source, step):
        if hasattr(source, "drain_events"):
            for event in source.drain_events():
                if not isinstance(event, dict):
                    raise TypeError("Adaptation events must be JSON objects")
                events.append(dict(event, event_index=len(events) + 1, at_attempt=step,
                                   event_source="model_adapter" if source is model else "controller"))

    def persist():
        elapsed = elapsed_before + time.perf_counter() - started
        snapshot = {"schema_version": 1, "config_sha256": config_id, "configuration": config,
                    "history": history, "events": events, "controller_state": controller.state_dict(),
                    "model_state": model.state_dict() if model is not None and hasattr(model, "state_dict") else None,
                    "elapsed_seconds": elapsed, "finalized": finalized}
        _atomic_json(checkpoint_path, snapshot)
        # Rebuild append-only views from the authoritative checkpoint. A crash
        # between these writes never charges or loses another evaluation.
        _write_jsonl(trajectory_path, history)
        _write_jsonl(events_path, events)

    if not resume:
        collect(controller, 0)
    persist()
    limit = budget if stop_after is None else stop_after
    for step in range(len(history) + 1, limit + 1):
        phase = "warm_start" if step <= warm_count else "search"
        raw, metadata, proposal = None, {}, {}
        if phase == "warm_start":
            ops = warm[step - 1]
        else:
            call_start = time.perf_counter()
            proposal = controller.propose(adaptive_observation(config, history))
            metadata["proposal_seconds"] = time.perf_counter() - call_start
            if not isinstance(proposal, dict):
                proposal = {"ops": None, "invalid_controller_response": str(proposal)}
            try:
                raw = _json(proposal)
            except (ValueError, TypeError):
                # A malformed model output still costs one final attempt. The
                # complete object may not be JSON serializable; retain its repr.
                proposal = {"ops": None, "invalid_controller_response": repr(proposal)}
                raw = _json(proposal)
            ops = proposal.get("ops")
            metadata.update({k: proposal[k] for k in ("model", "usage", "rationale") if k in proposal})
        row = evaluate_proposal(oracle, history, ops, phase, raw, metadata)
        if "controller_metadata" in proposal:
            row["controller_metadata"] = copy.deepcopy(proposal["controller_metadata"])
        # Never provide raw model text or private oracle fields to learning.
        update_start = time.perf_counter()
        controller.observe(public_row(row, feedback=True))
        row["adaptation_seconds"] = time.perf_counter() - update_start
        row.update({"method": method, "seed": seed, "data_label": oracle.label,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - started})
        history.append(row)
        collect(controller, step)
        if model is not None:
            collect(model, step)
        persist()
    if len(history) == budget and not finalized:
        if hasattr(controller, "finalize"):
            final_start = time.perf_counter()
            controller.finalize()
            events.append({"kind": "controller_finalized", "event_source": "controller",
                           "event_index": len(events) + 1, "at_attempt": budget,
                           "seconds": time.perf_counter() - final_start})
            collect(controller, budget)
        finalized = True
        persist()
    if not history:
        return {"method": method, "seed": seed, "complete": False, "completed_attempts": 0,
                "budget_attempts": budget, "checkpoint_file": str(checkpoint_path.relative_to(output))}
    result = summarize(config, history, trajectory_path.name)
    state = controller.state_dict()
    result.update({"search_elapsed_seconds": elapsed_before + time.perf_counter() - started,
                   "adaptation_seconds": sum(r.get("adaptation_seconds", 0.0) for r in history) +
                       sum(e.get("seconds", 0.0) for e in events if e.get("kind") == "controller_finalized"),
                   "controller_event_count": sum(e["event_source"] == "controller" for e in events),
                   "model_call_count": _model_call_count(state, events, history, model),
                   "events_file": events_path.name,
                   "checkpoint_file": str(checkpoint_path.relative_to(output)),
                   "config_sha256": config_id,
                   "adaptation_enabled": not method.endswith("-frozen"),
                   "adaptation_summary": _controller_summary(state)})
    _atomic_json(output / f"{name}_summary.json", result)
    return result
