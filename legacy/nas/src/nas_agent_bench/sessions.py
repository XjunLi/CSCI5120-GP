"""Auditable one-proposal-at-a-time interface for an external research assistant.

The caller must obey the observation-only protocol. This is information filtering,
not an OS security boundary. Session files must be inaccessible to a tool agent
when stronger adversarial isolation is required.
"""
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .architecture import warm_start
from .engine import write_summaries
from .oracles import load_oracle
from .protocol import configuration, evaluate_proposal, observation, public_row, summarize


def atomic_json(path, value):
    path = Path(path)
    descriptor, tmp = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextmanager
def locked(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield folder


def initialize(folder, backend, data, condition, seed, budget=50, warm_count=10, dataset="cifar10-valid", hp="200", model="ChatGPT assistant"):
    with locked(folder) as folder:
        if (folder / "private_state.json").exists():
            raise ValueError("Session already exists; use status to resume")
        oracle = load_oracle(backend, data, dataset, hp)
        method = "chatgpt-feedback" if condition == "feedback" else "chatgpt-no-feedback"
        config = configuration(oracle, method, seed, budget, warm_count)
        config["model"] = model
        history = []
        for ops in warm_start(seed, warm_count)[0]:
            history.append(evaluate_proposal(oracle, history, ops, phase="warm_start"))
        state = {"config": config, "oracle": {"backend": backend, "data": str(Path(data).expanduser().resolve()),
                                              "dataset": dataset, "hp": str(hp)},
                 "history": history, "exported": False}
        atomic_json(folder / "private_state.json", state)
        return observation(config, history)


def status(folder):
    with locked(folder) as folder:
        state = json.loads((folder / "private_state.json").read_text())
        return observation(state["config"], state["history"])


def propose(folder, expected_step, raw_ops, rationale=None, proposal_seconds=None):
    with locked(folder) as folder:
        state = json.loads((folder / "private_state.json").read_text())
        history, config = state["history"], state["config"]
        if state.get("exported"):
            raise ValueError("Session has been exported and locked")
        if len(history) >= config["budget"]:
            raise ValueError("Proposal budget exhausted")
        if expected_step != len(history) + 1:
            raise ValueError("Stale expected-step; call status before retrying")
        try:
            ops = json.loads(raw_ops)
        except json.JSONDecodeError:
            ops = None
        oracle = load_oracle(**state["oracle"])
        if oracle.metadata != config["provenance"]:
            raise ValueError("Oracle provenance changed during session")
        metadata = {"model": config["model"]}
        if rationale is not None:
            metadata["rationale"] = rationale
        if proposal_seconds is not None:
            metadata["proposal_seconds"] = proposal_seconds
        row = evaluate_proposal(oracle, history, ops, raw_response=raw_ops, metadata=metadata)
        history.append(row)
        atomic_json(folder / "private_state.json", state)
        feedback = config["method"] == "chatgpt-feedback"
        return {"observation": public_row(row, feedback),
                "next_attempt": len(history) + 1 if len(history) < config["budget"] else None,
                "remaining_attempts": config["budget"] - len(history),
                "complete": len(history) == config["budget"]}


def export_session(folder, output):
    """Evaluator-only phase: releases measured trajectories after search is over."""
    with locked(folder) as folder:
        state = json.loads((folder / "private_state.json").read_text())
        config, history = state["config"], state["history"]
        if len(history) != config["budget"]:
            raise ValueError("Cannot export an incomplete search")
        output = Path(output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("Export destination must be empty")
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        name = f"{config['method']}_seed{config['seed']}.jsonl"
        with (output / name).open("w", encoding="utf-8") as stream:
            for row in history:
                row = dict(row, method=config["method"], seed=config["seed"], data_label=config["benchmark"])
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        summary = summarize(config, history, name)
        write_summaries(output, [summary])
        (output / "locked_winners.json").write_text(json.dumps([summary], indent=2), encoding="utf-8")
        state["exported"] = True
        atomic_json(folder / "private_state.json", state)
        return {"exported": True, "output": str(output.resolve()), "test_evaluated": False}
