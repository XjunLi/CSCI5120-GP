"""Versioned evolution of a bounded, executable NAS harness program.

The evolving object is the *search procedure*: its workflow, memory policy,
parent selection, exploration rate and model instruction.  It is distinct from
the six-edge architecture proposed by that procedure.  Programs are interpreted
from a small schema, never executed as arbitrary Python.  A quality-diversity
archive stores the best *observed validation* result in each resource niche.

Incumbent and challenger programs share the public history and are interleaved
over equal proposal windows.  A challenger replaces the incumbent only when its
mean reward exceeds the incumbent's by ``promotion_margin``.  This is an online
selection heuristic, not an independent statistical test of improvement.  Every
trial uses the ordinary proposal budget; there are no hidden oracle calls.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random
from typing import Any

from .architecture import EDGES, OPS, SPACE_SIZE, random_unseen, validate_ops


WORKFLOWS = (
    ("random",),
    ("select", "mutate"),
    ("select", "mutate", "repair"),
    ("select", "crossover", "mutate"),
    ("select", "crossover", "mutate", "repair"),
)
MEMORY_POLICIES = ("top", "recent", "mixed", "archive")
PARENT_POLICIES = ("best", "tournament", "diverse")
INSTRUCTIONS = (
    "Balance high-scoring candidates with unexplored combinations.",
    "Use recent failures to avoid repeating unproductive search choices.",
    "Combine complementary strong candidates while preserving diversity.",
    "Explore different resource niches before refining the strongest candidates.",
)
_ROW_FIELDS = ("step", "phase", "status", "ops", "error", "val_accuracy", "flops", "params")
_CONFIG_FIELDS = {
    "evolve", "evaluation_window", "promotion_margin", "duplicate_penalty",
    "invalid_reward", "initial_program", "model_label", "flops_bins", "params_bins",
}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _bins(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) > 32:
        raise ValueError(f"{name} must be a list of at most 32 finite boundaries")
    result = [_finite_number(x, name) for x in value]
    if any(x < 0 for x in result) or any(a >= b for a, b in zip(result, result[1:])):
        raise ValueError(f"{name} boundaries must be nonnegative and strictly increasing")
    return result


@dataclass(frozen=True)
class HarnessProgram:
    """A small executable search language whose fields can be evolved."""

    workflow: tuple[str, ...] = ("select", "mutate")
    memory_policy: str = "mixed"
    memory_size: int = 12
    parent_policy: str = "tournament"
    tournament_size: int = 3
    exploration_rate: float = 0.20
    mutation_edges: int = 1
    duplicate_retries: int = 8
    instruction: str = INSTRUCTIONS[0]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["workflow"] = list(self.workflow)
        return value


def validate_program(value: dict | HarnessProgram, base: HarnessProgram | None = None) -> HarnessProgram:
    """Validate a complete program or a bounded patch against ``base``."""
    if isinstance(value, HarnessProgram):
        value = value.to_dict()
    if not isinstance(value, dict):
        raise ValueError("A harness program must be a JSON object")
    allowed = set(HarnessProgram.__dataclass_fields__)
    if set(value) - allowed:
        raise ValueError(f"Unknown harness fields: {sorted(set(value) - allowed)}")
    merged = (base or HarnessProgram()).to_dict()
    merged.update(deepcopy(value))
    workflow = merged["workflow"]
    if not isinstance(workflow, (list, tuple)) or tuple(workflow) not in WORKFLOWS:
        raise ValueError("workflow is not in the bounded search language")
    if merged["memory_policy"] not in MEMORY_POLICIES:
        raise ValueError("Unknown memory_policy")
    if merged["parent_policy"] not in PARENT_POLICIES:
        raise ValueError("Unknown parent_policy")
    _integer(merged["memory_size"], "memory_size", 1, 64)
    _integer(merged["tournament_size"], "tournament_size", 1, 16)
    _integer(merged["mutation_edges"], "mutation_edges", 1, 6)
    _integer(merged["duplicate_retries"], "duplicate_retries", 0, 64)
    exploration = _finite_number(merged["exploration_rate"], "exploration_rate")
    if not 0 <= exploration <= 1:
        raise ValueError("exploration_rate must lie in [0, 1]")
    instruction = merged["instruction"]
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 1000:
        raise ValueError("instruction must contain 1 to 1000 characters")
    merged["workflow"] = tuple(workflow)
    merged["exploration_rate"] = exploration
    return HarnessProgram(**merged)


def _tuple_tree(value):
    return tuple(_tuple_tree(x) for x in value) if isinstance(value, list) else value


def _public_row(row: dict, feedback: bool = True) -> dict:
    keys = _ROW_FIELDS if feedback else _ROW_FIELDS[:5]
    return {k: deepcopy(row[k]) for k in keys if k in row}


class HarnessEvolutionController:
    """CPU-runnable harness evolution, optionally driven by an LLM adapter.

    ``model(payload)`` returns a parsed JSON object.  For architecture calls the
    response contains ``ops``.  For harness-edit calls it contains ``patch``.
    Invalid architecture responses consume the next ordinary attempt; invalid
    edits fall back to a validated local program mutation and are logged.
    """

    def __init__(self, seed: int, config: dict | None = None, model=None):
        config = deepcopy(config or {})
        if not isinstance(config, dict) or set(config) - _CONFIG_FIELDS:
            raise ValueError("Unknown harness controller configuration field")
        if not isinstance(config.get("evolve", True), bool):
            raise ValueError("evolve must be a boolean")
        self.config = {
            "evolve": config.get("evolve", True),
            "evaluation_window": _integer(config.get("evaluation_window", 4), "evaluation_window", 1, 64),
            "promotion_margin": _finite_number(config.get("promotion_margin", 0.0), "promotion_margin"),
            "duplicate_penalty": _finite_number(config.get("duplicate_penalty", 1.0), "duplicate_penalty"),
            "invalid_reward": _finite_number(config.get("invalid_reward", 0.0), "invalid_reward"),
            "initial_program": validate_program(config.get("initial_program", {})).to_dict(),
            "model_label": config.get("model_label", "external-llm" if model else "local-harness-program"),
            "flops_bins": _bins(config.get("flops_bins", [50, 100, 150]), "flops_bins"),
            "params_bins": _bins(config.get("params_bins", [0.5, 1.0, 1.5]), "params_bins"),
        }
        if self.config["promotion_margin"] < 0 or self.config["duplicate_penalty"] < 0:
            raise ValueError("promotion_margin and duplicate_penalty must be nonnegative")
        if not isinstance(self.config["model_label"], str):
            raise ValueError("model_label must be a string")
        self.seed, self.model = int(seed), model
        # Keep proposal randomness independent of the program-edit scheduler.
        self.rng = random.Random(seed)
        self.evolution_rng = random.Random(seed ^ 0x5E1FCAFE)
        self.incumbent = validate_program(self.config["initial_program"])
        self.incumbent_version = 0
        self.next_version = 1
        self.versions = [{"version": 0, "parent": None, "program": self.incumbent.to_dict(), "status": "incumbent"}]
        self.trial = None
        self.pending = None
        self.history = []
        self.archive = {}
        self.events = []
        self.model_calls = {"proposal": 0, "harness_edit": 0}
        self.promotions = 0
        self.rejections = 0
        self.feedback_mode = None

    def _event(self, event_type: str, **fields):
        self.events.append({"type": event_type, "controller": "harness", **deepcopy(fields)})

    def drain_events(self) -> list[dict]:
        events, self.events = self.events, []
        return events

    def _update_archive(self, row: dict):
        if row.get("status") != "valid_new" or not row.get("ops"):
            return
        try:
            score = _finite_number(row.get("val_accuracy"), "val_accuracy")
            flops = _finite_number(row.get("flops"), "flops")
            params = _finite_number(row.get("params"), "params")
        except ValueError:
            return
        if flops < 0 or params < 0:
            return
        niche = (bisect_right(self.config["flops_bins"], flops), bisect_right(self.config["params_bins"], params))
        previous = self.archive.get(niche)
        if previous is None or score > previous["val_accuracy"]:
            self.archive[niche] = _public_row(row)
            self._event("archive_update", step=row.get("step"), niche=list(niche),
                        val_accuracy=score, previous_val_accuracy=previous["val_accuracy"] if previous else None,
                        occupied_niches=len(self.archive))

    def _memory(self, history: list[dict], program: HarnessProgram, feedback: bool = True) -> list[dict]:
        valid = [r for r in history if r.get("ops") is not None and r.get("status") != "invalid"]
        # No score is invented when a caller has deliberately hidden feedback.
        top = sorted(valid, key=lambda r: r.get("val_accuracy", float("-inf")), reverse=True)
        if program.memory_policy == "top":
            pool = top
        elif program.memory_policy == "recent":
            pool = list(reversed(valid))
        elif program.memory_policy == "archive" and feedback:
            pool = [deepcopy(self.archive[k]) for k in sorted(self.archive)] + top
        elif program.memory_policy == "archive":
            pool = list(reversed(valid))
        else:
            pool = []
            for i in range(max(len(top), len(valid))):
                if i < len(top):
                    pool.append(top[i])
                if i < len(valid):
                    pool.append(valid[-1-i])
        seen, result = set(), []
        for row in pool:
            try:
                key = validate_ops(row["ops"])
            except (ValueError, TypeError):
                continue
            if key not in seen:
                seen.add(key)
                result.append(deepcopy(row))
            if len(result) >= program.memory_size:
                break
        return result

    def _local_patch(self) -> HarnessProgram:
        choices = {
            "workflow": list(WORKFLOWS), "memory_policy": list(MEMORY_POLICIES),
            "memory_size": [4, 8, 12, 20, 32], "parent_policy": list(PARENT_POLICIES),
            "tournament_size": [1, 2, 3, 5, 8], "exploration_rate": [0.0, 0.1, 0.2, 0.4, 0.7],
            "mutation_edges": [1, 2, 3, 4], "duplicate_retries": [0, 4, 8, 16],
            "instruction": list(INSTRUCTIONS),
        }
        if self.model is None:
            # An instruction-only mutation cannot alter the local interpreter.
            choices.pop("instruction")
        fields = self.evolution_rng.sample(list(choices), self.evolution_rng.randint(1, 3))
        changes = {}
        for field in fields:
            alternatives = [v for v in choices[field] if v != getattr(self.incumbent, field)]
            changes[field] = self.evolution_rng.choice(alternatives)
        return validate_program(changes, self.incumbent)

    def _start_trial(self, observation: dict):
        source, candidate = "local_mutation", None
        if self.model is not None:
            self.model_calls["harness_edit"] += 1
            payload = {
                "task": "propose_harness_patch",
                "instruction": "Return a bounded patch to improve the NAS search procedure using only supplied observations.",
                "current_program": self.incumbent.to_dict(),
                "response_format": {"patch": {"one or more program fields": "new value"}},
                "allowed_fields": {
                    "workflow": [list(w) for w in WORKFLOWS], "memory_policy": list(MEMORY_POLICIES),
                    "memory_size": "integer 1..64", "parent_policy": list(PARENT_POLICIES),
                    "tournament_size": "integer 1..16", "exploration_rate": "number 0..1",
                    "mutation_edges": "integer 1..6", "duplicate_retries": "integer 0..64",
                    "instruction": "nonempty string, at most 1000 characters",
                },
                "history": [_public_row(r, observation.get("feedback_enabled", True)) for r in observation.get("history", [])],
                "previous_trials": [{k: deepcopy(v[k]) for k in ("version", "status", "mean_reward") if k in v}
                                    for v in self.versions[-6:]],
            }
            try:
                response = self.model(payload)
                if not isinstance(response, dict):
                    raise ValueError("Model edit response must be a JSON object")
                candidate = validate_program(response.get("patch"), self.incumbent)
                if candidate == self.incumbent:
                    raise ValueError("Model edit did not change the program")
                source = "model_patch"
                self._event("model_call", kind="harness_edit", usage=response.get("usage"), model=response.get("model"))
            except Exception as exc:
                candidate = None
                self._event("harness_edit_rejected", reason=type(exc).__name__, fallback="local_mutation")
        if candidate is None:
            candidate = self._local_patch()
        version = self.next_version
        self.next_version += 1
        schedule = []
        for _ in range(self.config["evaluation_window"]):
            pair = ["incumbent", "challenger"]
            self.evolution_rng.shuffle(pair)
            schedule.extend(pair)
        self.trial = {"version": version, "program": candidate.to_dict(), "schedule": schedule,
                      "position": 0, "rewards": {"incumbent": [], "challenger": []}}
        self.versions.append({"version": version, "parent": self.incumbent_version,
                              "program": candidate.to_dict(), "status": "trial", "source": source})
        old, new = self.incumbent.to_dict(), candidate.to_dict()
        self._event("harness_trial_started", step=observation.get("next_attempt"),
                    incumbent_version=self.incumbent_version, challenger_version=version, source=source,
                    changes={k: {"before": old[k], "after": new[k]} for k in old if old[k] != new[k]},
                    proposal_attempts=2 * self.config["evaluation_window"], schedule=schedule)

    def _parent(self, memory: list[dict], program: HarnessProgram, other=None):
        if other is not None and len(memory) > 1:
            memory = [row for row in memory if row["ops"] != other["ops"]]
        if program.parent_policy == "best":
            return max(memory, key=lambda r: r.get("val_accuracy", float("-inf")))
        if program.parent_policy == "diverse" and other is not None:
            return max(memory, key=lambda r: sum(a != b for a, b in zip(r["ops"], other["ops"])))
        if program.parent_policy == "diverse":
            return self.rng.choice(memory)
        candidates = self.rng.sample(memory, min(program.tournament_size, len(memory)))
        return max(candidates, key=lambda r: r.get("val_accuracy", float("-inf")))

    def _repair(self, ops: list[str]):
        reachable = {0}
        for edge, (source, destination) in enumerate(((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3))):
            if source in reachable and ops[edge] != "none":
                reachable.add(destination)
        if 3 not in reachable:
            index = self.rng.choice([3 + source for source in reachable])
            ops[index] = self.rng.choice(OPS[1:])

    def _local_proposal(self, history: list[dict], memory: list[dict], program: HarnessProgram):
        seen = {tuple(r["ops"]) for r in history if r.get("ops") is not None}
        if len(seen) >= SPACE_SIZE:
            raise RuntimeError("The NAS encoding space is exhausted")
        for _ in range(program.duplicate_retries + 1):
            if not memory or program.workflow == ("random",) or self.rng.random() < program.exploration_rate:
                return list(random_unseen(self.rng, seen))
            first = self._parent(memory, program)
            ops = list(first["ops"])
            if "crossover" in program.workflow:
                second = self._parent(memory, program, first)
                ops = [a if self.rng.random() < 0.5 else b for a, b in zip(ops, second["ops"])]
            if "mutate" in program.workflow:
                for edge in self.rng.sample(range(6), program.mutation_edges):
                    ops[edge] = self.rng.choice([op for op in OPS if op != ops[edge]])
            if "repair" in program.workflow:
                self._repair(ops)
            if tuple(ops) not in seen:
                return ops
        # Repeated outputs remain visible and count as attempts when retries are
        # exhausted; a proposed program cannot get free oracle evaluations.
        return ops

    def propose(self, observation: dict) -> dict:
        if self.pending is not None:
            raise RuntimeError("observe must be called before another proposal")
        remaining = observation.get("remaining_attempts", 1)
        if observation.get("complete") or remaining <= 0:
            raise RuntimeError("No proposal attempts remain")
        history = [_public_row(r, observation.get("feedback_enabled", True)) for r in observation.get("history", [])]
        feedback = observation.get("feedback_enabled", True)
        if self.feedback_mode is None:
            self.feedback_mode = feedback
        elif self.feedback_mode != feedback:
            raise ValueError("feedback_enabled cannot change within a harness run")
        if self.config["evolve"] and feedback and self.trial is None and remaining >= 2 * self.config["evaluation_window"]:
            self._start_trial(observation)
        role, version, program = "incumbent", self.incumbent_version, self.incumbent
        if self.trial is not None:
            role = self.trial["schedule"][self.trial["position"]]
            if role == "challenger":
                version, program = self.trial["version"], validate_program(self.trial["program"])
        step = observation.get("next_attempt", len(history) + 1)
        self.pending = {"step": step, "role": role, "version": version, "in_trial": self.trial is not None}
        memory = self._memory(history, program, feedback)
        response = {}
        if self.model is None:
            response["ops"] = self._local_proposal(history, memory, program)
            response["rationale"] = f"Harness v{version}: {' -> '.join(program.workflow)}; {program.memory_policy} memory."
        else:
            self.model_calls["proposal"] += 1
            payload = {
                "task": "Propose one NAS cell using the supplied executable harness program.",
                "harness_program": program.to_dict(), "history": memory,
                "attempted_ops": [r["ops"] for r in history if r.get("ops") is not None],
                "allowed_operations": list(OPS), "edge_order": list(EDGES),
                "response_format": {"ops": ["six operation tokens"], "rationale": "optional short explanation"},
                "feedback_enabled": observation.get("feedback_enabled", True),
                "next_attempt": step, "remaining_attempts": remaining,
                "rules": ["Use only supplied observations.", "Return exactly one candidate.",
                          "Invalid and duplicate outputs consume an attempt."],
            }
            try:
                result = self.model(payload)
                if not isinstance(result, dict):
                    raise ValueError("Model proposal must be a JSON object")
                response.update({k: deepcopy(result[k]) for k in ("ops", "rationale", "model", "usage") if k in result})
                response.setdefault("ops", None)
            except Exception as exc:
                response.update(ops=None, rationale=f"Model call failed: {type(exc).__name__}")
                self._event("model_call_failed", step=step, kind="proposal", reason=type(exc).__name__)
        response.setdefault("model", self.config["model_label"])
        response["controller_metadata"] = {"harness_version": version, "harness_role": role,
                                           "workflow": list(program.workflow), "memory_policy": program.memory_policy,
                                           "occupied_niches": len(self.archive)}
        try:
            self.pending["ops"] = list(validate_ops(response.get("ops")))
        except (ValueError, TypeError):
            self.pending["ops"] = None
        self._event("harness_proposal", step=step, **response["controller_metadata"])
        return response

    def observe(self, row: dict) -> None:
        row = _public_row(row)
        if type(row.get("step")) is not int or row["step"] != len(self.history) + 1:
            raise ValueError("Observations must be supplied exactly once in consecutive step order")
        if self.pending is not None and row.get("step") != self.pending["step"]:
            raise ValueError("Observed step does not match the pending harness proposal")
        if self.pending is None and self.feedback_mode is not None:
            raise ValueError("Search observations require a pending proposal")
        if self.pending is not None and row.get("ops") != self.pending["ops"]:
            raise ValueError("Observed encoding differs from the submitted harness proposal")
        self.history.append(row)
        self._update_archive(row)
        if self.pending is None:
            return  # Shared warm start has no incumbent/challenger credit.
        pending, self.pending = self.pending, None
        if not pending["in_trial"]:
            return
        reward = self.config["invalid_reward"]
        score = row.get("val_accuracy")
        if row.get("status") != "invalid" and isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
            reward = float(score) - (self.config["duplicate_penalty"] if row.get("status") == "duplicate" else 0)
        self.trial["rewards"][pending["role"]].append(reward)
        self.trial["position"] += 1
        self._event("harness_credit", step=row.get("step"), version=pending["version"],
                    role=pending["role"], reward=reward)
        if self.trial["position"] == len(self.trial["schedule"]):
            self._finish_trial(row.get("step"))

    def _finish_trial(self, step):
        trial = self.trial
        means = {role: sum(values) / len(values) for role, values in trial["rewards"].items()}
        promoted = means["challenger"] > means["incumbent"] + self.config["promotion_margin"]
        record = next(v for v in self.versions if v["version"] == trial["version"])
        record.update(status="promoted" if promoted else "rejected", mean_reward=means["challenger"])
        previous = self.incumbent_version
        if promoted:
            self.incumbent = validate_program(trial["program"])
            self.incumbent_version = trial["version"]
            self.promotions += 1
        else:
            self.rejections += 1
        self._event("harness_promoted" if promoted else "harness_rejected", step=step,
                    previous_incumbent_version=previous, challenger_version=trial["version"],
                    active_version=self.incumbent_version, mean_reward=means,
                    promotion_margin=self.config["promotion_margin"],
                    credit_samples={k: len(v) for k, v in trial["rewards"].items()},
                    rollback_to_version=None if promoted else previous)
        self.trial = None

    def state_dict(self) -> dict:
        return deepcopy({
            "schema_version": 1, "controller": "harness", "seed": self.seed, "config": self.config,
            "incumbent": self.incumbent.to_dict(), "incumbent_version": self.incumbent_version,
            "next_version": self.next_version, "versions": self.versions, "trial": self.trial,
            "pending": self.pending, "history": self.history,
            "archive": [{"niche": list(k), "row": v} for k, v in sorted(self.archive.items())],
            "events": self.events, "rng_state": self.rng.getstate(),
            "evolution_rng_state": self.evolution_rng.getstate(), "model_calls": self.model_calls,
            "promotions": self.promotions, "rejections": self.rejections,
            "feedback_mode": self.feedback_mode,
        })

    def load_state_dict(self, state: dict) -> None:
        if state.get("schema_version") != 1 or state.get("controller") != "harness":
            raise ValueError("Unsupported harness checkpoint")
        # Build and validate the replacement first, leaving this object intact on
        # malformed checkpoints.  No model calls occur while restoring state.
        restored = type(self)(state["seed"], state["config"], self.model)
        restored.incumbent = validate_program(state["incumbent"])
        for entry in state["versions"]:
            validate_program(entry["program"])
        if state["trial"] is not None:
            validate_program(state["trial"]["program"])
        for key in ("incumbent_version", "next_version", "versions", "trial", "pending", "history",
                    "events", "model_calls", "promotions", "rejections", "feedback_mode"):
            setattr(restored, key, deepcopy(state[key]))
        restored.archive = {tuple(entry["niche"]): deepcopy(entry["row"]) for entry in state["archive"]}
        restored.rng.setstate(_tuple_tree(state["rng_state"]))
        restored.evolution_rng.setstate(_tuple_tree(state["evolution_rng_state"]))
        self.__dict__.update(restored.__dict__)
