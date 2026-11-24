"""Budgeted, observation-only search over a measured Parameter Golf table.

These are *local search policies*, not measurements of LLM capability.  The
three adaptive conditions change a bounded harness program, an agent population,
or NumPy policy weights.  Each has a control that retains observations but does
not retain updates to the corresponding object.

The evaluator is a trusted boundary.  It is called once for each newly submitted
candidate, never to rank unobserved candidates.  If a virtual cost budget is used,
each public candidate must carry a ``charge_seconds`` quote; the evaluator must
return that same charge.  Timing measured during original training is reported
separately from the virtual cost of replaying a lookup table.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import random
from typing import Any, Callable

import numpy as np


DEFAULT_METHODS = (
    "random", "evolution", "memory-frontier-adaptive", "memory-frontier-frozen",
    "memory-frontier-no-memory", "memory-frontier-no-structure", "memory-frontier-no-frontier",
)

METHODS = DEFAULT_METHODS + (
    "memory-frontier-memory-only", "memory-frontier-structure-only", "memory-frontier-frontier-only",
    "harness-adaptive", "harness-frozen",
    "multiagent-adaptive", "multiagent-frozen", "weight-adaptive", "weight-frozen",
)


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class HarnessProgram:
    """An executable, bounded search configuration; no generated code is run."""

    exploration: float = 0.35
    parent: str = "tournament"
    mutation_hops: int = 1
    memory_size: int = 8


_PROGRAM_VALUES = {
    "exploration": (0.15, 0.35, 0.70),
    "parent": ("best", "tournament", "recent_best"),
    "mutation_hops": (1, 2, 3),
    "memory_size": (4, 8, 16),
}


def _mutate_program(program: HarnessProgram, rng: random.Random) -> HarnessProgram:
    field = rng.choice(list(_PROGRAM_VALUES))
    options = [value for value in _PROGRAM_VALUES[field] if value != getattr(program, field)]
    return replace(program, **{field: rng.choice(options)})


def _distance(left: dict, right: dict) -> int:
    # Derived fields (e.g. head count from width) must not count as additional
    # mutations of a single independent experimental factor.
    factors = {key for key in ("num_layers", "model_dim", "tied_embeddings", "lr_scale")
               if key in left or key in right}
    keys = factors or (left.keys() | right.keys())
    return sum(left.get(key) != right.get(key) for key in keys)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum()


class _MemoryFrontierController:
    """Adapt the typed harness to the same charged, observation-only runner."""

    def __init__(self, candidates: list[dict], method: str, seed: int):
        from .harness_graph import MemoryFrontierHarness
        flags = {
            "memory-frontier-adaptive": (True, True, True),
            "memory-frontier-frozen": (False, False, False),
            "memory-frontier-no-memory": (False, True, True),
            "memory-frontier-no-structure": (True, False, True),
            "memory-frontier-no-frontier": (True, True, False),
            "memory-frontier-memory-only": (True, False, False),
            "memory-frontier-structure-only": (False, True, False),
            "memory-frontier-frontier-only": (False, False, True),
        }[method]
        self.harness = MemoryFrontierHarness(
            candidates, seed,
            adapt_memory=flags[0], adapt_structure=flags[1], adapt_frontier=flags[2],
        )
        self.rng = random.Random(seed ^ 0xA6E17)
        self.observations: list[dict] = []

    def propose(self, available: list[str], attempt: int, remaining: int):
        if not available:
            return None, {}
        return self.harness.propose(deepcopy(self.observations), list(available), self.rng)

    def observe(self, row: dict, metadata: dict):
        public = {key: deepcopy(row[key]) for key in (
            "attempt", "phase", "candidate_id", "status", "val_bpb", "feasible",
            "charge_seconds", "train_seconds",
        ) if key in row}
        if row["status"] in ("valid_new", "infeasible_or_failed"):
            self.observations.append(public)
            self.harness.observe(public)

    def summary(self):
        output = self.harness.summary()
        output["controller_kind"] = "local_memory_frontier_harness"
        return output


class _LocalController:
    def __init__(self, candidates: list[dict], method: str, seed: int, weight_policy: Any = None):
        self.candidates = {candidate["candidate_id"]: candidate for candidate in candidates}
        self.ids = list(self.candidates)
        self.method = method
        self.adaptive = method.endswith("-adaptive")
        self.weight_policy = weight_policy
        if weight_policy is not None:
            weight_policy.start_run(seed, self.adaptive)
        self.rng = random.Random(seed ^ 0xA6E17)
        self.edit_rng = random.Random(seed ^ 0xC0FFEE)
        self.history: list[dict] = []
        self.events: list[dict] = []
        self.program = HarnessProgram()
        self.program_version = 0
        self.next_version = 1
        self.versions = [{"version": 0, "parent_version": None, "program": asdict(self.program)}]
        self.trial: dict | None = None
        self.search_count = 0
        self.agents = [
            {"id": i, "parent_id": None,
             "program": replace(HarnessProgram(), exploration=exploration), "memory": []}
            for i, exploration in enumerate((0.70, 0.15, 0.35))
        ]
        self.routes = [1, 2, 0]
        self.next_agent_id = 3
        self.epoch_credit: dict[int, list[float]] = {i: [] for i in range(3)}
        self.warm_history: list[dict] = []

        # Candidate features contain public design settings only.  In particular,
        # no evaluator values enter the feature matrix or policy initialization.
        keys = sorted({key for item in candidates for key in item["config"]})
        self.feature_terms = [(key, value) for key in keys for value in sorted({
            _canonical(item["config"].get(key)) for item in candidates
        })]
        self.features = np.array([
            [1.0] + [float(_canonical(item["config"].get(key)) == value)
                     for key, value in self.feature_terms]
            for item in candidates
        ], dtype=np.float64) / math.sqrt(max(1, len(keys)))
        self.id_to_index = {candidate_id: i for i, candidate_id in enumerate(self.ids)}
        self.weights = np.zeros(self.features.shape[1], dtype=np.float64)
        self.initial_weight_hash = _fingerprint(self.weights.tolist())
        self.gradient_updates = 0
        self.lineages = [{"agent_id": i, "parent_id": None, "created_at_attempt": 0}
                         for i in range(3)]

    def _choose_program(self, program: HarnessProgram, available: list[str],
                        memory: list[dict] | None = None) -> str:
        memory = self.history if memory is None else memory
        if not memory or self.rng.random() < program.exploration:
            return self.rng.choice(available)
        if program.parent == "recent_best":
            parents = memory[-program.memory_size:]
            parent = min(parents, key=lambda row: row["val_bpb"])
        else:
            parents = sorted(memory, key=lambda row: row["val_bpb"])[:program.memory_size]
            if program.parent == "best":
                parent = parents[0]
            else:
                parent = min(self.rng.sample(parents, min(3, len(parents))),
                             key=lambda row: row["val_bpb"])
        parent_config = self.candidates[parent["candidate_id"]]["config"]
        distances = {candidate_id: _distance(parent_config, self.candidates[candidate_id]["config"])
                     for candidate_id in available}
        closest_hop_error = min(abs(distance - program.mutation_hops) for distance in distances.values())
        return self.rng.choice([candidate_id for candidate_id in available
                                if abs(distances[candidate_id] - program.mutation_hops) == closest_hop_error])

    def _start_trial(self, attempt: int) -> None:
        challenger = _mutate_program(self.program, self.edit_rng)
        version = self.next_version
        self.next_version += 1
        self.versions.append({"version": version, "parent_version": self.program_version,
                              "program": asdict(challenger)})
        roles = ["incumbent", "challenger", "challenger", "incumbent"]
        if self.edit_rng.randrange(2):
            roles = list(reversed(["challenger", "incumbent", "incumbent", "challenger"]))
        self.trial = {
            "challenger": challenger, "version": version, "roles": roles, "position": 0,
            "rewards": {"incumbent": [], "challenger": []},
            "charges": {"incumbent": 0.0, "challenger": 0.0},
        }
        self.events.append({"type": "harness_trial_started", "attempt": attempt,
                            "incumbent_version": self.program_version,
                            "challenger_version": version, "role_order": roles})

    def _predict_from_memory(self, candidate_id: str, memory: list[dict]) -> float:
        config = self.candidates[candidate_id]["config"]
        closest = sorted(memory, key=lambda row: _distance(config, self.candidates[row["candidate_id"]]["config"]))[:3]
        weights = [1.0 / (1 + _distance(config, self.candidates[row["candidate_id"]]["config"]))
                   for row in closest]
        return sum(weight * row["val_bpb"] for weight, row in zip(weights, closest)) / sum(weights)

    def propose(self, available: list[str], attempt: int, remaining: int) -> tuple[str | None, dict]:
        if not available:
            return None, {}
        if self.method == "random":
            return self.rng.choice(available), {}
        if self.method == "evolution":
            # Aging population: only the last ten feasible evaluated candidates.
            return self._choose_program(replace(HarnessProgram(), exploration=0.0, memory_size=10),
                                        available, self.history[-10:]), {"population_size": min(10, len(self.history))}
        if self.method.startswith("harness-"):
            if self.trial is None and remaining >= 4:
                self._start_trial(attempt)
            if self.trial is not None:
                role = self.trial["roles"][self.trial["position"]]
                program = self.program if role == "incumbent" else self.trial["challenger"]
                version = self.program_version if role == "incumbent" else self.trial["version"]
            else:
                role, program, version = "incumbent", self.program, self.program_version
            return self._choose_program(program, available), {"harness_role": role, "program_version": version}
        if self.method.startswith("multiagent-"):
            proposer_slot = self.search_count % len(self.agents)
            critic_slot = self.routes[proposer_slot]
            proposer, critic = self.agents[proposer_slot], self.agents[critic_slot]
            proposer_memory = self.warm_history + proposer["memory"]
            critic_memory = self.warm_history + critic["memory"]
            first = self._choose_program(proposer["program"], available, proposer_memory)
            second = self._choose_program(critic["program"], available, critic_memory)
            selected = first
            if critic_memory and self._predict_from_memory(second, critic_memory) < self._predict_from_memory(first, critic_memory):
                selected = second
            return selected, {"proposer_id": proposer["id"], "critic_id": critic["id"],
                              "proposer_slot": proposer_slot, "critic_slot": critic_slot,
                              "original_candidate": first, "critic_candidate": second,
                              "critic_changed_candidate": selected != first}
        if self.method.startswith("weight-"):
            if self.weight_policy is not None:
                return self.weight_policy.propose([self.candidates[item] for item in available], self.history)
            indices = [self.id_to_index[candidate_id] for candidate_id in available]
            features = self.features[indices]
            probabilities = _softmax(features @ self.weights)
            target = self.rng.random()
            cumulative, selected = 0.0, len(available) - 1
            for i, probability in enumerate(probabilities):
                cumulative += float(probability)
                if target < cumulative:
                    selected = i
                    break
            prior = [row["val_bpb"] for row in self.history]
            mean = float(np.mean(prior)) if prior else 8.0
            scale = max(float(np.std(prior)), 0.05) if prior else 1.0
            # Keep the exact masked on-policy distribution for the one update
            # following this draw; no future outcome is used as a baseline.
            return available[selected], {
                "policy_probability": float(probabilities[selected]),
                "policy_version": self.gradient_updates,
                "_features": features, "_probabilities": probabilities, "_selected": selected,
                "_baseline_bpb": mean, "_baseline_scale": scale,
            }
        raise ValueError(f"Unknown method {self.method}")

    def observe(self, row: dict, metadata: dict) -> None:
        is_new_feasible = row["status"] == "valid_new"
        public_row = {key: row[key] for key in ("attempt", "candidate_id", "val_bpb")}
        if row["phase"] == "warm_start":
            if is_new_feasible:
                self.history.append(public_row)
                self.warm_history.append(public_row)
            return
        self.search_count += 1
        if self.method.startswith("harness-") and self.trial is not None:
            role = metadata["harness_role"]
            # Bounded monotone quality makes every feasible result preferable
            # to a failed attempt without guessing an upper bound for BPB.
            reward = 1.0 / (1.0 + row["val_bpb"]) if is_new_feasible else 0.0
            self.trial["rewards"][role].append(reward)
            self.trial["charges"][role] += row["charge_seconds"]
            self.trial["position"] += 1
            if self.trial["position"] == 4:
                scores = {role: float(np.mean(values)) for role, values in self.trial["rewards"].items()}
                would_promote = scores["challenger"] > scores["incumbent"] + 1e-12
                promoted = would_promote and self.adaptive
                self.events.append({"type": "harness_trial_completed", "attempt": row["attempt"],
                                    "incumbent_version": self.program_version,
                                    "challenger_version": self.trial["version"], "mean_reward": scores,
                                    "role_attempts": {role: len(values) for role, values in self.trial["rewards"].items()},
                                    "role_charge_seconds": dict(self.trial["charges"]),
                                    "would_promote": would_promote, "promoted": promoted,
                                    "decision": "promote" if promoted else "discard"})
                if promoted:
                    self.program = self.trial["challenger"]
                    self.program_version = self.trial["version"]
                self.trial = None
        if self.method.startswith("multiagent-"):
            # Joint candidate credit is shared by the two participating roles;
            # it does not estimate either role's separate causal contribution.
            reward = 1.0 / (1.0 + row["val_bpb"]) if is_new_feasible else 0.0
            for slot in (metadata["proposer_slot"], metadata["critic_slot"]):
                agent = self.agents[slot]
                self.epoch_credit.setdefault(agent["id"], []).append(reward)
                if is_new_feasible:
                    agent["memory"].append(public_row)
                    agent["memory"] = agent["memory"][-16:]
            if self.search_count % 6 == 0:
                scores = {agent["id"]: float(np.mean(self.epoch_credit.get(agent["id"], [0.0])))
                          for agent in self.agents}
                best_slot = max(range(3), key=lambda slot: (scores[self.agents[slot]["id"]], -slot))
                worst_slot = min(range(3), key=lambda slot: (scores[self.agents[slot]["id"]], slot))
                can_replace = best_slot != worst_slot and scores[self.agents[best_slot]["id"]] > scores[self.agents[worst_slot]["id"]] + 1e-12
                parent, replaced = self.agents[best_slot], self.agents[worst_slot]
                child_program = _mutate_program(parent["program"], self.edit_rng)
                new_route = self.edit_rng.choice([slot for slot in range(3) if slot != worst_slot])
                event = {"type": "population_selection", "attempt": row["attempt"],
                         "joint_credit": scores, "would_replace": can_replace,
                         "replaced": can_replace and self.adaptive,
                         "parent_id": parent["id"], "replaced_id": replaced["id"]}
                if can_replace and self.adaptive:
                    child_id = self.next_agent_id
                    self.next_agent_id += 1
                    self.agents[worst_slot] = {"id": child_id, "parent_id": parent["id"],
                                               "program": child_program, "memory": list(parent["memory"])}
                    self.routes[worst_slot] = new_route
                    self.lineages.append({"agent_id": child_id, "parent_id": parent["id"],
                                          "created_at_attempt": row["attempt"]})
                    event.update(child_id=child_id, child_program=asdict(child_program), route_to_slot=new_route)
                self.events.append(event)
                self.epoch_credit = {agent["id"]: [] for agent in self.agents}
        if self.method.startswith("weight-") and self.weight_policy is not None:
            self.weight_policy.observe(row, metadata)
        if self.method.startswith("weight-") and "_features" in metadata:
            features, probabilities = metadata["_features"], metadata["_probabilities"]
            centered_features = features - probabilities @ features
            advantage = ((metadata["_baseline_bpb"] - row["val_bpb"]) / metadata["_baseline_scale"]
                         if is_new_feasible else -5.0)
            advantage = float(np.clip(advantage, -5.0, 5.0))
            log_ratio = np.log(np.maximum(probabilities * len(probabilities), 1e-300))
            kl_gradient = (probabilities * log_ratio) @ centered_features
            gradient = advantage * centered_features[metadata["_selected"]] - 0.03 * kl_gradient
            norm = float(np.linalg.norm(gradient))
            clipped = gradient * min(1.0, 2.0 / max(norm, 1e-12))
            old = self.weights.copy()
            if self.adaptive:
                self.weights += 0.15 * clipped
                self.gradient_updates += 1
            self.events.append({"type": "policy_gradient", "attempt": row["attempt"],
                                "applied": self.adaptive, "policy_version": self.gradient_updates,
                                "advantage": advantage, "gradient_norm": norm,
                                "update_norm": float(np.linalg.norm(self.weights - old)),
                                "reference_kl": float(probabilities @ log_ratio),
                                "sample_probability": metadata["policy_probability"]})
        if is_new_feasible:
            self.history.append(public_row)

    def summary(self) -> dict:
        if self.weight_policy is not None:
            self.weight_policy.cancel_pending("search_finished_without_evaluation")
            return self.weight_policy.summary()
        output: dict[str, Any] = {"events": self.events, "controller_kind": "local_policy"}
        if self.method.startswith("harness-"):
            output.update(program_versions=self.versions, final_program_version=self.program_version,
                          final_program=asdict(self.program),
                          promotions=sum(bool(event.get("promoted")) for event in self.events),
                          incomplete_trial=self.trial is not None,
                          selection_reward="1/(1+val_bpb) for a new feasible result; 0 otherwise")
        if self.method.startswith("multiagent-"):
            output.update(agent_lineage=self.lineages, descendants=len(self.lineages) - 3,
                          final_agents=[{"id": agent["id"], "parent_id": agent["parent_id"],
                                         "program": asdict(agent["program"]), "memory_size": len(agent["memory"])}
                                        for agent in self.agents],
                          final_routes=self.routes, credit_type="joint_candidate_credit",
                          selection_reward="1/(1+val_bpb) for a new feasible result; 0 otherwise")
        if self.method.startswith("weight-"):
            output.update(policy_kind="numpy_linear_softmax_over_public_design_features",
                          trainable_parameters=int(self.weights.size), gradient_updates=self.gradient_updates,
                          initial_weight_sha256=self.initial_weight_hash,
                          final_weight_sha256=_fingerprint(self.weights.tolist()),
                          final_weight_norm=float(np.linalg.norm(self.weights)),
                          final_weights=self.weights.tolist(),
                          feature_terms=[list(item) for item in self.feature_terms])
        return output


def run_search(candidates: list[dict], evaluate: Callable[[str], dict], method: str,
               seed: int, budget: int = 12, warm_start: int = 4,
               cost_budget_seconds: float | None = None, weight_policy: Any = None) -> dict:
    """Run one reproducible search against an opaque measured-table evaluator.

    ``candidates`` exposes ``candidate_id`` (``id`` is an alias), ``config``, and
    optionally a public ``charge_seconds``.  Unknown top-level fields are
    discarded.  ``evaluate(candidate_id)`` returns ``val_bpb``, ``feasible``,
    ``status``, ``charge_seconds`` and optionally ``train_seconds``.  Extra fields
    including held-out scores are never copied or examined.

    Invalid proposals and repeats consume attempts.  A repeated ID reuses its
    observation and consumes zero virtual training cost.  A failed or infeasible
    new evaluation consumes its full quoted cost.  A cost stop happens before
    evaluation and leaves the unaffordable candidate unobserved.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if weight_policy is not None and not method.startswith("weight-"):
        raise ValueError("weight_policy is only valid for weight-adaptive or weight-frozen")
    _integer(seed, "seed")
    _integer(budget, "budget")
    _integer(warm_start, "warm_start")
    if cost_budget_seconds is not None:
        cost_budget_seconds = _number(cost_budget_seconds, "cost_budget_seconds", nonnegative=True)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a nonempty list")
    public = []
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id", candidate.get("id"))
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Each candidate needs a nonempty string candidate_id")
        config = candidate.get("config")
        if not isinstance(config, dict):
            raise ValueError("Each candidate needs a public config object")
        item = {"candidate_id": candidate_id, "config": json.loads(_canonical(config))}
        if "charge_seconds" in candidate:
            item["charge_seconds"] = _number(candidate["charge_seconds"], "charge_seconds", nonnegative=True)
        elif cost_budget_seconds is not None:
            raise ValueError("Cost-budgeted search requires public charge_seconds for every candidate")
        public.append(item)
    ids = [candidate["candidate_id"] for candidate in public]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique")
    by_id = {candidate["candidate_id"]: candidate for candidate in public}
    initialization = random.Random(seed).sample(ids, min(warm_start, budget, len(ids)))
    controller = (_MemoryFrontierController(public, method, seed)
                  if method.startswith("memory-frontier-") else
                  _LocalController(public, method, seed, weight_policy=weight_policy))
    trajectory, observed = [], {}
    charged = measured_train = 0.0
    winner = None
    stop_reason, stop_detail = "attempt_budget", None
    for attempt in range(1, budget + 1):
        if attempt <= len(initialization):
            candidate_id, metadata, phase = initialization[attempt - 1], {}, "warm_start"
        else:
            available = [candidate_id for candidate_id in ids if candidate_id not in observed]
            candidate_id, metadata = controller.propose(available, attempt, budget - attempt + 1)
            phase = "search"
            if candidate_id is None:
                stop_reason = "space_exhausted"
                break
        quoted = by_id.get(candidate_id, {}).get("charge_seconds") if isinstance(candidate_id, str) else None
        if isinstance(candidate_id, str) and candidate_id in by_id and candidate_id not in observed and cost_budget_seconds is not None:
            if charged + quoted > cost_budget_seconds + 1e-9:
                stop_reason = "cost_budget"
                stop_detail = {"candidate_id": candidate_id, "quoted_charge_seconds": quoted,
                               "remaining_charge_seconds": max(0.0, cost_budget_seconds - charged)}
                break
        row = {"attempt": attempt, "phase": phase, "candidate_id": candidate_id,
               "charge_seconds": 0.0, "train_seconds": 0.0, "val_bpb": None, "feasible": False,
               "controller": {key: deepcopy(value) for key, value in metadata.items() if not key.startswith("_")}}
        if not isinstance(candidate_id, str) or candidate_id not in by_id:
            row.update(status="invalid", error="unknown candidate_id")
        elif candidate_id in observed:
            cached = observed[candidate_id]
            row.update(status="duplicate", val_bpb=cached["val_bpb"], feasible=cached["feasible"],
                       original_attempt=cached["attempt"])
        else:
            result = evaluate(candidate_id)
            if not isinstance(result, dict):
                raise ValueError("Evaluator must return a result object")
            if result.get("candidate_id", candidate_id) != candidate_id:
                raise ValueError("Evaluator returned the wrong candidate_id")
            charge = _number(result.get("charge_seconds", quoted), "evaluator charge_seconds", nonnegative=True)
            if quoted is not None and not math.isclose(charge, quoted, rel_tol=0, abs_tol=1e-9):
                raise ValueError("Evaluator charge disagrees with public quote")
            train_seconds = _number(result.get("train_seconds", charge), "train_seconds", nonnegative=True)
            raw_status = result.get("status", "ok")
            if not isinstance(raw_status, str):
                raise ValueError("Evaluator status must be a string")
            feasible = result.get("feasible", raw_status in ("ok", "completed", "success", "valid"))
            if not isinstance(feasible, bool):
                raise ValueError("Evaluator feasible must be a boolean")
            score = result.get("val_bpb")
            if feasible:
                score = _number(score, "val_bpb", nonnegative=True)
            else:
                score = None
            row.update(status="valid_new" if feasible else "infeasible_or_failed",
                       evaluator_status=raw_status, val_bpb=score, feasible=feasible,
                       charge_seconds=charge, train_seconds=train_seconds)
            observed[candidate_id] = row
            charged += charge
            measured_train += train_seconds
            if feasible and (winner is None or score < winner["val_bpb"]):
                winner = {"candidate_id": candidate_id, "val_bpb": score,
                          "selected_at_attempt": attempt, "config": deepcopy(by_id[candidate_id]["config"])}
        row["cumulative_charge_seconds"] = charged
        row["best_val_bpb"] = None if winner is None else winner["val_bpb"]
        controller.observe(row, metadata)
        trajectory.append(row)
    lock_payload = {"method": method, "search_seed": seed, "candidate_id": None if winner is None else winner["candidate_id"],
                    "observations_sha256": _fingerprint([
                        {key: row[key] for key in ("attempt", "candidate_id", "status", "val_bpb", "charge_seconds")}
                        for row in trajectory
                    ])}
    output = {
        "schema_version": 1, "method": method, "search_seed": seed,
        "budget_attempts": budget, "budget_charge_seconds": cost_budget_seconds,
        "warm_start_requested": warm_start, "initial_candidate_ids": initialization,
        "attempts": len(trajectory), "new_evaluations": len(observed),
        "duplicate_attempts": sum(row["status"] == "duplicate" for row in trajectory),
        "invalid_attempts": sum(row["status"] == "invalid" for row in trajectory),
        "failed_or_infeasible_evaluations": sum(row["status"] == "infeasible_or_failed" for row in trajectory),
        "charged_seconds": charged, "observed_train_seconds": measured_train,
        "stop_reason": stop_reason, "stop_detail": stop_detail,
        "winner": winner, "winner_lock": dict(lock_payload, sha256=_fingerprint(lock_payload)),
        "trajectory": trajectory,
    }
    output.update(controller.summary())
    return output
