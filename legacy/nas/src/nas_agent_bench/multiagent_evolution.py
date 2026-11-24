"""Population evolution of collaborating NAS proposer/critic agents.

Only the public validation observations enter this controller. A round consists
of a draft and peer critique, followed by ONE final architecture evaluation by
the external runner. Internal drafts and model calls are reported separately.

The CPU backend is an explicit collection of search policies, not a simulated
LLM. Supplying ``model(payload) -> dict`` enables model-backed proposals and
critiques. Population replacement evolves roles, prompts, mutation/exploration
settings, critic revision rates, and communication shortcuts. ``evolve=False``
retains collaboration and the same call pattern with fixed agent genomes.
"""

from copy import deepcopy
import math
import random
import statistics

from .architecture import EDGES, OPS, SPACE_SIZE, validate_ops


_ROLES = ("explorer", "exploiter", "recombiner", "diversity")
_PROMPTS = {
    "explorer": "Explore underrepresented operations and distinct cell structures.",
    "exploiter": "Improve strong observed cells using small, evidence-based edits.",
    "recombiner": "Combine useful edges from different strong observed cells.",
    "diversity": "Preserve promising structure while testing uncommon operations.",
}
_DEFAULTS = {
    "evolve": True, "population_size": 4, "evaluations_per_agent": 3,
    "memory_size": 16, "share_top_k": 2, "share_every": 1,
    "replacement_margin": 0.0,
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _row(value):
    """An explicit boundary: test scores and raw evaluator fields cannot pass."""
    result = {k: value[k] for k in ("step", "phase", "status") if k in value}
    try:
        result["ops"] = list(validate_ops(value.get("ops")))
    except (ValueError, TypeError):
        result["ops"] = None
    for key in ("val_accuracy", "flops", "params"):
        if _finite(value.get(key)):
            result[key] = float(value[key])
    return result


def _tuple_tree(value):
    return tuple(_tuple_tree(x) for x in value) if isinstance(value, (list, tuple)) else value


def _connected(ops):
    reachable = {0}
    for (source, target), operation in zip(((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)), ops):
        if source in reachable and operation != "none":
            reachable.add(target)
    return 3 in reachable


class MultiAgentEvolutionController:
    """A validation-only, resumable population of communicating search agents."""

    def __init__(self, seed, config=None, model=None):
        supplied = dict(config or {})
        unknown = set(supplied) - set(_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown multi-agent configuration: {sorted(unknown)}")
        self.config = {**_DEFAULTS, **supplied}
        if not isinstance(self.config["evolve"], bool):
            raise ValueError("evolve must be boolean")
        bounds = {"population_size": (3, 16), "evaluations_per_agent": (1, 1000),
                  "memory_size": (2, 1024), "share_top_k": (1, 64), "share_every": (1, 1000)}
        for key, (lo, hi) in bounds.items():
            value = self.config[key]
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
        if self.config["share_top_k"] > self.config["memory_size"]:
            raise ValueError("share_top_k must not exceed memory_size")
        if not _finite(self.config["replacement_margin"]) or self.config["replacement_margin"] < 0:
            raise ValueError("replacement_margin must be finite and nonnegative")
        self.seed, self.model = int(seed), model
        self.rng = random.Random(seed)
        self.agents = [self._new_agent(i) for i in range(self.config["population_size"])]
        count = len(self.agents)
        self.shortcuts = [(i + 2) % count for i in range(count)]
        self.generation = 0
        self.births = 0
        self.proposal_count = 0
        self.model_calls = 0
        self.message_count = 0
        self._observed_step = 0
        self._seen = set()
        self._warm = []
        self._baseline = None
        self._scale = 1.0
        self._best = None
        self._pending = None
        self._events = []

    def _new_agent(self, slot, parent=None):
        role = _ROLES[slot % len(_ROLES)]
        return {
            "id": f"agent-{slot}-v0", "slot": slot, "parent_id": parent,
            "birth_generation": 0,
            "genome": {"role": role, "mutation_edges": 1 if role == "exploiter" else 2,
                       "exploration_rate": .6 if role == "explorer" else .1,
                       "critique_rate": .25, "prompt": _PROMPTS[role],
                       "critic_prompt": "Check connectivity, repetition, and support in your observed archive."},
            "memory": [], "proposal_rewards": [], "critic_rewards": [],
            "lifetime_proposals": 0, "lifetime_critiques": 0,
        }

    def _event(self, kind, **fields):
        self._events.append({"event": kind, "generation": self.generation,
                             "observed_step": self._observed_step, **deepcopy(fields)})

    def _remember(self, agent, row):
        if row.get("ops") is None or "val_accuracy" not in row:
            return
        key = tuple(row["ops"])
        agent["memory"] = [r for r in agent["memory"] if tuple(r["ops"]) != key]
        agent["memory"].append(deepcopy(row))
        # Keep half elite evidence and half most recent evidence, without repeats.
        limit = self.config["memory_size"]
        elite = sorted(agent["memory"], key=lambda r: -r["val_accuracy"])[:max(1, limit // 2)]
        keep = {tuple(r["ops"]) for r in elite}
        recent = [r for r in reversed(agent["memory"]) if tuple(r["ops"]) not in keep]
        agent["memory"] = sorted(elite + recent[:limit - len(elite)], key=lambda r: r["step"])

    def _recipients(self, slot):
        return sorted({(slot + 1) % len(self.agents), self.shortcuts[slot]} - {slot})

    def _elite(self, agent):
        rows = sorted(agent["memory"], key=lambda r: -r["val_accuracy"])
        return rows[:max(1, math.ceil(len(rows) / 3))]

    def _weights(self, rows, edge, inverse=False):
        values = [1.0] * len(OPS)
        for row in rows:
            values[OPS.index(row["ops"][edge])] += 1.0
        return [1.0 / v for v in values] if inverse else values

    def _sample_local(self, agent):
        if len(self._seen) >= SPACE_SIZE:
            raise RuntimeError("The encoding space is exhausted")
        genome, elite = agent["genome"], self._elite(agent)
        for draws in range(1, 129):
            if not elite or self.rng.random() < genome["exploration_rate"]:
                ops = [self.rng.choice(OPS) for _ in EDGES]
            elif genome["role"] == "diversity":
                ops = [self.rng.choices(OPS, weights=self._weights(agent["memory"], i, True))[0]
                       for i in range(6)]
            else:
                parent = elite[0] if genome["role"] == "exploiter" else self.rng.choice(elite)
                ops = list(parent["ops"])
                if genome["role"] == "recombiner" and len(elite) > 1:
                    other = self.rng.choice(elite)
                    ops = [op if self.rng.random() < .5 else other["ops"][i] for i, op in enumerate(ops)]
                for edge in self.rng.sample(range(6), genome["mutation_edges"]):
                    choices = [op for op in OPS if op != ops[edge]]
                    weights = self._weights(elite, edge)
                    ops[edge] = self.rng.choices(choices, weights=[weights[OPS.index(op)] for op in choices])[0]
            if tuple(ops) not in self._seen:
                return ops, draws
        # Finite and deterministic fallback when a local neighbourhood is saturated.
        start = self.rng.randrange(SPACE_SIZE)
        for offset in range(SPACE_SIZE):
            index, ops = (start + offset) % SPACE_SIZE, []
            for _ in EDGES:
                ops.append(OPS[index % len(OPS)])
                index //= len(OPS)
            if tuple(ops) not in self._seen:
                return ops, 128 + offset + 1
        raise RuntimeError("The encoding space is exhausted")

    def _local_critique(self, critic, draft):
        ops = list(draft)
        reasons = []
        if tuple(ops) in self._seen:
            ops, draws = self._sample_local(critic)
            return ops, "Replace a previously evaluated encoding.", draws
        if not _connected(ops):
            # This is a structural check, not an unevaluated accuracy claim.
            weights = self._weights(self._elite(critic), 3)
            ops[3] = self.rng.choices(OPS[1:], weights=weights[1:])[0]
            reasons.append("Add an input-to-output path; the draft has none.")
        elif critic["memory"] and self.rng.random() < critic["genome"]["critique_rate"]:
            elite = self._elite(critic)
            alternatives = []
            for edge in range(6):
                weights = self._weights(elite, edge)
                best = max(range(len(OPS)), key=lambda i: weights[i])
                support = weights[best] - weights[OPS.index(ops[edge])]
                if support > 0:
                    alternatives.append((support, edge, OPS[best]))
            if alternatives:
                _, edge, op = max(alternatives)
                ops[edge] = op
                reasons.append(f"Test the operation favoured by this peer's elite archive on edge {EDGES[edge]}.")
        if tuple(ops) in self._seen:
            return list(draft), "Keep the draft: the proposed revision repeats an evaluated encoding.", 1
        return ops, " ".join(reasons) or "Accept: valid structure with no repeated encoding.", 1

    def _payload(self, observation, agent, stage, draft=None, draft_rationale=""):
        context = observation.get("search_context", {})
        payload = {
            "task": "Propose one NAS cell." if stage == "proposer" else "Critique one NAS draft; accept it or revise once.",
            "stage": stage, "agent_id": agent["id"], "agent_genome": deepcopy(agent["genome"]),
            "allowed_operations": list(OPS), "edge_order": list(EDGES),
            "search_context": {k: context[k] for k in ("dataset", "hp", "training_seed_policy", "accuracy_unit", "objective") if k in context},
            "feedback_enabled": True, "history": deepcopy(agent["memory"]),
            "seen_ops": [list(x) for x in sorted(self._seen)],
            "next_attempt": observation.get("next_attempt"),
            "remaining_attempts": observation.get("remaining_attempts"),
            "rules": ["Only the final candidate will be evaluated, using validation only.",
                      "Use only the supplied observations; do not query files, tools, benchmarks or other runs.",
                      "Return one JSON object and at most one six-operation candidate."],
        }
        if stage == "critic":
            payload.update({"draft_ops": draft, "draft_rationale": draft_rationale,
                            "response_format": {"ops": "optional revised six-operation list", "critique": "short explanation"}})
        else:
            payload["response_format"] = {"ops": ["six operation tokens"], "rationale": "short explanation"}
        return payload

    def _call_model(self, payload):
        self.model_calls += 1
        try:
            response = self.model(deepcopy(payload))
            if not isinstance(response, dict):
                raise ValueError("Model response must be an object")
            return response, None
        except Exception as exc:
            return {}, type(exc).__name__

    def propose(self, observation):
        if self._pending is not None:
            raise RuntimeError("Observe the pending proposal before requesting another")
        if observation.get("complete") or observation.get("remaining_attempts") == 0:
            raise ValueError("The search budget is exhausted")
        if observation.get("feedback_enabled") is False:
            raise ValueError("Multi-agent evolution requires validation feedback; use evolve=False for the frozen-policy ablation")
        step = observation.get("next_attempt", self._observed_step + 1)
        if type(step) is not int or step != self._observed_step + 1:
            raise ValueError("The next proposal must follow the latest observed step")
        if self._baseline is None:
            values = [r["val_accuracy"] for r in self._warm if "val_accuracy" in r]
            self._baseline = statistics.fmean(values) if values else 0.0
            self._scale = max(statistics.pstdev(values), 1.0) if values else 1.0
        slot = self.proposal_count % len(self.agents)
        proposer = self.agents[slot]
        recipients = self._recipients(slot)
        # Alternate between backbone and shortcut: the learned route changes peers.
        critic_slot = recipients[(self.proposal_count // len(self.agents)) % len(recipients)]
        critic = self.agents[critic_slot]
        before_calls = self.model_calls
        failures, labels, usage = [], [], {}
        if self.model is None:
            draft, draws = self._sample_local(proposer)
            draft_rationale = proposer["genome"]["prompt"]
            final_ops, critique, critique_draws = self._local_critique(critic, draft)
            draws += critique_draws
        else:
            draft_response, error = self._call_model(self._payload(observation, proposer, "proposer"))
            if error:
                failures.append({"stage": "proposer", "error": error})
            try:
                draft = list(validate_ops(draft_response.get("ops")))
            except (TypeError, ValueError):
                draft = None
                failures.append({"stage": "proposer", "error": "invalid_ops"})
            draft_rationale = str(draft_response.get("rationale", ""))[:2000]
            peer_response, error = self._call_model(self._payload(observation, critic, "critic", draft, draft_rationale))
            if error:
                failures.append({"stage": "critic", "error": error})
            final_ops = draft
            if "ops" in peer_response:
                try:
                    final_ops = list(validate_ops(peer_response["ops"]))
                except (TypeError, ValueError):
                    failures.append({"stage": "critic", "error": "invalid_revision_ignored"})
            critique = str(peer_response.get("critique", peer_response.get("rationale", "")))[:2000]
            draws = 2
            for response in (draft_response, peer_response):
                if isinstance(response.get("model"), str):
                    labels.append(response["model"][:256])
                if isinstance(response.get("usage"), dict):
                    for key, value in response["usage"].items():
                        if isinstance(key, str) and _finite(value) and value >= 0:
                            usage[key] = usage.get(key, 0) + value
        revised = final_ops != draft
        self._pending = {"step": step, "proposer_slot": slot, "critic_slot": critic_slot,
                         "proposer_id": proposer["id"], "critic_id": critic["id"],
                         "best_before": self._best, "draft_ops": draft, "final_ops": final_ops,
                         "revised": revised}
        self.proposal_count += 1
        proposer["lifetime_proposals"] += 1
        critic["lifetime_critiques"] += 1
        metadata = {"backend": "model" if self.model is not None else "local_policy",
                    "generation": self.generation, "proposer_id": proposer["id"], "critic_id": critic["id"],
                    "model_calls": self.model_calls - before_calls, "internal_candidate_draws": draws,
                    "draft_ops": draft, "peer_critique": critique, "revised": revised,
                    "model_failures": failures}
        self._event("multiagent_round", step=step, **metadata, final_ops=final_ops)
        result = {"ops": final_ops, "rationale": critique or draft_rationale,
                  "controller_metadata": metadata}
        if usage:
            result["usage"] = usage
        if labels:
            result["model"] = "+".join(dict.fromkeys(labels))
        return result

    def _reward(self, row, best_before):
        if row.get("status") == "invalid" or "val_accuracy" not in row:
            return -2.0
        if row.get("status") == "duplicate":
            return -.5
        quality = max(-3.0, min(3.0, (row["val_accuracy"] - self._baseline) / self._scale))
        improvement = max(0.0, row["val_accuracy"] - best_before) / self._scale if best_before is not None else 0.0
        return quality + .25 * min(3.0, improvement) + .05

    def observe(self, value):
        row = _row(value)
        step = row.get("step")
        if type(step) is not int or step != self._observed_step + 1:
            raise ValueError("Observations must be supplied exactly once in consecutive step order")
        pending = self._pending
        if pending is not None and pending["step"] != step:
            raise ValueError("Observation does not match the pending proposal step")
        if pending is None and self.proposal_count:
            raise ValueError("Search observations require a pending proposal")
        if pending is not None and row.get("ops") != pending["final_ops"]:
            raise ValueError("Observation encoding differs from the submitted final candidate")
        self._observed_step = step
        if row.get("ops") is not None:
            self._seen.add(tuple(row["ops"]))
        if "val_accuracy" in row:
            self._best = max(self._best, row["val_accuracy"]) if self._best is not None else row["val_accuracy"]
        if pending is None:
            self._warm.append(row)
            for agent in self.agents:
                self._remember(agent, row)
            return
        proposer = self.agents[pending["proposer_slot"]]
        critic = self.agents[pending["critic_slot"]]
        reward = self._reward(row, pending["best_before"])
        proposer["proposal_rewards"].append(reward)
        critic["critic_rewards"].append(reward)
        for agent in (proposer, critic):
            self._remember(agent, row)
        self._event("multiagent_credit", step=step, proposer_id=proposer["id"], critic_id=critic["id"],
                    reward=reward, credit_rule="joint final-candidate reward; no counterfactual draft evaluation",
                    revised=pending["revised"], status=row.get("status"))
        if self.proposal_count % self.config["share_every"] == 0:
            shared = sorted(proposer["memory"], key=lambda r: -r["val_accuracy"])[:self.config["share_top_k"]]
            for recipient in self._recipients(pending["proposer_slot"]):
                for message in shared:
                    self._remember(self.agents[recipient], message)
                self.message_count += len(shared)
                self._event("multiagent_share", sender=proposer["id"], recipient=self.agents[recipient]["id"],
                            messages=shared)
        self._pending = None
        window = self.config["evaluations_per_agent"]
        if all(len(agent["proposal_rewards"]) >= window for agent in self.agents):
            self._evolve_population()

    def _fitness(self, agent):
        proposal = statistics.fmean(agent["proposal_rewards"]) if agent["proposal_rewards"] else -2.0
        peer = statistics.fmean(agent["critic_rewards"]) if agent["critic_rewards"] else 0.0
        return proposal + .25 * peer

    def _mutate_genome(self, parent):
        genome = deepcopy(parent)
        key = self.rng.choice(("role", "mutation_edges", "exploration_rate", "critique_rate"))
        if key == "role":
            genome[key] = self.rng.choice([role for role in _ROLES if role != genome[key]])
        elif key == "mutation_edges":
            genome[key] = self.rng.choice([count for count in (1, 2, 3) if count != genome[key]])
        elif key == "exploration_rate":
            genome[key] = self.rng.choice([rate for rate in (.05, .1, .3, .6, .9) if rate != genome[key]])
        else:
            genome[key] = self.rng.choice([rate for rate in (.0, .25, .5, .75, 1.0) if rate != genome[key]])
        # Prompts reflect the executable strategy, so model and local runs use the same genome.
        genome["prompt"] = (_PROMPTS[genome["role"]] +
                            f" Use approximately {genome['mutation_edges']} edge edits and "
                            f"{genome['exploration_rate']:.0%} exploration when appropriate.")
        genome["critic_prompt"] = ("Check connectivity, repetition, and support in your observed archive. "
                                   f"Consider an evidence-based revision with rate {genome['critique_rate']:.0%}.")
        return genome

    def _evolve_population(self):
        fitness = [self._fitness(agent) for agent in self.agents]
        winner = max(range(len(self.agents)), key=lambda i: (fitness[i], -i))
        loser = min(range(len(self.agents)), key=lambda i: (fitness[i], -i))
        self._event("multiagent_generation", fitness=[{"agent_id": a["id"], "fitness": f,
                    "proposal_evaluations": len(a["proposal_rewards"]), "critic_evaluations": len(a["critic_rewards"])}
                    for a, f in zip(self.agents, fitness)], evolve_enabled=self.config["evolve"])
        margin = fitness[winner] - fitness[loser]
        if self.config["evolve"] and winner != loser and margin > self.config["replacement_margin"]:
            parent, retired = self.agents[winner], self.agents[loser]
            child = deepcopy(parent)
            self.births += 1
            child.update({"id": f"agent-{loser}-v{self.births}", "slot": loser, "parent_id": parent["id"],
                          "birth_generation": self.generation + 1, "genome": self._mutate_genome(parent["genome"]),
                          "lifetime_proposals": 0, "lifetime_critiques": 0})
            old_routes = list(self.shortcuts)
            # Retain a ring backbone; replace an adaptive shortcut toward a successful peer.
            options = [i for i in range(len(self.agents)) if i not in (loser, (loser + 1) % len(self.agents))]
            if winner in options and self.shortcuts[loser] != winner:
                self.shortcuts[loser] = winner
            else:
                alternatives = [i for i in options if i != self.shortcuts[loser]]
                if alternatives:
                    self.shortcuts[loser] = self.rng.choice(alternatives)
            self.agents[loser] = child
            self._event("multiagent_replacement", parent_id=parent["id"], retired_id=retired["id"],
                        child_id=child["id"], fitness_margin=margin, inherited_memory=len(child["memory"]),
                        previous_genome=retired["genome"], parent_genome=parent["genome"], new_genome=child["genome"],
                        previous_shortcuts=old_routes, new_shortcuts=self.shortcuts)
        else:
            self._event("multiagent_retained", reason="frozen" if not self.config["evolve"] else "insufficient_fitness_margin",
                        fitness_margin=margin)
        for agent in self.agents:
            agent["proposal_rewards"], agent["critic_rewards"] = [], []
        self.generation += 1

    def drain_events(self):
        events, self._events = self._events, []
        return events

    def state_dict(self):
        return deepcopy({
            "schema_version": 1, "controller": "multiagent_evolution", "seed": self.seed,
            "config": self.config, "backend": "model" if self.model is not None else "local_policy",
            "rng_state": self.rng.getstate(), "agents": self.agents, "shortcuts": self.shortcuts,
            "generation": self.generation, "births": self.births, "proposal_count": self.proposal_count,
            "model_calls": self.model_calls, "message_count": self.message_count,
            "observed_step": self._observed_step, "seen": [list(x) for x in sorted(self._seen)],
            "warm": self._warm, "baseline": self._baseline, "scale": self._scale, "best": self._best,
            "pending": self._pending, "events": self._events,
        })

    def load_state_dict(self, state):
        if state.get("schema_version") != 1 or state.get("controller") != "multiagent_evolution":
            raise ValueError("Incompatible multi-agent checkpoint")
        if state.get("backend") != ("model" if self.model is not None else "local_policy"):
            raise ValueError("Restore with the same local/model backend")
        restored = type(self)(state["seed"], config=state["config"], model=self.model)
        restored.rng.setstate(_tuple_tree(state["rng_state"]))
        for key in ("agents", "shortcuts", "generation", "births", "proposal_count", "model_calls", "message_count"):
            setattr(restored, key, deepcopy(state[key]))
        for public, private in (("observed_step", "_observed_step"), ("warm", "_warm"),
                                ("baseline", "_baseline"), ("scale", "_scale"), ("best", "_best"),
                                ("pending", "_pending"), ("events", "_events")):
            setattr(restored, private, deepcopy(state[public]))
        restored._seen = {validate_ops(x) for x in state["seen"]}
        self.__dict__.update(restored.__dict__)


MultiAgentController = MultiAgentEvolutionController
