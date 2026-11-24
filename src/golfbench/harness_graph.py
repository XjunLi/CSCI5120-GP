"""An evolving evidence-memory and exploration-frontier harness.

The persistent objects are a memory schema with evidence clusters/hypotheses,
a typed executable workflow graph, and a frontier policy over public regions.
Only ``observe`` supplies scores.  The class has no oracle, table or file access.
All ablations retain the append-only observation ledger and ordinary feedback.

Edits are evidence-conditioned hypotheses, not independently validated
promotions.  Schema/frontier updates start after four queried candidates and
continue every two queries; workflow edits are limited to query counts 4 and 8.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import random
import statistics
from typing import Any


_FACTORS = ("num_layers", "model_dim", "tied_embeddings", "lr_scale")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _mean(values, default=0.0):
    values = list(values)
    return sum(values) / len(values) if values else default


class MemoryFrontierHarness:
    """Three independently switchable interventions over the same raw evidence.

    ``propose(observations, remaining_ids, rng)`` returns an ID and compact trace.
    ``observe(row)`` accepts a completed query, including common initialization.
    Passing the history to ``propose`` is optional/idempotent; it does not reveal
    any data beyond that explicit history.  Extra candidate/result fields are
    discarded rather than persisted or used as features.
    """

    def __init__(self, candidates: list[dict], seed: int,
                 adapt_memory: bool = True, adapt_structure: bool = True,
                 adapt_frontier: bool = True):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        flags = dict(memory=adapt_memory, structure=adapt_structure, frontier=adapt_frontier)
        if any(not isinstance(value, bool) for value in flags.values()):
            raise ValueError("Intervention switches must be booleans")
        self.candidates = {}
        for item in candidates:
            candidate_id = item.get("candidate_id", item.get("id"))
            if not isinstance(candidate_id, str) or not candidate_id or not isinstance(item.get("config"), dict):
                raise ValueError("Each candidate needs an ID and a public config")
            if candidate_id in self.candidates:
                raise ValueError("Candidate IDs must be unique")
            self.candidates[candidate_id] = json.loads(_json(item["config"]))
        if not self.candidates:
            raise ValueError("At least one public candidate is required")
        keys = set().union(*(config.keys() for config in self.candidates.values()))
        standard = [key for key in _FACTORS if key in keys]
        self.factors = [key for key in (standard or sorted(keys))
                        if len({_json(config.get(key)) for config in self.candidates.values()}) > 1]
        self.levels = {key: sorted({_json(config.get(key)) for config in self.candidates.values()})
                       for key in self.factors}
        self.seed, self.adapt = seed, flags
        self.rng = random.Random(seed)
        self.ledger: list[dict] = []
        self.seen: dict[str, dict] = {}
        self.proposal_count = 0
        self.events: list[dict] = []
        self.component_versions = dict(memory=0, structure=0, frontier=0)
        self.memory_policy = {"group_by": [], "representative_policy": "best_worst_latest",
                              "active_hypotheses": []}
        self.hypotheses: dict[str, dict] = {}
        self.clusters: dict[str, dict] = {}
        self.contrast_edges: list[dict] = []
        self.graph = {
            "entry": "entry",
            "nodes": {
                "entry": {"type": "public_pool", "params": {}},
                "retrieve": {"type": "evidence_retrieval", "params": {"neighbors": 3}},
                "frontier": {"type": "frontier_lookup", "params": {}},
                "rank": {"type": "rank", "params": {"frontier_weight": 0.35, "information_weight": 0.45}},
                "select": {"type": "select", "params": {"coverage_every": 4}},
            },
            "edges": [{"from": left, "to": right} for left, right in
                      (("entry", "retrieve"), ("retrieve", "frontier"),
                       ("frontier", "rank"), ("rank", "select"))],
        }
        self.frontier_policy = {"quality_weight": 0.25, "coverage_weight": 1.0,
                                "uncertainty_weight": 0.5, "focus_regions": [],
                                "focus_bonus": 0.0}
        self.frontier: dict = {}
        self.versions = {"memory": [{"version": 0, "policy": deepcopy(self.memory_policy)}],
                         "structure": [{"version": 0, "graph": deepcopy(self.graph)}],
                         "frontier": [{"version": 0, "policy": deepcopy(self.frontier_policy)}]}
        self.pending_structure_comparison: dict | None = None
        self.traces: list[dict] = []
        self._refresh_memory()
        self._refresh_frontier()

    def _distance(self, left: str, right: str) -> int:
        return sum(self.candidates[left].get(key) != self.candidates[right].get(key) for key in self.factors)

    def _valid(self) -> list[dict]:
        return [row for row in self.ledger if row["feasible"]]

    def _global(self) -> tuple[float, float]:
        values = [row["val_bpb"] for row in self._valid()]
        return _mean(values, 8.0), max(statistics.pstdev(values), 0.05) if len(values) > 1 else 1.0

    def _factor_statistics(self) -> dict:
        global_mean, scale = self._global()
        result = {}
        for factor in self.factors:
            groups = defaultdict(list)
            for row in self._valid():
                groups[_json(self.candidates[row["candidate_id"]].get(factor))].append(row)
            means = {value: _mean(row["val_bpb"] for row in rows) for value, rows in groups.items()}
            between = sum(len(groups[value]) * (mean - global_mean) ** 2 for value, mean in means.items()) / max(1, sum(map(len, groups.values())))
            coverage = len(groups) / len(self.levels[factor])
            support = min((len(rows) for rows in groups.values()), default=0)
            association = (math.sqrt(between) / scale) * coverage * min(1.0, support / 2)
            if len(groups) < 2:
                association = 0.0
            result[factor] = {"groups": groups, "means": means,
                              "association": association, "coverage": coverage}
        return result

    def _matched_contrasts(self) -> list[dict]:
        result = []
        rows = self._valid()
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                if self._distance(left["candidate_id"], right["candidate_id"]) != 1:
                    continue
                factor = next(key for key in self.factors
                              if self.candidates[left["candidate_id"]].get(key) != self.candidates[right["candidate_id"]].get(key))
                result.append({"factor": factor,
                               "left_value": _json(self.candidates[left["candidate_id"]].get(factor)),
                               "right_value": _json(self.candidates[right["candidate_id"]].get(factor)),
                               "delta_bpb": right["val_bpb"] - left["val_bpb"],
                               "support_ids": [left["evidence_id"], right["evidence_id"]]})
        return result

    def _refresh_memory(self) -> None:
        grouped = defaultdict(list)
        dimensions = self.memory_policy["group_by"]
        for row in self._valid():
            key = _json({factor: self.candidates[row["candidate_id"]].get(factor) for factor in dimensions})
            grouped[key].append(row)
        clusters = {}
        for key, rows in sorted(grouped.items()):
            best = min(rows, key=lambda row: row["val_bpb"])
            worst = max(rows, key=lambda row: row["val_bpb"])
            representatives = list(dict.fromkeys(row["evidence_id"] for row in (best, worst, rows[-1])))
            values = [row["val_bpb"] for row in rows]
            cluster_id = "cluster-" + _hash([dimensions, key])[:12]
            clusters[cluster_id] = {"key": json.loads(key), "support_count": len(rows),
                                    "support_ids": [row["evidence_id"] for row in rows],
                                    "representative_ids": representatives,
                                    "mean_bpb": _mean(values), "best_bpb": min(values),
                                    "dispersion": statistics.pstdev(values) if len(values) > 1 else 0.0}
        self.clusters = clusters
        # Edges preserve exact observed one-factor comparisons.  Generic cluster
        # associations are never labelled causal factor effects.
        self.contrast_edges = self._matched_contrasts()

    def _record_edit(self, component: str, before: dict, after: dict, reason: str,
                     operations: list[dict]) -> None:
        if before == after:
            return
        self.component_versions[component] += 1
        version = self.component_versions[component]
        event = {"type": "harness_component_edit", "component": component, "version": version,
                 "query_count": len(self.ledger), "attempt": self.ledger[-1]["attempt"],
                 "before_hash": _hash(before), "after_hash": _hash(after),
                 "before": deepcopy(before), "after": deepcopy(after),
                 "reason": reason, "operations": operations,
                 "evidence_ids": [row["evidence_id"] for row in self.ledger],
                 "validation_status": "evidence_conditioned_edit_without_independent_validation"}
        self.events.append(event)
        item = {"version": version, "query_count": len(self.ledger), "state_hash": _hash(after)}
        item["graph" if component == "structure" else "policy"] = deepcopy(after)
        self.versions[component].append(item)

    def _evolve_memory(self) -> None:
        statistics_by_factor = self._factor_statistics()
        ranked = sorted(self.factors, key=lambda key: (-statistics_by_factor[key]["association"], key))
        dimensions = ranked[:1]
        if len(self._valid()) >= 8 and len(ranked) >= 2:
            proposed = ranked[:2]
            joint_counts = Counter(tuple(_json(self.candidates[row["candidate_id"]].get(key)) for key in proposed)
                                   for row in self._valid())
            if sum(count >= 2 for count in joint_counts.values()) >= 2:
                dimensions = proposed
        before = deepcopy(self.memory_policy)
        active = []
        operations = []
        for factor in dimensions:
            stats = statistics_by_factor[factor]
            if len(stats["means"]) < 2:
                continue
            value = min(stats["means"], key=lambda item: (stats["means"][item], item))
            other_rows = [row for level, rows in stats["groups"].items() if level != value for row in rows]
            rows = stats["groups"][value]
            other_mean = _mean(row["val_bpb"] for row in other_rows)
            hypothesis_id = "hyp-" + _hash([factor, value])[:12]
            active.append(hypothesis_id)
            support_ids = [row["evidence_id"] for row in rows + other_rows]
            counterexamples = [row["evidence_id"] for row in rows if row["val_bpb"] > other_mean]
            matched = [edge for edge in self._matched_contrasts() if edge["factor"] == factor
                       and value in (edge["left_value"], edge["right_value"])]
            confidence = min(1.0, (len(matched) + 0.25 * min(len(rows), len(other_rows))) / 3)
            if hypothesis_id not in self.hypotheses:
                operations.append({"operation": "create_hypothesis", "hypothesis_id": hypothesis_id})
            self.hypotheses[hypothesis_id] = {
                "id": hypothesis_id, "factor": factor, "value": json.loads(value),
                "claim": "lower_observed_group_mean", "inference_type": "association",
                "group_mean_bpb": stats["means"][value], "comparison_mean_bpb": other_mean,
                "support_ids": support_ids, "counterexample_ids": counterexamples,
                "matched_contrast_count": len(matched), "confidence_heuristic": confidence,
                "status": "active", "updated_at_query": len(self.ledger),
            }
        for hypothesis_id, hypothesis in self.hypotheses.items():
            if hypothesis_id not in active and hypothesis["status"] == "active":
                hypothesis["status"] = "retired"
                hypothesis["retired_at_query"] = len(self.ledger)
                operations.append({"operation": "retire_hypothesis", "hypothesis_id": hypothesis_id})
        if dimensions != before["group_by"]:
            operation = "specialize_schema" if len(dimensions) > len(before["group_by"]) else "repartition_schema"
            if len(dimensions) < len(before["group_by"]):
                operation = "merge_schema"
            operations.append({"operation": operation, "from_dimensions": before["group_by"],
                               "to_dimensions": dimensions})
        self.memory_policy = {"group_by": dimensions, "representative_policy": "best_worst_latest",
                              "active_hypotheses": sorted(active)}
        self._refresh_memory()
        self._record_edit("memory", before, self.memory_policy,
                          "Reorganize observed evidence by supported factor associations; retain the raw ledger and counterexamples.",
                          operations)

    def _structure_signals(self) -> dict:
        valid = self._valid()
        # The window is two actual queries, not two successful measurements.
        previous = [row for row in self.ledger[:-2] if row["feasible"]]
        improved = bool(previous and valid) and min(row["val_bpb"] for row in valid) < min(row["val_bpb"] for row in previous) - 1e-12
        contrasts = self._matched_contrasts()
        signed = defaultdict(list)
        for edge in contrasts:
            levels = tuple(sorted((edge["left_value"], edge["right_value"])))
            delta = edge["delta_bpb"] if edge["left_value"] == levels[0] else -edge["delta_bpb"]
            signed[(edge["factor"], levels)].append((delta, edge["support_ids"]))
        conflicts = []
        for (factor, levels), values in sorted(signed.items()):
            positive = [support for delta, support in values if delta > 1e-12]
            negative = [support for delta, support in values if delta < -1e-12]
            if positive and negative:
                conflicts.append({"factor": factor, "levels": [json.loads(value) for value in levels],
                                  "positive_support_ids": sorted({item for pair in positive for item in pair}),
                                  "negative_support_ids": sorted({item for pair in negative for item in pair})})
        regions = list(self.frontier["regions"].values())
        return {"recent_improvement": improved, "recent_plateau": bool(previous) and not improved,
                "max_coverage_gap": max((region["coverage_gap"] for region in regions), default=0.0),
                "regions_without_feasible_evidence": sum(not region["observed_count"] and bool(region["remaining_candidates"])
                                                         for region in regions),
                "matched_contrast_count": len(contrasts), "contrast_conflicts": conflicts,
                "query_window_evidence_ids": [row["evidence_id"] for row in self.ledger[-2:]]}

    def _evolve_structure(self) -> None:
        before = deepcopy(self.graph)
        graph = deepcopy(self.graph)
        signals = self._structure_signals()
        matched = signals["matched_contrast_count"] > 0
        conflicting = bool(signals["contrast_conflicts"])
        operation_type = "matched_contrast" if matched and not conflicting else "contrast_target"
        graph["nodes"]["contrast"] = {"type": operation_type, "params": {"blend": 0.6}}
        graph["edges"] = [edge for edge in graph["edges"] if edge != {"from": "retrieve", "to": "frontier"}]
        for edge in ({"from": "retrieve", "to": "contrast"}, {"from": "contrast", "to": "frontier"}):
            if edge not in graph["edges"]:
                graph["edges"].append(edge)
        action = "insert_matched_contrast" if operation_type == "matched_contrast" else "insert_comparison_target"
        reason = ("Observed one-factor comparisons support a matched-contrast operator."
                  if operation_type == "matched_contrast" else
                  "Missing or conflicting one-factor comparisons favor comparison targets over additive extrapolation.")
        add_coverage_route = (len(self.ledger) == 8 and signals["recent_plateau"]
                              and signals["max_coverage_gap"] >= 0.7 and not conflicting)
        if add_coverage_route:
            graph["nodes"]["route"] = {"type": "coverage_router", "params": {"gap_threshold": 0.7, "period": 3}}
            graph["nodes"]["coverage"] = {"type": "coverage_proposal", "params": {}}
            graph["edges"] = [edge for edge in graph["edges"] if edge != {"from": "entry", "to": "retrieve"}]
            graph["edges"] += [{"from": "entry", "to": "route"},
                               {"from": "route", "to": "coverage", "when": "coverage"},
                               {"from": "route", "to": "retrieve", "when": "evidence"},
                               {"from": "coverage", "to": "frontier"}]
            action = "insert_coverage_route"
            reason = "The last two queries produced no new best score while a public factor region remains at least 70% unqueried."
        elif len(self.ledger) == 8:
            previous_type = before["nodes"].get("contrast", {}).get("type")
            if conflicting:
                action = "prefer_comparison_targets"
                reason = "Observed matched comparisons disagree in sign across contexts; avoid unsupported additive extrapolation."
            elif previous_type != operation_type:
                action = "upgrade_evidence_operator"
                reason = "New matched comparisons support a different evidence operator; a coverage detour is not supported by the plateau/gap condition."
            else:
                action = "retain_evidence_path"
                reason = ("Recent queries still improve the incumbent; retain the evidence path."
                          if signals["recent_improvement"] else
                          "The plateau and coverage-gap condition does not support adding a coverage route.")
        added_nodes = sorted(set(graph["nodes"]) - set(before["nodes"]))
        removed_nodes = sorted(set(before["nodes"]) - set(graph["nodes"]))
        changed_nodes = sorted(key for key in set(before["nodes"]) & set(graph["nodes"])
                               if before["nodes"][key] != graph["nodes"][key])
        old_edges, new_edges = {_json(edge) for edge in before["edges"]}, {_json(edge) for edge in graph["edges"]}
        operations = [{"operation": "edit_workflow", "added_nodes": added_nodes, "removed_nodes": removed_nodes,
                       "changed_nodes": changed_nodes,
                       "added_edges": [json.loads(edge) for edge in sorted(new_edges - old_edges)],
                       "removed_edges": [json.loads(edge) for edge in sorted(old_edges - new_edges)]}]
        self.graph = graph
        self.events.append({"type": "structure_decision", "query_count": len(self.ledger),
                            "attempt": self.ledger[-1]["attempt"], "signals": signals,
                            "action": action, "reason": reason, "graph_changed": before != graph,
                            "before_hash": _hash(before), "after_hash": _hash(graph),
                            "evidence_ids": [row["evidence_id"] for row in self.ledger]})
        self._record_edit("structure", before, graph,
                          reason,
                          operations)
        if before != graph:
            self.pending_structure_comparison = {"before": before, "after": deepcopy(graph),
                                                 "version": self.component_versions["structure"]}

    def _refresh_frontier(self) -> None:
        global_mean, scale = self._global()
        regions = {}
        for factor in self.factors:
            for value in self.levels[factor]:
                members = [candidate_id for candidate_id, config in self.candidates.items()
                           if _json(config.get(factor)) == value]
                rows = [self.seen[candidate_id] for candidate_id in members
                        if candidate_id in self.seen and self.seen[candidate_id]["feasible"]]
                remaining = [candidate_id for candidate_id in members if candidate_id not in self.seen]
                region_id = "region-" + _hash([factor, value])[:12]
                # Shrink small observed groups toward the global mean.
                mean = (_mean(row["val_bpb"] for row in rows) * len(rows) + 2 * global_mean) / (len(rows) + 2)
                quality = (global_mean - mean) / scale
                gap = len(remaining) / len(members)
                uncertainty = 1.0 / math.sqrt(1 + len(rows))
                policy = self.frontier_policy
                priority = (policy["quality_weight"] * quality + policy["coverage_weight"] * gap
                            + policy["uncertainty_weight"] * uncertainty
                            + policy["focus_bonus"] * (region_id in policy["focus_regions"]))
                regions[region_id] = {"factor": factor, "value": json.loads(value),
                                      "total_candidates": len(members), "remaining_candidates": remaining,
                                      "support_ids": [row["evidence_id"] for row in rows],
                                      "observed_count": len(rows), "estimated_bpb": mean,
                                      "coverage_gap": gap, "uncertainty": uncertainty,
                                      "quality": quality, "priority": priority}
        queue = []
        for candidate_id in self.candidates:
            if candidate_id in self.seen:
                continue
            memberships = [region_id for region_id, region in regions.items()
                           if candidate_id in region["remaining_candidates"]]
            queue.append({"candidate_id": candidate_id, "region_ids": memberships,
                          "priority": _mean(regions[key]["priority"] for key in memberships),
                          "coverage_gap": _mean(regions[key]["coverage_gap"] for key in memberships),
                          "uncertainty": _mean(regions[key]["uncertainty"] for key in memberships)})
        queue.sort(key=lambda item: (-item["priority"], item["candidate_id"]))
        archive = []
        for row in self._valid():
            dominated = any(other["val_bpb"] <= row["val_bpb"] and other["charge_seconds"] <= row["charge_seconds"]
                            and (other["val_bpb"] < row["val_bpb"] or other["charge_seconds"] < row["charge_seconds"])
                            for other in self._valid())
            if not dominated:
                archive.append({key: row[key] for key in ("candidate_id", "evidence_id", "val_bpb", "charge_seconds")})
        self.frontier = {"policy": deepcopy(self.frontier_policy), "regions": regions,
                         "unqueried_queue": queue, "observed_pareto_archive": archive}

    def _evolve_frontier(self) -> None:
        before = deepcopy(self.frontier_policy)
        valid = self._valid()
        previous = [row for row in self.ledger[:-2] if row["feasible"]]
        improvement = bool(previous) and min(row["val_bpb"] for row in valid) < min(row["val_bpb"] for row in previous)
        regions = sorted(self.frontier["regions"].items(), key=lambda item: (-item[1]["quality"], item[0]))
        focus = [key for key, region in regions if region["observed_count"] and region["quality"] > 0][:2]
        self.frontier_policy = {
            "quality_weight": 1.2 if improvement else 0.5,
            "coverage_weight": 0.8 if improvement else 1.5,
            "uncertainty_weight": 0.6 if improvement else 1.0,
            "focus_regions": focus, "focus_bonus": 0.2,
        }
        self._refresh_frontier()
        self._record_edit("frontier", before, self.frontier_policy,
                          "Prioritize promising observed regions after progress; increase coverage and uncertainty after a plateau. Keep every unqueried candidate.",
                          [{"operation": "reallocate_frontier_channels", "recent_improvement": improvement,
                            "focus_regions_added": sorted(set(focus) - set(before["focus_regions"])),
                            "focus_regions_removed": sorted(set(before["focus_regions"]) - set(focus))}])

    def observe(self, observation: dict) -> None:
        candidate_id = observation.get("candidate_id", observation.get("id"))
        status = observation.get("status", "valid_new")
        if status in ("invalid", "duplicate"):
            return
        if candidate_id not in self.candidates:
            raise ValueError("Observation is not a registered public candidate")
        feasible = observation.get("feasible", status in ("valid_new", "valid", "ok", "completed", "success"))
        if not isinstance(feasible, bool):
            raise ValueError("feasible must be a boolean")
        score = observation.get("val_bpb")
        if feasible:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
                raise ValueError("Feasible observations need finite nonnegative val_bpb")
            score = float(score)
        else:
            score = None
        charge = observation.get("charge_seconds", 0.0)
        if isinstance(charge, bool) or not isinstance(charge, (int, float)) or not math.isfinite(charge) or charge < 0:
            raise ValueError("charge_seconds must be finite and nonnegative")
        if candidate_id in self.seen:
            existing = self.seen[candidate_id]
            if existing["feasible"] != feasible or existing["val_bpb"] != score:
                raise ValueError("An observed candidate cannot be overwritten with a different result")
            return
        attempt = observation.get("attempt", observation.get("step", len(self.ledger) + 1))
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        row = {"candidate_id": candidate_id, "evidence_id": f"q{attempt}:{candidate_id}",
               "attempt": attempt, "val_bpb": score, "feasible": feasible,
               "charge_seconds": float(charge), "phase": observation.get("phase", "search")}
        self.ledger.append(row)
        self.seen[candidate_id] = row
        self._refresh_memory()
        self._refresh_frontier()
        count = len(self.ledger)
        if count >= 4 and count % 2 == 0:
            if self.adapt["memory"]:
                self._evolve_memory()
            if self.adapt["structure"] and count in (4, 8):
                self._evolve_structure()
            if self.adapt["frontier"]:
                self._evolve_frontier()

    def _retrieve(self, candidate_id: str, neighbors: int = 3) -> tuple[float, list[str]]:
        nearest = sorted(self._valid(), key=lambda row: (self._distance(candidate_id, row["candidate_id"]), row["attempt"]))[:neighbors]
        if not nearest:
            return 8.0, []
        weights = [1.0 / (1 + self._distance(candidate_id, row["candidate_id"])) for row in nearest]
        prediction = sum(weight * row["val_bpb"] for weight, row in zip(weights, nearest)) / sum(weights)
        evidence = [row["evidence_id"] for row in nearest]
        if self.memory_policy["group_by"]:
            for cluster in self.clusters.values():
                if all(self.candidates[candidate_id].get(key) == value for key, value in cluster["key"].items()):
                    weight = cluster["support_count"] / (cluster["support_count"] + 2)
                    prediction = (1 - weight) * prediction + weight * cluster["mean_bpb"]
                    # The mean depends on every supporting observation; the
                    # representative subset is only a display summary.
                    evidence += cluster["support_ids"]
                    break
        return prediction, list(dict.fromkeys(evidence))

    def _contrast_prediction(self, candidate_id: str, default: float, blend: float) -> tuple[float, list[str]]:
        if not self._valid():
            return default, []
        parent = min(self._valid(), key=lambda row: (self._distance(candidate_id, row["candidate_id"]), row["val_bpb"]))
        changed = [key for key in self.factors
                   if self.candidates[candidate_id].get(key) != self.candidates[parent["candidate_id"]].get(key)]
        predicted = parent["val_bpb"]
        supports = [parent["evidence_id"]]
        for factor in changed:
            source = _json(self.candidates[parent["candidate_id"]].get(factor))
            target = _json(self.candidates[candidate_id].get(factor))
            deltas = []
            for edge in self.contrast_edges:
                if edge["factor"] != factor:
                    continue
                if (edge["left_value"], edge["right_value"]) == (source, target):
                    deltas.append(edge["delta_bpb"])
                    supports += edge["support_ids"]
                elif (edge["left_value"], edge["right_value"]) == (target, source):
                    deltas.append(-edge["delta_bpb"])
                    supports += edge["support_ids"]
            if not deltas:
                return default, []
            predicted += _mean(deltas)
        return (1 - blend) * default + blend * max(0.0, predicted), list(dict.fromkeys(supports))

    def _comparison_opportunity(self, candidate_id: str) -> float:
        opportunities = []
        for row in self._valid():
            if self._distance(candidate_id, row["candidate_id"]) == 1:
                factor = next(key for key in self.factors
                              if self.candidates[candidate_id].get(key) != self.candidates[row["candidate_id"]].get(key))
                count = sum(edge["factor"] == factor for edge in self.contrast_edges)
                opportunities.append(1.0 / math.sqrt(1 + count))
        return max(opportunities, default=0.0)

    def _execute(self, graph: dict, remaining: list[str], rng: random.Random) -> tuple[str, dict]:
        global_mean, scale = self._global()
        queue = {item["candidate_id"]: item for item in self.frontier["unqueried_queue"]}
        context = {candidate_id: {"predicted_bpb": global_mean, "evidence_ids": [], "information_bonus": 0.0}
                   for candidate_id in remaining}
        node_id, traversed, route = graph["entry"], [], None
        coverage_guard = False
        while True:
            if node_id in traversed:
                raise ValueError("Workflow must be acyclic")
            traversed.append(node_id)
            node = graph["nodes"][node_id]
            kind, params = node["type"], node["params"]
            route = None
            if kind == "public_pool":
                pass
            elif kind == "evidence_retrieval":
                for candidate_id, item in context.items():
                    item["predicted_bpb"], item["evidence_ids"] = self._retrieve(candidate_id, params["neighbors"])
            elif kind == "matched_contrast":
                for candidate_id, item in context.items():
                    item["predicted_bpb"], support = self._contrast_prediction(candidate_id, item["predicted_bpb"], params["blend"])
                    item["evidence_ids"] += support
            elif kind == "contrast_target":
                for candidate_id, item in context.items():
                    item["information_bonus"] = self._comparison_opportunity(candidate_id)
            elif kind == "coverage_router":
                gap = max((queue[candidate_id]["coverage_gap"] for candidate_id in remaining), default=0.0)
                route = "coverage" if gap >= params["gap_threshold"] and (self.proposal_count + 1) % params["period"] == 0 else "evidence"
            elif kind == "coverage_proposal":
                for candidate_id, item in context.items():
                    item["information_bonus"] = queue[candidate_id]["coverage_gap"] + self._comparison_opportunity(candidate_id)
            elif kind == "frontier_lookup":
                for candidate_id, item in context.items():
                    item["frontier_priority"] = queue[candidate_id]["priority"]
                    item["coverage_gap"] = queue[candidate_id]["coverage_gap"]
            elif kind == "rank":
                for item in context.values():
                    item["utility"] = ((global_mean - item["predicted_bpb"]) / scale
                                       + params["frontier_weight"] * item["frontier_priority"]
                                       + params["information_weight"] * item["information_bonus"])
            elif kind == "select":
                coverage_guard = (self.proposal_count + 1) % params["coverage_every"] == 0
                if coverage_guard:
                    best = max(context[candidate_id]["coverage_gap"] + self._comparison_opportunity(candidate_id)
                               for candidate_id in remaining)
                    tied = [candidate_id for candidate_id in remaining
                            if abs(context[candidate_id]["coverage_gap"] + self._comparison_opportunity(candidate_id) - best) < 1e-12]
                else:
                    best = max(item["utility"] for item in context.values())
                    tied = [candidate_id for candidate_id in remaining if abs(context[candidate_id]["utility"] - best) < 1e-12]
                selected = rng.choice(tied)
                item = context[selected]
                return selected, {"traversed_nodes": traversed, "coverage_guard": coverage_guard,
                                  "predicted_bpb": item["predicted_bpb"], "frontier_priority": item["frontier_priority"],
                                  "information_bonus": item["information_bonus"], "utility": item["utility"],
                                  "retrieved_evidence_ids": list(dict.fromkeys(item["evidence_ids"])),
                                  # Global normalization and frontier coverage
                                  # also depend on the complete observed ledger.
                                  "evidence_ids": [row["evidence_id"] for row in self.ledger]}
            else:
                raise ValueError(f"Unsupported workflow node type: {kind}")
            edges = [edge for edge in graph["edges"] if edge["from"] == node_id
                     and ("when" not in edge or edge["when"] == route)]
            if len(edges) != 1:
                raise ValueError("Workflow nodes must choose exactly one outgoing edge")
            node_id = edges[0]["to"]

    def propose(self, observations: list[dict] | None, remaining_ids: list[str],
                rng: random.Random | None = None) -> tuple[str | None, dict]:
        for row in observations or []:
            self.observe(row)
        if len(remaining_ids) != len(set(remaining_ids)):
            raise ValueError("Remaining IDs must be unique")
        if any(candidate_id not in self.candidates or candidate_id in self.seen for candidate_id in remaining_ids):
            raise ValueError("Remaining IDs must be registered and unobserved")
        if not remaining_ids:
            return None, {"reason": "space_exhausted"}
        rng = self.rng if rng is None else rng
        if self.pending_structure_comparison is not None:
            comparison = self.pending_structure_comparison
            before_rng, after_rng = random.Random(), random.Random()
            before_rng.setstate(rng.getstate())
            after_rng.setstate(rng.getstate())
            old_id, old_trace = self._execute(comparison["before"], remaining_ids, before_rng)
            new_id, new_trace = self._execute(comparison["after"], remaining_ids, after_rng)
            self.events.append({"type": "structure_counterfactual", "structure_version": comparison["version"],
                                "query_count": len(self.ledger), "proposal_number": self.proposal_count + 1,
                                "before_candidate": old_id, "after_candidate": new_id, "candidate_changed": old_id != new_id,
                                "before_traversed_nodes": old_trace["traversed_nodes"],
                                "after_traversed_nodes": new_trace["traversed_nodes"],
                                "evidence_ids": [row["evidence_id"] for row in self.ledger],
                                "oracle_calls": 0})
            self.pending_structure_comparison = None
        selected, trace = self._execute(self.graph, remaining_ids, rng)
        self.proposal_count += 1
        metadata = {"controller_kind": "memory_frontier_harness", "proposal_number": self.proposal_count,
                    "candidate_origin": "harness_proposal", "candidate_id": selected,
                    "component_versions": dict(self.component_versions), "graph_sha256": _hash(self.graph),
                    "memory_policy_sha256": _hash(self.memory_policy), "frontier_policy_sha256": _hash(self.frontier_policy),
                    "remaining_candidates": len(remaining_ids), **trace}
        self.traces.append(deepcopy(metadata))
        return selected, metadata

    def summary(self) -> dict:
        state = {"raw_observations": deepcopy(self.ledger),
                 "memory": {"policy": deepcopy(self.memory_policy), "clusters": deepcopy(self.clusters),
                            "hypotheses": deepcopy(self.hypotheses), "matched_contrast_edges": deepcopy(self.contrast_edges)},
                 "graph": deepcopy(self.graph), "frontier": deepcopy(self.frontier)}
        comparisons = [event for event in self.events if event["type"] == "structure_counterfactual"]
        return {"controller_kind": "memory_frontier_harness", "interventions": dict(self.adapt),
                "component_versions": dict(self.component_versions), "edit_counts": dict(self.component_versions),
                "events": deepcopy(self.events), "versions": deepcopy(self.versions), "final_state": state,
                "state_sha256": _hash(state), "candidate_manifest_sha256": _hash(self.candidates),
                "independent_factors": list(self.factors), "evidence_queries": len(self.ledger),
                "proposal_count": self.proposal_count,
                "node_visit_counts": dict(Counter(node for trace in self.traces for node in trace["traversed_nodes"])),
                "structure_counterfactuals": len(comparisons),
                "structure_counterfactual_candidate_changes": sum(event["candidate_changed"] for event in comparisons),
                "claim_scope": "bounded evidence-conditioned harness edits; no independent validation or recursive meta-improvement claim"}
