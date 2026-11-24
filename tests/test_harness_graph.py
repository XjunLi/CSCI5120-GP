"""Software fixtures for memory/workflow/frontier intervention boundaries."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from golfbench.harness_graph import MemoryFrontierHarness


def designs(standard=True):
    result = []
    for index, (depth, width, tied, lr) in enumerate(itertools.product([1, 2, 3], [32, 48], [False, True], [0.7, 1.3])):
        config = {"num_layers": depth, "model_dim": width, "tied_embeddings": tied,
                  "lr_scale": lr, "num_heads": width // 8, "num_kv_heads": width // 8} if standard else {
                      "depth": depth, "width": width, "tied": tied, "lr": lr}
        result.append({"candidate_id": f"c{index:02d}", "config": config, "hidden_score": object()})
    return result


def row(candidate_id, attempt, score=None):
    # Synthetic rewards are test fixtures and never enter reported experiments.
    return {"candidate_id": candidate_id, "attempt": attempt, "phase": "warm_start" if attempt <= 4 else "search",
            "status": "valid_new", "feasible": True, "val_bpb": 4 + int(candidate_id[1:]) / 50 if score is None else score,
            "charge_seconds": 1.0, "private_test_bpb": object()}


def warm(harness):
    # Contains observed one-factor contrasts and two values of several factors.
    for attempt, candidate_id in enumerate(("c00", "c01", "c04", "c12"), 1):
        harness.observe(row(candidate_id, attempt))


class HarnessGraphTests(unittest.TestCase):
    def test_all_three_persistent_components_can_change_after_common_warm_start(self):
        harness = MemoryFrontierHarness(designs(), 0)
        warm(harness)
        summary = harness.summary()
        self.assertTrue(all(version >= 1 for version in summary["component_versions"].values()))
        self.assertNotEqual(summary["final_state"]["memory"]["policy"]["group_by"], [])
        self.assertIn("contrast", summary["final_state"]["graph"]["nodes"])
        self.assertEqual(len(summary["final_state"]["raw_observations"]), 4)
        remaining = [item["candidate_id"] for item in designs() if item["candidate_id"] not in harness.seen]
        chosen, trace = harness.propose(None, remaining, random.Random(3))
        self.assertIn(chosen, remaining)
        self.assertIn("contrast", trace["traversed_nodes"])
        self.assertEqual(harness.summary()["structure_counterfactuals"], 1)
        event = harness.events[-1]
        self.assertEqual(event["type"], "structure_counterfactual")
        self.assertEqual(event["oracle_calls"], 0)
        self.assertEqual(event["after_candidate"], chosen)

    def test_frozen_components_keep_raw_feedback_and_refresh_fixed_statistics(self):
        harness = MemoryFrontierHarness(designs(), 0, False, False, False)
        initial = harness.summary()
        warm(harness)
        summary = harness.summary()
        self.assertEqual(summary["edit_counts"], {"memory": 0, "structure": 0, "frontier": 0})
        self.assertEqual(summary["final_state"]["graph"], initial["final_state"]["graph"])
        self.assertEqual(summary["final_state"]["memory"]["policy"], initial["final_state"]["memory"]["policy"])
        clusters = summary["final_state"]["memory"]["clusters"]
        self.assertEqual(len(clusters), 1)
        self.assertEqual(next(iter(clusters.values()))["support_count"], 4)
        self.assertEqual(len(summary["final_state"]["frontier"]["unqueried_queue"]), 20)
        self.assertTrue(summary["final_state"]["frontier"]["observed_pareto_archive"])
        remaining = [item["candidate_id"] for item in designs() if item["candidate_id"] not in harness.seen]
        _, trace = harness.propose(None, remaining, random.Random(1))
        self.assertTrue(trace["evidence_ids"])

    def test_each_intervention_operates_when_the_other_two_are_frozen(self):
        for active in ("memory", "structure", "frontier"):
            with self.subTest(active=active):
                flags = {"adapt_" + component: component == active for component in ("memory", "structure", "frontier")}
                harness = MemoryFrontierHarness(designs(), 2, **flags)
                warm(harness)
                counts = harness.summary()["edit_counts"]
                self.assertGreater(counts[active], 0)
                self.assertTrue(all(counts[component] == 0 for component in counts if component != active))
                self.assertEqual(len(harness.ledger), 4)

    def test_typed_workflow_has_at_most_two_edits_and_traversal_is_real(self):
        harness = MemoryFrontierHarness(designs(), 4)
        warm(harness)
        rng = random.Random(4)
        for attempt in range(5, 13):
            remaining = [item["candidate_id"] for item in designs() if item["candidate_id"] not in harness.seen]
            chosen, trace = harness.propose(None, remaining, rng)
            self.assertEqual(trace["traversed_nodes"][0], "entry")
            self.assertEqual(trace["traversed_nodes"][-1], "select")
            if attempt >= 9 and "route" in harness.graph["nodes"]:
                self.assertIn("route", trace["traversed_nodes"])
            harness.observe(row(chosen, attempt))
        edits = [event for event in harness.events if event["type"] == "harness_component_edit" and event["component"] == "structure"]
        self.assertEqual(edits[0]["query_count"], 4)
        self.assertTrue(all(event["query_count"] in (4, 8) for event in edits))
        self.assertLessEqual(len(edits), 2)
        for event in edits:
            self.assertNotEqual(event["before_hash"], event["after_hash"])
            self.assertTrue(event["operations"][0]["added_nodes"] or event["operations"][0]["changed_nodes"])
        self.assertEqual(harness.summary()["structure_counterfactuals"], len(edits))
        self.assertGreater(harness.summary()["node_visit_counts"]["contrast"], 0)

    def test_same_public_space_and_query_count_choose_different_structures_from_feedback(self):
        plateau = MemoryFrontierHarness(designs(), 0, False, True, False)
        progress = MemoryFrontierHarness(designs(), 0, False, True, False)
        queries = ("c00", "c01", "c04", "c05", "c08", "c09", "c12", "c13")
        for attempt, candidate_id in enumerate(queries, 1):
            plateau.observe(row(candidate_id, attempt, 4 + attempt / 10))
            progress.observe(row(candidate_id, attempt, 8 - attempt / 10))
        plateau_decision = [event for event in plateau.events if event["type"] == "structure_decision"][-1]
        progress_decision = [event for event in progress.events if event["type"] == "structure_decision"][-1]
        self.assertEqual(plateau_decision["query_count"], progress_decision["query_count"])
        self.assertEqual(plateau_decision["action"], "insert_coverage_route")
        self.assertEqual(progress_decision["action"], "retain_evidence_path")
        self.assertTrue(plateau_decision["signals"]["recent_plateau"])
        self.assertTrue(progress_decision["signals"]["recent_improvement"])
        self.assertIn("route", plateau.graph["nodes"])
        self.assertNotIn("route", progress.graph["nodes"])

    def test_failed_queries_are_included_in_the_recent_progress_window(self):
        harness = MemoryFrontierHarness(designs(), 0, False, False, True)
        for attempt, candidate_id in enumerate(("c00", "c01", "c04", "c05"), 1):
            harness.observe(row(candidate_id, attempt, 8 - attempt))
        self.assertEqual(harness.frontier_policy["quality_weight"], 1.2)
        for attempt, candidate_id in enumerate(("c08", "c09"), 5):
            failed = row(candidate_id, attempt)
            failed.update(status="infeasible_or_failed", feasible=False, val_bpb=None)
            harness.observe(failed)
        self.assertEqual(harness.frontier_policy["quality_weight"], 0.5)
        self.assertEqual(harness.frontier_policy["coverage_weight"], 1.5)
        event = [event for event in harness.events if event.get("component") == "frontier"][-1]
        self.assertFalse(event["operations"][0]["recent_improvement"])

    def test_unqueried_scores_and_hidden_result_fields_cannot_change_the_policy(self):
        first_catalog, second_catalog = designs(), designs()
        for index, item in enumerate(second_catalog):
            item["hidden_score"] = 10000 - index * 500
        first, second = MemoryFrontierHarness(first_catalog, 8), MemoryFrontierHarness(second_catalog, 8)
        warm(first)
        warm(second)
        remaining = [item["candidate_id"] for item in first_catalog if item["candidate_id"] not in first.seen]
        a = first.propose(None, remaining, random.Random(8))
        b = second.propose(None, remaining, random.Random(8))
        self.assertEqual(a, b)
        self.assertEqual(first.summary(), second.summary())
        serialized = json.dumps(first.summary(), allow_nan=False)
        self.assertNotIn("hidden_score", serialized)
        self.assertNotIn("private_test_bpb", serialized)

    def test_memory_support_and_edit_evidence_never_include_future_queries(self):
        harness = MemoryFrontierHarness(designs(), 9)
        warm(harness)
        for attempt, candidate_id in enumerate(("c08", "c09", "c16", "c17"), 5):
            harness.observe(row(candidate_id, attempt))
        summary = harness.summary()
        known = {item["evidence_id"] for item in harness.ledger}
        support = [evidence for cluster in harness.clusters.values() for evidence in cluster["support_ids"]]
        self.assertEqual(set(support), known)
        self.assertEqual(len(support), len(known))
        for event in summary["events"]:
            for evidence in event["evidence_ids"]:
                self.assertLessEqual(int(evidence.split(":", 1)[0][1:]), event["query_count"])
        for hypothesis in summary["final_state"]["memory"]["hypotheses"].values():
            self.assertLessEqual(set(hypothesis["support_ids"]), known)
            self.assertLessEqual(set(hypothesis["counterexample_ids"]), known)
            self.assertEqual(hypothesis["inference_type"], "association")

    def test_cluster_mean_trace_lists_all_support_not_only_representatives(self):
        harness = MemoryFrontierHarness(designs(), 0, False, False, False)
        harness.memory_policy["group_by"] = ["model_dim"]
        for attempt, candidate_id in enumerate(("c00", "c01", "c02", "c03", "c08", "c09"), 1):
            harness.observe(row(candidate_id, attempt))
        _, support = harness._retrieve("c10")
        known = {item["evidence_id"] for item in harness.ledger}
        self.assertEqual(set(support), known)
        cluster = next(iter(harness.clusters.values()))
        self.assertGreater(len(cluster["support_ids"]), len(cluster["representative_ids"]))
        remaining = [item["candidate_id"] for item in designs() if item["candidate_id"] not in harness.seen]
        _, trace = harness.propose(None, remaining, random.Random(0))
        self.assertEqual(set(trace["evidence_ids"]), known)

    def test_queue_retains_every_unqueried_candidate_and_coverage_guard_is_shared(self):
        for adaptive in (True, False):
            harness = MemoryFrontierHarness(designs(), 3, adaptive, adaptive, adaptive)
            warm(harness)
            rng = random.Random(3)
            for attempt in range(5, 13):
                remaining = [item["candidate_id"] for item in designs() if item["candidate_id"] not in harness.seen]
                self.assertEqual({item["candidate_id"] for item in harness.frontier["unqueried_queue"]}, set(remaining))
                selected, trace = harness.propose(None, remaining, rng)
                self.assertEqual(trace["coverage_guard"], (attempt - 4) % 4 == 0)
                harness.observe(row(selected, attempt))

    def test_idempotent_history_sync_and_read_only_summary(self):
        harness = MemoryFrontierHarness(designs(), 6)
        warm(harness)
        before = harness.summary()
        for item in list(harness.ledger):
            harness.observe(item)
        self.assertEqual(before, harness.summary())
        before["final_state"]["raw_observations"][0]["val_bpb"] = -100
        self.assertGreater(harness.ledger[0]["val_bpb"], 0)
        with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
            harness.observe(row("c00", 1, 999))
        with self.assertRaisesRegex(ValueError, "registered and unobserved"):
            harness.propose(None, ["c00"], random.Random(0))

    def test_independent_factors_exclude_derived_heads_and_generic_configs_work(self):
        standard = MemoryFrontierHarness(designs(), 0)
        self.assertNotIn("num_heads", standard.factors)
        self.assertNotIn("num_kv_heads", standard.factors)
        self.assertEqual(standard._distance("c00", "c04"), 1)
        generic = MemoryFrontierHarness(designs(standard=False), 0)
        self.assertEqual(set(generic.factors), {"depth", "width", "tied", "lr"})
        warm(generic)
        self.assertGreater(generic.summary()["edit_counts"]["memory"], 0)


if __name__ == "__main__":
    unittest.main()
