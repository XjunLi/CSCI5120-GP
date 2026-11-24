"""Protocol tests.  Numerical rewards here are software fixtures, not results."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from golfbench.search import METHODS, _LocalController, run_search


def designs(count=24, quote=600.0):
    return [{"candidate_id": f"c{i:02d}",
             "config": {"depth": i % 3, "width": (i // 3) % 2, "lr": (i // 6) % 4},
             "charge_seconds": quote}
            for i in range(count)]


class Evaluator:
    def __init__(self, candidates, failures=()):
        self.ids = [item["candidate_id"] for item in candidates]
        self.quotes = {item["candidate_id"]: item["charge_seconds"] for item in candidates}
        self.calls = []
        self.failures = set(failures)

    def __call__(self, candidate_id):
        self.calls.append(candidate_id)
        index = self.ids.index(candidate_id)
        return {"candidate_id": candidate_id, "val_bpb": 4.0 + ((index * 11) % 23) / 20,
                "charge_seconds": self.quotes[candidate_id], "train_seconds": 0.02,
                "status": "failed" if candidate_id in self.failures else "ok",
                "feasible": candidate_id not in self.failures,
                "test_bpb": object(), "private_result_payload": object()}


class SearchProtocolTests(unittest.TestCase):
    def test_all_methods_share_initial_candidates_and_are_reproducible(self):
        initializations = []
        for method in METHODS:
            with self.subTest(method=method):
                candidates = designs()
                first_evaluator, second_evaluator = Evaluator(candidates), Evaluator(candidates)
                first = run_search(candidates, first_evaluator, method, seed=7)
                second = run_search(candidates, second_evaluator, method, seed=7)
                self.assertEqual(first, second)
                self.assertEqual(first_evaluator.calls, second_evaluator.calls)
                self.assertEqual(len(first_evaluator.calls), 12)
                self.assertEqual(first["attempts"], 12)
                self.assertEqual(first["charged_seconds"], 7200)
                initializations.append(first["initial_candidate_ids"])
        self.assertTrue(all(initial == initializations[0] for initial in initializations))

    def test_cost_budget_stops_before_unaffordable_query(self):
        for method in METHODS:
            with self.subTest(method=method):
                candidates = designs()
                evaluator = Evaluator(candidates)
                run = run_search(candidates, evaluator, method, seed=3,
                                 budget=18, cost_budget_seconds=3600)
                self.assertEqual(len(evaluator.calls), 6)
                self.assertEqual(run["new_evaluations"], 6)
                self.assertEqual(run["charged_seconds"], 3600)
                self.assertEqual(run["stop_reason"], "cost_budget")
                self.assertNotIn(run["stop_detail"]["candidate_id"], evaluator.calls)
        candidates = designs()
        evaluator = Evaluator(candidates)
        run = run_search(candidates, evaluator, "random", seed=3, cost_budget_seconds=0)
        self.assertEqual(evaluator.calls, [])
        self.assertIsNone(run["winner"])

    def test_public_cost_quote_is_mandatory_and_cannot_change(self):
        candidates = designs()
        missing = deepcopy(candidates)
        del missing[-1]["charge_seconds"]
        evaluator = Evaluator(candidates)
        with self.assertRaisesRegex(ValueError, "public charge_seconds"):
            run_search(missing, evaluator, "random", 0, cost_budget_seconds=1000)
        self.assertEqual(evaluator.calls, [])

        def bad_cost(candidate_id):
            result = evaluator(candidate_id)
            result["charge_seconds"] += 1
            return result

        with self.assertRaisesRegex(ValueError, "disagrees"):
            run_search(candidates, bad_cost, "random", 0, cost_budget_seconds=1000)
        self.assertEqual(len(evaluator.calls), 1)

    def test_evaluator_is_queried_only_for_submitted_ids_and_hidden_data_is_discarded(self):
        candidates = designs()
        for item in candidates:
            item["test_bpb"] = object()
        evaluator = Evaluator(candidates)
        run = run_search(candidates, evaluator, "weight-adaptive", seed=2)
        self.assertEqual(evaluator.calls, [row["candidate_id"] for row in run["trajectory"]])
        serialized = json.dumps(run, allow_nan=False)
        self.assertNotIn("test_bpb", serialized)
        self.assertNotIn("private_result_payload", serialized)
        winner = run["winner"]
        feasible = [row for row in run["trajectory"] if row["feasible"]]
        self.assertEqual(winner["val_bpb"], min(row["val_bpb"] for row in feasible))
        self.assertIn(winner["candidate_id"], evaluator.calls)

    def test_duplicates_and_invalid_proposals_consume_attempts(self):
        candidates = designs()
        evaluator = Evaluator(candidates)
        warm_id = run_search(candidates, Evaluator(candidates), "random", 0,
                             budget=1, warm_start=1)["initial_candidate_ids"][0]
        new_id = next(item["candidate_id"] for item in candidates if item["candidate_id"] != warm_id)
        answers = iter([(warm_id, {}), ("not-a-candidate", {}), (new_id, {})])
        with patch.object(_LocalController, "propose", side_effect=lambda *_: next(answers)):
            run = run_search(candidates, evaluator, "random", 0, budget=4, warm_start=1)
        self.assertEqual(run["attempts"], 4)
        self.assertEqual(run["duplicate_attempts"], 1)
        self.assertEqual(run["invalid_attempts"], 1)
        self.assertEqual(evaluator.calls, [warm_id, new_id])
        self.assertEqual(run["charged_seconds"], 1200)
        self.assertEqual(run["trajectory"][1]["charge_seconds"], 0)

    def test_failed_evaluations_are_charged_and_cannot_win(self):
        candidates = designs()
        evaluator = Evaluator(candidates, failures=[item["candidate_id"] for item in candidates])
        run = run_search(candidates, evaluator, "random", 0, budget=7)
        self.assertEqual(run["charged_seconds"], 4200)
        self.assertEqual(run["failed_or_infeasible_evaluations"], 7)
        self.assertIsNone(run["winner"])

    def test_harness_controls_pay_equal_trial_slots_but_only_adaptive_retains_update(self):
        for method in ("harness-adaptive", "harness-frozen"):
            controller = _LocalController(designs(), method, 9)
            available = list(controller.ids)
            for attempt in range(1, 5):
                candidate_id, metadata = controller.propose(available, attempt, 5 - attempt)
                available.remove(candidate_id)
                row = {"attempt": attempt, "phase": "search", "candidate_id": candidate_id,
                       "status": "valid_new", "val_bpb": 2.0 if metadata["harness_role"] == "challenger" else 5.0,
                       "charge_seconds": 600.0}
                controller.observe(row, metadata)
            summary = controller.summary()
            event = next(event for event in summary["events"] if event["type"] == "harness_trial_completed")
            self.assertEqual(event["role_attempts"], {"incumbent": 2, "challenger": 2})
            self.assertEqual(event["role_charge_seconds"], {"incumbent": 1200.0, "challenger": 1200.0})
            self.assertTrue(event["would_promote"])
            self.assertEqual(summary["promotions"], int(method == "harness-adaptive"))
            self.assertEqual(summary["final_program_version"], int(method == "harness-adaptive"))

    def test_weight_updates_are_real_and_frozen_control_keeps_identical_weights(self):
        candidates = designs()
        adaptive = run_search(candidates, Evaluator(candidates), "weight-adaptive", 17,
                              budget=18, warm_start=4)
        frozen = run_search(candidates, Evaluator(candidates), "weight-frozen", 17,
                            budget=18, warm_start=4)
        self.assertEqual(adaptive["gradient_updates"], 14)
        self.assertNotEqual(adaptive["initial_weight_sha256"], adaptive["final_weight_sha256"])
        self.assertGreater(adaptive["final_weight_norm"], 0)
        self.assertEqual(frozen["gradient_updates"], 0)
        self.assertEqual(frozen["initial_weight_sha256"], frozen["final_weight_sha256"])
        self.assertTrue(all(event["update_norm"] == 0 for event in frozen["events"]))
        self.assertTrue(any(event["gradient_norm"] > 0 for event in frozen["events"]))

    def test_multiagent_control_preserves_population_and_routes(self):
        candidates = designs()
        frozen = run_search(candidates, Evaluator(candidates), "multiagent-frozen", 4,
                            budget=24, warm_start=4)
        adaptive = run_search(candidates, Evaluator(candidates), "multiagent-adaptive", 4,
                              budget=24, warm_start=4)
        self.assertEqual(frozen["descendants"], 0)
        self.assertEqual([agent["id"] for agent in frozen["final_agents"]], [0, 1, 2])
        self.assertEqual(frozen["final_routes"], [1, 2, 0])
        self.assertTrue(all(not event["replaced"] for event in frozen["events"]))
        self.assertGreater(adaptive["descendants"], 0)
        self.assertTrue(all(item["parent_id"] is not None for item in adaptive["agent_lineage"][3:]))

    def test_space_exhaustion_and_earliest_winner_tie(self):
        candidates = designs(count=3)

        def tied(candidate_id):
            return {"val_bpb": 4.0, "charge_seconds": 600.0, "feasible": True, "status": "ok"}

        run = run_search(candidates, tied, "random", 5, budget=10, warm_start=1)
        self.assertEqual(run["attempts"], 3)
        self.assertEqual(run["stop_reason"], "space_exhausted")
        self.assertEqual(run["winner"]["candidate_id"], run["initial_candidate_ids"][0])


if __name__ == "__main__":
    unittest.main()
