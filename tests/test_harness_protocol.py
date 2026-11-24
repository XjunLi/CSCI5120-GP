"""Independent search-boundary checks; every score here is a software fixture."""
import itertools
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from golfbench.search import METHODS, run_search


HARNESS_METHODS = tuple(method for method in METHODS if method.startswith("memory-frontier-"))


def public_candidates():
    return [
        {"candidate_id": f"candidate-{index:02d}",
         "config": {"num_layers": depth, "model_dim": width,
                    "tied_embeddings": tied, "lr_scale": lr},
         "charge_seconds": 3.0 + index / 4}
        for index, (depth, width, tied, lr) in enumerate(
            itertools.product((2, 4, 6), (32, 48), (False, True), (0.7, 1.3)))
    ]


def fixture_scores(candidates):
    return {item["candidate_id"]: 2.0 + ((index * 11) % 29) / 17
            for index, item in enumerate(candidates)}


class Poison:
    """A hidden payload must never be copied or interpreted by a controller."""

    def __deepcopy__(self, memo):
        raise AssertionError("hidden payload was copied")

    def __float__(self):
        raise AssertionError("hidden payload was interpreted as a score")

    def __bool__(self):
        raise AssertionError("hidden payload was inspected")


class RecordingOracle:
    def __init__(self, candidates, scores, *, all_failed=False, poison=False):
        self.quotes = {item["candidate_id"]: item["charge_seconds"] for item in candidates}
        self.scores = scores
        self.all_failed = all_failed
        self.poison = poison
        self.calls = []

    def __call__(self, candidate_id):
        if candidate_id in self.calls:
            raise AssertionError("a cached candidate was evaluated twice")
        self.calls.append(candidate_id)
        result = {"candidate_id": candidate_id, "status": "failed" if self.all_failed else "ok",
                  "feasible": not self.all_failed,
                  "val_bpb": float("nan") if self.all_failed else self.scores[candidate_id],
                  "charge_seconds": self.quotes[candidate_id],
                  "train_seconds": self.quotes[candidate_id]}
        if self.poison:
            result.update(test_bpb=Poison(), private_result_payload=Poison())
        return result


class HarnessProtocolTests(unittest.TestCase):
    def test_permuting_only_unqueried_scores_preserves_the_complete_run(self):
        for method in HARNESS_METHODS:
            with self.subTest(method=method):
                candidates = public_candidates()
                scores = fixture_scores(candidates)
                first_oracle = RecordingOracle(candidates, scores)
                first = run_search(candidates, first_oracle, method, seed=17, budget=12)
                unseen = [item["candidate_id"] for item in candidates
                          if item["candidate_id"] not in first_oracle.calls]
                self.assertGreater(len(unseen), 1)
                permuted = dict(scores)
                for index, candidate_id in enumerate(unseen):
                    permuted[candidate_id] = scores[unseen[(index + 1) % len(unseen)]]
                self.assertTrue(any(permuted[candidate_id] != scores[candidate_id]
                                    for candidate_id in unseen))
                second_oracle = RecordingOracle(candidates, permuted)
                second = run_search(candidates, second_oracle, method, seed=17, budget=12)
                self.assertEqual(first_oracle.calls, second_oracle.calls)
                self.assertEqual(first, second)

    def test_counterfactual_graph_execution_adds_no_oracle_calls(self):
        for method in HARNESS_METHODS:
            with self.subTest(method=method):
                candidates = public_candidates()
                oracle = RecordingOracle(candidates, fixture_scores(candidates))
                run = run_search(candidates, oracle, method, seed=7, budget=12)
                submitted = [row["candidate_id"] for row in run["trajectory"]
                             if row["status"] in ("valid_new", "infeasible_or_failed")]
                self.assertEqual(oracle.calls, submitted)
                self.assertEqual(run["new_evaluations"], len(oracle.calls))
                self.assertEqual(run["evidence_queries"], len(oracle.calls))
                self.assertEqual([row["candidate_id"] for row in run["final_state"]["raw_observations"]],
                                 oracle.calls)
                comparisons = [event for event in run["events"]
                               if event["type"] == "structure_counterfactual"]
                if run["interventions"]["structure"]:
                    self.assertTrue(comparisons)
                self.assertTrue(all(event["oracle_calls"] == 0 for event in comparisons))

    def test_hidden_candidate_and_result_payloads_do_not_enter_any_state(self):
        for method in HARNESS_METHODS:
            with self.subTest(method=method):
                candidates = public_candidates()
                for item in candidates:
                    item.update(test_bpb=Poison(), private_candidate_payload=Poison())
                oracle = RecordingOracle(candidates, fixture_scores(candidates), poison=True)
                run = run_search(candidates, oracle, method, seed=19, budget=12)
                serialized = json.dumps(run, allow_nan=False)
                for key in ("test_bpb", "private_candidate_payload", "private_result_payload"):
                    self.assertNotIn(key, serialized)
                observed_ids = {row["evidence_id"] for row in run["final_state"]["raw_observations"]}
                for event in run["events"]:
                    self.assertLessEqual(set(event.get("evidence_ids", [])), observed_ids)

    def test_all_failed_queries_remain_charged_and_cannot_produce_a_winner(self):
        for method in HARNESS_METHODS:
            with self.subTest(method=method):
                candidates = public_candidates()
                oracle = RecordingOracle(candidates, fixture_scores(candidates), all_failed=True)
                run = run_search(candidates, oracle, method, seed=5, budget=11)
                self.assertEqual(run["attempts"], 11)
                self.assertEqual(run["new_evaluations"], 11)
                self.assertEqual(run["failed_or_infeasible_evaluations"], 11)
                self.assertEqual(run["charged_seconds"], sum(oracle.quotes[cid] for cid in oracle.calls))
                self.assertIsNone(run["winner"])
                self.assertTrue(all(not row["feasible"] and row["val_bpb"] is None
                                    for row in run["final_state"]["raw_observations"]))
                json.dumps(run, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
