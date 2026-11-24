"""Harness behavior tests. Synthetic rewards below are software fixtures only."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nas_agent_bench.architecture import OPS, validate_ops
from nas_agent_bench.harness_evolution import HarnessEvolutionController, validate_program


def warm_row(step=1, score=50.0, flops=30.0, params=0.3, ops=None):
    return {"step": step, "phase": "warm_start", "status": "valid_new", "ops": ops or [OPS[0]] * 6,
            "val_accuracy": score, "flops": flops, "params": params}


def observation(history, remaining=16, feedback=True):
    return {"history": history, "remaining_attempts": remaining, "next_attempt": len(history) + 1,
            "complete": remaining == 0, "feedback_enabled": feedback}


def evaluate(controller, history, remaining=16, incumbent_score=60.0, challenger_score=80.0):
    response = controller.propose(observation(history, remaining))
    role = response["controller_metadata"]["harness_role"]
    row = {"step": len(history) + 1, "phase": "search", "status": "valid_new", "ops": response["ops"],
           "val_accuracy": challenger_score if role == "challenger" else incumbent_score,
           "flops": 40.0, "params": 0.4}
    controller.observe(row)
    history.append(row)
    return response


class HarnessEvolutionTests(unittest.TestCase):
    def test_actual_program_is_mutated_and_promoted_using_equal_credit(self):
        controller = HarnessEvolutionController(4, {"evaluation_window": 2})
        history = [warm_row()]
        controller.observe(history[0])
        initial = controller.incumbent.to_dict()
        responses = [evaluate(controller, history, 4-i) for i in range(4)]
        self.assertEqual(sorted(r["controller_metadata"]["harness_role"] for r in responses),
                         ["challenger", "challenger", "incumbent", "incumbent"])
        self.assertNotEqual(controller.incumbent.to_dict(), initial)
        self.assertEqual(controller.incumbent_version, 1)
        event = next(e for e in controller.drain_events() if e["type"] == "harness_promoted")
        self.assertEqual(event["mean_reward"], {"incumbent": 60.0, "challenger": 80.0})
        self.assertEqual(event["credit_samples"], {"incumbent": 2, "challenger": 2})

    def test_failed_challenger_rolls_back_and_ties_do_not_promote(self):
        for challenger_score in (20.0, 60.0):
            controller = HarnessEvolutionController(1, {"evaluation_window": 1})
            history = [warm_row()]
            controller.observe(history[0])
            initial = controller.incumbent
            for i in range(2):
                evaluate(controller, history, 2-i, challenger_score=challenger_score)
            self.assertEqual(controller.incumbent, initial)
            self.assertEqual(controller.rejections, 1)
            event = next(e for e in controller.drain_events() if e["type"] == "harness_rejected")
            self.assertEqual(event["rollback_to_version"], 0)

    def test_frozen_program_and_too_short_budget_do_not_mutate(self):
        for config, remaining in (({"evolve": False}, 20), ({"evaluation_window": 4}, 7)):
            controller = HarnessEvolutionController(9, config)
            history = [warm_row()]
            controller.observe(history[0])
            initial = controller.incumbent
            for i in range(remaining):
                evaluate(controller, history, remaining-i)
            self.assertEqual(controller.incumbent, initial)
            self.assertEqual(len(controller.versions), 1)
            self.assertFalse(any(e["type"] == "harness_trial_started" for e in controller.drain_events()))

    def test_niche_archive_keeps_best_observed_validation_and_earliest_ties(self):
        controller = HarnessEvolutionController(5, {"flops_bins": [50], "params_bins": [0.5]})
        for row in (warm_row(1, 50), warm_row(2, 40), warm_row(3, 60),
                    warm_row(4, 60), warm_row(5, 10, 70, 0.8, [OPS[1]] * 6)):
            controller.observe(row)
        self.assertEqual(len(controller.archive), 2)
        self.assertEqual(controller.archive[(0, 0)]["step"], 3)
        self.assertEqual(controller.archive[(1, 1)]["val_accuracy"], 10)
        memory = controller._memory(controller.history, validate_program({"memory_policy": "archive"}))
        self.assertEqual({tuple(row["ops"]) for row in memory}, {(OPS[0],) * 6, (OPS[1],) * 6})

    def test_json_checkpoint_resumes_pending_credit_and_rng_exactly(self):
        first = HarnessEvolutionController(17, {"evaluation_window": 2})
        history = [warm_row()]
        first.observe(history[0])
        proposal = first.propose(observation(history))
        state = json.loads(json.dumps(first.state_dict(), allow_nan=False))
        second = HarnessEvolutionController(999)
        second.load_state_dict(state)
        row = {"step": 2, "phase": "search", "status": "valid_new", "ops": proposal["ops"],
               "val_accuracy": 70.0, "flops": 70.0, "params": 0.7}
        first.observe(row)
        second.observe(row)
        history.append(row)
        for i in range(12):
            a = evaluate(first, history, 12-i)
            other_history = history[:-1]
            b = evaluate(second, other_history, 12-i)
            self.assertEqual(a, b)
            self.assertEqual(first.state_dict(), second.state_dict())

    def test_model_patch_is_bounded_and_model_never_receives_hidden_fields(self):
        calls = []
        def model(payload):
            calls.append(payload)
            if payload["task"] == "propose_harness_patch":
                return {"patch": {"workflow": ["select", "crossover", "mutate", "repair"], "memory_policy": "archive"}}
            return {"ops": [OPS[3]] * 6, "usage": {"input_tokens": 10, "output_tokens": 6}}
        controller = HarnessEvolutionController(0, {"evaluation_window": 1}, model)
        row = warm_row()
        row.update(test_accuracy=987654321, raw_response="PRIVATE EVALUATOR")
        controller.observe(row)
        response = controller.propose(observation([row]))
        self.assertEqual(len(calls), 2)
        serialized = json.dumps(calls)
        self.assertNotIn("987654321", serialized)
        self.assertNotIn("PRIVATE EVALUATOR", serialized)
        self.assertEqual(validate_ops(response["ops"]), (OPS[3],) * 6)
        self.assertEqual(controller.versions[-1]["source"], "model_patch")
        self.assertEqual(controller.versions[-1]["program"]["workflow"], ["select", "crossover", "mutate", "repair"])

    def test_bad_model_edits_fall_back_but_invalid_architecture_stays_invalid(self):
        def model(payload):
            if payload["task"] == "propose_harness_patch":
                return {"patch": {"python": "raise SystemExit"}}
            raise RuntimeError("backend unavailable")
        controller = HarnessEvolutionController(3, {"evaluation_window": 1}, model)
        proposal = controller.propose(observation([], 2))
        self.assertIsNone(proposal["ops"])
        controller.observe({"step": 1, "phase": "search", "status": "invalid", "ops": None})
        self.assertIsNone(controller.pending)
        events = controller.drain_events()
        self.assertTrue(any(e["type"] == "harness_edit_rejected" for e in events))
        self.assertTrue(any(e["type"] == "model_call_failed" for e in events))
        credit = next(e for e in events if e["type"] == "harness_credit")
        self.assertEqual(credit["reward"], 0.0)

    def test_noop_model_patch_falls_back_to_a_changed_program(self):
        for patch in ({}, {"memory_policy": "mixed"}):
            def model(payload):
                if payload["task"] == "propose_harness_patch":
                    return {"patch": patch}
                return {"ops": [OPS[2]] * 6}
            controller = HarnessEvolutionController(8, {"evaluation_window": 1}, model)
            controller.propose(observation([], 2))
            self.assertNotEqual(controller.versions[0]["program"], controller.versions[1]["program"])
            event = next(e for e in controller.drain_events() if e["type"] == "harness_trial_started")
            self.assertTrue(event["changes"])
            self.assertEqual(event["source"], "local_mutation")

    def test_hidden_feedback_not_reintroduced_from_archive(self):
        calls = []
        def model(payload):
            calls.append(payload)
            return {"ops": [OPS[2]] * 6}
        controller = HarnessEvolutionController(2, {"initial_program": {"memory_policy": "archive"}}, model)
        row = warm_row(score=88.12345)
        controller.observe(row)
        response = controller.propose(observation([row], feedback=False))
        self.assertEqual(len(calls), 1)
        self.assertNotIn("val_accuracy", json.dumps(calls))
        self.assertNotIn("88.12345", json.dumps(calls))
        self.assertEqual(response["controller_metadata"]["harness_version"], 0)

    def test_validation_rejects_executable_code_unknown_fields_and_nonfinite_numbers(self):
        for patch in ({"workflow": ["exec"]}, {"source_code": "print('bad')"},
                      {"exploration_rate": float("nan")}, {"mutation_edges": True},
                      {"instruction": ""}, {"memory_size": 10000}):
            with self.assertRaises(ValueError):
                validate_program(patch)
        with self.assertRaises(ValueError):
            HarnessEvolutionController(0, {"flops_bins": [50, 10]})

    def test_step_guard_prevents_misattributed_credit(self):
        controller = HarnessEvolutionController(1)
        controller.propose(observation([], 1))
        with self.assertRaises(RuntimeError):
            controller.propose(observation([], 1))
        with self.assertRaises(ValueError):
            controller.observe(warm_row(step=9))
        self.assertIsNotNone(controller.pending)
        pending_ops = controller.pending["ops"]
        different = [OPS[1] if op != OPS[1] else OPS[2] for op in pending_ops]
        with self.assertRaises(ValueError):
            controller.observe(warm_row(step=1, ops=different))

    def test_executable_repair_workflow_connects_input_to_output(self):
        for seed in range(10):
            controller = HarnessEvolutionController(seed, {"evolve": False, "initial_program": {
                "workflow": ["select", "mutate", "repair"], "exploration_rate": 0.0}})
            history = [warm_row()]
            controller.observe(history[0])
            response = controller.propose(observation(history, 1))
            reachable = {0}
            for op, (source, target) in zip(response["ops"], ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3))):
                if op != "none" and source in reachable:
                    reachable.add(target)
            self.assertIn(3, reachable)


if __name__ == "__main__":
    unittest.main()
