"""Behavioural invariants for evolving, communicating NAS agents."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nas_agent_bench.architecture import OPS, warm_start
from nas_agent_bench.multiagent_evolution import MultiAgentEvolutionController


def make_row(step, ops, score=None, phase="search", status="valid_new"):
    row = {"step": step, "phase": phase, "status": status, "ops": ops}
    if score is not None:
        row.update({"val_accuracy": score, "flops": 1., "params": 2.})
    return row


def initialize(controller):
    for i, ops in enumerate(warm_start(9, 4)[0], 1):
        controller.observe(make_row(i, list(ops), 70. + i, "warm_start"))


def observation(step, history=None):
    return {"next_attempt": step, "remaining_attempts": 100 - step,
            "feedback_enabled": True, "search_context": {"dataset": "synthetic", "hp": "200"},
            "history": history or []}


def round_trip(controller, step, score=None):
    proposal = controller.propose(observation(step))
    ops = proposal["ops"]
    if score is None:
        score = 60. + sum(OPS.index(op) for op in ops)
    controller.observe(make_row(step, ops, score))
    return proposal


class MultiagentEvolutionTests(unittest.TestCase):
    def test_reproduction_changes_strategy_and_routing_with_lineage(self):
        controller = MultiAgentEvolutionController(13, {"evaluations_per_agent": 1})
        initialize(controller)
        original_ids = [agent["id"] for agent in controller.agents]
        original_routes = list(controller.shortcuts)
        for step, score in zip(range(5, 9), (75., 90., 55., 65.)):
            round_trip(controller, step, score)
        state = controller.state_dict()
        events = controller.drain_events()
        replacement = [event for event in events if event["event"] == "multiagent_replacement"]
        self.assertEqual(state["generation"], 1)
        self.assertEqual(state["births"], 1)
        self.assertEqual(len(replacement), 1)
        event = replacement[0]
        self.assertIn(event["parent_id"], original_ids)
        self.assertIn(event["retired_id"], original_ids)
        self.assertNotIn(event["child_id"], original_ids)
        self.assertNotEqual(event["new_genome"], event["parent_genome"])
        self.assertNotEqual(state["shortcuts"], original_routes)
        self.assertGreater(state["message_count"], 0)
        self.assertTrue(any(event["event"] == "multiagent_share" for event in events))
        self.assertTrue(all(agent["memory"] for agent in controller.agents))

    def test_frozen_ablation_keeps_genomes_but_runs_peer_critique(self):
        controller = MultiAgentEvolutionController(13, {"evolve": False, "evaluations_per_agent": 1})
        initialize(controller)
        genomes = [json.dumps(agent["genome"], sort_keys=True) for agent in controller.agents]
        routes = list(controller.shortcuts)
        rounds = [round_trip(controller, step, 60. + step * 2) for step in range(5, 13)]
        self.assertEqual(genomes, [json.dumps(agent["genome"], sort_keys=True) for agent in controller.agents])
        self.assertEqual(routes, controller.shortcuts)
        self.assertEqual(controller.births, 0)
        self.assertEqual(controller.generation, 2)
        self.assertTrue(all("peer_critique" in row["controller_metadata"] for row in rounds))
        self.assertTrue(all(row["controller_metadata"]["model_calls"] == 0 for row in rounds))

    def test_checkpoint_resume_reproduces_rounds_and_evolution(self):
        controller = MultiAgentEvolutionController(21, {"evaluations_per_agent": 1})
        initialize(controller)
        for step in range(5, 10):
            round_trip(controller, step)
        controller.drain_events()
        snapshot = json.loads(json.dumps(controller.state_dict(), allow_nan=False))
        restored = MultiAgentEvolutionController(999)
        restored.load_state_dict(snapshot)
        for step in range(10, 25):
            self.assertEqual(round_trip(controller, step), round_trip(restored, step))
        self.assertEqual(controller.state_dict(), restored.state_dict())

    def test_pending_checkpoint_and_step_guard(self):
        controller = MultiAgentEvolutionController(2)
        initialize(controller)
        proposal = controller.propose(observation(5))
        restored = MultiAgentEvolutionController(0)
        restored.load_state_dict(json.loads(json.dumps(controller.state_dict())))
        with self.assertRaises(RuntimeError):
            restored.propose(observation(6))
        with self.assertRaises(ValueError):
            restored.observe(make_row(6, proposal["ops"], 80.))
        for item in (controller, restored):
            item.observe(make_row(5, proposal["ops"], 80.))
        self.assertEqual(round_trip(controller, 6), round_trip(restored, 6))

    def test_bad_feedback_or_candidate_mismatch_does_not_mutate_state(self):
        controller = MultiAgentEvolutionController(19)
        initialize(controller)
        before = controller.state_dict()
        hidden = observation(5)
        hidden["feedback_enabled"] = False
        with self.assertRaises(ValueError):
            controller.propose(hidden)
        self.assertEqual(before, controller.state_dict())
        proposal = controller.propose(observation(5))
        before = controller.state_dict()
        wrong_ops = list(proposal["ops"])
        wrong_ops[0] = OPS[(OPS.index(wrong_ops[0]) + 1) % len(OPS)]
        with self.assertRaises(ValueError):
            controller.observe(make_row(5, wrong_ops, 90.))
        self.assertEqual(before, controller.state_dict())
        controller.observe(make_row(5, proposal["ops"], 80.))

    def test_model_peer_can_repair_draft_without_extra_evaluations_or_leakage(self):
        payloads = []
        def model(payload):
            payloads.append(payload)
            if payload["stage"] == "proposer":
                return {"ops": ["none"] * 6, "usage": {"total_tokens": 7}, "model": "fake-model"}
            return {"ops": ["nor_conv_1x1"] * 6, "critique": "Add useful paths.",
                    "usage": {"total_tokens": 5}, "model": "fake-model"}
        controller = MultiAgentEvolutionController(0, model=model)
        for step in range(1, 3):
            row = make_row(step, [OPS[step]] * 6, 70. + step, "warm_start")
            row["test_accuracy"] = "SECRET_TEST_SCORE"
            row["raw_response"] = {"private": "SECRET_TEST_SCORE"}
            controller.observe(row)
        obs = observation(3, history=[{"test_accuracy": "SECRET_TEST_SCORE"}])
        obs["test"] = {"secret": "SECRET_TEST_SCORE"}
        obs["search_context"]["test_accuracy"] = "SECRET_TEST_SCORE"
        proposal = controller.propose(obs)
        self.assertEqual(len(payloads), 2)
        self.assertEqual([payload["stage"] for payload in payloads], ["proposer", "critic"])
        self.assertNotEqual(payloads[0]["agent_id"], payloads[1]["agent_id"])
        self.assertNotIn("SECRET_TEST_SCORE", json.dumps(payloads))
        self.assertEqual(proposal["ops"], ["nor_conv_1x1"] * 6)
        self.assertEqual(proposal["usage"]["total_tokens"], 12)
        self.assertEqual(proposal["controller_metadata"]["model_calls"], 2)
        self.assertTrue(proposal["controller_metadata"]["revised"])
        controller.observe(make_row(3, proposal["ops"], 85.))
        self.assertEqual(controller.state_dict()["observed_step"], 3)
        self.assertEqual(controller.proposal_count, 1)

    def test_model_failures_preserve_call_accounting_and_consume_final_attempt(self):
        seen = []
        def broken(payload):
            seen.append(payload["stage"])
            if payload["stage"] == "proposer":
                raise TimeoutError("timeout")
            return {"ops": ["invalid"] * 6}
        controller = MultiAgentEvolutionController(7, model=broken)
        initialize(controller)
        proposal = controller.propose(observation(5))
        self.assertIsNone(proposal["ops"])
        self.assertEqual(seen, ["proposer", "critic"])
        self.assertEqual(proposal["controller_metadata"]["model_calls"], 2)
        self.assertGreaterEqual(len(proposal["controller_metadata"]["model_failures"]), 2)
        controller.observe(make_row(5, None, status="invalid"))
        self.assertEqual(controller.agents[0]["proposal_rewards"], [-2.])

    def test_duplicate_gets_no_positive_credit_and_bad_critic_keeps_valid_draft(self):
        draft = ["none"] * 6
        def model(payload):
            return {"ops": draft} if payload["stage"] == "proposer" else {"ops": "malformed"}
        controller = MultiAgentEvolutionController(5, model=model)
        controller.observe(make_row(1, draft, 99., "warm_start"))
        proposal = controller.propose(observation(2))
        self.assertEqual(proposal["ops"], draft)
        controller.observe(make_row(2, draft, 99., status="duplicate"))
        self.assertEqual(controller.agents[0]["proposal_rewards"], [-.5])


if __name__ == "__main__":
    unittest.main()
