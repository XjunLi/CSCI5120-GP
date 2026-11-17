"""Numerical and information-boundary checks for online weight adaptation."""
import copy
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nas_agent_bench.architecture import OPS
from nas_agent_bench.weight_evolution import WeightEvolutionController, entropic_advantages


def observation(step):
    return {"next_attempt": step, "remaining_attempts": 100, "feedback_enabled": True, "history": []}


def evaluate(controller, step, reward=None, extra=None):
    proposal = controller.propose(observation(step))
    if reward is None:
        reward = sum(OPS.index(op) for op in proposal["ops"])
    row = {"step": step, "phase": "search", "status": "valid_new", "ops": proposal["ops"],
           "val_accuracy": float(reward), **(extra or {})}
    controller.observe(row)
    return proposal


class WeightEvolutionTests(unittest.TestCase):
    def test_adaptive_entropic_constraint_and_scale_invariance(self):
        advantages, info = entropic_advantages([1, 2, 3, 4, 7], kl_budget=0.35)
        scaled, _ = entropic_advantages([13, 23, 33, 43, 73], kl_budget=0.35)
        self.assertAlmostEqual(info["tilt_kl"], 0.35, places=9)
        self.assertAlmostEqual(float(advantages.mean()), 0, places=12)
        np.testing.assert_allclose(advantages, scaled, atol=1e-12)
        self.assertGreater(advantages[-1], advantages[0])
        equal, _ = entropic_advantages([5, 5, 5])
        np.testing.assert_equal(equal, np.zeros(3))

    def test_gradient_matches_finite_difference(self):
        c = WeightEvolutionController(4)
        p = c.policy
        p.weights[:] = np.random.default_rng(6).normal(size=p.weights.shape) * 0.03
        batch = [{"indices": [0, 1, 2, 3, 4, 0]}, {"indices": [4, 3, 2, 1, 0, 4]}]
        adv = np.array([-0.7, 0.7])
        _, grad, _ = p.loss_and_gradient(batch, adv)
        for index in [(0, 0, 0), (1, 1, 1), (3, 8, 3), (5, 0, 4), (5, 20, 1)]:
            original, epsilon = p.weights[index], 1e-5
            p.weights[index] = original + epsilon
            plus = p.loss_and_gradient(batch, adv)[0]
            p.weights[index] = original - epsilon
            minus = p.loss_and_gradient(batch, adv)[0]
            p.weights[index] = original
            self.assertAlmostEqual(grad[index], (plus - minus) / (2 * epsilon), places=7)

    def test_actual_updates_frozen_control_and_autoregressive_dependence(self):
        adaptive = WeightEvolutionController(8)
        frozen = WeightEvolutionController(8, {"adaptive": False})
        for step in range(1, 17):
            evaluate(adaptive, step)
            evaluate(frozen, step)
        self.assertEqual(adaptive.total_updates, 4)
        self.assertGreater(adaptive.state_dict()["weight_delta_l2"], 0)
        self.assertEqual(frozen.total_updates, 0)
        self.assertEqual(frozen.state_dict()["weight_delta_l2"], 0)
        # Earlier operations genuinely change later-edge conditional distributions.
        a = adaptive.policy.distributions([0, 0, 0, 0, 0, 0])[1]
        b = adaptive.policy.distributions([4, 0, 0, 0, 0, 0])[1]
        self.assertGreater(float(np.max(np.abs(a - b))), 0)

    def test_checkpoint_restores_partial_batch_rng_and_adam_exactly(self):
        original = WeightEvolutionController(12)
        for step in range(1, 7):
            evaluate(original, step)
        state = json.loads(json.dumps(original.state_dict()))
        restored = WeightEvolutionController(12)
        restored.load_state_dict(state)
        for step in range(7, 17):
            self.assertEqual(evaluate(original, step), evaluate(restored, step))
        self.assertEqual(original.state_dict(), restored.state_dict())
        reset = WeightEvolutionController(12)
        self.assertEqual(reset.state_dict()["weight_delta_l2"], 0)
        self.assertEqual(reset.total_updates, 0)

    def test_reward_information_boundary_and_masking(self):
        clean, contaminated = WeightEvolutionController(5), WeightEvolutionController(5)
        for step in range(1, 9):
            a = evaluate(clean, step)
            b = evaluate(contaminated, step, extra={"test_accuracy": 987654321, "secret": "heldout-secret"})
            self.assertEqual(a, b)
        self.assertEqual(clean.state_dict(), contaminated.state_dict())
        self.assertNotIn("heldout-secret", json.dumps(contaminated.state_dict()))
        masked = WeightEvolutionController(5)
        for step in range(1, 9):
            payload = {**observation(step), "feedback_enabled": False}
            proposal = masked.propose(payload)
            masked.observe({"step": step, "status": "valid_new", "ops": proposal["ops"], "val_accuracy": 99})
        self.assertEqual(masked.total_updates, 0)
        self.assertEqual(masked.state_dict()["weight_delta_l2"], 0)

    def test_warm_start_not_misrepresented_as_on_policy_data(self):
        c = WeightEvolutionController(3)
        for step in range(1, 11):
            c.observe({"step": step, "phase": "warm_start", "status": "valid_new",
                       "ops": [OPS[-1]] * 6, "val_accuracy": 99})
        self.assertEqual(c.batch, [])
        self.assertEqual(c.total_updates, 0)
        evaluate(c, 11)
        with self.assertRaises(ValueError):
            c.observe({"step": 11, "phase": "warm_start"})

    def test_partial_final_batch_and_stale_policy_detection(self):
        c = WeightEvolutionController(2)
        evaluate(c, 1, reward=0)
        evaluate(c, 2, reward=2)
        c.finalize()
        self.assertEqual(c.total_updates, 1)
        self.assertEqual(len(c.batch), 0)
        stale = WeightEvolutionController(2)
        for step in range(1, 4):
            evaluate(stale, step)
        stale.policy.weights[0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "off-policy"):
            evaluate(stale, 4)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch backend unavailable")
class LoRAWeightTests(unittest.TestCase):
    @staticmethod
    def tiny_model():
        import torch
        # A tiny frozen causal model, with exact cache/non-cache equivalence.
        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 12)
                self.q_proj = torch.nn.Linear(12, 12)
                self.v_proj = torch.nn.Linear(12, 12)
                self.lm_head = torch.nn.Linear(12, 16)

            def forward(self, input_ids, past_key_values=None, use_cache=False):
                hidden = self.embedding(input_ids).cumsum(dim=1)
                if past_key_values is not None:
                    hidden = hidden + past_key_values
                output = self.lm_head(torch.tanh(self.q_proj(hidden) + self.v_proj(hidden)))
                return SimpleNamespace(logits=output, past_key_values=hidden[:, -1:].detach())

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [int(text)] if text in ("0", "1", "2", "3", "4") else [5, 6, 7]

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            model = TinyLM()
        return {"model": model, "tokenizer": Tokenizer(), "identifier": "deterministic-tiny-causal-model"}

    def make(self, adaptive=True):
        return WeightEvolutionController(13, {"backend": "hf_lora", "adaptive": adaptive,
                                              "batch_size": 2, "learning_rate": 0.01}, model=self.tiny_model())

    def test_real_lora_gradient_base_frozen_and_action_mask(self):
        import torch
        c = self.make()
        frozen_before = {name: p.detach().clone() for name, p in c.policy.model.named_parameters() if not p.requires_grad}
        evaluate(c, 1, 0)
        evaluate(c, 2, 10)
        self.assertGreater(c.state_dict()["weight_delta_l2"], 0)
        update = [event for event in c.drain_events() if event["type"] == "weight_update"][0]
        self.assertGreater(update["gradient_norm"], 0)
        self.assertEqual(c.policy._all_action_distributions([0] * 6).shape, (6, 5))
        for name, p in c.policy.model.named_parameters():
            if name in frozen_before:
                self.assertTrue(torch.equal(p.detach(), frozen_before[name]), name)
                self.assertIsNone(p.grad)
        frozen = self.make(adaptive=False)
        evaluate(frozen, 1, 0)
        evaluate(frozen, 2, 10)
        self.assertEqual(frozen.state_dict()["weight_delta_l2"], 0)

    def test_lora_optimizer_rng_and_pending_action_roundtrip(self):
        a = self.make()
        evaluate(a, 1, 0)
        evaluate(a, 2, 10)
        pending = a.propose(observation(3))
        state = json.loads(json.dumps(a.state_dict()))
        b = self.make()
        b.load_state_dict(state)
        row = {"step": 3, "status": "valid_new", "ops": pending["ops"], "val_accuracy": 3}
        a.observe(row)
        b.observe(row)
        for step in range(4, 9):
            self.assertEqual(evaluate(a, step), evaluate(b, step))
        self.assertEqual(a.state_dict(), b.state_dict())

    def test_checkpoint_rejects_changed_frozen_base(self):
        c = self.make()
        state = c.state_dict()
        state["policy"]["base_manifest"]["parameters_sha256"] = "changed-base"
        with self.assertRaisesRegex(ValueError, "frozen base"):
            self.make().load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
