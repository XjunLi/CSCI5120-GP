"""Actual causal-model/gradient tests; all weights and rewards are test fixtures.

These randomly initialized tiny GPT-2 models are never experimental evidence of
pretrained-LLM search capability.  No external pretrained weights are downloaded.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ("torch", "transformers", "tokenizers"))

if AVAILABLE:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from golfbench.model_policy import HuggingFaceLoRAPolicy
    from golfbench.search import run_search


def candidates(count=8):
    return [{"candidate_id": f"design{i:02d}",
             "config": {"num_layers": 1 + i // 2, "model_dim": 32 * (1 + i % 2)},
             "charge_seconds": 10.0}
            for i in range(count)]


def fixture_components():
    vocabulary = {word: i for i, word in enumerate(
        ["[UNK]", "[PAD]", "[EOS]"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        + ["Choice", ":", "Available", "Observed", "id", "config", "label", "val_bpb"])}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                       pad_token="[PAD]", eos_token="[EOS]")
    with torch.random.fork_rng():
        torch.manual_seed(12345)
        config = GPT2Config(vocab_size=len(vocabulary), n_layer=1, n_head=2, n_embd=32,
                            n_positions=2048, n_ctx=2048, bos_token_id=2, eos_token_id=2,
                            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config)
    return model, tokenizer


def fixture_policy(count=8, **kwargs):
    model, tokenizer = fixture_components()
    return HuggingFaceLoRAPolicy(model, tokenizer, candidates(count),
                                model_id="random-gpt2-software-fixture", revision="test-only",
                                rank=2, alpha=4, max_input_tokens=2048, **kwargs)


def evaluate(candidate_id):
    index = int(candidate_id.removeprefix("design"))
    return {"candidate_id": candidate_id, "val_bpb": 4.0 + ((index * 7) % 11) / 10,
            "charge_seconds": 10.0, "train_seconds": 0.01, "feasible": True, "status": "ok",
            "held_out_test": object()}


def observation(candidate_id, attempt=1):
    return {"candidate_id": candidate_id, "val_bpb": evaluate(candidate_id)["val_bpb"],
            "status": "valid_new", "phase": "search", "attempt": attempt,
            "hidden_test_bpb": object()}


@unittest.skipUnless(AVAILABLE, "Optional torch/transformers/tokenizers dependencies are absent")
class ModelPolicyTests(unittest.TestCase):
    def test_real_h200_space_keeps_complete_action_prompt_within_limit(self):
        from golfbench.space import make_space
        model, tokenizer=fixture_components()
        public=make_space("gpu")["candidates"]
        policy=HuggingFaceLoRAPolicy(model,tokenizer,public,model_id="random-gpt2-software-fixture",
                                    revision="test-only",rank=2,alpha=4,max_input_tokens=2048)
        prompt,tokens,ids=policy._prompt(public,[])
        self.assertEqual(len(ids),24)
        self.assertLessEqual(len(tokens),2048)
        self.assertIn("Common config:",prompt)

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_twenty_four_choices_have_unique_verified_single_token_labels(self):
        policy = fixture_policy(24)
        labels = policy.provenance["action_map"]
        self.assertEqual(len({row["token_id"] for row in labels}), 24)
        self.assertEqual(len({row["label"] for row in labels}), 24)
        chosen, metadata = policy.propose(candidates(24), [])
        self.assertIn(chosen, [row["candidate_id"] for row in candidates(24)])
        self.assertIn(metadata["action_token_id"], {row["token_id"] for row in labels})
        self.assertGreater(metadata["input_tokens"], 0)
        self.assertEqual(policy.summary()["gradient_updates"], 0)

    def test_real_lora_updates_leave_base_frozen_and_control_unchanged(self):
        policy = fixture_policy()
        frozen_base = {name: parameter.detach().clone() for name, parameter in policy.model.named_parameters()
                       if "lora_" not in name}
        adaptive = run_search(candidates(), evaluate, "weight-adaptive", seed=12, budget=7,
                              warm_start=3, weight_policy=policy)
        self.assertEqual(adaptive["gradient_updates"], 4)
        self.assertEqual(adaptive["gradient_evaluations"], 4)
        self.assertNotEqual(adaptive["initial_weight_sha256"], adaptive["final_weight_sha256"])
        self.assertTrue(any(event.get("update_norm", 0) > 0 for event in adaptive["events"]))
        self.assertGreater(adaptive["trainable_parameter_count"], 0)
        self.assertLess(adaptive["trainable_parameter_count"], adaptive["model_provenance"]["base_parameters"])
        for name, parameter in policy.model.named_parameters():
            if "lora_" not in name:
                self.assertFalse(parameter.requires_grad)
                self.assertTrue(torch.equal(parameter.detach(), frozen_base[name]), name)
        frozen = run_search(candidates(), evaluate, "weight-frozen", seed=12, budget=7,
                            warm_start=3, weight_policy=policy)
        self.assertEqual(frozen["gradient_updates"], 0)
        self.assertEqual(frozen["gradient_evaluations"], 4)
        self.assertEqual(frozen["initial_weight_sha256"], frozen["final_weight_sha256"])
        self.assertTrue(all(event.get("update_norm", 0) == 0 for event in frozen["events"]))
        self.assertEqual(adaptive["initial_candidate_ids"], frozen["initial_candidate_ids"])
        self.assertEqual(adaptive["trajectory"][3]["candidate_id"], frozen["trajectory"][3]["candidate_id"])
        self.assertEqual(adaptive["initial_weight_sha256"], frozen["initial_weight_sha256"])
        self.assertNotIn("held_out_test", json.dumps(adaptive, allow_nan=False))

    def test_warm_starts_and_unevaluated_budget_stop_never_update(self):
        policy = fixture_policy()
        warm_only = run_search(candidates(), evaluate, "weight-adaptive", seed=4,
                               budget=3, warm_start=3, weight_policy=policy)
        self.assertEqual(warm_only["gradient_updates"], 0)
        self.assertEqual(warm_only["policy_usage"]["forward_passes"], 0)
        calls = []

        def tracked(candidate_id):
            calls.append(candidate_id)
            return evaluate(candidate_id)

        stopped = run_search(candidates(), tracked, "weight-adaptive", seed=4,
                             budget=7, warm_start=3, cost_budget_seconds=30, weight_policy=policy)
        self.assertEqual(len(calls), 3)
        self.assertEqual(stopped["gradient_updates"], 0)
        self.assertEqual(stopped["gradient_evaluations"], 0)
        self.assertEqual(stopped["initial_weight_sha256"], stopped["final_weight_sha256"])
        self.assertEqual(stopped["policy_usage"]["forward_passes"], 1)
        self.assertFalse(stopped["pending_unobserved_proposal"])
        self.assertEqual(stopped["events"][-1]["type"], "policy_proposal_cancelled")

    def test_pending_checkpoint_resumes_same_optimizer_and_sampler(self):
        policy = fixture_policy()
        policy.start_run(18, True)
        pool = candidates()
        history = []
        chosen, _ = policy.propose(pool, history)
        row = observation(chosen)
        policy.observe(row)
        history.append({key: row[key] for key in ("candidate_id", "val_bpb")})
        pool = [item for item in pool if item["candidate_id"] != chosen]
        pending, _ = policy.propose(pool, history)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            policy.save_checkpoint(checkpoint)
            restored = fixture_policy()
            restored.load_checkpoint(checkpoint)
            next_row = observation(pending, attempt=2)
            policy.observe(next_row)
            restored.observe(next_row)
            self.assertEqual(policy.summary()["final_weight_sha256"], restored.summary()["final_weight_sha256"])
            self.assertEqual(policy.gradient_updates, restored.gradient_updates)
            history.append({key: next_row[key] for key in ("candidate_id", "val_bpb")})
            pool = [item for item in pool if item["candidate_id"] != pending]
            first, first_meta = policy.propose(pool, history)
            second, second_meta = restored.propose(pool, history)
            self.assertEqual(first, second)
            self.assertEqual(first_meta, second_meta)
            self.assertEqual(policy.optimizer.state_dict()["param_groups"], restored.optimizer.state_dict()["param_groups"])

    def test_checkpoint_rejects_another_model_and_identity_is_defensive(self):
        policy = fixture_policy()
        identity = policy.identity
        identity["adapter_settings"]["rank"] = 999
        self.assertEqual(policy.identity["adapter_settings"]["rank"], 2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            policy.save_checkpoint(checkpoint)
            incompatible = fixture_policy(7)
            with self.assertRaisesRegex(ValueError, "does not match"):
                incompatible.load_checkpoint(checkpoint)

    def test_local_loader_roundtrip_uses_only_fixture_weights(self):
        model, tokenizer = fixture_components()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            policy = HuggingFaceLoRAPolicy.from_local(directory, candidates(), rank=2, alpha=4)
            self.assertTrue(policy.provenance["local_files_only"])
            self.assertFalse(policy.provenance["trust_remote_code"])
            self.assertGreater(policy.provenance["model_loading_seconds"], 0)
            chosen, _ = policy.propose(candidates(), [])
            self.assertIn(chosen, policy.candidate_by_id)
        with self.assertRaisesRegex(ValueError, "existing local model directory"):
            HuggingFaceLoRAPolicy.from_local("nonexistent/hub-model-id", candidates())

    def test_stale_sample_and_wrong_observation_are_rejected(self):
        policy = fixture_policy()
        policy.start_run(3, True)
        chosen, _ = policy.propose(candidates(), [])
        with self.assertRaisesRegex(ValueError, "does not match"):
            policy.observe(observation("design99"))
        with torch.no_grad():
            policy.parameters[0].add_(0.01)
        with self.assertRaisesRegex(ValueError, "off-policy"):
            policy.observe(observation(chosen))


if __name__ == "__main__":
    unittest.main()
