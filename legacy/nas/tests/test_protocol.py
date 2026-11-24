"""Protocol invariants, using synthetic data explicitly separated from results."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nas_agent_bench.architecture import OPS, canonical, parse_architecture
from nas_agent_bench.engine import run_search
from nas_agent_bench.oracles import TableOracle, NatsOracle
from nas_agent_bench.protocol import configuration, evaluate_proposal, observation, summarize
from nas_agent_bench.sessions import initialize, propose, status, export_session


class ToyOracle:
    label = "TOY SOFTWARE TEST ONLY"
    dataset, hp = "cifar10-valid", "200"
    training_seed_policy = "synthetic"
    metadata = {"source": "synthetic", "training_seed_policy": "synthetic"}
    def __init__(self):
        self.test_calls = 0
    def validation(self, ops):
        return {"val_accuracy": float(sum(OPS.index(x) for x in ops)), "flops": 111.0, "params": 222.0}
    def final_test(self, ops):
        self.test_calls += 1
        return {"test_score": -self.validation(ops)["val_accuracy"]}


class ProtocolTests(unittest.TestCase):
    def test_known_encoding(self):
        ops = ("nor_conv_3x3", "none", "skip_connect", "nor_conv_1x1", "avg_pool_3x3", "skip_connect")
        expected = "|nor_conv_3x3~0|+|none~0|skip_connect~1|+|nor_conv_1x1~0|avg_pool_3x3~1|skip_connect~2|"
        self.assertEqual(canonical(ops), expected)
        self.assertEqual(parse_architecture(expected), ops)

    def test_budget_invalid_duplicate_and_validation_selection(self):
        oracle, history = ToyOracle(), []
        for ops in ([OPS[0]] * 6, [OPS[-1]] * 6, [OPS[-1]] * 6, ["bad"] * 6):
            history.append(evaluate_proposal(oracle, history, ops))
        config = configuration(oracle, "chatgpt-feedback", 0, 4, 1)
        summary = summarize(config, history)
        self.assertEqual([r["status"] for r in history], ["valid_new", "valid_new", "duplicate", "invalid"])
        self.assertEqual(summary["completed_attempts"], 4)
        self.assertEqual(summary["unique_evaluations_including_warm"], 2)
        self.assertEqual(summary["winner_ops"], [OPS[-1]] * 6)
        self.assertEqual(oracle.test_calls, 0)
        self.assertIsNone(summary["model_seconds"])

    def test_feedback_mask_includes_warm_start(self):
        oracle = ToyOracle()
        history = [evaluate_proposal(oracle, [], [OPS[-1]] * 6, phase="warm_start")]
        for method in ("llm-no-feedback", "chatgpt-no-feedback"):
            payload = observation(configuration(oracle, method, 1, 50, 1), history)
            for key in ("val_accuracy", "flops", "params", "best_so_far_val_accuracy", "111.0", "222.0"):
                self.assertNotIn(key, json.dumps(payload))
        visible = observation(configuration(oracle, "chatgpt-feedback", 1, 50, 1), history)
        self.assertIn("val_accuracy", visible["history"][0])

    def test_baselines_share_initialization_and_defer_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            oracle = ToyOracle()
            summaries = [run_search(oracle, method, 12, 50, 10, tmp) for method in ("random", "evolution")]
            trajectories = [[json.loads(line) for line in (Path(tmp) / s["trajectory_file"]).read_text().splitlines()] for s in summaries]
            self.assertEqual([x["ops"] for x in trajectories[0][:10]], [x["ops"] for x in trajectories[1][:10]])
            self.assertEqual(oracle.test_calls, 0)
            self.assertTrue(all(s["completed_attempts"] == 50 for s in summaries))
            self.assertEqual(summaries[0]["unique_evaluations_including_warm"], 50)

    def test_real_oracle_whitelist_not_raw_api(self):
        class FakeAPI:
            def query_index_by_arch(self, text): return 0
            def get_more_info(self, *a, **kw): return {"valid-accuracy": 1, "test-accuracy": 987654321}
            def get_cost_info(self, *a, **kw): return {"flops": 2, "params": 3}
        oracle = NatsOracle.__new__(NatsOracle)
        oracle.api, oracle.dataset, oracle.hp, oracle.cache = FakeAPI(), "cifar10-valid", "200", {}
        actual = oracle.validation(["none"] * 6)
        self.assertEqual(set(actual), {"val_accuracy", "flops", "params"})
        self.assertNotIn(987654321, actual.values())

    def test_table_validation_and_test_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.json"
            path.write_text(json.dumps({"metadata": {"source": "synthetic", "training_seed_policy": "synthetic", "hp": "200"},
                                        "rows": [{"ops": ["none"] * 6, "val_accuracy": 42, "test": {"cifar10": 99}}]}))
            oracle = TableOracle(path)
            self.assertEqual(oracle.validation(["none"] * 6)["val_accuracy"], 42)
            self.assertEqual(oracle.final_test(["none"] * 6)["cifar10_test_accuracy_hp200"], 99)

    def test_interactive_atomic_budget_mask_and_stale_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "session"
            oracle = ToyOracle()
            with patch("nas_agent_bench.sessions.load_oracle", return_value=oracle):
                initial = initialize(folder, "table", "/unused", "no-feedback", 3, 4, 2)
                self.assertEqual(initial["remaining_attempts"], 2)
                first = initial["history"][0]["ops"]
                duplicate = propose(folder, 3, json.dumps(first))
                self.assertEqual(duplicate["observation"]["status"], "duplicate")
                self.assertNotIn("val_accuracy", json.dumps(duplicate))
                with self.assertRaises(ValueError):
                    propose(folder, 3, json.dumps(first))
                invalid = propose(folder, 4, "NOT JSON")
                self.assertEqual(invalid["observation"]["status"], "invalid")
                self.assertTrue(invalid["complete"])
                with self.assertRaises(ValueError):
                    propose(folder, 5, json.dumps(first))
                self.assertEqual(oracle.test_calls, 0)
                result = export_session(folder, Path(tmp) / "export")
                self.assertFalse(result["test_evaluated"])
                self.assertEqual(status(folder)["remaining_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
