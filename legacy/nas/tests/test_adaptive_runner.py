"""Integration tests for adaptation accounting and resumable validation-only runs.

All oracle scores in this module are synthetic software fixtures, not NAS
results. Real subprocesses are used only for tiny local adapter-error fixtures.
"""
from copy import deepcopy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nas_agent_bench import adaptive_runner, model_adapters
from nas_agent_bench.adaptive_runner import run_adaptive_search
from nas_agent_bench.architecture import OPS
from nas_agent_bench.model_adapters import SubprocessJSONModel, parse_json_response


SECRET = "HIDDEN_TEST_SENTINEL_963714"


class ToyOracle:
    label = "SYNTHETIC INTEGRATION FIXTURE"
    dataset, hp = "cifar10-valid", "200"
    training_seed_policy = "synthetic fixture"
    metadata = {"source": "synthetic", "training_seed_policy": "synthetic fixture",
                "normalized_data_sha256": "f" * 64, "private_provenance": SECRET}

    def __init__(self):
        self.validation_calls, self.test_calls = 0, 0

    def validation(self, ops):
        self.validation_calls += 1
        score = 30.0 + sum((i + 1) * OPS.index(op) for i, op in enumerate(ops)) / 2
        return {"val_accuracy": score, "flops": 20 + score, "params": score / 100,
                "test_accuracy": SECRET}

    def final_test(self, ops):
        self.test_calls += 1
        return {"hidden_test_score": SECRET}


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if not k.endswith("seconds") and k != "seconds"}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


class RecordingController:
    def __init__(self):
        self.observations, self.rows = [], []

    def propose(self, observation):
        self.observations.append(deepcopy(observation))
        return {"ops": [OPS[3]] * 6}

    def observe(self, row):
        self.rows.append(deepcopy(row))

    def state_dict(self):
        return {"observations": self.observations, "rows": self.rows}

    def load_state_dict(self, state):
        self.observations, self.rows = deepcopy(state["observations"]), deepcopy(state["rows"])

    def drain_events(self):
        return []


class AdaptiveRunnerTests(unittest.TestCase):
    def test_harness_and_multiagent_resume_exactly_and_share_warm_start(self):
        beginnings = []
        with tempfile.TemporaryDirectory() as tmp:
            for method in ("harness", "harness-frozen", "multiagent", "multiagent-frozen"):
                with self.subTest(method=method):
                    full, resumed = Path(tmp) / (method + "_full"), Path(tmp) / (method + "_resumed")
                    settings = {"evaluation_window": 2} if method.startswith("harness") else {
                        "population_size": 3, "evaluations_per_agent": 1}
                    oracle_full, oracle_resume = ToyOracle(), ToyOracle()
                    expected = run_adaptive_search(oracle_full, method, 11, 18, 3, full, settings)
                    partial = run_adaptive_search(oracle_resume, method, 11, 18, 3, resumed, settings, stop_after=8)
                    self.assertFalse(partial["complete"])
                    actual = run_adaptive_search(oracle_resume, method, 11, 18, 3, resumed, settings, resume=True)
                    name = f"{method}_seed11"
                    first, second = read_lines(full / f"{name}.jsonl"), read_lines(resumed / f"{name}.jsonl")
                    self.assertEqual(without_timing(first), without_timing(second))
                    self.assertEqual(without_timing(read_lines(full / f"{name}_events.jsonl")),
                                     without_timing(read_lines(resumed / f"{name}_events.jsonl")))
                    self.assertEqual(json.loads((full / "checkpoints" / f"{name}.json").read_text())["controller_state"],
                                     json.loads((resumed / "checkpoints" / f"{name}.json").read_text())["controller_state"])
                    self.assertEqual(expected["winner_ops"], actual["winner_ops"])
                    self.assertEqual(actual["completed_attempts"], 18)
                    self.assertEqual(oracle_full.validation_calls, oracle_resume.validation_calls)
                    self.assertEqual(oracle_resume.validation_calls, actual["unique_evaluations_including_warm"])
                    self.assertEqual(oracle_full.test_calls + oracle_resume.test_calls, 0)
                    beginnings.append([r["ops"] for r in first[:3]])
            self.assertTrue(all(warm == beginnings[0] for warm in beginnings))

    def test_checkpoint_recovers_stale_views_without_replaying_committed_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            oracle = ToyOracle()
            output = Path(tmp)
            original_write = adaptive_runner._write_jsonl
            def fail_after_checkpoint(path, rows):
                if Path(path).name == "harness_seed7.jsonl" and len(rows) == 3:
                    raise OSError("Simulated crash rebuilding a derived trajectory")
                return original_write(path, rows)
            with patch.object(adaptive_runner, "_write_jsonl", side_effect=fail_after_checkpoint):
                with self.assertRaises(OSError):
                    run_adaptive_search(oracle, "harness", 7, 9, 2, output, {"evaluation_window": 1})
            checkpoint = json.loads((output / "checkpoints/harness_seed7.json").read_text())
            self.assertEqual(len(checkpoint["history"]), 3)
            self.assertEqual(len(read_lines(output / "harness_seed7.jsonl")), 2)
            result = run_adaptive_search(oracle, "harness", 7, 9, 2, output, {"evaluation_window": 1}, resume=True)
            self.assertEqual(len(read_lines(output / "harness_seed7.jsonl")), 9)
            self.assertEqual(oracle.validation_calls, result["unique_evaluations_including_warm"])
            self.assertEqual(oracle.test_calls, 0)

    def test_injected_controller_never_receives_private_or_test_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            oracle, controller = ToyOracle(), RecordingController()
            result = run_adaptive_search(oracle, "harness", 2, 7, 2, tmp, controller=controller)
            public_data = json.dumps(controller.observations + controller.rows)
            self.assertNotIn(SECRET, public_data)
            self.assertNotIn("test_accuracy", public_data)
            self.assertNotIn("private_provenance", public_data)
            self.assertEqual(result["completed_attempts"], 7)
            self.assertEqual(oracle.test_calls, 0)

    def test_subprocess_failures_cost_attempts_and_all_calls_are_counted(self):
        command = [sys.executable, "-c", "import sys; sys.stderr.write('fixture failure'); sys.exit(7)"]
        with tempfile.TemporaryDirectory() as tmp:
            for method, expected_calls in (("harness-frozen", 4), ("multiagent-frozen", 8), ("harness", 6)):
                with self.subTest(method=method):
                    model = SubprocessJSONModel(command, model_label="failing-fixture", seed=1)
                    oracle = ToyOracle()
                    config = {"evaluation_window": 1} if method.startswith("harness") else {}
                    result = run_adaptive_search(oracle, method, 1, 6, 2, Path(tmp) / method,
                                                 config, model=model, model_label="failing-fixture")
                    self.assertEqual(result["completed_attempts"], 6)
                    self.assertEqual(result["invalid_excluding_warm"], 4)
                    self.assertEqual(result["model_call_count"], expected_calls)
                    self.assertEqual(model.calls, expected_calls)
                    self.assertEqual(model.errors, expected_calls)
                    self.assertEqual(oracle.validation_calls, 2)
                    self.assertEqual(oracle.test_calls, 0)
                    requests = [e["request"] for e in read_lines(Path(tmp) / method / result["events_file"])
                                if e.get("event_source") == "model_adapter"]
                    self.assertNotIn(SECRET, json.dumps(requests))

    def test_subprocess_call_counter_and_error_log_resume(self):
        command = [sys.executable, "-c", "print('not-json')"]
        with tempfile.TemporaryDirectory() as tmp:
            first = SubprocessJSONModel(command, model_label="invalid-json", seed=6)
            run_adaptive_search(ToyOracle(), "harness-frozen", 6, 8, 2, tmp,
                                model=first, model_label="invalid-json", stop_after=5)
            self.assertEqual(first.calls, 3)
            second = SubprocessJSONModel(command, model_label="invalid-json", seed=6)
            result = run_adaptive_search(ToyOracle(), "harness-frozen", 6, 8, 2, tmp,
                                         model=second, model_label="invalid-json", resume=True)
            self.assertEqual(second.calls, 6)
            self.assertEqual(second.errors, 6)
            self.assertEqual(result["model_call_count"], 6)
            calls = [e for e in read_lines(Path(tmp) / result["events_file"])
                     if e.get("event_source") == "model_adapter"]
            self.assertEqual([e["call_index"] for e in calls], list(range(1, 7)))

    def test_plain_callable_model_calls_are_not_reported_as_zero(self):
        calls = []
        def model(payload):
            calls.append(deepcopy(payload))
            return {"ops": [OPS[2]] * 6}
        with tempfile.TemporaryDirectory() as tmp:
            result = run_adaptive_search(ToyOracle(), "harness-frozen", 3, 6, 2, tmp, model=model)
            self.assertEqual(len(calls), 4)
            self.assertEqual(result["model_call_count"], len(calls))
            self.assertNotIn(SECRET, json.dumps(calls))

    def test_resume_rejects_model_to_local_switch_and_changed_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = lambda payload: {"ops": [OPS[2]] * 6}
            run_adaptive_search(ToyOracle(), "harness-frozen", 3, 8, 2, tmp, model=model, stop_after=4)
            with self.assertRaises(ValueError):
                run_adaptive_search(ToyOracle(), "harness-frozen", 3, 8, 2, tmp, model=None, resume=True)
            with self.assertRaises(ValueError):
                run_adaptive_search(ToyOracle(), "harness-frozen", 3, 10, 2, tmp, model=model, resume=True)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_adaptive_search(ToyOracle(), "harness-frozen", 0, 4, 2, tmp)
            before = (Path(tmp) / "harness-frozen_seed0.jsonl").read_bytes()
            with self.assertRaises(FileExistsError):
                run_adaptive_search(ToyOracle(), "harness-frozen", 0, 4, 2, tmp)
            self.assertEqual((Path(tmp) / "harness-frozen_seed0.jsonl").read_bytes(), before)

    def test_model_parser_rejects_nonfinite_and_nonobject_json(self):
        for text in ('{"ops": null, "usage": {"x": NaN}}', '{"x": Infinity}', '{"x": -Infinity}', '{"x":1e999}',
                     '[]', '{"a": 1} {"b": 2}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_json_response(text)
        self.assertEqual(parse_json_response('```json\n{"ops": null}\n```'), {"ops": None})

    def test_nonfinite_subprocess_response_becomes_one_invalid_attempt(self):
        for number in ("NaN", "1e999"):
            command = [sys.executable, "-c", f"print('{{\"ops\": null, \"usage\": {{\"x\": {number}}}}}')"]
            with self.subTest(number=number), tempfile.TemporaryDirectory() as tmp:
                model = SubprocessJSONModel(command, seed=0)
                result = run_adaptive_search(ToyOracle(), "harness-frozen", 0, 3, 2, tmp, model=model)
                self.assertEqual(result["completed_attempts"], 3)
                self.assertEqual(result["invalid_excluding_warm"], 1)
                self.assertEqual(result["model_call_count"], 1)
                self.assertEqual(model.errors, 1)

    def test_plain_callable_invalid_metadata_is_rejected_before_pending_ops(self):
        for bad_metadata in (float("nan"), object()):
            def model(payload):
                return {"ops": [OPS[2]] * 6, "usage": {"bad": bad_metadata}}
            with tempfile.TemporaryDirectory() as tmp:
                result = run_adaptive_search(ToyOracle(), "harness-frozen", 0, 4, 2, tmp, model=model)
                self.assertEqual(result["completed_attempts"], 4)
                self.assertEqual(result["invalid_excluding_warm"], 2)
                self.assertEqual(result["model_call_count"], 2)

    def test_local_model_manifest_detects_changed_weights_and_tokenizer(self):
        manifest = getattr(model_adapters, "local_model_manifest", None)
        self.assertTrue(callable(manifest), "Local HF identity needs a model-content manifest")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text('{"model_type":"fixture"}')
            (path / "tokenizer.json").write_text('{"fixture":1}')
            (path / "model.safetensors").write_bytes(b"FAKE TEST WEIGHTS v1")
            initial = manifest(path)
            (path / "model.safetensors").write_bytes(b"FAKE TEST WEIGHTS v2")
            changed_weights = manifest(path)
            self.assertNotEqual(initial, changed_weights)
            (path / "tokenizer.json").write_text('{"fixture":2}')
            self.assertNotEqual(changed_weights, manifest(path))

    def test_finalization_rejects_partial_declared_batch_before_test_access(self):
        # This is a batch boundary check, not a search-quality experiment.
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location("finalization_under_test", ROOT / "scripts/finalize_results.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row = {"method": "harness", "seed": 0, "complete": True, "completed_attempts": 2,
                   "budget_attempts": 2, "data_sha256": "f" * 64, "search_hp": "200",
                   "dataset_for_search": "cifar10-valid", "winner_ops": [OPS[1]] * 6}
            (folder / "config.json").write_text(json.dumps({"methods": ["harness", "harness-frozen"], "seeds": [0]}))
            (folder / "locked_winners.json").write_text(json.dumps([row]))
            oracle = ToyOracle()
            with patch.object(sys, "argv", ["finalize_results.py", str(folder), "--data", "/unused"]), \
                 patch.object(module, "load_oracle", return_value=oracle):
                with self.assertRaises(ValueError):
                    module.main()
            self.assertEqual(oracle.test_calls, 0)


if __name__ == "__main__":
    unittest.main()
