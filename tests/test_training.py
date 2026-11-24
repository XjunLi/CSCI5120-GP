"""Check actual training, score provenance, split isolation and adapter parsing."""
from pathlib import Path
import json
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from golfbench.training import (ROOT, _config, parse_upstream_log, patch_upstream,
                                prepare_fixture, run_training, sha256_file, execution_context, fingerprint)
from golfbench.space import make_manifest, make_space
from golfbench.table import validate_record


def tiny_config():
    return make_space("cpu_smoke")["candidates"][0]["config"]


def test_fixture_documents_and_hashes_are_disjoint(tmp_path):
    m = prepare_fixture(tmp_path)
    assert m["source"] == "authored_test_fixture"
    assert len({x["sha256"] for x in m["splits"].values()}) == 3
    sets = [set(x["document_ids"]) for x in m["splits"].values()]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert prepare_fixture(tmp_path) == m
    for item in m["splits"].values():
        assert sha256_file(tmp_path / item["file"]) == item["sha256"]


def test_true_training_roundtrip_and_root_contract(tmp_path):
    pytest.importorskip("torch")
    data = tmp_path / "data"
    prepare_fixture(data)
    job = make_manifest(make_space("cpu_smoke"), "CPU")["jobs"][0]
    job = dict(job, max_steps=5)
    args = dict(config=job["config"], candidate_id=job["candidate_id"], training_seed=job["training_seed"],
                train_budget_seconds=job["train_budget_seconds"], max_steps=5, data_dir=data)
    a = run_training(**args, output_dir=tmp_path / "a")
    b = run_training(**args, output_dir=tmp_path / "b")
    assert a["status"] == "completed", a
    assert a["steps"] == 5
    validate_record(a, job)
    assert a["config"] == job["config"]
    assert a["quantized_bpb"] == b["quantized_bpb"]
    assert 0 < a["quantized_bpb"] < 20
    assert a["artifact_bytes"] == a["model_bytes"] + a["code_bytes"]
    assert (tmp_path / "a" / a["artifact_file"]).stat().st_size == a["model_bytes"]
    assert a["model_sha256"] == sha256_file(tmp_path / "a" / a["artifact_file"])
    assert a["end_to_end_seconds"] >= a["train_seconds"] > 0
    assert a["provenance"]["claim_scope"] == "pipeline_smoke_only"
    assert a["execution_fingerprint"] == fingerprint(execution_context("cpu_tiny_lm", data))
    rows = [json.loads(line) for line in (tmp_path / "a/train.jsonl").read_text().splitlines()]
    assert rows[-1]["loss"] < rows[0]["loss"]


def test_hidden_split_changes_fingerprint(tmp_path):
    pytest.importorskip("torch")
    prepare_fixture(tmp_path / "data")
    args = dict(config=tiny_config(), candidate_id="check", training_seed=0, max_steps=1, data_dir=tmp_path / "data")
    a = run_training(**args, output_dir=tmp_path / "search", split="search")
    b = run_training(**args, output_dir=tmp_path / "confirm", split="confirm")
    assert a["status"] == b["status"] == "completed"
    assert a["evaluation_fingerprint"] != b["evaluation_fingerprint"]
    assert a["hardware_fingerprint"] == b["hardware_fingerprint"]


def test_corrupted_data_never_produces_score(tmp_path):
    pytest.importorskip("torch")
    prepare_fixture(tmp_path / "data")
    (tmp_path / "data/search.txt").write_text("wrong data")
    result = run_training(tiny_config(), "bad-data", 0, tmp_path / "run", max_steps=1, data_dir=tmp_path / "data")
    assert result["status"] == "failed"
    assert "hash mismatch" in result["error"]
    assert result["quantized_bpb"] is None


def test_manifest_corruption_blocks_resume_identity(tmp_path):
    pytest.importorskip("torch")
    prepare_fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"] = "changed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        execution_context("cpu_tiny_lm", tmp_path)


def test_refuses_overwrite_and_invalid_configuration(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run_training(tiny_config(), "existing", 0, tmp_path)
    with pytest.raises(ValueError):
        _config(dict(tiny_config(), model_dim=33), "cpu_tiny_lm")
    with pytest.raises(ValueError):
        _config(dict(tiny_config(), eval_seq_len=128), "cpu_tiny_lm")


def test_unavailable_gpu_has_no_scores(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = make_space("gpu")["candidates"][0]["config"]
    result = run_training(config, "gpu-unavailable", 0, tmp_path, backend="parameter_golf", train_budget_seconds=600, max_steps=None)
    assert result["status"] == "unavailable"
    assert result["backend"] == "parameter_golf"
    assert result["max_steps"] is None
    assert result["quantized_bpb"] is None and result["val_bpb"] is None


def test_upstream_pin_and_patch_compile():
    root = ROOT / "vendor/parameter_golf"
    pin = json.loads((root / "UPSTREAM.json").read_text())
    for name, expected in pin["files"].items():
        assert sha256_file(root / name) == expected
    source = patch_upstream((root / "train_gpt.py").read_text())
    compile(source, "train_gpt.py", "exec")
    assert "GOLFBENCH_WARMDOWN_FRACTION" in source
    assert "golfbench_evaluation_seconds" in source
    assert "torch.cuda.synchronize()\n        approx_training_time_ms" in source


def exact_log():
    return """step:100/20000 val_loss:2.9 val_bpb:1.9000 train_time:60000ms
final_int8_zlib_roundtrip val_loss:3.0000 val_bpb:2.0000 eval_time:100ms
golfbench_float_bpb:1.91234567
golfbench_train_seconds:599.9
golfbench_steps:100
golfbench_warmup_seconds:23.5
golfbench_export_seconds:1.5
golfbench_evaluation_seconds:7.9
Total submission size int8+zlib: 15000000 bytes
final_int8_zlib_roundtrip_exact val_loss:3.00001234 val_bpb:2.00004321
"""


def test_parse_requires_exact_roundtrip_and_uses_actual_bytes():
    r = parse_upstream_log(exact_log())
    assert r["quantized_bpb"] == 2.00004321
    assert r["val_bpb"] == 1.91234567
    assert r["artifact_bytes"] == 15000000
    assert r["train_seconds"] == 599.9
    with pytest.raises(ValueError):
        parse_upstream_log(exact_log().replace("final_int8_zlib_roundtrip_exact", "missing"))
    with pytest.raises(ValueError):
        parse_upstream_log(exact_log() + "final_int8_zlib_roundtrip_exact val_loss:2.9 val_bpb:1.9\n")
    with pytest.raises(ValueError):
        parse_upstream_log(exact_log().replace("2.00004321", "nan"))
