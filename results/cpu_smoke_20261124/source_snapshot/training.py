"""Measured LM backends. CPU fixture results are never Parameter Golf scores.

The GPU adapter runs a pinned, self-contained upstream script in an isolated run
directory. It requires prepared, disjoint development/confirmation data splits.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "f5c079314c4877fbb0af378c0abade5a8ca33d3a"
ARTIFACT_LIMIT_BYTES = 16_000_000
RESULT_SCHEMA = "golfbench.measurement.v1"
_VERIFIED_FILE_HASHES: dict[tuple, str] = {}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _verified_hash(path: Path) -> str:
    """Re-use a full hash only while file identity, size, mtime and ctime agree."""
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    key = (str(resolved), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _VERIFIED_FILE_HASHES:
        _VERIFIED_FILE_HASHES[key] = sha256_file(resolved)
    return _VERIFIED_FILE_HASHES[key]


def _verify_manifest(data_dir: Path, backend: str) -> tuple[dict, list[dict]]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    expected = manifest.get("dataset_fingerprint")
    if expected != fingerprint({k: v for k, v in manifest.items() if k != "dataset_fingerprint"}):
        raise ValueError("Data manifest fingerprint mismatch")
    if not manifest.get("documents_disjoint"):
        raise ValueError("Prepared data must identify its split isolation protocol")
    verified = []
    if backend == "cpu_tiny_lm":
        if manifest.get("source") != "authored_test_fixture":
            raise ValueError("CPU backend accepts the explicitly authored fixture only")
        sets = []
        for split in ["train", "search", "confirm"]:
            item = manifest["splits"][split]
            path = data_dir / item["file"]
            if _verified_hash(path) != item["sha256"]:
                raise ValueError(f"Data hash mismatch: {split}")
            sets.append(set(item["document_ids"]))
            verified.append({"role": split, "sha256": item["sha256"]})
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("Fixture document IDs overlap")
    else:
        if manifest.get("source") != "parameter_golf_fineweb_partition":
            raise ValueError("GPU backend requires prepared FineWeb partition manifest")
        tokenizer = Path(manifest["tokenizer_path"])
        if _verified_hash(tokenizer) != manifest["tokenizer_sha256"]:
            raise ValueError("Tokenizer hash mismatch")
        verified.append({"role": "tokenizer", "sha256": manifest["tokenizer_sha256"]})
        entries = manifest["splits"]["train"]["files"]
        if fingerprint(entries) != manifest["splits"]["train"]["sha256"]:
            raise ValueError("Training shard manifest mismatch")
        names = {entry["name"] for entry in entries}
        for entry in entries:
            source = Path(entry["path"]).resolve(strict=True)
            if source.stat().st_size != entry["bytes"] or _verified_hash(source) != entry["sha256"]:
                raise ValueError(f"Training shard hash mismatch: {entry['name']}")
            verified.append({"role": "train", "name": entry["name"], "sha256": entry["sha256"]})
        for split in ["search", "confirm"]:
            item = manifest["splits"][split]
            path = data_dir / item["file"]
            if _verified_hash(path) != item["sha256"]:
                raise ValueError(f"Data hash mismatch: {split}")
            directory = data_dir / split
            if {p.name for p in directory.glob("fineweb_train_*.bin")} != names:
                raise ValueError(f"Unexpected training shard set in {split}")
            if {p.resolve() for p in directory.glob("fineweb_val_*.bin")} != {path.resolve()}:
                raise ValueError(f"Unexpected validation shard set in {split}")
            for entry in entries:
                link = directory / entry["name"]
                if link.resolve(strict=True) != Path(entry["path"]).resolve(strict=True):
                    raise ValueError(f"Training data link changed: {link}")
            verified.append({"role": split, "sha256": item["sha256"]})
        a, b = manifest["splits"]["search"], manifest["splits"]["confirm"]
        if not (a["document_index_start"] == 0 < a["document_index_end_exclusive"] == b["document_index_start"] < b["document_index_end_exclusive"] == manifest["source_validation_documents"]):
            raise ValueError("Validation document ranges are not disjoint")
    if manifest["splits"]["search"]["sha256"] == manifest["splits"]["confirm"]["sha256"]:
        raise ValueError("Search and confirmation data hashes must differ")
    return manifest, verified


def execution_context(backend: str, data_dir: str | Path, upstream_dir: str | Path | None = None) -> dict:
    """Immutable execution identity for batch resume and table consistency."""
    import torch
    if backend not in {"cpu_tiny_lm", "parameter_golf", "upstream_single_gpu"}:
        raise ValueError("Unknown training backend")
    normalized = "cpu_tiny_lm" if backend == "cpu_tiny_lm" else "upstream_single_gpu"
    manifest, verified = _verify_manifest(Path(data_dir).resolve(), normalized)
    sources = {"training.py": sha256_file(__file__), "train.py": sha256_file(ROOT / "scripts/golf/train.py"),
               "prepare_data.py": sha256_file(ROOT / "scripts/golf/prepare_data.py")}
    if normalized == "upstream_single_gpu":
        folder = Path(upstream_dir or ROOT / "vendor/parameter_golf")
        pin = json.loads((folder / "UPSTREAM.json").read_text())
        if pin["commit"] != UPSTREAM_COMMIT:
            raise ValueError("Unexpected upstream commit")
        for name, value in pin["files"].items():
            if _verified_hash(folder / name) != value:
                raise ValueError(f"Vendored file differs from pin: {name}")
        sources["upstream_train_gpt.py"] = sha256_file(folder / "train_gpt.py")
        sources["effective_train_gpt.py"] = hashlib.sha256(patch_upstream((folder / "train_gpt.py").read_text()).encode()).hexdigest()
    software = {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "platform": platform.platform()}
    return {"schema": "golfbench.execution.v1", "backend": "cpu_tiny_lm" if normalized == "cpu_tiny_lm" else "parameter_golf",
            "implementation_sha256": fingerprint(sources), "source_hashes": sources, "software": software,
            "data_manifest_sha256": sha256_file(Path(data_dir) / "manifest.json"),
            "dataset_fingerprint": manifest["dataset_fingerprint"], "verified_data": verified}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def prepare_fixture(output_dir: str | Path) -> dict:
    """Make deterministic, authored text documents; this is synthetic test data.

    Document IDs and complete document strings are disjoint across the splits.
    Shared grammatical templates are intentional. This fixture tests plumbing,
    not natural-language performance or transfer to FineWeb.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    subjects = ["the astronomer", "the gardener", "the student", "the engineer", "the artist", "the sailor"]
    objects = ["a quiet river", "a small garden", "a distant star", "a wooden bridge", "a silver clock", "a mountain path"]
    actions = ["carefully measured", "patiently described", "quietly observed", "repeatedly examined", "accurately recorded"]
    endings = [
        "The first observation suggested a pattern, and the second observation tested it.",
        "A simple explanation was useful only when the next measurement agreed with it.",
        "They kept the original notes so that another person could repeat the experiment.",
        "The result changed after the temperature fell, but the order of events stayed the same.",
    ]
    split_ids = {"train": range(0, 160), "search": range(160, 192), "confirm": range(192, 224)}
    manifest = {"schema": "golfbench.data.v1", "source": "authored_test_fixture", "tokenizer": "identity_utf8_bytes_256", "documents_disjoint": True, "splits": {}}
    all_docs: set[str] = set()
    for split, ids in split_ids.items():
        docs = []
        for i in ids:
            rng = random.Random(94123 + i)
            sentence = f"Record {i:04d}: {rng.choice(subjects)} {rng.choice(actions)} {rng.choice(objects)}. "
            sentence += rng.choice(endings)
            sentence += f" The notebook contained {rng.randint(2, 19)} measurements from day {i + 1}.\n"
            if sentence in all_docs:
                raise RuntimeError("Fixture document overlap")
            all_docs.add(sentence)
            docs.append(sentence)
        path = out / f"{split}.txt"
        text = "\n".join(docs)
        if path.exists() and path.read_text() != text:
            raise ValueError(f"Refusing to overwrite different fixture: {path}")
        path.write_text(text, encoding="utf-8")
        manifest["splits"][split] = {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size, "document_ids": list(ids)}
    manifest["dataset_fingerprint"] = fingerprint(manifest)
    _write_json(out / "manifest.json", manifest)
    return manifest


def _config(config: dict, backend: str) -> dict:
    c = dict(config)
    if "train_seq_len" in c:
        if "seq_len" in c and c["seq_len"] != c["train_seq_len"]:
            raise ValueError("Conflicting train_seq_len and seq_len")
        c["seq_len"] = c["train_seq_len"]
    defaults = {"seq_len": 64 if backend == "cpu_tiny_lm" else 1024, "vocab_size": 256 if backend == "cpu_tiny_lm" else 1024,
                "mlp_mult": 2, "train_batch_tokens": 256 if backend == "cpu_tiny_lm" else 8192,
                "warmdown_fraction": 0.3, "lr_scale": 1.0, "tied_embeddings": True}
    for key, value in defaults.items():
        c.setdefault(key, value)
    c.setdefault("eval_seq_len", c["seq_len"])
    if c["eval_seq_len"] != c["seq_len"]:
        raise ValueError("This pinned backend requires equal, fixed train/eval sequence lengths")
    for key in ["num_layers", "model_dim", "num_heads", "num_kv_heads", "seq_len", "vocab_size", "mlp_mult", "train_batch_tokens"]:
        if not isinstance(c.get(key), int) or isinstance(c[key], bool) or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(c["tied_embeddings"], bool):
        raise ValueError("tied_embeddings must be a boolean")
    if c["model_dim"] % c["num_heads"] or c["num_heads"] % c["num_kv_heads"]:
        raise ValueError("Invalid attention head dimensions")
    if (c["model_dim"] // c["num_heads"]) % 2:
        raise ValueError("Head dimension must be even")
    if not 0 < float(c["warmdown_fraction"]) <= 1 or not math.isfinite(float(c["lr_scale"])) or float(c["lr_scale"]) <= 0:
        raise ValueError("Invalid learning-rate scale or warmdown fraction")
    accum = 1 if backend == "cpu_tiny_lm" else 8
    if c["train_batch_tokens"] % (accum * c["seq_len"]):
        raise ValueError("train_batch_tokens must be divisible by accumulation * seq_len")
    if backend == "cpu_tiny_lm" and (c["vocab_size"] != 256 or c["num_heads"] != c["num_kv_heads"]):
        raise ValueError("CPU fixture requires byte vocabulary 256 and ordinary multihead attention")
    return c


def _environment(torch, backend: str) -> tuple[dict, dict]:
    if backend == "cpu_tiny_lm":
        cpu_model = platform.processor() or platform.machine()
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            matches = re.findall(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE)
            if matches:
                cpu_model = matches[0]
        hardware = {"device_type": "cpu", "device_name": cpu_model, "threads": 1, "architecture": platform.machine()}
    else:
        props = torch.cuda.get_device_properties(0)
        hardware = {"device_type": "cuda", "device_name": props.name, "memory_bytes": props.total_memory,
                    "compute_capability": [props.major, props.minor], "gpu_count": 1}
    software = {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "platform": platform.platform()}
    return hardware, software


def _load_prepared_data(data_dir: Path, split: str, backend: str, seq_len: int) -> tuple[dict, str]:
    manifest, _ = _verify_manifest(data_dir, backend)
    if split not in {"search", "confirm"}:
        raise ValueError("split must be search or confirm")
    if not manifest.get("documents_disjoint"):
        raise ValueError("Prepared data must declare verified disjoint documents")
    if manifest["splits"]["search"]["sha256"] == manifest["splits"]["confirm"]["sha256"]:
        raise ValueError("Search and confirmation data hashes must differ")
    if backend == "cpu_tiny_lm":
        if manifest.get("source") != "authored_test_fixture":
            raise ValueError("CPU backend accepts the explicitly authored fixture only")
        for label in ["train", split]:
            item = manifest["splits"][label]
            if sha256_file(data_dir / item["file"]) != item["sha256"]:
                raise ValueError(f"Data hash mismatch: {label}")
    else:
        if manifest.get("source") != "parameter_golf_fineweb_partition":
            raise ValueError("GPU backend requires prepared FineWeb partition manifest")
        item = manifest["splits"][split]
        if sha256_file(data_dir / item["file"]) != item["sha256"]:
            raise ValueError(f"Data hash mismatch: {split}")
        tokenizer = Path(manifest["tokenizer_path"])
        if sha256_file(tokenizer) != manifest["tokenizer_sha256"]:
            raise ValueError("Tokenizer hash mismatch")
    protocol = {"dataset": manifest["dataset_fingerprint"], "source": manifest["source"], "split": split,
                "evaluation_sha256": manifest["splits"][split]["sha256"], "tokenizer": manifest.get("tokenizer_sha256", manifest.get("tokenizer")),
                "eval_seq_len": seq_len, "windowing": "nonoverlapping_next_token_drop_incomplete_tail", "metric": "bits_per_byte"}
    return manifest, fingerprint(protocol)


def _cpu_train(c: dict, seed: int, out: Path, data_dir: Path, split: str, budget: float, max_steps: int) -> dict:
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            d = c["model_dim"]
            self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.mlp = nn.Sequential(nn.Linear(d, c["mlp_mult"] * d), nn.GELU(), nn.Linear(c["mlp_mult"] * d, d))

        def forward(self, x):
            b, t, d = x.shape
            q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
            shape = (b, t, c["num_heads"], d // c["num_heads"])
            q, k, v = [a.reshape(shape).transpose(1, 2) for a in (q, k, v)]
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
            return x + self.mlp(self.norm2(x))

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(256, c["model_dim"])
            self.pos = nn.Embedding(c["seq_len"], c["model_dim"])
            self.blocks = nn.ModuleList([Block() for _ in range(c["num_layers"])])
            self.norm = nn.LayerNorm(c["model_dim"])
            self.head = None if c["tied_embeddings"] else nn.Linear(c["model_dim"], 256, bias=False)
            for parameter in self.parameters():
                if parameter.ndim > 1:
                    nn.init.normal_(parameter, mean=0.0, std=0.02)

        def forward(self, x):
            x = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            return F.linear(x, self.tok.weight) if self.head is None else self.head(x)

    train_tokens = torch.tensor(list((data_dir / "train.txt").read_bytes()), dtype=torch.long)
    eval_tokens = torch.tensor(list((data_dir / f"{split}.txt").read_bytes()), dtype=torch.long)
    seq = c["seq_len"]
    if min(len(train_tokens), len(eval_tokens)) <= seq:
        raise ValueError("Text split is shorter than the sequence length")
    model = TinyLM()
    base_lr = float(c.get("base_lr", 0.003)) * float(c["lr_scale"])
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    batch = c["train_batch_tokens"] // seq
    generator = torch.Generator().manual_seed(seed + 100_003)
    train_log = []
    train_start = time.perf_counter()
    longest_step = 0.0
    for step in range(max_steps):
        elapsed = time.perf_counter() - train_start
        if elapsed >= budget or (step > 0 and elapsed + 1.2 * longest_step >= budget):
            break
        step_start = time.perf_counter()
        starts = torch.randint(len(train_tokens) - seq, (batch,), generator=generator)
        offsets = starts[:, None] + torch.arange(seq)[None]
        x, y = train_tokens[offsets], train_tokens[offsets + 1]
        progress = step / max(1, max_steps - 1)
        wd = float(c["warmdown_fraction"])
        multiplier = min(1.0, max(0.0, (1.0 - progress) / wd))
        for group in opt.param_groups:
            group["lr"] = base_lr * multiplier
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        duration = time.perf_counter() - step_start
        longest_step = max(longest_step, duration)
        train_log.append({"step": step + 1, "loss": float(loss.detach()), "lr": base_lr * multiplier, "seconds": time.perf_counter() - train_start})
    train_seconds = time.perf_counter() - train_start
    if not train_log:
        raise RuntimeError("No optimizer step completed within the training budget")

    def evaluate() -> float:
        model.eval()
        usable = ((len(eval_tokens) - 1) // seq) * seq
        x = eval_tokens[:usable].reshape(-1, seq)
        y = eval_tokens[1:usable + 1].reshape(-1, seq)
        nll = 0.0
        with torch.inference_mode():
            for start in range(0, len(x), 16):
                logits = model(x[start:start + 16])
                nll += float(F.cross_entropy(logits.reshape(-1, 256), y[start:start + 16].reshape(-1), reduction="sum"))
        return nll / usable / math.log(2)

    eval_start = time.perf_counter()
    val_bpb = evaluate()
    first_eval_seconds = time.perf_counter() - eval_start
    export_start = time.perf_counter()
    state = {}
    for name, weight in model.state_dict().items():
        scale = float(weight.abs().max()) / 127.0 or 1.0
        state[name] = {"q": torch.round(weight / scale).clamp(-127, 127).to(torch.int8), "scale": scale}
    buf = io.BytesIO()
    torch.save({"format": "golfbench_cpu_int8_v1", "config": c, "state": state}, buf)
    artifact = out / "model.int8.ptz"
    artifact.write_bytes(zlib.compress(buf.getvalue(), level=9))
    # Re-read the exact bytes that are counted toward the artifact.
    loaded = torch.load(io.BytesIO(zlib.decompress(artifact.read_bytes())), map_location="cpu", weights_only=True)
    model.load_state_dict({name: item["q"].float() * item["scale"] for name, item in loaded["state"].items()}, strict=True)
    export_seconds = time.perf_counter() - export_start
    eval_start = time.perf_counter()
    quantized_bpb = evaluate()
    evaluation_seconds = first_eval_seconds + time.perf_counter() - eval_start
    code_paths = [Path(__file__), ROOT / "scripts/golf/train.py"]
    code_bytes = sum(path.stat().st_size for path in code_paths if path.is_file())
    (out / "train.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in train_log))
    return {"train_seconds": train_seconds, "evaluation_seconds": evaluation_seconds, "export_seconds": export_seconds,
            "compile_warmup_seconds": 0.0, "steps": len(train_log), "tokens_seen": len(train_log) * c["train_batch_tokens"],
            "val_bpb": val_bpb, "quantized_bpb": quantized_bpb, "artifact_bytes": artifact.stat().st_size + code_bytes,
            "model_bytes": artifact.stat().st_size, "code_bytes": code_bytes, "parameter_count": sum(p.numel() for p in model.parameters()),
            "model_sha256": sha256_file(artifact), "artifact_file": artifact.name,
            "stop_reason": "max_steps" if len(train_log) == max_steps else "time_guard", "optimizer": "AdamW",
            "learning_rate_schedule": "linear_step_fraction_warmdown", "data_protocol": "authored_byte_fixture",
            "status": "completed", "budget_compliant": train_seconds <= budget,
            "budget_tolerance_seconds": 0.0, "budget_overshoot_seconds": max(0.0, train_seconds - budget)}


def patch_upstream(source: str) -> str:
    """Apply narrowly scoped, checked patches; never edit the vendored source."""
    start = source.index("    def lr_mul(step: int, elapsed_ms: float) -> float:")
    end = source.index("    # Warmup primes", start)
    source = source[:start] + '''    def lr_mul(step: int, elapsed_ms: float) -> float:
        fraction = float(os.environ["GOLFBENCH_WARMDOWN_FRACTION"])
        if max_wallclock_ms is None:
            raise ValueError("Golfbench requires an explicit positive training time budget")
        return min(1.0, max(0.0, (max_wallclock_ms - elapsed_ms) / (fraction * max_wallclock_ms)))

''' + source[end:]
    replacements = {
        "    # Warmup primes": "    _golfbench_warmup_start = time.perf_counter()\n    # Warmup primes",
        "    training_time_ms = 0.0": "    torch.cuda.synchronize()\n    log0(f'golfbench_warmup_seconds:{time.perf_counter() - _golfbench_warmup_start:.8f}')\n    training_time_ms = 0.0",
        "        approx_training_time_ms =": "        torch.cuda.synchronize()\n        approx_training_time_ms =",
        "    quant_obj, quant_stats =": "    _golfbench_export_start = time.perf_counter()\n    quant_obj, quant_stats =",
        "    t_qeval = time.perf_counter()": "    log0(f'golfbench_export_seconds:{time.perf_counter() - _golfbench_export_start:.8f}')\n    t_qeval = time.perf_counter()",
        '    log0(f"final_int8_zlib_roundtrip_exact': '    log0(f"golfbench_train_seconds:{training_time_ms / 1000.0:.8f}")\n    log0(f"golfbench_steps:{step}")\n    log0(f"golfbench_float_bpb:{val_bpb:.8f}")\n    log0(f"final_int8_zlib_roundtrip_exact',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f"Unexpected pinned source structure at {old[:55]!r}")
        source = source.replace(old, new, 1)
    # Every evaluation pass contributes to a separate measured evaluation total.
    source = source.replace("def eval_val(\n", "def _golfbench_original_eval_val(\n", 1)
    marker = "# POST-TRAINING QUANTIZATION"
    wrapper = '''_golfbench_eval_seconds = 0.0
def eval_val(*args, **kwargs):
    global _golfbench_eval_seconds
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = _golfbench_original_eval_val(*args, **kwargs)
    torch.cuda.synchronize()
    _golfbench_eval_seconds += time.perf_counter() - started
    return result

'''
    source = source.replace(marker, wrapper + marker, 1)
    source = source.replace('    log0(f"golfbench_steps:{step}")', '    log0(f"golfbench_steps:{step}")\n    log0(f"golfbench_evaluation_seconds:{_golfbench_eval_seconds:.8f}")', 1)
    compile(source, "train_gpt.py", "exec")
    return source


def parse_upstream_log(text: str) -> dict:
    """Require exact final roundtrip, measured timing and measured artifact size."""
    def number(pattern: str, convert=float):
        values = re.findall(pattern, text)
        if len(values) != 1:
            raise ValueError(f"Expected exactly one result matching {pattern!r}, found {len(values)}")
        value = convert(values[0])
        if not math.isfinite(value):
            raise ValueError("Non-finite upstream metric")
        return value
    return {"quantized_bpb": number(r"(?m)^final_int8_zlib_roundtrip_exact val_loss:\S+ val_bpb:(\S+)\s*$"),
            "val_bpb": number(r"(?m)^golfbench_float_bpb:(\S+)\s*$"),
            "train_seconds": number(r"(?m)^golfbench_train_seconds:(\S+)\s*$"),
            "steps": number(r"(?m)^golfbench_steps:(\d+)\s*$", int),
            "compile_warmup_seconds": number(r"(?m)^golfbench_warmup_seconds:(\S+)\s*$"),
            "evaluation_seconds": number(r"(?m)^golfbench_evaluation_seconds:(\S+)\s*$"),
            "export_seconds": number(r"(?m)^golfbench_export_seconds:(\S+)\s*$"),
            "artifact_bytes": number(r"(?m)^Total submission size int8\+zlib: (\d+) bytes\s*$", int)}


def _gpu_train(c: dict, seed: int, out: Path, data_dir: Path, manifest: dict, split: str, budget: float, upstream_dir: Path) -> dict:
    source_path = upstream_dir / "train_gpt.py"
    pin = json.loads((upstream_dir / "UPSTREAM.json").read_text())
    if pin["commit"] != UPSTREAM_COMMIT or sha256_file(source_path) != pin["files"]["train_gpt.py"]:
        raise ValueError("Upstream source does not match the pinned commit and SHA256")
    source = patch_upstream(source_path.read_text())
    script = out / "train_gpt.py"
    script.write_text(source)
    env = os.environ.copy()
    inherited_configuration_keys = set(re.findall(r'''os\.environ\.get\(\s*["']([^"']+)["']''', source))
    for key in inherited_configuration_keys:
        env.pop(key, None)
    # Force a single process even when the caller runs under a distributed launcher.
    for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        env.pop(key, None)
    values = {"DATA_PATH": str((data_dir / split).resolve()), "TOKENIZER_PATH": manifest["tokenizer_path"], "RUN_ID": "run",
              "SEED": seed, "MAX_WALLCLOCK_SECONDS": budget, "ITERATIONS": 1_000_000,
              "VOCAB_SIZE": c["vocab_size"], "NUM_LAYERS": c["num_layers"], "MODEL_DIM": c["model_dim"],
              "NUM_HEADS": c["num_heads"], "NUM_KV_HEADS": c["num_kv_heads"], "MLP_MULT": c["mlp_mult"],
              "TRAIN_SEQ_LEN": c["seq_len"], "TRAIN_BATCH_TOKENS": c["train_batch_tokens"], "VAL_BATCH_SIZE": 524288,
              "TIE_EMBEDDINGS": int(c["tied_embeddings"]), "GOLFBENCH_WARMDOWN_FRACTION": c["warmdown_fraction"],
              "VAL_LOSS_EVERY": 0, "TRAIN_LOG_EVERY": 50, "WARMUP_STEPS": 2,
              "MATRIX_LR": 0.04 * c["lr_scale"], "SCALAR_LR": 0.04 * c["lr_scale"], "EMBED_LR": 0.6 * c["lr_scale"],
              "HEAD_LR": 0.008 * c["lr_scale"], "TIED_EMBED_LR": 0.05 * c["lr_scale"]}
    env.update({key: str(value) for key, value in values.items()})
    _write_json(out / "launch.json", {"argv": [sys.executable, "train_gpt.py"], "overrides": values,
                "cleared_upstream_environment_keys": sorted(inherited_configuration_keys),
                "default_values_source": "pinned_self_contained_script", "script_sha256": sha256_file(script), "upstream_commit": UPSTREAM_COMMIT})
    with (out / "stdout.log").open("w") as log:
        completed = subprocess.run([sys.executable, "train_gpt.py"], cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=budget + 7200, check=False)
    if completed.returncode:
        raise RuntimeError(f"Upstream training exited with {completed.returncode}; see stdout.log")
    measured = parse_upstream_log((out / "stdout.log").read_text())
    artifact = out / "final_model.int8.ptz"
    actual_bytes = script.stat().st_size + artifact.stat().st_size
    if measured["artifact_bytes"] != actual_bytes:
        raise ValueError("Logged submission size differs from actual script + compressed model bytes")
    measured.update({"model_bytes": artifact.stat().st_size, "code_bytes": script.stat().st_size,
                     "model_sha256": sha256_file(artifact), "artifact_file": artifact.name,
                     "effective_script_sha256": sha256_file(script), "tokens_seen": measured["steps"] * c["train_batch_tokens"],
                     "optimizer": "upstream_Muon_Adam", "learning_rate_schedule": "linear_wallclock_fraction_warmdown",
                     "stop_reason": "target_reached_after_current_step", "data_protocol": "fineweb_disjoint_validation_partition",
                     "status": "completed", "budget_compliant": measured["train_seconds"] <= budget + 60.0,
                     "time_policy": "approximate_target", "artifact_compliant": actual_bytes <= ARTIFACT_LIMIT_BYTES, "budget_tolerance_seconds": 60.0,
                     "budget_overshoot_seconds": max(0.0, measured["train_seconds"] - budget)})
    return measured


def run_training(config: dict, candidate_id: str, training_seed: int, output_dir: str | Path,
                 backend: str = "cpu_tiny_lm", train_budget_seconds: float = 10.0, max_steps: int | None = 40,
                 split: str = "search", data_dir: str | Path | None = None,
                 upstream_dir: str | Path | None = None) -> dict:
    """Train one candidate and persist strict JSON plus the measured artifact.

    CPU uses true gradient training on an authored byte-text fixture. GPU results
    require CUDA and prepared FineWeb data; absence of CUDA never creates scores.
    """
    started = time.perf_counter()
    if backend == "parameter_golf":
        backend = "upstream_single_gpu"
    if backend not in {"cpu_tiny_lm", "upstream_single_gpu"}:
        raise ValueError("backend must be cpu_tiny_lm or upstream_single_gpu")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", candidate_id):
        raise ValueError("Invalid candidate_id")
    if not isinstance(training_seed, int) or isinstance(training_seed, bool) or training_seed < 0:
        raise ValueError("training_seed must be a nonnegative integer")
    budget = float(train_budget_seconds)
    if backend != "cpu_tiny_lm" and max_steps is None:
        max_steps = 1_000_000
    if not math.isfinite(budget) or budget <= 0 or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("Positive finite budget and positive integer max_steps required")
    c = _config(config, backend)
    if split not in {"search", "confirm"}:
        raise ValueError("split must be search or confirm")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise FileExistsError(f"Result already exists: {out / 'result.json'}")
    result = {"schema": RESULT_SCHEMA, "candidate_id": candidate_id, "config": dict(config), "actual_config": c,
              "training_seed": training_seed, "backend": "parameter_golf" if backend == "upstream_single_gpu" else backend, "split": split, "train_budget_seconds": budget,
              "max_steps": max_steps if backend == "cpu_tiny_lm" else None, "status": "running", "train_seconds": None,
              "val_bpb": None, "quantized_bpb": None, "artifact_bytes": None, "evaluation_fingerprint": None,
              "device_class": "cpu" if backend == "cpu_tiny_lm" else "cuda", "hardware_fingerprint": None,
              "provenance": {"implementation_sha256": sha256_file(__file__), "upstream_commit": UPSTREAM_COMMIT if backend == "upstream_single_gpu" else None,
                             "claim_scope": "pipeline_smoke_only" if backend == "cpu_tiny_lm" else "single_gpu_proxy_not_official_leaderboard"}}
    try:
        import torch
        if backend == "upstream_single_gpu" and not torch.cuda.is_available():
            result.update(status="unavailable", error="CUDA is unavailable; GPU training was not run")
        else:
            hardware, software = _environment(torch, backend)
            result.update(hardware=hardware, software=software, hardware_fingerprint=fingerprint(hardware), software_fingerprint=fingerprint(software))
            result["device_class"] = "cpu" if backend == "cpu_tiny_lm" else hardware["device_name"]
            if data_dir is None:
                if backend != "cpu_tiny_lm":
                    raise ValueError("GPU training requires --data-dir with a prepared split manifest")
                dataset = out / "fixture"
                prepare_fixture(dataset)
            else:
                dataset = Path(data_dir).resolve()
            manifest, eval_fp = _load_prepared_data(dataset, split, backend, c["seq_len"])
            context = execution_context(backend, dataset, upstream_dir)
            result["execution_fingerprint"] = fingerprint(context)
            result["provenance"]["execution_context"] = context
            result["evaluation_fingerprint"] = eval_fp
            result["provenance"].update(data_source=manifest["source"], dataset_fingerprint=manifest["dataset_fingerprint"],
                                        training_data_sha256=manifest["splits"]["train"]["sha256"], evaluation_data_sha256=manifest["splits"][split]["sha256"])
            if backend == "cpu_tiny_lm":
                result.update(_cpu_train(c, training_seed, out, dataset, split, budget, max_steps))
            else:
                result.update(_gpu_train(c, training_seed, out, dataset, manifest, split, budget, Path(upstream_dir or ROOT / "vendor/parameter_golf")))
    except Exception as exc:
        status = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "oom" if "out of memory" in str(exc).lower() else "failed"
        result.update(status=status, error=f"{type(exc).__name__}: {exc}", val_bpb=None, quantized_bpb=None)
        # Failed jobs still consume budget. This is an elapsed upper bound, not a
        # claimed measurement of optimizer time; keep that distinction explicit.
        result["train_seconds"] = time.perf_counter() - started
        result["failed_train_time_basis"] = "elapsed_run_upper_bound"
        result["budget_compliant"] = False
    result["end_to_end_seconds"] = time.perf_counter() - started
    if result["train_seconds"] is not None:
        result["overhead_seconds"] = max(0.0, result["end_to_end_seconds"] - result["train_seconds"])
    _write_json(out / "result.json", result)
    return result
