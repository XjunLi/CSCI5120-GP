"""Audited frozen-model callables for adaptive harness and multi-agent searches.

No model receives a tool, evaluator handle or benchmark path. Invalid model
outputs are returned unchanged as invalid proposals, not replaced by random NAS
candidates. Controllers may reject malformed *program patches* separately.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path


_MODEL_MANIFEST_CACHE = {}


def local_model_manifest(model_path):
    """Content identity of frozen weights and tokenizer, cached by file metadata.

    No timestamps enter the saved identity: a byte-identical copied snapshot
    remains compatible. Weight files, tokenizer files and all JSON model
    configuration are included. Python code in model folders is not executed.
    """
    path = Path(model_path).expanduser().resolve()
    files = sorted(p for p in path.iterdir() if p.is_file() and (
        p.suffix in {".json", ".safetensors", ".bin", ".model", ".tiktoken"}
        or p.name in {"vocab.txt", "merges.txt"}))
    if not files:
        raise ValueError("Local model snapshot contains no recognized model/tokenizer files")
    signature = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in files)
    cached = _MODEL_MANIFEST_CACHE.get(str(path))
    if cached and cached[0] == signature:
        return copy.deepcopy(cached[1])
    manifest = {}
    for file in files:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest[file.name] = {"bytes": file.stat().st_size, "sha256": digest.hexdigest()}
    _MODEL_MANIFEST_CACHE[str(path)] = (signature, manifest)
    return copy.deepcopy(manifest)


def parse_json_response(text):
    """Accept one object, optionally inside a single Markdown JSON fence."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        first, _, rest = text.partition("\n")
        if first.strip().lower() not in ("```", "```json"):
            raise ValueError("Expected a JSON code fence")
        text = rest.rsplit("```", 1)[0].strip()
    def reject_nonfinite(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    result = json.loads(text, parse_constant=reject_nonfinite)
    if not isinstance(result, dict):
        raise ValueError("Expected exactly one JSON object")
    json.dumps(result, allow_nan=False)  # Also catches overflow literals such as 1e999.
    return result


class SubprocessJSONModel:
    """A stateless JSON-stdin/JSON-stdout process with complete call accounting."""
    def __init__(self, command, timeout=180, model_label=None, seed=0):
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        if not self.command or timeout <= 0:
            raise ValueError("A nonempty command and positive timeout are required")
        self.timeout, self.model_label, self.seed = timeout, model_label, seed
        self.calls, self.errors, self._events = 0, 0, []

    def configuration(self):
        return {"backend": "subprocess-json", "command": self.command,
                "timeout": self.timeout, "model_label": self.model_label,
                "seed": self.seed, "resume_semantics": "adapter counters restored; external service RNG is not controlled"}

    def _invoke(self, payload):
        completed = subprocess.run(self.command, input=json.dumps(payload, allow_nan=False),
                                   text=True, encoding="utf-8", capture_output=True, timeout=self.timeout)
        if completed.returncode:
            raise RuntimeError(f"Model adapter exited {completed.returncode}: {completed.stderr[-4000:]}")
        return completed.stdout, {}

    def __call__(self, payload):
        started, raw, extra, error = time.perf_counter(), "", {}, None
        self.calls += 1
        try:
            raw, extra = self._invoke(copy.deepcopy(payload))
            result = parse_json_response(raw)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.errors += 1
            result = {"ops": None, "adapter_error": error}
        result.setdefault("model", self.model_label)
        result.update(extra)
        result["raw_model_response"] = raw
        event = {"kind": "model_call", "call_index": self.calls, "request": payload,
                 "response": result, "error": error, "seconds": time.perf_counter() - started}
        self._events.append(event)
        return result

    def drain_events(self):
        events, self._events = self._events, []
        return events

    def state_dict(self):
        return {"version": 1, "calls": self.calls, "errors": self.errors, "seed": self.seed}

    def load_state_dict(self, state):
        if state.get("version") != 1 or state["seed"] != self.seed:
            raise ValueError("Model adapter checkpoint identity mismatch")
        self.calls, self.errors = int(state["calls"]), int(state["errors"])
        self._events = []


class HuggingFaceJSONModel(SubprocessJSONModel):
    """One locally loaded, frozen causal LM, reused for all calls of one run.

    This free-text interface can generate both harness programs and peer
    critiques. It is distinct from the weight controller's constrained LoRA
    policy. Torch and Transformers are imported only if this backend is used.
    """
    def __init__(self, model_path, seed=0, model_label=None, max_new_tokens=256,
                 temperature=0.7, num_threads=2):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError("HF backend requires a locally downloaded model directory")
        if not 1 <= max_new_tokens <= 4096 or temperature < 0 or num_threads < 1:
            raise ValueError("Invalid model sampling settings")
        self.path, self.seed = str(path), int(seed)
        self.model_label = model_label or path.name
        self.max_new_tokens, self.temperature, self.num_threads = max_new_tokens, temperature, num_threads
        self.calls, self.errors, self._events = 0, 0, []
        self.torch = torch
        torch.set_num_threads(num_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
                                                        trust_remote_code=False, dtype=torch.float32)
        self.model.eval()
        self.model.requires_grad_(False)
        self.base_manifest = local_model_manifest(path)

    def configuration(self):
        return {"backend": "huggingface-json-frozen", "model_path": self.path,
                "model_label": self.model_label, "seed": self.seed,
                "base_manifest": self.base_manifest, "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature, "num_threads": self.num_threads,
                "resume_semantics": "generation RNG seeded by run and call index"}

    def _invoke(self, payload):
        system = ("You are a neural architecture search agent. Follow the supplied task and response schema. "
                  "Use only the information in this request. Reply with exactly one JSON object and no prose. "
                  "A harness patch changes the search program; a NAS proposal contains exactly six allowed operations. "
                  "A peer critique follows its requested critique/revision schema. Do not invent evaluation scores.")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        length = inputs["input_ids"].shape[1]
        max_context = getattr(self.model.config, "max_position_embeddings", 32768)
        if length + self.max_new_tokens > max_context:
            raise ValueError("Request exceeds model context; reduce the controller memory size")
        generation_seed = (self.seed * 1_000_003 + self.calls * 97) % (2**31)
        kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0,
                  "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id}
        if self.temperature > 0:
            kwargs.update(temperature=self.temperature, top_p=0.95)
        with self.torch.random.fork_rng(devices=[]):
            self.torch.manual_seed(generation_seed)
            with self.torch.inference_mode():
                tokens = self.model.generate(**inputs, **kwargs)
        generated = tokens[0, length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True), {
            "usage": {"input_tokens": length, "output_tokens": len(generated)},
            "generation_seed": generation_seed}
