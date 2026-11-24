"""Optional, real causal-LM LoRA policy for the measured design search.

``from_local`` loads explicitly provided local model/tokenizer files only.  It
never downloads weights or calls a hosted API.  The frozen causal model assigns
next-token probabilities to verified single-token labels for the available
designs.  Low-rank adapters learn from the selected design's validation reward;
unqueried results and held-out scores are never part of a prompt or update.

The objective is online REINFORCE with a prior-observation baseline, reference
policy KL, and entropy regularization.  This is a small task-specific policy
learning implementation, not a reproduction of the complete TTT-Discover
system.  LoRA follows https://arxiv.org/abs/2106.09685 .
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import string
import tempfile
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


_DEFAULT_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "c_attn", "c_proj", "c_fc",
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(_json([name, str(tensor.dtype), list(tensor.shape)]).encode())
        # The byte view also supports bfloat16 without a lossy float conversion.
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


class _LoRALinear(nn.Module):
    """Adapter for torch Linear and HuggingFace's transposed GPT-2 Conv1D."""

    def __init__(self, base: nn.Module, rank: int, alpha: float):
        super().__init__()
        self.base = base
        if isinstance(base, nn.Linear):
            in_features, out_features = base.in_features, base.out_features
        elif type(base).__name__ == "Conv1D" and base.weight.ndim == 2:
            in_features, out_features = base.weight.shape
        else:
            raise TypeError("LoRA targets must be Linear or HuggingFace Conv1D modules")
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32,
                                               device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32,
                                               device=base.weight.device))
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if self.enabled:
            update = F.linear(F.linear(inputs.to(self.lora_A.dtype), self.lora_A), self.lora_B)
            output = output + (self.scale * update).to(output.dtype)
        return output


class HuggingFaceLoRAPolicy:
    """A resettable, checkpointable policy with an immutable pretrained base.

    The component constructor is also useful for small randomly initialized
    causal-model *software tests*.  Reported pretrained-model experiments should
    use :meth:`from_local` so their local model identity is recorded.
    """

    def __init__(self, model: nn.Module, tokenizer: Any, candidates: list[dict], *,
                 model_id: str = "provided-components", revision: str | None = None,
                 device: str = "cpu", rank: int = 4, alpha: float = 8.0,
                 learning_rate: float = 1e-3, temperature: float = 1.0,
                 kl_coefficient: float = 0.02, entropy_coefficient: float = 0.01,
                 max_gradient_norm: float = 1.0, max_input_tokens: int = 2048,
                 max_history: int = 16, target_modules: tuple[str, ...] = _DEFAULT_TARGETS):
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if isinstance(max_input_tokens, bool) or not isinstance(max_input_tokens, int) or max_input_tokens < 16:
            raise ValueError("max_input_tokens must be an integer >=16")
        if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 0:
            raise ValueError("max_history must be a nonnegative integer")
        self.settings = {
            "rank": rank, "alpha": _positive(alpha, "alpha"),
            "learning_rate": _positive(learning_rate, "learning_rate"),
            "temperature": _positive(temperature, "temperature"),
            "kl_coefficient": _positive(kl_coefficient, "kl_coefficient", allow_zero=True),
            "entropy_coefficient": _positive(entropy_coefficient, "entropy_coefficient", allow_zero=True),
            "max_gradient_norm": _positive(max_gradient_norm, "max_gradient_norm"),
            "max_input_tokens": max_input_tokens, "max_history": max_history,
            "target_modules": list(target_modules),
        }
        self.candidates = []
        for candidate in candidates:
            candidate_id = candidate.get("candidate_id", candidate.get("id"))
            if not isinstance(candidate_id, str) or not candidate_id or not isinstance(candidate.get("config"), dict):
                raise ValueError("Candidates need an ID and a public config object")
            self.candidates.append({"candidate_id": candidate_id,
                                    "config": json.loads(_json(candidate["config"]))})
        if not self.candidates or len({item["candidate_id"] for item in self.candidates}) != len(self.candidates):
            raise ValueError("Candidates must have unique nonempty IDs")
        self.candidate_by_id = {item["candidate_id"]: item for item in self.candidates}
        self.candidate_sha256 = _digest(self.candidates)
        self.tokenizer = tokenizer
        self.model = model
        if any(isinstance(module, _LoRALinear) for module in model.modules()):
            raise ValueError("This model already contains golfbench LoRA adapters")
        self.base_sha256 = _tensor_digest(dict(model.state_dict()))
        config = json.loads(model.config.to_json_string(use_diff=False))
        self.provenance = {
            "model_id": str(model_id), "model_revision": revision,
            "base_model_sha256": self.base_sha256,
            "model_class": type(model).__name__, "model_config": config,
            "model_config_sha256": _digest(config),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_vocabulary_sha256": _digest(tokenizer.get_vocab()),
            "base_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "local_files_only": True, "trust_remote_code": False,
            "model_loading_seconds": None,
        }
        self.device = torch.device(device)
        self.model.requires_grad_(False)
        self.adapters: dict[str, _LoRALinear] = {}
        for name, module in list(model.named_modules()):
            if name.rsplit(".", 1)[-1] not in target_modules:
                continue
            if not isinstance(module, nn.Linear) and type(module).__name__ != "Conv1D":
                continue
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            adapter = _LoRALinear(module, rank, self.settings["alpha"])
            setattr(parent, leaf, adapter)
            self.adapters[name] = adapter
        if not self.adapters:
            raise ValueError("No supported LoRA target modules found; provide explicit target_modules")
        self.model.to(self.device)
        self.model.eval()  # Frozen-base dropout must not change the sampling policy.
        self.parameters = [parameter for module in self.adapters.values()
                           for parameter in (module.lora_A, module.lora_B)]
        self.labels = self._discover_labels(len(self.candidates))
        self.action_by_id = {item["candidate_id"]: label
                             for item, label in zip(self.candidates, self.labels)}
        configured_context = next((int(getattr(model.config, name))
                                   for name in ("max_position_embeddings", "n_positions", "max_sequence_length")
                                   if getattr(model.config, name, None)), max_input_tokens)
        self.input_limit = min(max_input_tokens, configured_context)
        self.provenance.update(
            device=str(self.device), torch_version=torch.__version__,
            target_modules=list(self.adapters),
            trainable_parameters=sum(parameter.numel() for parameter in self.parameters),
            action_map=[{"candidate_id": item["candidate_id"], **label}
                        for item, label in zip(self.candidates, self.labels)],
        )
        self._identity = {
            "model_id": self.provenance["model_id"],
            "model_revision": self.provenance["model_revision"],
            "base_model_sha256": self.base_sha256,
            "model_config_sha256": self.provenance["model_config_sha256"],
            "tokenizer_vocabulary_sha256": self.provenance["tokenizer_vocabulary_sha256"],
            "action_map_sha256": _digest(self.provenance["action_map"]),
            "candidate_manifest_sha256": self.candidate_sha256,
            "adapter_settings_sha256": _digest(self.settings),
            "adapter_settings": deepcopy(self.settings),
        }
        try:
            import transformers
            self.provenance["transformers_version"] = transformers.__version__
        except ImportError:
            self.provenance["transformers_version"] = None
        self.start_run(seed=0, adaptive=False)

    @property
    def identity(self) -> dict:
        """A defensive copy of immutable model/configuration identity, without timings."""
        return deepcopy(self._identity)

    @classmethod
    def from_local(cls, model_path: str | Path, candidates: list[dict], **kwargs) -> "HuggingFaceLoRAPolicy":
        """Load an existing directory; network access is disabled in both loaders."""
        directory = Path(model_path).expanduser().resolve()
        if not directory.is_dir() or not (directory / "config.json").is_file():
            raise ValueError("model_path must be an existing local model directory containing config.json")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        revision = getattr(model.config, "_commit_hash", None)
        if revision is None and directory.parent.name == "snapshots":
            revision = directory.name
        policy = cls(model, tokenizer, candidates, model_id=str(directory), revision=revision, **kwargs)
        policy.provenance["model_loading_seconds"] = time.perf_counter() - started
        return policy

    def _discover_labels(self, count: int) -> list[dict]:
        prefix = "\nChoice:"
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        vocabulary_size = self.model.get_output_embeddings().weight.shape[0]
        special = set(self.tokenizer.all_special_ids)
        labels, used = [], set()
        for character in string.ascii_uppercase + string.ascii_lowercase + string.digits:
            for text in (" " + character, character):
                encoded = self.tokenizer.encode(text, add_special_tokens=False)
                if len(encoded) != 1 or encoded[0] in used or encoded[0] in special:
                    continue
                token_id = int(encoded[0])
                if not 0 <= token_id < vocabulary_size:
                    continue
                decoded = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                if decoded.strip() != character:
                    continue
                if self.tokenizer.encode(prefix + text, add_special_tokens=False) != prefix_ids + [token_id]:
                    continue
                labels.append({"label": character, "completion": text, "token_id": token_id})
                used.add(token_id)
                break
            if len(labels) == count:
                return labels
        raise ValueError(f"Tokenizer provides only {len(labels)} verified single-token labels; need {count}")

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _adapter_tensors(self) -> dict[str, torch.Tensor]:
        return {f"{name}.{field}": getattr(module, field)
                for name, module in self.adapters.items() for field in ("lora_A", "lora_B")}

    def start_run(self, seed: int, adaptive: bool) -> None:
        """Reset every trainable parameter, optimizer, sampler and accounting field."""
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(adaptive, bool):
            raise ValueError("adaptive must be a boolean")
        started = time.perf_counter()
        generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x10A)
        with torch.no_grad():
            for adapter in self.adapters.values():
                bound = 1.0 / math.sqrt(adapter.lora_A.shape[1])
                values = torch.empty(adapter.lora_A.shape, dtype=torch.float32).uniform_(
                    -bound, bound, generator=generator)
                adapter.lora_A.copy_(values.to(adapter.lora_A.device))
                adapter.lora_B.zero_()
                adapter.enabled = True
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.settings["learning_rate"], weight_decay=0.0)
        self.sample_generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x10ABEE)
        self.seed, self.adaptive = seed, adaptive
        self.gradient_updates = self.gradient_evaluations = self.proposals = 0
        self.pending: dict | None = None
        self.events: list[dict] = []
        self.usage = {"forward_passes": 0, "forward_input_tokens": 0,
                      "proposal_input_tokens": 0, "sampled_action_tokens": 0,
                      "proposal_seconds": 0.0, "update_seconds": 0.0,
                      "api_requests": 0}
        self.initial_adapter_sha256 = _tensor_digest(self._adapter_tensors())
        self.run_reset_seconds = time.perf_counter() - started

    def _prompt(self, available: list[dict], history: list[dict]) -> tuple[str, list[int], list[str]]:
        ids, options = [], []
        all_configs = [item["config"] for item in self.candidate_by_id.values()]
        common = {key:value for key,value in all_configs[0].items()
                  if all(config.get(key)==value for config in all_configs)}
        for item in available:
            candidate_id = item.get("candidate_id", item.get("id"))
            if candidate_id not in self.candidate_by_id or candidate_id in ids:
                raise ValueError("Available actions must be unique registered candidate IDs")
            registered = self.candidate_by_id[candidate_id]
            if item.get("config") != registered["config"]:
                raise ValueError("A candidate config changed after policy initialization")
            ids.append(candidate_id)
            options.append({"label": self.action_by_id[candidate_id]["label"],
                            "id": candidate_id, "config": {k:v for k,v in registered["config"].items() if k not in common}})
        if not ids:
            raise ValueError("No available actions")
        observations = []
        for row in history:
            if row.get("candidate_id") not in self.candidate_by_id:
                raise ValueError("History contains an unregistered candidate")
            score = _positive(row.get("val_bpb"), "observed val_bpb", allow_zero=True)
            observations.append({"id": row["candidate_id"], "val_bpb": score})
        observations = observations[-self.settings["max_history"]:] if self.settings["max_history"] else []
        while True:
            prompt = ("Choose one available language-model design to minimize validation bits per byte. "
                      "Only evaluated designs have known scores. Return its label.\nCommon config:"
                      + _json(common) + "\nAvailable:" + _json(options)
                      + "\nObserved:" + _json(observations) + "\nChoice:")
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) <= self.input_limit:
                break
            if not observations:
                raise ValueError("The complete action list exceeds max_input_tokens; increase the limit")
            observations.pop(0)
        if not tokens:
            raise ValueError("The policy prompt tokenized to an empty sequence")
        for candidate_id in ids:
            label = self.action_by_id[candidate_id]
            if self.tokenizer.encode(prompt + label["completion"], add_special_tokens=False) != tokens + [label["token_id"]]:
                raise ValueError("Action label is not a single-token continuation of the actual prompt")
        return prompt, tokens, ids

    def _log_probabilities(self, tokens: list[int], action_ids: list[int]) -> torch.Tensor:
        inputs = torch.tensor([tokens], dtype=torch.long, device=self.device)
        logits = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            use_cache=False).logits[0, -1, action_ids].float()
        self.usage["forward_passes"] += 1
        self.usage["forward_input_tokens"] += len(tokens)
        return F.log_softmax(logits / self.settings["temperature"], dim=-1)

    @contextmanager
    def _reference_policy(self):
        for module in self.adapters.values():
            module.enabled = False
        try:
            yield
        finally:
            for module in self.adapters.values():
                module.enabled = True

    def propose(self, available: list[dict], history: list[dict]) -> tuple[str, dict]:
        if self.pending is not None:
            raise ValueError("Observe or cancel the previous proposal before sampling again")
        self._sync()
        started = time.perf_counter()
        prompt, tokens, ids = self._prompt(available, history)
        action_ids = [self.action_by_id[candidate_id]["token_id"] for candidate_id in ids]
        with torch.no_grad():
            log_probabilities = self._log_probabilities(tokens, action_ids)
        probabilities = log_probabilities.exp().cpu()
        selected = int(torch.multinomial(probabilities, 1, generator=self.sample_generator).item())
        prior = [float(row["val_bpb"]) for row in history]
        baseline = sum(prior) / len(prior) if prior else 8.0
        scale = max(math.sqrt(sum((value - baseline) ** 2 for value in prior) / len(prior)), 0.05) if prior else 1.0
        self.proposals += 1
        self.pending = {"candidate_id": ids[selected], "tokens": tokens, "action_ids": action_ids,
                        "available_ids": ids, "selected": selected,
                        "sample_log_probability": float(log_probabilities[selected].cpu()),
                        "baseline_bpb": baseline, "baseline_scale": scale,
                        "adapter_sha256": _tensor_digest(self._adapter_tensors()),
                        "policy_version": self.gradient_updates, "proposal_number": self.proposals,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        self.usage["proposal_input_tokens"] += len(tokens)
        self.usage["sampled_action_tokens"] += 1
        self._sync()
        self.usage["proposal_seconds"] += time.perf_counter() - started
        return ids[selected], {"policy_backend": "huggingface_lora", "policy_version": self.gradient_updates,
                               "policy_probability": float(probabilities[selected]),
                               "proposal_number": self.proposals, "input_tokens": len(tokens),
                               "prompt_sha256": self.pending["prompt_sha256"],
                               "action_token_id": action_ids[selected]}

    def cancel_pending(self, reason: str = "not_evaluated") -> None:
        if self.pending is not None:
            self.events.append({"type": "policy_proposal_cancelled", "reason": str(reason),
                                "candidate_id": self.pending["candidate_id"],
                                "proposal_number": self.pending["proposal_number"]})
            self.pending = None

    def observe(self, row: dict, metadata: dict | None = None) -> None:
        if row.get("phase") == "warm_start":
            return  # The common initialization was not sampled from this policy.
        if self.pending is None or row.get("candidate_id") != self.pending["candidate_id"]:
            raise ValueError("Observation does not match a pending on-policy proposal")
        if row.get("status") not in ("valid_new", "infeasible_or_failed"):
            self.cancel_pending("not_a_new_evaluation")
            return
        pending = self.pending
        if pending["adapter_sha256"] != _tensor_digest(self._adapter_tensors()):
            raise ValueError("Adapter weights changed after sampling; refusing an off-policy update")
        self._sync()
        started = time.perf_counter()
        logp = self._log_probabilities(pending["tokens"], pending["action_ids"])
        if not torch.isfinite(logp).all():
            raise ValueError("Non-finite policy probabilities")
        if abs(float(logp[pending["selected"]].detach().cpu()) - pending["sample_log_probability"]) > 1e-4:
            raise ValueError("The sampling distribution changed before its update")
        with torch.no_grad(), self._reference_policy():
            reference_logp = self._log_probabilities(pending["tokens"], pending["action_ids"])
        probabilities = logp.exp()
        kl = torch.sum(probabilities * (logp - reference_logp))
        entropy = -torch.sum(probabilities * logp)
        if row["status"] == "valid_new":
            score = _positive(row.get("val_bpb"), "val_bpb", allow_zero=True)
            advantage = (pending["baseline_bpb"] - score) / pending["baseline_scale"]
            advantage = min(5.0, max(-5.0, advantage))
        else:
            advantage = -5.0
        loss = (-advantage * logp[pending["selected"]]
                + self.settings["kl_coefficient"] * kl
                - self.settings["entropy_coefficient"] * entropy)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.settings["max_gradient_norm"],
                                                       error_if_nonfinite=True)
        before = [parameter.detach().clone() for parameter in self.parameters]
        self.gradient_evaluations += 1
        if self.adaptive:
            self.optimizer.step()
            self.gradient_updates += 1
        update_norm = math.sqrt(sum(float(torch.sum((parameter.detach() - old) ** 2).cpu())
                                    for parameter, old in zip(self.parameters, before)))
        self._sync()
        elapsed = time.perf_counter() - started
        self.usage["update_seconds"] += elapsed
        self.events.append({"type": "lora_policy_gradient", "attempt": row.get("attempt"),
                            "candidate_id": row["candidate_id"], "applied": self.adaptive,
                            "policy_version": self.gradient_updates, "advantage": advantage,
                            "loss": float(loss.detach().cpu()), "reference_kl": float(kl.detach().cpu()),
                            "entropy": float(entropy.detach().cpu()),
                            "gradient_norm": float(gradient_norm.detach().cpu()), "update_norm": update_norm,
                            "update_seconds": elapsed, "proposal_number": pending["proposal_number"]})
        self.pending = None

    def summary(self) -> dict:
        return {
            "controller_kind": "causal_lm_lora_policy", "policy_kind": "huggingface_lora_single_token_choice",
            "model_provenance": deepcopy(self.provenance), "policy_settings": deepcopy(self.settings),
            "policy_model": self.provenance["model_id"],
            "trainable_parameters": self.provenance["trainable_parameters"],
            "trainable_parameter_count": self.provenance["trainable_parameters"],
            "initial_weight_sha256": self.initial_adapter_sha256,
            "final_weight_sha256": _tensor_digest(self._adapter_tensors()),
            "gradient_updates": self.gradient_updates, "gradient_evaluations": self.gradient_evaluations,
            "events": deepcopy(self.events), "policy_usage": dict(self.usage),
            "policy_wall_seconds": self.usage["proposal_seconds"] + self.usage["update_seconds"],
            "run_reset_seconds": self.run_reset_seconds,
            "pending_unobserved_proposal": self.pending is not None,
            "candidate_manifest_sha256": self.candidate_sha256,
            "reference_policy": "same frozen base with LoRA disabled",
            "objective": "REINFORCE + KL(policy||reference) - entropy_bonus",
        }

    def save_checkpoint(self, path: str | Path) -> dict:
        """Atomically save adapters, optimizer, sampling RNG and pending proposal."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1, "base_sha256": self.base_sha256,
            "candidate_sha256": self.candidate_sha256,
            "tokenizer_sha256": self.provenance["tokenizer_vocabulary_sha256"],
            "model_config_sha256": self.provenance["model_config_sha256"],
            "settings": deepcopy(self.settings), "seed": self.seed, "adaptive": self.adaptive,
            "adapters": {name: tensor.detach().cpu().clone() for name, tensor in self._adapter_tensors().items()},
            "optimizer": self.optimizer.state_dict(), "sample_rng": self.sample_generator.get_state(),
            "gradient_updates": self.gradient_updates, "gradient_evaluations": self.gradient_evaluations,
            "proposals": self.proposals, "pending": deepcopy(self.pending), "events": deepcopy(self.events),
            "usage": dict(self.usage), "initial_adapter_sha256": self.initial_adapter_sha256,
        }
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return {"path": str(destination.resolve()), "size_bytes": destination.stat().st_size,
                "adapter_sha256": _tensor_digest(self._adapter_tensors()),
                "gradient_updates": self.gradient_updates}

    def load_checkpoint(self, path: str | Path) -> None:
        """Load tensor-only checkpoint data and reject another model or design space."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected = {"format_version": 1, "base_sha256": self.base_sha256,
                    "candidate_sha256": self.candidate_sha256,
                    "tokenizer_sha256": self.provenance["tokenizer_vocabulary_sha256"],
                    "model_config_sha256": self.provenance["model_config_sha256"],
                    "settings": self.settings}
        if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("Checkpoint does not match the base model, tokenizer, candidates or policy settings")
        current = self._adapter_tensors()
        if set(payload.get("adapters", {})) != set(current):
            raise ValueError("Checkpoint adapter names do not match")
        for name, tensor in current.items():
            restored = payload["adapters"][name]
            if not isinstance(restored, torch.Tensor) or restored.shape != tensor.shape or not torch.isfinite(restored).all():
                raise ValueError("Checkpoint has an invalid adapter tensor")
        self.start_run(payload["seed"], payload["adaptive"])
        with torch.no_grad():
            for name, tensor in self._adapter_tensors().items():
                tensor.copy_(payload["adapters"][name].to(tensor.device))
        self.optimizer.load_state_dict(payload["optimizer"])
        self.sample_generator.set_state(payload["sample_rng"])
        for name in ("gradient_updates", "gradient_evaluations", "proposals", "pending", "events", "usage", "initial_adapter_sha256"):
            setattr(self, name, deepcopy(payload[name]))
