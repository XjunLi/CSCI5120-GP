"""Online policy-weight adaptation for NAS, inspired by TTT-Discover.

The default backend is an autoregressive 405-parameter policy trained on CPU.
The optional ``hf_lora`` backend updates real low-rank weights in a local causal
language model. Both use the same on-policy, validation-reward objective. This
is a small NAS adaptation, not a reproduction of TTT-Discover or TTT layers.

Only completed proposals made by this controller can enter the gradient batch.
Warm starts, duplicates of observations, hidden tests, and arbitrary observation
fields cannot become training data. Candidate CNN weights are never trained.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from .architecture import OPS, validate_ops


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


def _softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    value = np.exp(shifted)
    return value / value.sum(axis=-1, keepdims=True)


def entropic_advantages(rewards, kl_budget=math.log(2), max_beta=64.0):
    """Adaptive exponential tilt, constrained against the uniform batch measure.

Returns mean-baselined score-function advantages ``N * softmax(beta*r) - 1``.
Unlike the paper's leave-one-out estimator, this uses a finite-batch mean
baseline. Rewards are centered/scaled for stable bisection. The resulting beta
is in standardized reward units. All weights are treated as stop-gradient.
"""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or not len(rewards) or not np.isfinite(rewards).all():
        raise ValueError("rewards must be a finite, nonempty vector")
    if not math.isfinite(kl_budget) or kl_budget < 0 or not math.isfinite(max_beta) or max_beta <= 0:
        raise ValueError("invalid entropic temperature bounds")
    n = len(rewards)
    scale = float(rewards.std())
    if n < 2 or scale < 1e-12 or kl_budget == 0:
        return np.zeros(n), {"beta": 0.0, "tilt_kl": 0.0, "reward_scale": scale}
    centered = (rewards - rewards.max()) / scale

    def distribution(beta):
        q = _softmax(beta * centered)
        positive = q > 0
        return q, float(np.sum(q[positive] * np.log(n * q[positive])))

    lo, hi = 0.0, float(max_beta)
    if distribution(hi)[1] > kl_budget:
        for _ in range(60):
            mid = (lo + hi) / 2
            if distribution(mid)[1] > kl_budget:
                hi = mid
            else:
                lo = mid
        beta = lo
    else:
        beta = hi
    q, tilt_kl = distribution(beta)
    return n * q - 1.0, {"beta": beta, "tilt_kl": tilt_kl, "reward_scale": scale}


class AutoregressiveNumpyPolicy:
    """Softmax policy with edge-specific bias and all earlier edge/op features.

There are 30 biases and 375 active interaction weights. Unused future-prefix
entries stay zero. Initial weights are uniform, independent of benchmark data.
Analytic gradients implement the same objective as the PyTorch backend.
"""
    backend = "numpy"

    def __init__(self, seed, config):
        self.config = config
        self.rng = random.Random(seed)
        self.weights = np.zeros((6, 31, 5), dtype=np.float64)
        self.initial = self.weights.copy()
        self.m = np.zeros_like(self.weights)
        self.v = np.zeros_like(self.weights)
        self.optimizer_step = 0
        self.trainable_parameters = 405

    @staticmethod
    def features(prefix):
        value = np.zeros(31)
        value[0] = 1.0
        for edge, op in enumerate(prefix):
            value[1 + edge * 5 + op] = 1.0
        return value / math.sqrt(len(prefix) + 1)

    def distributions(self, indices, initial=False):
        weights = self.initial if initial else self.weights
        return np.stack([_softmax(self.features(indices[:edge]) @ weights[edge] / self.config["temperature"])
                         for edge in range(6)])

    def sample(self):
        indices, log_probability = [], 0.0
        for edge in range(6):
            probability = _softmax(self.features(indices) @ self.weights[edge] / self.config["temperature"])
            u, cumulative, action = self.rng.random(), 0.0, 4
            for index, p in enumerate(probability):
                cumulative += float(p)
                if u < cumulative:
                    action = index
                    break
            indices.append(action)
            log_probability += math.log(float(probability[action]))
        return {"indices": indices, "log_probability": log_probability}

    def log_probability(self, indices):
        probabilities = self.distributions(indices)
        return float(np.log(probabilities[np.arange(6), indices]).sum())

    def loss_and_gradient(self, batch, advantages):
        gradient = np.zeros_like(self.weights)
        policy_loss = kl_loss = entropy = 0.0
        n = len(batch)
        for sample, advantage in zip(batch, advantages):
            actions = sample["indices"]
            probabilities = self.distributions(actions)
            references = self.distributions(actions, initial=True)
            logp = np.log(np.maximum(probabilities, 1e-300))
            logq = np.log(np.maximum(references, 1e-300))
            policy_loss -= float(advantage) * float(logp[np.arange(6), actions].sum()) / n
            kl_loss += float(np.sum(probabilities * (logp - logq))) / n
            entropy -= float(np.sum(probabilities * logp)) / n
            for edge, action in enumerate(actions):
                p = probabilities[edge]
                one_hot = np.zeros(5)
                one_hot[action] = 1.0
                dz = float(advantage) * (p - one_hot)
                ratio = logp[edge] - logq[edge]
                dz += self.config["kl_coefficient"] * p * (ratio - np.dot(p, ratio))
                dz += self.config["entropy_coefficient"] * p * (logp[edge] - np.dot(p, logp[edge]))
                gradient[edge] += np.outer(self.features(actions[:edge]), dz) / (n * self.config["temperature"])
        loss = policy_loss + self.config["kl_coefficient"] * kl_loss - self.config["entropy_coefficient"] * entropy
        return float(loss), gradient, {"policy_loss": policy_loss, "reference_kl": kl_loss, "entropy": entropy}

    def update(self, batch, advantages):
        for sample in batch:
            if abs(self.log_probability(sample["indices"]) - sample["log_probability"]) > 1e-8:
                raise ValueError("stale/off-policy sample: weights must stay fixed throughout a batch")
        loss, gradient, stats = self.loss_and_gradient(batch, advantages)
        norm = float(np.linalg.norm(gradient))
        gradient *= min(1.0, self.config["max_grad_norm"] / max(norm, 1e-12))
        before = self.weights.copy()
        self.optimizer_step += 1
        self.m = 0.9 * self.m + 0.1 * gradient
        self.v = 0.999 * self.v + 0.001 * gradient ** 2
        mhat = self.m / (1 - 0.9 ** self.optimizer_step)
        vhat = self.v / (1 - 0.999 ** self.optimizer_step)
        self.weights -= self.config["learning_rate"] * mhat / (np.sqrt(vhat) + 1e-8)
        if not np.isfinite(self.weights).all():
            raise FloatingPointError("nonfinite policy weights")
        return {"loss": loss, "gradient_norm": norm,
                "step_weight_delta_l2": float(np.linalg.norm(self.weights - before)), **stats}

    def weight_summary(self):
        return {"trainable_parameters": self.trainable_parameters,
                "initial_weight_sha256": hashlib.sha256(self.initial.tobytes()).hexdigest(),
                "current_weight_sha256": hashlib.sha256(self.weights.tobytes()).hexdigest(),
                "weight_delta_l2": float(np.linalg.norm(self.weights - self.initial))}

    def state_dict(self):
        return {"backend": self.backend, "weights": self.weights.tolist(), "initial": self.initial.tolist(),
                "m": self.m.tolist(), "v": self.v.tolist(), "optimizer_step": self.optimizer_step,
                "rng_state": self.rng.getstate()}

    def load_state_dict(self, state):
        if state["backend"] != self.backend:
            raise ValueError("policy backend mismatch")
        for name in ("weights", "initial", "m", "v"):
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != self.weights.shape or not np.isfinite(value).all():
                raise ValueError("invalid policy array")
            setattr(self, name, value.copy())
        self.optimizer_step = int(state["optimizer_step"])
        self.rng.setstate(_tuples(state["rng_state"]))


class HuggingFaceLoRAPolicy:
    """Six constrained op tokens, real LoRA updates in a frozen causal LM.

The fixed prompt excludes scores and other histories, so the frozen/adaptive
comparison isolates parameter learning. Only action positions enter the loss.
Sampling and training both normalize over exactly the same five digit tokens.
"""
    backend = "hf_lora"
    PROMPT = ("Design a neural network cell. Select one operation for each of six edges "
              "(0->1,0->2,1->2,0->3,1->3,2->3). Operation codes: "
              "0=none; 1=skip_connect; 2=nor_conv_1x1; 3=nor_conv_3x3; 4=avg_pool_3x3. "
              "Output exactly six operation code digits.\nCodes:")

    def __init__(self, seed, config, supplied_model=None):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("hf_lora requires torch and transformers; see adapters/ttt/README.md") from exc
        self.torch, self.config = torch, config
        torch.set_num_threads(config["num_threads"])
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        if supplied_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from .model_adapters import local_model_manifest
            path = Path(config["model_path"]).expanduser().resolve()
            if not path.is_dir():
                raise ValueError("model_path must name an existing local HuggingFace model directory")
            self.base_manifest = local_model_manifest(path)
            self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
                                                            trust_remote_code=False, dtype=torch.float32,
                                                            attn_implementation="eager")
            self.base_identifier = str(path)
        else:
            if not isinstance(supplied_model, dict) or not {"model", "tokenizer"} <= supplied_model.keys():
                raise TypeError("supplied model must contain model and tokenizer (test injection only)")
            self.model, self.tokenizer = supplied_model["model"], supplied_model["tokenizer"]
            self.base_identifier = str(supplied_model.get("identifier", "injected-test-model"))
        self.model.to(device="cpu", dtype=torch.float32)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if supplied_model is not None:
            frozen_hash = hashlib.sha256()
            for name, parameter in self.model.named_parameters():
                frozen_hash.update(name.encode())
                frozen_hash.update(parameter.detach().numpy().tobytes())
            self.base_manifest = {"identifier": self.base_identifier, "parameters_sha256": frozen_hash.hexdigest()}
        token_lists = [self.tokenizer.encode(str(index), add_special_tokens=False) for index in range(5)]
        if any(len(tokens) != 1 for tokens in token_lists) or len({tokens[0] for tokens in token_lists}) != 5:
            raise ValueError("tokenizer must encode each operation digit as one distinct token")
        self.action_tokens = [tokens[0] for tokens in token_lists]
        self.prompt_tokens = self.tokenizer.encode(self.PROMPT, add_special_tokens=True)
        if not self.prompt_tokens:
            raise ValueError("empty prompt tokenization")
        self.adapters = {}
        rank, alpha = config["lora_rank"], config["lora_alpha"]
        generator = self.generator

        class LoRALinear(torch.nn.Module):
            def __init__(module, base):
                super().__init__()
                module.base, module.enabled = base, True
                module.scale = alpha / rank
                module.lora_A = torch.nn.Parameter(torch.randn((rank, base.in_features), generator=generator,
                                                              dtype=torch.float32) * 0.01)
                module.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))

            def forward(module, value):
                output = module.base(value)
                if module.enabled:
                    output = output + (value @ module.lora_A.T @ module.lora_B.T) * module.scale
                return output

        selected = [(name, module) for name, module in self.model.named_modules()
                    if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in config["target_modules"]]
        if not selected:
            raise ValueError("no torch Linear module matched target_modules")
        for name, module in selected:
            parent_name, _, leaf = name.rpartition(".")
            parent = self.model.get_submodule(parent_name) if parent_name else self.model
            wrapper = LoRALinear(module)
            setattr(parent, leaf, wrapper)
            self.adapters[name] = wrapper
        self.parameters = {name: parameter for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        self.initial = {name: parameter.detach().clone() for name, parameter in self.parameters.items()}
        self.trainable_parameters = sum(parameter.numel() for parameter in self.parameters.values())
        self.optimizer = torch.optim.Adam(list(self.parameters.values()), lr=config["learning_rate"])
        self.optimizer_step = 0

    def _set_adapters(self, enabled):
        for module in self.adapters.values():
            module.enabled = enabled

    def _all_action_distributions(self, indices, reference=False):
        torch = self.torch
        tokens = self.prompt_tokens + [self.action_tokens[index] for index in indices]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        self._set_adapters(not reference)
        try:
            output = self.model(input_ids=input_ids, use_cache=False)
            start = len(self.prompt_tokens) - 1
            logits = output.logits[0, start:start + 6, self.action_tokens] / self.config["temperature"]
            return torch.log_softmax(logits, dim=-1)
        finally:
            self._set_adapters(True)

    def sample(self):
        torch = self.torch
        self.model.eval()
        indices, log_probability, cache = [], 0.0, None
        input_ids = torch.tensor([self.prompt_tokens], dtype=torch.long)
        with torch.no_grad():
            for _ in range(6):
                output = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                logits = output.logits[0, -1, self.action_tokens] / self.config["temperature"]
                logp = torch.log_softmax(logits, dim=-1)
                index = int(torch.multinomial(logp.exp(), 1, generator=self.generator).item())
                indices.append(index)
                log_probability += float(logp[index].item())
                input_ids = torch.tensor([[self.action_tokens[index]]], dtype=torch.long)
        return {"indices": indices, "log_probability": log_probability,
                "prompt_tokens": list(self.prompt_tokens),
                "action_tokens": [self.action_tokens[index] for index in indices]}

    def log_probability(self, indices):
        with self.torch.no_grad():
            logp = self._all_action_distributions(indices)
            return float(logp[self.torch.arange(6), indices].sum().item())

    def update(self, batch, advantages):
        torch = self.torch
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        before = {name: p.detach().clone() for name, p in self.parameters.items()}
        stats = {"loss": 0.0, "policy_loss": 0.0, "reference_kl": 0.0, "entropy": 0.0}
        for sample, advantage in zip(batch, advantages):
            if sample.get("prompt_tokens") != self.prompt_tokens:
                raise ValueError("policy prompt changed inside an on-policy batch")
            indices = sample["indices"]
            with torch.no_grad():
                reference = self._all_action_distributions(indices, reference=True)
            logp = self._all_action_distributions(indices)
            chosen = logp[torch.arange(6), indices].sum()
            if abs(float(chosen.detach()) - sample["log_probability"]) > 2e-4:
                raise ValueError("stale/off-policy LM sample: constrained log probability does not match")
            probability = logp.exp()
            kl = (probability * (logp - reference)).sum()
            entropy = -(probability * logp).sum()
            policy_loss = -float(advantage) * chosen
            loss = (policy_loss + self.config["kl_coefficient"] * kl - self.config["entropy_coefficient"] * entropy) / len(batch)
            loss.backward()
            for key, value in (("loss", loss), ("policy_loss", policy_loss / len(batch)),
                               ("reference_kl", kl / len(batch)), ("entropy", entropy / len(batch))):
                stats[key] += float(value.detach().item())
        norm = float(torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), self.config["max_grad_norm"]).item())
        if not math.isfinite(norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("nonfinite LoRA gradient")
        self.optimizer.step()
        self.optimizer_step += 1
        delta = sum(float(((p.detach() - before[name]) ** 2).sum()) for name, p in self.parameters.items())
        if not all(torch.isfinite(p).all().item() for p in self.parameters.values()):
            raise FloatingPointError("nonfinite LoRA weights")
        return {**stats, "gradient_norm": norm, "step_weight_delta_l2": math.sqrt(delta)}

    def weight_summary(self):
        initial_hash, current_hash, delta = hashlib.sha256(), hashlib.sha256(), 0.0
        for name, parameter in self.parameters.items():
            initial_hash.update(name.encode())
            initial_hash.update(self.initial[name].numpy().tobytes())
            current_hash.update(name.encode())
            current_hash.update(parameter.detach().numpy().tobytes())
            delta += float(((parameter.detach() - self.initial[name]) ** 2).sum())
        return {"trainable_parameters": self.trainable_parameters, "initial_weight_sha256": initial_hash.hexdigest(),
                "current_weight_sha256": current_hash.hexdigest(), "weight_delta_l2": math.sqrt(delta),
                "base_model": self.base_identifier, "base_manifest": self.base_manifest,
                "lora_modules": list(self.adapters)}

    def state_dict(self):
        optimizer_state = {}
        for name, parameter in self.parameters.items():
            state = self.optimizer.state.get(parameter, {})
            optimizer_state[name] = {key: value.detach().cpu().tolist() if self.torch.is_tensor(value) else value
                                     for key, value in state.items()}
        return {"backend": self.backend, "base_identifier": self.base_identifier, "base_manifest": self.base_manifest,
                "prompt_tokens": self.prompt_tokens, "action_tokens": self.action_tokens,
                "parameters": {name: p.detach().tolist() for name, p in self.parameters.items()},
                "initial": {name: p.tolist() for name, p in self.initial.items()},
                "optimizer": optimizer_state, "optimizer_step": self.optimizer_step,
                "generator_state": self.generator.get_state().tolist()}

    def load_state_dict(self, state):
        torch = self.torch
        if (state["backend"] != self.backend or state["base_identifier"] != self.base_identifier
                or state.get("base_manifest") != self.base_manifest
                or state["prompt_tokens"] != self.prompt_tokens or state["action_tokens"] != self.action_tokens
                or set(state["parameters"]) != set(self.parameters)):
            raise ValueError("checkpoint and frozen base/tokenizer/adapter configuration differ")
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                value = torch.tensor(state["parameters"][name], dtype=torch.float32)
                initial = torch.tensor(state["initial"][name], dtype=torch.float32)
                if (value.shape != parameter.shape or initial.shape != parameter.shape
                        or not torch.isfinite(value).all() or not torch.isfinite(initial).all()):
                    raise ValueError("invalid LoRA checkpoint tensor")
                parameter.copy_(value)
                self.initial[name] = initial
                saved = state["optimizer"].get(name, {})
                self.optimizer.state[parameter] = {key: torch.tensor(item, dtype=torch.float32)
                                                   if key in ("step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq") else item
                                                   for key, item in saved.items()}
        self.optimizer_step = int(state["optimizer_step"])
        self.generator.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))


class WeightEvolutionController:
    """A strictly batched on-policy learner implementing the controller contract."""
    DEFAULTS = {"backend": "numpy", "adaptive": True, "batch_size": 4, "learning_rate": None,
                "kl_coefficient": 0.02, "entropy_coefficient": 0.001, "entropic_kl": math.log(2),
                "max_beta": 64.0, "max_grad_norm": 1.0, "temperature": 1.0,
                "model_path": None, "lora_rank": 4, "lora_alpha": 8.0,
                "target_modules": ["q_proj", "v_proj"], "num_threads": 2}

    def __init__(self, seed: int, config: dict | None = None, model=None):
        self.seed = int(seed)
        supplied = copy.deepcopy(config or {})
        unknown = set(supplied) - set(self.DEFAULTS)
        if unknown:
            raise ValueError("unknown weight configuration: " + ", ".join(sorted(unknown)))
        self.config = {**copy.deepcopy(self.DEFAULTS), **supplied}
        backend = self.config["backend"]
        if backend not in ("numpy", "hf_lora"):
            raise ValueError("backend must be numpy or hf_lora")
        if not isinstance(self.config["adaptive"], bool):
            raise ValueError("adaptive must be boolean")
        for key in ("batch_size", "lora_rank", "num_threads"):
            if not isinstance(self.config[key], int) or isinstance(self.config[key], bool) or self.config[key] < 1:
                raise ValueError(key + " must be a positive integer")
        if self.config["batch_size"] < 2:
            raise ValueError("batch_size must be >= 2 for reward contrasts")
        if self.config["learning_rate"] is None:
            self.config["learning_rate"] = 0.03 if backend == "numpy" else 0.0001
        for key in ("learning_rate", "max_beta", "max_grad_norm", "temperature", "lora_alpha"):
            if not isinstance(self.config[key], (int, float)) or not math.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(key + " must be finite and positive")
        for key in ("kl_coefficient", "entropy_coefficient", "entropic_kl"):
            if not isinstance(self.config[key], (int, float)) or not math.isfinite(self.config[key]) or self.config[key] < 0:
                raise ValueError(key + " must be finite and nonnegative")
        if not isinstance(self.config["target_modules"], list) or not self.config["target_modules"] or not all(
                isinstance(name, str) and name for name in self.config["target_modules"]):
            raise ValueError("target_modules must be a nonempty list of names")
        if backend == "numpy":
            if model is not None:
                raise ValueError("numpy policy does not accept an external LLM callable")
            self.policy = AutoregressiveNumpyPolicy(self.seed, self.config)
        else:
            if model is None and not isinstance(self.config["model_path"], str):
                raise ValueError("hf_lora requires model_path")
            self.policy = HuggingFaceLoRAPolicy(self.seed, self.config, supplied_model=model)
        self.pending = None
        self.batch = []
        self.total_updates = self.total_proposals = self.observed_rows = 0
        self.observed_steps = set()
        self.events = [{"type": "weight_policy_initialized", "backend": backend,
                        "adaptive": self.config["adaptive"], **self.policy.weight_summary()}]

    def propose(self, observation: dict) -> dict:
        if self.pending is not None:
            raise RuntimeError("observe the previous proposal before requesting another")
        if observation.get("complete") or observation.get("remaining_attempts", 1) <= 0:
            raise ValueError("search budget exhausted")
        sample = self.policy.sample()
        sample["step"] = observation.get("next_attempt", self.observed_rows + 1)
        sample["feedback_enabled"] = bool(observation.get("feedback_enabled", True))
        self.pending = sample
        self.total_proposals += 1
        return {"ops": [OPS[index] for index in sample["indices"]],
                "model": "autoregressive-nas-policy" if self.config["backend"] == "numpy" else "local-causal-lm-lora",
                "rationale": "Sampled from the current constrained autoregressive policy.",
                "usage": {"model_calls": 0 if self.config["backend"] == "numpy" else 1,
                          "generated_action_tokens": 6, "weight_updates_so_far": self.total_updates}}

    def observe(self, row: dict) -> None:
        step = row.get("step", self.observed_rows + 1)
        if step in self.observed_steps:
            raise ValueError("an evaluation may be observed only once")
        # Do not copy arbitrary fields: held-out test values cannot enter state.
        if self.pending is None:
            if row.get("phase") != "warm_start":
                raise ValueError("only warm starts may be observed without a pending proposal")
            self.observed_steps.add(step)
            self.observed_rows += 1
            return
        sample = self.pending
        if step != sample["step"]:
            raise ValueError("feedback step does not match pending proposal")
        self.pending = None
        self.observed_steps.add(step)
        self.observed_rows += 1
        try:
            indices = [OPS.index(op) for op in validate_ops(row.get("ops"))]
        except (TypeError, ValueError):
            indices = None
        reward = row.get("val_accuracy")
        if (indices != sample["indices"] or row.get("status") not in ("valid_new", "duplicate")
                or not sample["feedback_enabled"] or not isinstance(reward, (int, float))
                or isinstance(reward, bool) or not math.isfinite(reward)):
            self.events.append({"type": "weight_feedback_skipped", "step": step,
                                "reason": "missing, invalid, mismatched, or masked validation feedback"})
            return
        sample.pop("feedback_enabled")
        sample["reward"] = float(reward)
        self.batch.append(sample)
        if len(self.batch) >= self.config["batch_size"]:
            self._update()

    def _update(self):
        if len(self.batch) < 2:
            return
        rewards = [sample["reward"] for sample in self.batch]
        advantages, entropic = entropic_advantages(rewards, self.config["entropic_kl"], self.config["max_beta"])
        event = {"type": "weight_update" if self.config["adaptive"] else "weight_update_skipped_frozen",
                 "batch_steps": [sample["step"] for sample in self.batch], "batch_size": len(self.batch),
                 "reward_min": min(rewards), "reward_max": max(rewards), **entropic}
        if self.config["adaptive"]:
            event.update(self.policy.update(self.batch, advantages))
            self.total_updates += 1
        event.update({"total_updates": self.total_updates, **self.policy.weight_summary()})
        self.events.append(event)
        self.batch = []

    def finalize(self):
        """Train a final partial on-policy batch with >=2 actions; log singleton."""
        if self.pending is not None:
            raise RuntimeError("cannot finalize with an unevaluated proposal")
        if len(self.batch) >= 2:
            self._update()
        elif self.batch:
            self.events.append({"type": "weight_singleton_not_trained", "batch_steps": [self.batch[0]["step"]],
                                "reason": "a single reward supplies no within-batch contrast"})
            self.batch = []

    def state_dict(self) -> dict:
        return _json_copy({"schema_version": 1, "controller": "weight_evolution", "seed": self.seed,
                           "config": self.config, "policy": self.policy.state_dict(), "pending": self.pending,
                           "batch": self.batch, "observed_rows": self.observed_rows,
                           "observed_steps": sorted(self.observed_steps), "total_updates": self.total_updates,
                           "total_proposals": self.total_proposals, "events": self.events,
                           **self.policy.weight_summary()})

    def load_state_dict(self, state) -> None:
        if state.get("schema_version") != 1 or state.get("controller") != "weight_evolution":
            raise ValueError("unsupported weight controller checkpoint")
        if state["seed"] != self.seed or state["config"] != self.config:
            raise ValueError("checkpoint seed/config mismatch")
        self.policy.load_state_dict(state["policy"])
        self.pending, self.batch = copy.deepcopy(state["pending"]), copy.deepcopy(state["batch"])
        self.observed_rows = int(state["observed_rows"])
        self.observed_steps = set(state["observed_steps"])
        self.total_updates, self.total_proposals = int(state["total_updates"]), int(state["total_proposals"])
        self.events = copy.deepcopy(state["events"])

    def drain_events(self) -> list[dict]:
        events, self.events = self.events, []
        return _json_copy(events)


TTTController = WeightEvolutionController
