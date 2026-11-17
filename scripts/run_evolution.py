#!/usr/bin/env python3
"""Run matched harness, multi-agent and test-time weight adaptation experiments."""
import _bootstrap
import argparse
import contextlib
import hashlib
import json
from pathlib import Path

from nas_agent_bench.adaptive_runner import METHODS, run_adaptive_search
from nas_agent_bench.engine import write_summaries
from nas_agent_bench.model_adapters import SubprocessJSONModel, HuggingFaceJSONModel
from nas_agent_bench.oracles import load_oracle


@contextlib.contextmanager
def experiment_lock(output):
    import fcntl
    with (output / ".run.lock").open("w") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process owns this experiment directory") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "src" / "nas_agent_bench").glob("*.py")) + [Path(__file__).resolve()]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["table", "nats"], default="table")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--warm-start", type=int, default=10)
    p.add_argument("--dataset", default="cifar10-valid")
    p.add_argument("--hp", default="200")
    p.add_argument("--controller-config", help="JSON object with harness, multiagent and/or weight sections")
    p.add_argument("--proposer-backend", choices=["local", "command", "hf"], default="local")
    p.add_argument("--adapter-command", help="JSON stdin/stdout command supporting all controller task schemas")
    p.add_argument("--adapter-timeout", type=float, default=180)
    p.add_argument("--model-path", help="Local frozen HF model for harness/multi-agent text generation")
    p.add_argument("--model-label")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--weight-backend", choices=["numpy", "hf_lora"], help="Override the weight configuration")
    p.add_argument("--weight-model-path", help="Local trainable HF model, independent of the frozen proposer")
    p.add_argument("--resume", action="store_true", help="Resume the same settings and original absolute budget")
    p.add_argument("--stop-after", type=int, help="Checkpoint at this absolute attempt without finalizing")
    a = p.parse_args()
    if not 1 <= a.warm_start <= a.budget <= 15625:
        p.error("Require 1 <= warm-start <= budget <= 15625")
    if len(set(a.methods)) != len(a.methods) or len(set(a.seeds)) != len(a.seeds):
        p.error("Methods and seeds must not repeat")
    if a.stop_after is not None and not a.warm_start <= a.stop_after <= a.budget:
        p.error("stop-after must be between warm-start and budget")
    if a.proposer_backend == "command" and not a.adapter_command:
        p.error("command backend requires --adapter-command")
    if a.proposer_backend == "hf" and not a.model_path:
        p.error("hf backend requires --model-path")
    settings = json.loads(Path(a.controller_config).read_text()) if a.controller_config else {}
    if not isinstance(settings, dict) or set(settings) - {"harness", "multiagent", "weight"}:
        p.error("Controller config must contain only harness, multiagent and weight objects")
    if any(not isinstance(v, dict) for v in settings.values()):
        p.error("Each controller config section must be an object")
    if a.weight_backend:
        settings.setdefault("weight", {})["backend"] = a.weight_backend
    if a.weight_model_path:
        settings.setdefault("weight", {})["model_path"] = str(Path(a.weight_model_path).expanduser().resolve())
    output = Path(a.output).expanduser().resolve()
    if not a.resume and output.exists() and any(output.iterdir()):
        p.error("Output directory must be empty; use --resume for an existing experiment")
    if a.resume and not (output / "config.json").is_file():
        p.error("Resume requires an existing experiment config.json")
    output.mkdir(parents=True, exist_ok=True)
    oracle = load_oracle(a.backend, a.data, a.dataset, a.hp)
    args_identity = {k: v for k, v in vars(a).items() if k not in {"resume", "stop_after", "output", "controller_config"}}
    experiment = dict(args_identity, controller_settings=settings, provenance=oracle.metadata,
                      source_manifest=source_manifest())
    with experiment_lock(output):
        if a.resume:
            saved = json.loads((output / "config.json").read_text())
            if saved != experiment:
                p.error("Resume settings, benchmark or source files differ from the saved experiment")
        else:
            (output / "config.json").write_text(json.dumps(experiment, indent=2, allow_nan=False) + "\n")
        summaries = []
        for seed in a.seeds:
            for method in a.methods:
                family = method.split("-")[0]
                model = None
                if family != "weight" and a.proposer_backend == "command":
                    model = SubprocessJSONModel(a.adapter_command, a.adapter_timeout, a.model_label, seed)
                elif family != "weight" and a.proposer_backend == "hf":
                    model = HuggingFaceJSONModel(a.model_path, seed, a.model_label,
                                                 a.max_new_tokens, a.temperature, a.num_threads)
                label = a.model_label if model is not None else (
                    "local-policy" if family != "weight" else settings.get("weight", {}).get("backend", "numpy") + "-policy")
                print(f"{method}: seed={seed}, budget={a.budget}, proposer={label}", flush=True)
                run_exists = (output / "checkpoints" / f"{method}_seed{seed}.json").exists()
                result = run_adaptive_search(oracle, method, seed, a.budget, a.warm_start, output,
                                             settings.get(family, {}), model, label,
                                             resume=a.resume and run_exists, stop_after=a.stop_after)
                summaries.append(result)
                write_summaries(output, summaries)
                print(f"  {result['completed_attempts']} attempts; best validation={result.get('best_val_accuracy')}", flush=True)
                del model
        if all(row["complete"] for row in summaries):
            (output / "locked_winners.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
            print("All winners locked. Held-out test lookup remains a separate command.")
        else:
            print("Checkpointed incomplete runs. Resume with the same settings and --resume.")


if __name__ == "__main__":
    main()
