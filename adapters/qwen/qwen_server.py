#!/usr/bin/env python3
"""CPU-only, persistent Qwen text inference. No tools or benchmark access."""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def messages_for(payload):
    feedback = payload.get("feedback_enabled", True)
    search_context = payload.get("search_context", payload.get("task_context", {}))
    history = []
    for row in payload.get("history", []):
        item = {key: row[key] for key in ("step", "phase", "status", "ops", "error") if key in row}
        if feedback:
            for key in ("val_accuracy", "validation_accuracy", "flops", "params"):
                if key in row:
                    item[key] = row[key]
        history.append(item)
    context = {
        "task": "Choose the next neural-network cell to maximize validation accuracy on image classification.",
        "cell": "Four nodes 0,1,2,3. Node 0 is input. Each other node sums transformed incoming feature maps. The surrounding network is fixed.",
        "edge_order": payload.get("edge_order", ["0->1", "0->2", "1->2", "0->3", "1->3", "2->3"]),
        "allowed_operations": payload.get("allowed_operations", OPS),
        "feedback_enabled": feedback,
        "remaining_attempts": payload.get("remaining_attempts_including_this_one", payload.get("remaining_attempts", payload.get("remaining_budget"))),
        "search_context": {k: search_context[k] for k in ("dataset", "hp", "accuracy_unit", "training_seed_policy") if k in search_context},
        "history": history,
    }
    return [
        {"role": "system", "content": "You select neural network architectures. Return ONLY one JSON object with an ops array of exactly six operation strings. Use only the allowed operations. Avoid previously submitted architectures. No prose or Markdown."},
        {"role": "user", "content": json.dumps(context, separators=(",", ":"))},
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--port", type=int, default=8712)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    args = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model_path = str(Path(args.model_path).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, local_files_only=True,
    ).to("cpu").eval()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond({"status": "ready", "model": MODEL_ID, "revision": REVISION, "device": "cpu"})

        def respond(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 2_000_000:
                    raise ValueError("Expected JSON payload smaller than 2 MB")
                payload = json.loads(self.rfile.read(size))
                messages = messages_for(payload)
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt")
                input_tokens = inputs["input_ids"].shape[-1]
                if input_tokens + args.max_new_tokens > 32768:
                    raise ValueError("Prompt exceeds model context; no silent truncation")
                run_seed = int(payload.get("run_seed", payload.get("seed", 0)))
                attempt = int(payload.get("next_attempt", len(payload.get("history", [])) + 1))
                generation_seed = (run_seed * 1_000_003 + attempt) % (2**31)
                torch.manual_seed(generation_seed)
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=True, temperature=args.temperature, top_p=args.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = output[0, input_tokens:]
                raw = tokenizer.decode(generated, skip_special_tokens=True)
                self.respond({
                    "raw_response": raw, "model": MODEL_ID,
                    "model_revision": REVISION, "generation_seed": generation_seed,
                    "usage": {"input_tokens": input_tokens, "output_tokens": len(generated)},
                    "inference_seconds": time.perf_counter() - started,
                    "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens},
                })
            except Exception as exc:
                self.respond({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps({"status": "ready", "port": args.port, "model": MODEL_ID, "revision": REVISION}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
