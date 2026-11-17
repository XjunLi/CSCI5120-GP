#!/usr/bin/env python3
"""JSON stdin -> Qwen localhost server -> uncorrected model JSON on stdout."""
import argparse
import json
import sys
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--url", default="http://127.0.0.1:8712")
p.add_argument("--timeout", type=float, default=180)
args = p.parse_args()
payload = json.load(sys.stdin)
request = urllib.request.Request(args.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=args.timeout) as response:
    result = json.load(response)
raw = result["raw_response"]
try:
    parsed = json.loads(raw)
except json.JSONDecodeError as exc:
    # Null ops remain invalid; keep raw text and actual usage instead of repairing.
    parsed = {"ops": None, "parse_error": str(exc)}
if not isinstance(parsed, dict):
    parsed = {"ops": None, "parse_error": "Model response was not a JSON object"}
for field in ("model", "usage", "model_revision", "generation_seed", "inference_seconds", "sampling"):
    parsed[field] = result[field]
parsed["raw_model_response"] = raw
print(json.dumps(parsed))
