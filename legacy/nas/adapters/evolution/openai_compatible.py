#!/usr/bin/env python3
"""JSON adapter for a user-specified OpenAI-compatible chat-completions server.

Credentials stay in NAS_LLM_API_KEY. This adapter has no evaluator tools and
passes only the runner-provided payload to the configured model service.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=os.environ.get("NAS_LLM_BASE_URL"))
    p.add_argument("--model", default=os.environ.get("NAS_LLM_MODEL"))
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--json-mode", action="store_true", help="Request response_format=json_object if supported")
    args = p.parse_args()
    if not args.base_url or not args.model:
        p.error("Supply --base-url and --model, or NAS_LLM_BASE_URL and NAS_LLM_MODEL")
    payload = json.load(sys.stdin)
    system = ("You are a neural architecture search agent. Execute the supplied task using only its observations. "
              "Follow response_format and return exactly one JSON object. A proposer outputs six allowed ops; "
              "a harness editor outputs a patch; a critic outputs critique and optional revised ops. "
              "Never invent evaluation scores or access external benchmark data.")
    body = {"model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(payload, allow_nan=False)}]}
    if args.json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("NAS_LLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["NAS_LLM_API_KEY"]
    request = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            returned = json.load(response)
        raw = returned["choices"][0]["message"]["content"]
        if not isinstance(raw, str):
            raise ValueError("The server did not return text content")
        text = raw.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.partition("\n")[2].rsplit("```", 1)[0].strip()
        try:
            result = json.loads(text)
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            json.dumps(result, allow_nan=False)
        except (ValueError, TypeError):
            result = {"ops": None, "parse_error": "Model output was not a finite JSON object"}
        result.update(model=returned.get("model", args.model), raw_model_response=raw,
                      usage=returned.get("usage", {}))
        print(json.dumps(result, allow_nan=False))
    except (urllib.error.URLError, KeyError, IndexError, ValueError, TypeError) as exc:
        print(f"LLM adapter failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
