#!/usr/bin/env python3
"""Start persistent Qwen, run a command in the same process environment, stop Qwen."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model-path", default=str(root.parents[1] / "models/Qwen2.5-0.5B-Instruct"))
p.add_argument("--model-python", default=sys.executable)
p.add_argument("--port", type=int, default=8712)
p.add_argument("--threads", type=int, default=4)
p.add_argument("command", nargs=argparse.REMAINDER)
args = p.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command:
    p.error("Provide the experiment command after --")
env = os.environ.copy()
env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1")
server_command = [args.model_python, str(root / "qwen_server.py"), "--model-path", args.model_path, "--port", str(args.port), "--threads", str(args.threads)]
server = subprocess.Popen(server_command, env=env, stdout=sys.stderr, stderr=sys.stderr)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Qwen server exited with code {server.returncode}")
        try:
            with opener.open(f"http://127.0.0.1:{args.port}", timeout=1) as response:
                health = json.load(response)
            print("Qwen ready: " + json.dumps(health), file=sys.stderr, flush=True)
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise TimeoutError("Qwen initialization timed out")
    returncode = subprocess.run(command, env=env, check=False).returncode
finally:
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
sys.exit(returncode)
