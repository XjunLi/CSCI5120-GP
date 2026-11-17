#!/usr/bin/env python3
"""Root creates/exports sessions; an agent uses only status/propose."""
import _bootstrap
import argparse
import json
from nas_agent_bench.sessions import initialize, status, propose, export_session


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--session", required=True)
    init.add_argument("--backend", choices=["table", "nats"], default="table")
    init.add_argument("--data", required=True)
    init.add_argument("--condition", choices=["feedback", "no-feedback"], required=True)
    init.add_argument("--seed", type=int, required=True)
    init.add_argument("--budget", type=int, default=50)
    init.add_argument("--warm-start", type=int, default=10)
    init.add_argument("--dataset", default="cifar10-valid")
    init.add_argument("--hp", default="200")
    init.add_argument("--model", default="ChatGPT assistant; runtime model identity recorded separately")
    stat = sub.add_parser("status")
    stat.add_argument("--session", required=True)
    step = sub.add_parser("propose")
    step.add_argument("--session", required=True)
    step.add_argument("--expected-step", type=int, required=True)
    step.add_argument("--ops", required=True, help="JSON list of six operation tokens")
    step.add_argument("--rationale")
    step.add_argument("--proposal-seconds", type=float)
    export = sub.add_parser("export")
    export.add_argument("--session", required=True)
    export.add_argument("--output", required=True)
    a = p.parse_args()
    try:
        if a.command == "init":
            if not 1 <= a.warm_start <= a.budget <= 15625:
                p.error("Require 1 <= warm-start <= budget <= 15625")
            result = initialize(a.session, a.backend, a.data, a.condition, a.seed, a.budget, a.warm_start, a.dataset, a.hp, a.model)
        elif a.command == "status":
            result = status(a.session)
        elif a.command == "propose":
            result = propose(a.session, a.expected_step, a.ops, a.rationale, a.proposal_seconds)
        else:
            result = export_session(a.session, a.output)
    except (ValueError, FileNotFoundError) as exc:
        p.exit(2, f"Session error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
