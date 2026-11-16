"""Probe the configured advisor with synthetic JSON only; never load experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.agents.base_agent import AdvisoryAgentBase
from agent.agents.advisory_response import exception_diagnostics
from agent.hpo.resources import CampaignResourceBudget


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="GLM-4.7")
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--proxy-mode", choices=["environment", "direct"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.requests <= 5:
        parser.error("requests must be between 1 and 5")
    if args.proxy_mode:
        os.environ["SR_AGENT_LLM_PROXY_MODE"] = args.proxy_mode
    agent = AdvisoryAgentBase(args.model, 0, 1, False)
    rows = []
    with tempfile.TemporaryDirectory(prefix="sr-advisor-probe-") as directory:
        budget = CampaignResourceBudget(Path(directory) / "budget.json", limits={"max_wall_seconds": 90})
        agent._resource_budget = budget
        try:
            for index in range(args.requests):
                started = time.monotonic()
                try:
                    value = agent._request_json_advice('Return only the JSON object {"ok":true}.')
                    row = {"probe": index + 1, "success": value == {"ok": True}, "result": value}
                except Exception as exc:
                    row = {"probe": index + 1, "success": False, **exception_diagnostics(exc)}
                row.update(duration_seconds=time.monotonic() - started,
                           transport_attempts=getattr(agent, "_advisor_transport_events", []))
                rows.append(row)
                if not row["success"]:
                    break
        finally:
            budget.close()
    report = {"model": args.model, "proxy_mode": os.getenv("SR_AGENT_LLM_PROXY_MODE", "environment"), "probes": rows}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return int(any(not row["success"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
