"""Manual smoke test for the advisory LLM boundary.

Run this test explicitly when debugging API connectivity:

    SR_AGENT_LLM_SMOKE=1 python -m pytest tests/smoke/test_base_agent_llm.py -s

It uses AdvisoryAgentBase directly, so failures here isolate the .env,
ChatOpenAI, base_url, authentication, and network path before the HPO workflow.
"""

from __future__ import annotations

import os
import time
import traceback

import pytest

from agent.agents.base_agent import AdvisoryAgentBase


pytestmark = pytest.mark.skipif(
    os.getenv("SR_AGENT_LLM_SMOKE") != "1",
    reason="set SR_AGENT_LLM_SMOKE=1 to call the real advisory LLM",
)


def test_base_agent_llm_connectivity() -> None:
    model_name = os.getenv("SR_AGENT_LLM_SMOKE_MODEL", "GLM-4.7")
    prompt = os.getenv("SR_AGENT_LLM_SMOKE_PROMPT", "Reply with only pong.")
    agent = AdvisoryAgentBase(
        model_name=model_name,
        temperature=0.2,
        max_iterations=1,
        verbose=True,
    )
    llm = agent.llm

    print("model:", model_name)
    print("ZHIPUAI_API_BASE_URL:", os.getenv("ZHIPUAI_API_BASE_URL"))
    print("ZHIPUAI_API_KEY set:", bool(os.getenv("ZHIPUAI_API_KEY")))
    start = time.monotonic()

    try:
        response = llm.invoke(prompt)
    except Exception:
        print("LLM call failed after %.2fs" % (time.monotonic() - start))
        traceback.print_exc()
        raise

    elapsed = time.monotonic() - start
    content = agent._extract_message_content(response)
    print("elapsed_seconds:", "%.2f" % elapsed)
    print("response_type:", type(response).__name__)
    print("response_content:", content)

    assert content.strip()
