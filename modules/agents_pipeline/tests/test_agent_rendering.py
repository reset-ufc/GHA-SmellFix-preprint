"""The user turn reaches the model byte for byte.

Agno renders every message with ``string.Template.safe_substitute`` unless the
agent sets ``resolve_in_context=False``; with rendering on, a workflow's ``$$``
arrives at the model as ``$``. Each factory's agent is run against a mocked
OpenAI-compatible endpoint and the captured request body is checked. No LLM, no
network.
"""

from __future__ import annotations

import contextlib
import json

import gha_agents.agents as agents
import httpx
import pytest
from agno.models.openai import OpenAIChat
from gha_agents.agents import (
    ValidSmells,
    make_combined_agent,
    make_detection_agent,
    make_refactor_agent,
)

_SMELL = next(iter(ValidSmells))
_TURN = (
    "<workflow>\n"
    "on: push\n"
    "jobs:\n"
    "  clean:\n"
    "    runs-on: ubuntu-latest\n"
    "    steps:\n"
    '      - run: TMPFILE=/tmp/tmp.$$ && echo "$HOME ${VAR} ${{ github.sha }}"\n'
    "</workflow>"
)


def _capturing_model(requests: list[dict]) -> OpenAIChat:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            400, json={"error": {"message": "captured", "type": "invalid_request_error"}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAIChat(id="capture", api_key="test", http_client=client)


@pytest.mark.parametrize(
    "make",
    [
        make_combined_agent,
        lambda smell: make_detection_agent(smell, "D2"),
        lambda smell: make_refactor_agent(smell, "D2"),
    ],
    ids=["combined", "detection", "refactor"],
)
def test_user_turn_reaches_the_model_verbatim(monkeypatch, make):
    requests: list[dict] = []
    monkeypatch.setenv("AGNO_TELEMETRY", "false")
    monkeypatch.setattr(agents._MODEL_FACTORY, "build", lambda role: _capturing_model(requests))
    agent = make(_SMELL)
    agent.telemetry = False
    with contextlib.suppress(Exception):  # the mocked 400 ends the run; only the request matters
        agent.run(_TURN)
    user_turns = [m["content"] for m in requests[0]["messages"] if m["role"] == "user"]
    assert user_turns == [_TURN]
