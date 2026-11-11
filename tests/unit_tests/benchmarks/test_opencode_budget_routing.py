# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from urllib.parse import urlsplit

from fastapi.testclient import TestClient

from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import close_body, seed, setup  # noqa: F401
from responses_api_models.vllm_model.tests.test_turn_budget import BODY, model


def test_stock_session_url_dispatches_attempt_budget_with_observability_and_capture_disabled(setup):
    agent, sandbox = setup
    agent.server_client.global_config_dict.update(
        {"observability_enabled": False, "token_id_capture": {"enabled": False}, "rollout_correlation": True}
    )
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 200, response.text
        config = json.loads(sandbox.params()["env"]["OPENCODE_CONFIG_CONTENT"])
        provider = config["provider"]["nemo_gym"]
        url = provider["options"]["baseURL"]
        assert url == "http://model.example:9000/ng-rollout/opencode-smoke-a2/v1"
        assert not any("assistant_message_header" in plugin for plugin in config["plugin"])
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text

    gateway, requests, _ = model(
        budget={"regular_turns": 1, "grace_turns": 1, "grace_tool_turns": 0, "grace_tool_calls": 0}
    )
    path = urlsplit(url).path + "/chat/completions"
    with TestClient(gateway.setup_webserver()) as client:
        assert client.post(path, json=BODY).status_code == 200
        assert client.post(path, json=BODY).status_code == 200
        stopped = client.post(path, json=BODY)
        assert stopped.status_code == 422, stopped.text
        assert stopped.json()["detail"]["successful_requests"] == 2
    assert len(requests) == 2
