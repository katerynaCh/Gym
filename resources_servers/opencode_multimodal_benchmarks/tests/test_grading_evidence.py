# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scores must bind the saved answer and only actually parsed judge replies."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.core import rollout_context
from nemo_gym.server_runtime import observe_verifier
from nemo_gym.trajectory import grading_result_binding
from nemo_gym.verification import judge_failsafe
from resources_servers.opencode_multimodal_benchmarks.app import VisualBenchmarksServer
from resources_servers.opencode_multimodal_benchmarks.tests.test_app import request, server  # noqa: F401


def judge_reply(text: str, identifier: str = "verdict") -> MagicMock:
    payload = {
        "id": identifier,
        "object": "chat.completion",
        "created": 0,
        "model": "judge",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
    }
    http = MagicMock(status=200)
    http.read = AsyncMock(return_value=json.dumps(payload).encode())
    http.json = AsyncMock(return_value=payload)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=None)
    return http


async def observed(server: VisualBenchmarksServer, body: object) -> dict:
    with rollout_context("grading-evidence"):
        result = await observe_verifier(
            judge_failsafe(server.verify),
            enabled=True,
            verifier_name=server.config.name,
            uninstrumented_models=server.grading_may_use_uninstrumented_model,
        )(body)
    return json.loads(result.body) if hasattr(result, "body") else result.model_dump(mode="json")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "benchmark,answer,reply,expected",
    [
        ("VStarBench", "the fuzzy feline", "A", 1),
        ("HRBench8K", "the fuzzy feline", "B", 0),
        ("VTCBench", "an uncertain animal", "A", 1),
    ],
)
async def test_native_judge_binds_actual_response_and_reward(server, benchmark, answer, reply, expected):
    server.config.judge_server = "judge"
    server.server_client.post = AsyncMock(return_value=judge_reply(reply))
    result = await observed(server, request(benchmark, answer))
    assert result["reward"] == expected and result["mask_sample"] is False
    (invocation,) = result["ng_verifier_invocations"]
    (grade,) = invocation["grading_operations"]
    assert grade["kind"] == "llm_judge" and grade["outcome"] == "completed"
    assert grade["dependencies"] == [
        {
            "model_ref": {"name": "judge", "type": "responses_api_models"},
            "model_call_id": None,
            "response_id": "verdict",
        }
    ]
    assert grade["result_binding_sha256"] == grading_result_binding(result)
    assert grade["result_binding_sha256"] != grading_result_binding(result | {"reward": 1 - expected})
    changed = request(benchmark, "different answer").response.model_dump(mode="json")
    assert grade["result_binding_sha256"] != grading_result_binding(result | {"response": changed})


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,expected", [("A", 1), ("B", 0), ("", 0)])
async def test_programmatic_score_never_needs_a_judge(server, answer, expected):
    server.server_client.post = AsyncMock(side_effect=AssertionError("Unexpected judge"))
    result = await observed(server, request("VStarBench", answer))
    assert result["reward"] == expected and not result["mask_sample"]
    (grade,) = result["ng_verifier_invocations"][0]["grading_operations"]
    assert grade["kind"] == "programmatic_calculation" and grade["outcome"] == "completed"
    assert grade["dependencies"] == []
    assert grade["result_binding_sha256"] == grading_result_binding(result)
    server.server_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_judge_is_not_certified_as_model_free(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-fixture")
    monkeypatch.setattr(
        "resources_servers.opencode_multimodal_benchmarks.app.request", AsyncMock(return_value=judge_reply("A"))
    )
    result = await observed(server, request("VStarBench", "the fuzzy feline"))
    assert result["reward"] == 1
    (invocation,) = result["ng_verifier_invocations"]
    assert invocation.get("grading_record") is None
    (grade,) = invocation["grading_operations"]
    assert grade["kind"] == "llm_judge" and grade["outcome"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [None, "not a verdict"])
async def test_judge_fault_keeps_the_answer_and_can_be_rejudged(server, reply):
    server.config.judge_server = "judge"
    server.config.judge_attempts = 1
    body = request("VStarBench", "the fuzzy feline")
    server.server_client.post = AsyncMock(
        return_value=judge_reply(reply or ""), side_effect=TimeoutError("scripted outage") if reply is None else None
    )
    failed = await observed(server, body)
    assert failed["mask_sample"] is True and failed["failure_kind"] == "judge_failed"
    assert failed["response"] == body.response.model_dump(mode="json")
    assert failed["ng_verifier_invocations"][0]["grading_operations"][0]["outcome"] == "failed"
    server.server_client.post = AsyncMock(return_value=judge_reply("A", "rejudged"))
    repaired = await observed(server, body)
    assert repaired["reward"] == 1 and not repaired["mask_sample"]
    assert repaired["response"] == failed["response"]
    assert repaired["ng_verifier_invocations"][0]["grading_operations"][0]["outcome"] == "completed"


@pytest.mark.asyncio
async def test_malformed_reply_before_valid_verdict_is_not_falsely_certified(server):
    server.config.judge_server = "judge"
    server.server_client.post = AsyncMock(
        side_effect=[judge_reply("unparseable", "invalid"), judge_reply("A", "valid")]
    )
    result = await observed(server, request("VTCBench", "an uncertain animal"))
    assert result["reward"] == 1 and not result["mask_sample"]
    (operation,) = result["ng_verifier_invocations"][0]["grading_operations"]
    assert operation["outcome"] == "unknown"
    assert {ref["response_id"] for ref in operation["dependencies"]} == {"invalid", "valid"}


@pytest.mark.parametrize("benchmark", ["VStarBench", "VTCBench"])
@pytest.mark.parametrize("failed", [False, True])
def test_http_option_text_distinguishes_api_failure_from_a_valid_answer(server, benchmark, failed):
    server.server_client.global_config_dict = {"observability_enabled": True}
    server.config.judge_server = "judge"
    server.server_client.post = AsyncMock(return_value=judge_reply("Z"))
    prediction = "Failed to obtain answer via API. a small cat" if failed else "The API returned a small cat"
    body = request(benchmark, prediction, options={"A": "a small cat", "B": "a very large dog"})
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/ng-rollout/api-option/verify", json=body.model_dump(mode="json"))
    assert response.status_code == 200
    result = response.json()
    assert result["reward"] == (0 if failed else 1)
    assert result["mask_sample"] is False and result["evaluation_completed"] is True
    assert result["response"] == body.response.model_dump(mode="json")
    (grade,) = result["ng_verifier_invocations"][0]["grading_operations"]
    assert grade["outcome"] == "completed"
    if failed:
        assert result["extraction_method"] == "llm" and grade["kind"] == "llm_judge"
        server.server_client.post.assert_awaited_once()
    else:
        assert grade["kind"] == "programmatic_calculation"
        server.server_client.post.assert_not_awaited()


def test_http_api_failure_judge_replies_remain_masked_judge_failures(server):
    server.server_client.global_config_dict = {"observability_enabled": True}
    server.config.judge_server = "judge"
    failed = "Failed to obtain answer via API. a small cat"
    server.server_client.post = AsyncMock(side_effect=[judge_reply(failed, "first"), judge_reply(failed, "second")])
    body = request("VStarBench", "uncertain animal", options={"A": "a small cat", "B": "a very large dog"})
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/ng-rollout/api-judge/verify", json=body.model_dump(mode="json"))
    assert response.status_code == 200
    result = response.json()
    assert result["reward"] == 0 and result["mask_sample"] is True
    assert result["failure_kind"] == result["_ng_failure_class"] == "judge_failed"
    assert result["response"] == body.response.model_dump(mode="json")
    assert server.server_client.post.await_count == 2
    prompts = [call.kwargs["json"]["messages"][0]["content"] for call in server.server_client.post.await_args_list]
    assert prompts[1].startswith(prompts[0]) and prompts[1] != prompts[0]
    (grade,) = result["ng_verifier_invocations"][0]["grading_operations"]
    assert grade["kind"] == "llm_judge" and grade["outcome"] == "failed"


def test_http_rule_only_vtc_does_not_credit_api_failure_text(server):
    server.server_client.global_config_dict = {"observability_enabled": True}
    server.config.vtc_llm_fallback = False
    server.server_client.post = AsyncMock(side_effect=AssertionError("Rule-only verification must not judge"))
    body = request(
        "VTCBench",
        "Failed to obtain answer via API. a small cat",
        options={"A": "a small cat", "B": "a very large dog"},
    )
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/ng-rollout/api-rule/verify", json=body.model_dump(mode="json"))
    assert response.status_code == 200
    result = response.json()
    assert result["reward"] == 0 and result["mask_sample"] is False
    assert result["extraction_method"] == "rule" and result["scorer"] == "vtcbench-final-answer-rules-v4"
    server.server_client.post.assert_not_awaited()
