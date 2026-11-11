# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from multidict import CIMultiDict
from omegaconf import DictConfig, OmegaConf
from yarl import URL

from nemo_gym.client import NeMoGymAsyncOpenAI, ServerClient
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming, rollout_context
from nemo_gym.models.token_capture import FileLineageStore, assistant_fingerprint
from nemo_gym.models.token_capture.staging import CaptureAdmission, RolloutTokenCapture, StageResult
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


BUDGET = {"regular_turns": 55, "grace_turns": 10, "grace_tool_turns": 2, "grace_tool_calls": 2}
NOTICE = (
    "This is your final turn. You cannot use any more tools. "
    "Stop using tools and provide your final answer now using the evidence already available."
)
TOOL = {"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}}
BODY = {"model": "policy", "messages": [{"role": "user", "content": "inspect the image"}], "tools": [TOOL]}


def completion(index: int) -> dict:
    return {
        "id": f"chatcmpl-{index}",
        "created": index,
        "object": "chat.completion",
        "model": "policy",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "tool_calls": [
                        {
                            "id": f"call-{index}-{call}",
                            "type": "function",
                            "function": {"name": "inspect", "arguments": "{}"},
                        }
                        for call in range(3)
                    ],
                    "function_call": {"name": "legacy", "arguments": "{}"},
                },
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }


def model(*, budget: dict | None = BUDGET, global_config: dict | None = None, **overrides):
    config = VLLMModelConfig(
        **(
            {
                "host": "127.0.0.1",
                "port": 8080,
                "entrypoint": "app.py",
                "name": "policy",
                "base_url": "http://worker.test/v1",
                "api_key": "dummy_key",  # pragma: allowlist secret
                "model": "policy",
                "return_token_id_information": False,
                "uses_reasoning_parser": False,
                "turn_budget": budget,
            }
            | overrides
        )
    )
    server = VLLMModel(
        config=config, server_client=MagicMock(spec=ServerClient, global_config_dict=global_config or {})
    )
    requests = []

    async def complete(**body):
        requests.append(deepcopy(body))
        return completion(len(requests))

    backend = MagicMock(spec=NeMoGymAsyncOpenAI)
    backend.create_chat_completion = AsyncMock(side_effect=complete)
    server._clients = [backend]
    return server, requests, backend


def test_http_55_plus_10_limits_successful_calls_and_preserves_regular_tools():
    server, requests, _ = model()
    path = "/ng-rollout/vtc-task-a1/v1/chat/completions"
    with TestClient(server.setup_webserver()) as client:
        replies = []
        for _ in range(65):
            response = client.post(path, json=BODY)
            assert response.status_code == 200, response.text
            replies.append(response.json()["choices"][0])
        for _ in range(2):
            refused = client.post(path, json={**BODY, "metadata": {"rollout_id": "different"}})
            assert refused.status_code == 422
            assert refused.json() == {
                "detail": {
                    "code": "nemo_gym_turn_budget_exhausted",
                    "successful_requests": 65,
                    "hard_limit": 65,
                }
            }
        assert len(requests) == 65
        for turn in (0, 54):
            assert len(replies[turn]["message"]["tool_calls"]) == 3
            assert replies[turn]["message"]["function_call"]["name"] == "legacy"
        assert requests[53]["messages"] == BODY["messages"]
        assert requests[54]["messages"] == [*BODY["messages"], {"role": "user", "content": NOTICE}]
        for turn in (55, 56):
            assert requests[turn]["tool_choice"] == "auto"
            assert requests[turn]["parallel_tool_calls"] is False
            assert len(replies[turn]["message"]["tool_calls"]) == 1
            assert replies[turn]["message"].get("function_call") is None
        for turn in range(57, 65):
            assert requests[turn]["tool_choice"] == "none"
            assert not replies[turn]["message"].get("tool_calls")
            assert replies[turn]["message"].get("function_call") is None
            assert replies[turn]["finish_reason"] == "stop"
        fresh = client.post("/ng-rollout/vtc-task-a2/v1/chat/completions", json=BODY)
        assert fresh.status_code == 200
        assert len(fresh.json()["choices"][0]["message"]["tool_calls"]) == 3


def test_disabled_budget_keeps_stock_calls_and_does_not_require_rollout_route():
    server, requests, _ = model(budget=None)
    with TestClient(server.setup_webserver()) as client:
        for _ in range(66):
            response = client.post("/v1/chat/completions", json=BODY)
            assert response.status_code == 200, response.text
            assert len(response.json()["choices"][0]["message"]["tool_calls"]) == 3
    assert len(requests) == 66
    assert all(body["messages"] == BODY["messages"] for body in requests)


def test_request_and_config_overrides_cannot_restore_grace_tools_or_multiple_choices():
    server, requests, _ = model(
        budget={"regular_turns": 1, "grace_turns": 2, "grace_tool_turns": 1, "grace_tool_calls": 1},
        extra_body={
            "n": 5,
            "parallel_tool_calls": True,
            "tool_choice": "required",
            "functions": [{"name": "legacy", "parameters": {"type": "object"}}],
            "function_call": "auto",
        },
        sampling_overrides={"n": 7},
    )
    body = {
        **BODY,
        "metadata": {"extra_body": json.dumps({"n": 9, "parallel_tool_calls": True, "tool_choice": "required"})},
    }
    with TestClient(server.setup_webserver()) as client:
        for _ in range(3):
            response = client.post("/ng-rollout/override/v1/chat/completions", json=body)
            assert response.status_code == 200, response.text
    assert [request["n"] for request in requests] == [1, 1, 1]
    assert requests[-1]["tool_choice"] == "none"
    assert requests[-1]["parallel_tool_calls"] is False
    assert requests[-1]["function_call"] == "none"
    assert requests[-1]["tools"] == [TOOL]


@pytest.mark.parametrize(
    "stream,endpoint", [(False, "chat/completions"), (True, "chat/completions"), (False, "responses")]
)
def test_public_routes_enforce_the_same_budget(stream, endpoint):
    server, requests, _ = model(
        budget={"regular_turns": 1, "grace_turns": 1, "grace_tool_turns": 0, "grace_tool_calls": 0}
    )
    body = {**BODY, "stream": stream} if endpoint == "chat/completions" else {"input": "task", "model": "policy"}
    with TestClient(server.setup_webserver()) as client:
        path = f"/ng-rollout/routes/v1/{endpoint}"
        for _ in range(2):
            response = client.post(path, json=body)
            assert response.status_code == 200, response.text
        refused = client.post(path, json=body)
        assert refused.status_code == 422, refused.text
    assert len(requests) == 2


@pytest.mark.parametrize(
    "failure", [HTTPException(503, "upstream unavailable"), TimeoutError(), asyncio.CancelledError()]
)
async def test_upstream_failure_timeout_and_cancellation_do_not_spend_a_turn(failure):
    server, requests, backend = model(
        budget={"regular_turns": 1, "grace_turns": 1, "grace_tool_turns": 0, "grace_tool_calls": 0}
    )
    original = backend.create_chat_completion.side_effect
    backend.create_chat_completion.side_effect = [failure]
    request = Request({"type": "http", "headers": [], "session": {}})
    params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(BODY)
    with rollout_context("failure"):
        with pytest.raises(type(failure)):
            await server.chat_completions(request, params)
        backend.create_chat_completion.side_effect = original
        first = await server.chat_completions(request, params)
        second = await server.chat_completions(request, params)
        with pytest.raises(HTTPException) as stopped:
            await server.chat_completions(request, params)
    assert stopped.value.status_code == 422
    assert len(first.choices[0].message.tool_calls) == 3
    assert not second.choices[0].message.tool_calls
    assert len(requests) == 2


@pytest.mark.parametrize("successful_before_refusal", [0, 1])
def test_handled_context_refusal_preserves_length_and_leaves_the_same_successful_turn(successful_before_refusal):
    server, requests, backend = model(
        budget={"regular_turns": 1, "grace_turns": 1, "grace_tool_turns": 0, "grace_tool_calls": 0}
    )
    original = backend.create_chat_completion.side_effect
    info = RequestInfo(URL("http://worker.test/v1/chat/completions"), "POST", CIMultiDict(), URL("http://worker.test"))
    refusal = ClientResponseError(info, (), status=400, message="Bad Request")
    refusal.response_content = b'{"error":{"message":"maximum context length","code":400}}'
    with TestClient(server.setup_webserver()) as client:
        replies = [
            client.post("/ng-rollout/refused/v1/chat/completions", json=BODY) for _ in range(successful_before_refusal)
        ]
        backend.create_chat_completion.side_effect = refusal
        refused = client.post("/ng-rollout/refused/v1/chat/completions", json=BODY)
        assert refused.status_code == 200
        assert refused.json()["choices"][0]["finish_reason"] == "length"
        backend.create_chat_completion.side_effect = original
        replies.extend(
            client.post("/ng-rollout/refused/v1/chat/completions", json=BODY)
            for _ in range(2 - successful_before_refusal)
        )
        stopped = client.post("/ng-rollout/refused/v1/chat/completions", json=BODY)
    assert all(response.status_code == 200 for response in replies)
    assert len(replies[0].json()["choices"][0]["message"]["tool_calls"]) == 3
    assert not replies[1].json()["choices"][0]["message"].get("tool_calls")
    assert stopped.status_code == 422
    assert backend.create_chat_completion.await_count == 3
    assert len(requests) == 2


def test_valid_zero_usage_completion_still_consumes_a_successful_request():
    server, requests, backend = model(
        budget={"regular_turns": 1, "grace_turns": 1, "grace_tool_turns": 0, "grace_tool_calls": 0}
    )
    payload = completion(1)
    payload["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    backend.create_chat_completion.side_effect = None
    backend.create_chat_completion.return_value = payload
    with TestClient(server.setup_webserver()) as client:
        for _ in range(2):
            assert client.post("/ng-rollout/zero-usage/v1/chat/completions", json=BODY).status_code == 200
        stopped = client.post("/ng-rollout/zero-usage/v1/chat/completions", json=BODY)
    assert stopped.status_code == 422
    assert backend.create_chat_completion.await_count == 2
    assert not requests


def test_local_empty_assistant_response_does_not_consume_a_successful_model_turn():
    server, requests, backend = model(
        budget={"regular_turns": 2, "grace_turns": 1, "grace_tool_turns": 0, "grace_tool_calls": 0},
        sequential_reasoning_allowed=False,
    )
    with TestClient(server.setup_webserver()) as client:
        local = client.post(
            "/ng-rollout/local/v1/chat/completions",
            json={"model": "policy", "messages": [{"role": "assistant", "content": ""}]},
        )
        assert local.status_code == 200
        assert local.json()["choices"][0]["finish_reason"] == "content_filter"
        assert backend.create_chat_completion.await_count == 0
        for _ in range(3):
            assert client.post("/ng-rollout/local/v1/chat/completions", json=BODY).status_code == 200
        stopped = client.post("/ng-rollout/local/v1/chat/completions", json=BODY)
    assert stopped.status_code == 422 and len(requests) == 3


async def test_concurrent_last_turn_is_not_admitted_twice():
    server, requests, backend = model(
        budget={"regular_turns": 1, "grace_turns": 1, "grace_tool_turns": 0, "grace_tool_calls": 0}
    )
    original = backend.create_chat_completion.side_effect

    def request():
        return Request({"type": "http", "headers": [], "session": {}})

    params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(BODY)
    entered, release = asyncio.Event(), asyncio.Event()

    async def held(**body):
        entered.set()
        await release.wait()
        return await original(**body)

    with rollout_context("concurrent"):
        await server.chat_completions(request(), params)
        backend.create_chat_completion.side_effect = held
        first = asyncio.create_task(server.chat_completions(request(), params))
        await entered.wait()
        second = asyncio.create_task(server.chat_completions(request(), params))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
    assert sum(isinstance(result, HTTPException) and result.status_code == 422 for result in results) == 1
    assert len(requests) == 2


async def test_cancelling_active_and_queued_requests_releases_the_same_unspent_turn():
    server, requests, backend = model(
        budget={"regular_turns": 1, "grace_turns": 1, "grace_tool_turns": 0, "grace_tool_calls": 0}
    )
    original = backend.create_chat_completion.side_effect
    entered = asyncio.Event()

    async def held(**body):
        entered.set()
        await asyncio.Future()

    backend.create_chat_completion.side_effect = held
    params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(BODY)

    def request():
        return Request({"type": "http", "headers": [], "session": {}})

    with rollout_context("cancelled"):
        active = asyncio.create_task(server.chat_completions(request(), params))
        await entered.wait()
        queued = asyncio.create_task(server.chat_completions(request(), params))
        await asyncio.sleep(0)
        for pending in (queued, active):
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        backend.create_chat_completion.side_effect = original
        first = await server.chat_completions(request(), params)
        second = await server.chat_completions(request(), params)
        with pytest.raises(HTTPException) as stopped:
            await server.chat_completions(request(), params)
    assert len(first.choices[0].message.tool_calls) == 3
    assert not second.choices[0].message.tool_calls
    assert stopped.value.status_code == 422
    assert len(requests) == 2


async def test_a_malformed_backend_completion_does_not_spend_a_turn():
    server, requests, backend = model(
        budget={"regular_turns": 1, "grace_turns": 1, "grace_tool_turns": 0, "grace_tool_calls": 0}
    )
    original = backend.create_chat_completion.side_effect
    bad = completion(0)
    bad["choices"] = []
    backend.create_chat_completion.side_effect = [bad]
    params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(BODY)
    request = Request({"type": "http", "headers": [], "session": {}})
    with rollout_context("malformed"):
        with pytest.raises((ValueError, IndexError)):
            await server.chat_completions(request, params)
        backend.create_chat_completion.side_effect = original
        first = await server.chat_completions(request, params)
        second = await server.chat_completions(request, params)
    assert len(first.choices[0].message.tool_calls) == 3
    assert not second.choices[0].message.tool_calls
    assert len(requests) == 2


def test_budget_identity_comes_from_the_route_and_capacity_does_not_reset_existing_attempts():
    server, requests, _ = model(
        budget={"regular_turns": 1, "grace_turns": 1, "grace_tool_turns": 0, "grace_tool_calls": 0, "max_rollouts": 1}
    )
    with TestClient(server.setup_webserver()) as client:
        missing = client.post("/v1/chat/completions", json={**BODY, "metadata": {"rollout_id": "a"}})
        assert missing.status_code == 400
        assert client.post("/ng-rollout/a/v1/chat/completions", json=BODY).status_code == 200
        full = client.post("/ng-rollout/b/v1/chat/completions", json=BODY)
        assert full.status_code == 503
        assert client.post("/ng-rollout/a/v1/chat/completions", json=BODY).status_code == 200
        assert client.post("/ng-rollout/a/v1/chat/completions", json=BODY).status_code == 422
    assert len(requests) == 2


@pytest.mark.parametrize(
    "override", [{"num_workers": 2}, {"use_completions_api": True}, {"is_responses_native": True}]
)
def test_unsupported_budget_backends_fail_at_configuration_load(override):
    with pytest.raises(ValueError, match="turn_budget requires"):
        model(**override)
    model(budget=None, **override)


def test_partial_checkpointing_is_explicitly_refused_instead_of_resetting_budget_state():
    with pytest.raises(ValueError, match="turn_budget does not support checkpoint.enabled=true"):
        model(global_config={"checkpoint": {"enabled": True, "control_auth_token": "test-control"}})
    model(global_config={"checkpoint": {"enabled": False}})


def test_n4_profile_clears_only_advisory_steps_and_enables_the_explicit_model_budget():
    def resolve(profile):
        return GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=DictConfig(
                    {
                        "config_paths": [profile, "responses_api_models/vllm_model/configs/vllm_model.yaml"],
                        "policy_base_url": "http://worker.test/v1",
                        "policy_api_key": "dummy_key",  # pragma: allowlist secret
                        "policy_model_name": "policy",
                        "visual_agent": {
                            "responses_api_agents": {"opencode_agent": {"sandbox_config": {"image": "test-image"}}}
                        },
                    }
                ),
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )

    base = resolve("benchmarks/opencode_multimodal_benchmarks/vtcbench/opencode.yaml")
    strict = resolve("benchmarks/opencode_multimodal_benchmarks/vtcbench/opencode_hard_budget.yaml")
    plain_agent = OmegaConf.to_container(base.visual_agent.responses_api_agents.opencode_agent)
    hard_agent = OmegaConf.to_container(strict.visual_agent.responses_api_agents.opencode_agent)
    assert plain_agent["opencode_config"]["agent"]["build"]["steps"] == 65
    assert hard_agent["opencode_config"]["agent"]["build"]["steps"] is None
    hard_agent["opencode_config"]["agent"]["build"]["steps"] = 65
    assert hard_agent == plain_agent
    hard_model = strict.policy_model.responses_api_models.vllm_model
    assert dict(hard_model.turn_budget) == {**BUDGET, "max_rollouts": 10000}
    assert hard_model.num_workers == 1
    assert strict.rollout_correlation is True


def test_external_capture_records_the_sanitized_response_that_the_harness_receives(monkeypatch, tmp_path):
    records = {}

    class Sink:
        def stage(self, record, *, attachments=None):
            key = f"{record.capture_key}/{record.model_call_id}"
            records[key] = record
            return StageResult(ok=True, staging_key=key)

    capture = RolloutTokenCapture(sink=Sink(), weight_version_fn=lambda: 7)
    global_config = {
        "token_id_capture": {
            "enabled": True,
            "external_staging": True,
            "parent_resolution": "source_history",
            "rebuild_response": False,
            "lineage_store": "nemo_gym.models.token_capture.lineage:FileLineageStore",
            "lineage_store_kwargs": {"root": str(tmp_path)},
        }
    }
    monkeypatch.setenv("NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN", "test-control")
    monkeypatch.setattr("nemo_gym.client.server_client.get_global_config_dict", lambda: global_config)
    server, _, backend = model(
        budget={"regular_turns": 1, "grace_turns": 2, "grace_tool_turns": 1, "grace_tool_calls": 1},
        global_config=global_config,
    )
    requests = []

    async def complete(**body):
        requests.append(body)
        admission = CaptureAdmission.model_validate(body["ng_capture"])
        prefix = [token for key in admission.staging_chain for token in records[key].token_ids_delta]
        active = capture.begin_call(admission, prefix_token_ids=prefix)
        index = len(requests)
        coords = capture.complete_call(
            active,
            prompt_token_ids=prefix + [10 * index],
            generated_token_ids=[100 + index],
            generated_logprobs=[-0.1],
        )
        payload = completion(index)
        payload["ng_commit_coords"] = coords.model_dump(mode="json")
        return payload

    backend.create_chat_completion.side_effect = complete
    history = deepcopy(BODY["messages"])
    ledger = FileLineageStore(tmp_path)
    with TestClient(server.setup_webserver()) as client:
        for expected_calls in (3, 1, 0):
            response = client.post(
                "/ng-rollout/budget-capture/training-token-capture/v1/chat/completions",
                json={**BODY, "messages": history},
            )
            assert response.status_code == 200, response.text
            served = response.json()["choices"][0]["message"]
            assert len(served.get("tool_calls") or []) == expected_calls
            manifest = asyncio.run(ledger.manifest("budget-capture"))
            assert not manifest["failures"] and not manifest["pending_call_ids"]
            recorded = [row for row in ledger.export_rows("budget-capture") if row.get("staging_key")][-1]
            assert recorded["fingerprint"] == assistant_fingerprint(history + [served])
            if expected_calls < 3:
                assert recorded["fingerprint"] != assistant_fingerprint(
                    history + [completion(len(requests))["choices"][0]["message"]]
                )
            history.append({key: value for key, value in served.items() if value is not None})
            history.extend(
                {"role": "tool", "tool_call_id": call["id"], "content": "result"}
                for call in served.get("tool_calls") or []
            )
            history.append({"role": "user", "content": "continue"})
        stopped = client.post(
            "/ng-rollout/budget-capture/training-token-capture/v1/chat/completions",
            json={**BODY, "messages": history},
        )
        assert stopped.status_code == 422
        assert stopped.json()["detail"]["code"] == "nemo_gym_turn_budget_exhausted"
        manifest = asyncio.run(ledger.manifest("budget-capture"))
        assert len(manifest["records"]) == len(manifest["attempted_call_ids"]) == 3
        assert not manifest["pending_call_ids"]
        assert not manifest["failures"]
        refusals = [row for row in ledger.export_rows("budget-capture") if row.get("refused_before_generation")]
        assert len(refusals) == 1
        assert refusals[0]["model_call_id"] not in manifest["attempted_call_ids"]
        assert backend.create_chat_completion.await_count == 3
