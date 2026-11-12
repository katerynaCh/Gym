# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_gym.models.token_capture.lineage import FileLineageStore
from nemo_gym.models.token_capture.sink import (
    CaptureContext,
    capture_model_refusal,
    reset_token_sink,
    set_token_sink,
)
from nemo_gym.models.token_capture.store import TokenCaptureStore
from nemo_gym.models.turn_budget import ModelTurnBudget, ModelTurnBudgetConfig, model_turn_limit_details


@pytest.mark.parametrize(
    "status,detail",
    [
        (400, {"code": "nemo_gym_turn_budget_exhausted", "successful_requests": 65, "hard_limit": 65}),
        (422, {"code": "provider_refusal", "successful_requests": 65, "hard_limit": 65}),
        (422, {"code": "nemo_gym_turn_budget_exhausted", "successful_requests": 64, "hard_limit": 65}),
        (422, {"code": "nemo_gym_turn_budget_exhausted", "successful_requests": True, "hard_limit": 1}),
        (422, {"code": "nemo_gym_turn_budget_exhausted"}),
        (422, {"successful_requests": 65, "hard_limit": 65}),
        (422, "nemo_gym_turn_budget_exhausted"),
    ],
)
def test_only_explicit_exhausted_gateway_evidence_is_a_model_turn_limit(status, detail):
    assert model_turn_limit_details(status_code=status, response_body=json.dumps({"detail": detail})) is None


def test_exhausted_gateway_evidence_preserves_the_recorded_successful_count():
    valid = model_turn_limit_details(
        status_code=422,
        response_body='{"detail":{"code":"nemo_gym_turn_budget_exhausted","successful_requests":65,"hard_limit":65}}',
    )
    assert valid is not None and valid.successful_requests == valid.hard_limit == 65


async def test_close_releases_the_registry_without_admitting_a_reset_budget():
    budget = ModelTurnBudget(
        ModelTurnBudgetConfig(regular_turns=1, grace_turns=1, grace_tool_turns=0, grace_tool_calls=0),
        global_config={},
    )
    async with budget.next_turn("attempt") as turn:
        turn.commit(grace_tool_calls=0)
    budget.close()
    with pytest.raises(RuntimeError, match="model turn budget is closed"):
        async with budget.next_turn("attempt"):
            pytest.fail("a closed model budget cannot be reused as a fresh counter")


@pytest.mark.parametrize("stage", ["unprepared", "prepared", "dispatched", "responded", "committed"])
async def test_model_refusal_settles_only_its_own_undispatched_capture_intent(tmp_path, stage):
    ledger = FileLineageStore(tmp_path)
    await ledger.begin_call("attempt-a1", "call1")
    context = CaptureContext(
        capture_key="attempt-a1",
        model_call_id="call1",
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        intent_registered=True,
    )
    if stage == "prepared":
        from nemo_gym.models.token_capture.staging.records import CaptureAdmission

        context.capture_admission = CaptureAdmission(
            capture_key="attempt-a1", model_call_id="call1", mode="text", staging_chain=[]
        )
    elif stage == "responded":
        context.external_worker_response_seen = True
    elif stage == "committed":
        context.committed = True
    token = set_token_sink(context)
    try:
        await capture_model_refusal(code="nemo_gym_turn_budget_exhausted", upstream_attempted=stage == "dispatched")
    finally:
        reset_token_sink(token)
    manifest = await ledger.manifest("attempt-a1")
    if stage in {"unprepared", "prepared"}:
        assert context.refused
        assert not manifest["attempted_call_ids"] and not manifest["pending_call_ids"]
        assert ledger.export_rows("attempt-a1")[-1] == {"model_call_id": "call1", "refused_before_generation": True}
    else:
        assert not context.refused
        assert manifest["pending_call_ids"] == ["call1"]
        assert manifest["attempted_call_ids"] == ["call1"]
    assert not manifest["records"] and not manifest["failures"]


async def test_local_model_refusal_preserves_the_typed_evidence_and_settles_its_intent(tmp_path):
    store = TokenCaptureStore(tmp_path)
    await store.begin_call("attempt-a1", "call1")
    context = CaptureContext(capture_key="attempt-a1", model_call_id="call1", token_sink=store, intent_registered=True)
    token = set_token_sink(context)
    try:
        await capture_model_refusal(code="nemo_gym_turn_budget_exhausted", upstream_attempted=False)
    finally:
        reset_token_sink(token)
    snapshot = await store.freeze("attempt-a1")
    assert not snapshot.incomplete and not snapshot.entries
    assert len(snapshot.engine_refusals) == 1
    refusal = snapshot.engine_refusals[0]
    assert refusal.model_call_id == "call1"
    assert refusal.code == "nemo_gym_turn_budget_exhausted"
    assert context.engine_refusal_recorded
