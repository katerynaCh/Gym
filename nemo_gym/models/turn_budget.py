# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Successful model-request budgets and their explicit terminal wire evidence."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.core.episode_types import EpisodeId
from nemo_gym.server_runtime._checkpoint.settings import checkpoint_settings


MODEL_TURN_LIMIT_CODE = "nemo_gym_turn_budget_exhausted"
FINAL_MODEL_TURN_NOTICE = (
    "This is your final turn. You cannot use any more tools. "
    "Stop using tools and provide your final answer now using the evidence already available."
)


class ModelTurnBudgetConfig(BaseModel):
    """Limit successful requests of each physical episode attempt, independently of client steps.

    State is retained until the model server shuts down. At capacity, new attempts are refused;
    existing counters are never evicted or reset to admit another attempt.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    regular_turns: int = Field(gt=0, strict=True)
    grace_turns: int = Field(gt=0, strict=True)
    grace_tool_turns: int = Field(ge=0, strict=True)
    grace_tool_calls: int = Field(ge=0, strict=True)
    max_rollouts: int = Field(default=10000, gt=0, strict=True)

    @model_validator(mode="after")
    def reserve_text_only_turn(self) -> Self:
        if self.grace_tool_turns >= self.grace_turns:
            raise ValueError("The grace period must end with at least one text-only turn")
        return self

    @property
    def hard_limit(self) -> int:
        return self.regular_turns + self.grace_turns

    def tool_allowance(self, *, turn: int, used_grace_calls: int) -> int | None:
        """Preserve regular parallel tools; grace permits at most one call on each allowed turn."""
        if turn <= self.regular_turns:
            return None
        if turn <= self.regular_turns + self.grace_tool_turns:
            return min(1, max(0, self.grace_tool_calls - used_grace_calls))
        return 0


class ModelTurnLimitDetails(BaseModel):
    """Evidence in the gateway's HTTP 422 detail, distinct from an ordinary provider refusal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: Literal["nemo_gym_turn_budget_exhausted"] = MODEL_TURN_LIMIT_CODE
    successful_requests: int = Field(ge=0, strict=True)
    hard_limit: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def require_exhausted_budget(self) -> Self:
        if self.successful_requests < self.hard_limit:
            raise ValueError("The successful request count has not reached the hard limit")
        return self


def model_turn_limit_details(*, status_code: object, response_body: object) -> ModelTurnLimitDetails | None:
    """Read only the gateway's exact typed refusal from a saved native APIError body."""
    if type(status_code) is not int or status_code != 422 or not isinstance(response_body, str):
        return None
    if len(response_body) > 32768:
        return None
    try:
        body = json.loads(response_body)
        if (
            not isinstance(body, dict)
            or not isinstance(body.get("detail"), dict)
            or body["detail"].get("code") != MODEL_TURN_LIMIT_CODE
        ):
            return None
        return ModelTurnLimitDetails.model_validate(body["detail"])
    except ValueError:
        return None


class ModelTurnLimitReached(Exception):
    """The attempt has consumed its successful-request budget."""

    def __init__(self, details: ModelTurnLimitDetails) -> None:
        self.details = details
        super().__init__("Successful model-request turn limit reached")


class ModelTurnBudgetCapacityReached(Exception):
    """The bounded registry cannot admit another episode attempt without losing an existing budget."""


@dataclass
class _TurnState:
    successful_requests: int = 0
    grace_tool_calls: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(kw_only=True)
class ModelBudgetTurn:
    """One serialized attempt to produce a valid model completion; exceptions leave it uncommitted."""

    number: int
    tool_allowance: int | None
    _state: _TurnState
    _committed: bool = field(default=False, init=False)

    def commit(self, *, grace_tool_calls: int) -> None:
        """Advance once, after the caller validates and sanitizes its served completion."""
        if self._committed:
            raise RuntimeError("The model budget turn was already committed")
        if type(grace_tool_calls) is not int or grace_tool_calls < 0:
            raise ValueError("grace_tool_calls must be a nonnegative integer")
        if (self.tool_allowance is None and grace_tool_calls) or (
            self.tool_allowance is not None and grace_tool_calls > self.tool_allowance
        ):
            raise ValueError("The served tool calls exceed this turn's grace allowance")
        self._state.successful_requests = self.number
        self._state.grace_tool_calls += grace_tool_calls
        self._committed = True


class ModelTurnBudget:
    """Own bounded, non-evicting per-attempt counters for one model-server process."""

    def __init__(self, config: ModelTurnBudgetConfig, *, global_config: object) -> None:
        if checkpoint_settings(global_config) is not None:
            raise ValueError(
                "turn_budget does not support checkpoint.enabled=true: model budget counters are not exported; "
                "use complete-rollout recovery, which starts a fresh physical attempt, instead of partial continuation"
            )
        self.config = config
        self._states: dict[str, _TurnState] = {}
        self._closed = False

    @asynccontextmanager
    async def next_turn(self, capture_key: str) -> AsyncIterator[ModelBudgetTurn]:
        """Serialize requests of the route's physical attempt, without accepting client metadata as identity."""
        key = EpisodeId.from_capture_key(capture_key).capture_key
        if self._closed:
            raise RuntimeError("The model turn budget is closed")
        state = self._states.get(key)
        if state is None:
            if len(self._states) >= self.config.max_rollouts:
                raise ModelTurnBudgetCapacityReached("Model turn-budget capacity reached; existing counters retained")
            state = self._states[key] = _TurnState()
        async with state.lock:
            if self._closed:
                raise RuntimeError("The model turn budget is closed")
            if state.successful_requests >= self.config.hard_limit:
                raise ModelTurnLimitReached(
                    ModelTurnLimitDetails(
                        successful_requests=state.successful_requests,
                        hard_limit=self.config.hard_limit,
                    )
                )
            number = state.successful_requests + 1
            yield ModelBudgetTurn(
                number=number,
                tool_allowance=self.config.tool_allowance(turn=number, used_grace_calls=state.grace_tool_calls),
                _state=state,
            )

    def close(self) -> None:
        """Release counters after the server's serving lifespan drains; a closed registry admits no calls."""
        self._closed = True
        self._states.clear()
