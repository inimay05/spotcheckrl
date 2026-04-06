"""
Pydantic v2 schemas for SpotCheckRL.
All constants imported from app/config.py — no magic numbers here.
"""

from __future__ import annotations

import json
from typing import List, Literal

from pydantic import BaseModel, Field, computed_field, model_validator

from app.config import IDEAL_STEPS


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "price": 11.3,
                    "progress": 0.4,
                    "last_checkpoint": 0.2,
                    "running": True,
                    "bid": 13.0,
                }
            ]
        }
    }

    price: float = Field(..., ge=0, description="Current spot price")
    progress: float = Field(..., ge=0, le=1, description="Job completion 0–1")
    last_checkpoint: float = Field(..., ge=0, le=1, description="Last saved progress 0–1")
    running: bool = Field(..., description="True = instance is active")
    bid: float = Field(..., ge=0, description="Interruption threshold for this task")

    # ---- computed fields (populated by model_validator) ----
    valid_actions: List[str] = Field(default_factory=list)
    exposure: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def _compute_derived(self) -> "Observation":
        # valid_actions depends on running state
        if self.running:
            self.valid_actions = ["CONTINUE", "CHECKPOINT", "TERMINATE"]
        else:
            self.valid_actions = ["RESUME"]

        # exposure = unsaved progress (clamp to [0, 1] for safety)
        raw = self.progress - self.last_checkpoint
        self.exposure = round(max(0.0, min(1.0, raw)), 4)
        return self


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"action_type": "CONTINUE"},
            ]
        }
    }

    action_type: Literal["CONTINUE", "CHECKPOINT", "TERMINATE", "RESUME"]


# ---------------------------------------------------------------------------
# Reward breakdown
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    progress_reward: float = Field(..., description="Reward for progress made this step")
    checkpoint_penalty: float = Field(..., le=0, description="Cost of checkpointing (always <= 0)")
    rollback_penalty: float = Field(..., le=0, description="Loss from rollback on interruption (always <= 0)")
    completion_bonus: float = Field(..., ge=0, description="Bonus awarded on job completion (always >= 0)")
    time_penalty: float = Field(..., le=0, description="Per-step cost for running (always <= 0)")
    total: float = Field(..., description="Sum of all reward terms")

    @model_validator(mode="after")
    def _verify_total(self) -> "RewardBreakdown":
        expected = round(
            self.progress_reward
            + self.checkpoint_penalty
            + self.rollback_penalty
            + self.completion_bonus
            + self.time_penalty,
            6,
        )
        self.total = expected
        return self


class Reward(BaseModel):
    value: float
    breakdown: RewardBreakdown


# ---------------------------------------------------------------------------
# Step response
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict = Field(
        ...,
        description="Must contain: interrupted (bool), reason (str), action_was_valid (bool)",
    )


# ---------------------------------------------------------------------------
# Episode result (used by graders)
# ---------------------------------------------------------------------------

class EpisodeResult(BaseModel):
    task_id: str
    total_reward: float
    steps_taken: int = Field(..., ge=0)
    completed: bool
    interruption_count: int = Field(..., ge=0)
    checkpoint_count: int = Field(..., ge=0)
    final_progress: float = Field(..., ge=0, le=1)
    total_rollback_loss: float = Field(..., description="Cumulative rollback penalty this episode")
    ideal_steps: int = Field(
        default=IDEAL_STEPS,
        description="Target steps for perfect score — from config",
    )


# ---------------------------------------------------------------------------
# Schema preview
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Observation JSON Schema ===")
    print(json.dumps(Observation.model_json_schema(), indent=2))
    print()
    print("=== Action JSON Schema ===")
    print(json.dumps(Action.model_json_schema(), indent=2))
