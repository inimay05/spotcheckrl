# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Spotcheckr1 Environment.

Wraps the SpotCheckRL observation and action schemas for use with the
openenv server infrastructure.
"""

from typing import List

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class Spotcheckr1Action(Action):
    """Action for the SpotCheckRL environment."""

    action_type: str = Field(..., description="One of: CONTINUE, CHECKPOINT, TERMINATE, RESUME")


class Spotcheckr1Observation(Observation):
    """Observation from the SpotCheckRL environment."""

    price: float = Field(default=0.0, description="Current spot market price")
    progress: float = Field(default=0.0, description="Job completion fraction [0, 1]")
    last_checkpoint: float = Field(default=0.0, description="Progress saved to durable storage [0, 1]")
    running: bool = Field(default=True, description="True if the instance is currently active")
    bid: float = Field(default=0.0, description="Interruption threshold — price above this triggers rollback")
    valid_actions: List[str] = Field(default_factory=list, description="Legal actions in the current state")
    exposure: float = Field(default=0.0, description="Unsaved progress at risk: progress - last_checkpoint")
    reward: float = Field(default=0.0, description="Reward received this step")
    done: bool = Field(default=False, description="True if the episode has ended")
