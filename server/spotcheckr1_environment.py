# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Spotcheckr1 Environment Implementation.

Wraps the real SpotCheckRL environment (app/environment.py) for use with
the openenv HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import Spotcheckr1Action, Spotcheckr1Observation
except ImportError:
    from models import Spotcheckr1Action, Spotcheckr1Observation

from app.environment import SpotSchedulingEnv
from app.models import Action as SpotAction


class Spotcheckr1Environment(Environment):
    """
    SpotCheckRL environment wrapped for the openenv server interface.

    Manages a single long-running cloud job on a spot instance whose price
    evolves according to a mean-reverting jump-diffusion process. The agent
    must decide each step whether to CONTINUE, CHECKPOINT, TERMINATE, or RESUME.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the SpotCheckRL environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._env = SpotSchedulingEnv(task_id="easy", seed=42)

    def reset(self) -> Spotcheckr1Observation:
        """
        Reset the environment and return the initial observation.

        Returns:
            Spotcheckr1Observation with initial SpotCheckRL state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        obs = self._env.reset()

        return Spotcheckr1Observation(
            price=obs.price,
            progress=obs.progress,
            last_checkpoint=obs.last_checkpoint,
            running=obs.running,
            bid=obs.bid,
            valid_actions=obs.valid_actions,
            exposure=obs.exposure,
            reward=0.0,
            done=False,
        )

    def step(self, action: Spotcheckr1Action) -> Spotcheckr1Observation:  # type: ignore[override]
        """
        Execute one step in the environment.

        Args:
            action: Spotcheckr1Action with action_type string

        Returns:
            Spotcheckr1Observation with updated state, reward, and done flag
        """
        self._state.step_count += 1

        spot_action = SpotAction(action_type=action.action_type)
        response = self._env.step(spot_action)

        obs = response.observation
        done = response.done

        return Spotcheckr1Observation(
            price=obs.price,
            progress=obs.progress,
            last_checkpoint=obs.last_checkpoint,
            running=obs.running,
            bid=obs.bid,
            valid_actions=obs.valid_actions,
            exposure=obs.exposure,
            reward=response.reward.value,
            done=done,
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
