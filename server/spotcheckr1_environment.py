import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any, Optional
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
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._env = SpotSchedulingEnv(task_id="easy", seed=42)
        self._last_reward = 0.0
        self._done = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Spotcheckr1Observation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._last_reward = 0.0
        self._done = False
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

    def step(
        self,
        action: Spotcheckr1Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Spotcheckr1Observation:
        self._state.step_count += 1
        spot_action = SpotAction(action_type=action.action_type)
        response = self._env.step(spot_action)
        obs = response.observation
        self._last_reward = response.reward.value
        self._done = response.done
        return Spotcheckr1Observation(
            price=obs.price,
            progress=obs.progress,
            last_checkpoint=obs.last_checkpoint,
            running=obs.running,
            bid=obs.bid,
            valid_actions=obs.valid_actions,
            exposure=obs.exposure,
            reward=self._last_reward,
            done=self._done,
        )

    @property
    def state(self) -> State:
        return self._state
