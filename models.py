from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional

class Spotcheckr1Action(Action):
    action_type: str = Field(default="CONTINUE", description="One of: CONTINUE, CHECKPOINT, TERMINATE, RESUME")

class Spotcheckr1Observation(Observation):
    price: float = Field(default=0.0)
    progress: float = Field(default=0.0)
    last_checkpoint: float = Field(default=0.0)
    running: bool = Field(default=True)
    bid: float = Field(default=18.0)
    valid_actions: List[str] = Field(default_factory=list)
    exposure: float = Field(default=0.0)
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
