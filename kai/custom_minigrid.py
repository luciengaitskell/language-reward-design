# custom_parking_env.py
from typing import Callable

from gymnasium.envs.registration import register
from minigrid.envs.lockedroom import LockedRoomEnv

from gymnasium import spaces

import numpy as np


class CustomMinigridEnv(LockedRoomEnv):
    def __init__(
        self,
        *args,
        compute_reward: Callable[["CustomMinigridEnv", spaces.Dict, dict], float],
        **kwargs
    ):
        self.compute_reward_func = compute_reward
        super().__init__(*args, **kwargs)
        
    def compute_reward(self, current_state: spaces.Dict, info: dict) -> float:
        return self.compute_reward_func(self, current_state, info)


register(
    id="CustomLockedRoom-v0",
    entry_point="custom_lockedroom:CustomMinigridEnv",
    # entry_point='highway_env.envs:ParkingEnv',
)

### to make work with VLM, we will have to customize our reward as something based on colors, etc.
### eg VLM1 -> color purple is best, higher reward in that direction, VLM2 -> translate that into code; distance from purple coordinate in observation, distance from key, distance from door
