# custom_parking_env.py
from typing import Callable, List

from gymnasium.envs.registration import register
from minigrid.envs.lockedroom import LockedRoomEnv

from gymnasium import spaces

import numpy as np


class CustomMinigridEnv(LockedRoomEnv):
    def __init__(
        self,
        *args,
        compute_reward: Callable[["CustomMinigridEnv", spaces.Dict], float],
        **kwargs
    ):  
        self.compute_reward_func = compute_reward
        super().__init__(*args, **kwargs)
        
        
    def _reward(self, current_state: spaces.Dict) -> float:
        return None#self.compute_reward_func[0](self, current_state)
    
    def get_reward(self, obs):
        return self._reward(obs)


register( id="CustomLockedRoom-v0", entry_point=CustomMinigridEnv)
# register( id="CustomLockedRoom-v0", entry_point=CustomMinigridEnv)
    # entry_point='highway_env.envs:ParkingEnv')

### to make work with VLM, we will have to customize our reward as something based on colors, etc.
### eg VLM1 -> color purple is best, higher reward in that direction, VLM2 -> translate that into code; distance from purple coordinate in observation, distance from key, distance from door
