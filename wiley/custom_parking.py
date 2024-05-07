# custom_parking_env.py
from typing import Callable

from gymnasium.envs.registration import register
from highway_env.envs import ParkingEnv

import numpy as np


class CustomParkingEnv(ParkingEnv):
    def __init__(
        self,
        *args,
        compute_reward: Callable[["CustomParkingEnv", np.ndarray, np.ndarray, dict, float], float],
        **kwargs
    ):
        self.compute_reward_func = compute_reward
        super().__init__(*args, **kwargs)
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        return self.compute_reward_func(self, achieved_goal, desired_goal, info, p)


register(
    id="CustomParking-v0",
    entry_point="custom_parking:CustomParkingEnv",
    # entry_point='highway_env.envs:ParkingEnv',
)
