# custom_parking_env.py
from typing import Callable

from highway_env.envs import ParkingEnv

import numpy as np


class CustomParkingEnv(ParkingEnv):
    def __init__(
        self,
        *args,
        compute_reward: Callable[[np.ndarray, np.ndarray, dict, float], float],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.compute_reward = compute_reward
