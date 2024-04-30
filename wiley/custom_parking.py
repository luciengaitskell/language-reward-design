# custom_parking_env.py
from highway_env.envs.parking_env import ParkingEnv
import numpy as np
from gym.envs.registration import register

class CustomParkingEnv(ParkingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _reward(self, action):
        cur_pos = self.vehicle.position
        desired_pos = self.goal_position
        reward = -np.linalg.norm(cur_pos - desired_pos)
        return reward
    
register(
    id='CustomParking-v0',
    entry_point='custom_parking_env:CustomParkingEnv',
)