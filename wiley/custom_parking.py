# custom_parking_env.py
from typing import Callable

from gymnasium.envs.registration import register, register_envs, registry
from highway_env.envs import ParkingEnv

import numpy as np


class CustomParkingEnv(ParkingEnv):
    PARKING_OBS = {"observation": {
            "type": "KinematicsGoal",
            "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False
        }}
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
       "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -5,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "controlled_vehicles": 1,
            "vehicles_count": 1,
            "add_walls": True
        })
        return config
    
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

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        # print(f"inside _reward, action is : {action}")
        # print(f"inside _reward, obs is : {obs}")
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {}) for agent_obs in obs)
        reward += self.config['collision_reward'] * sum(v.crashed for v in self.controlled_vehicles)
        return reward
    
    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """
        Defining success same as in original environment
        Vehicle is essentailly at the goal position
        """
        weighted_diff = -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), 0.5)
        return weighted_diff > -self.config["success_goal_reward"]


class CustomSparseParkingSparse(ParkingEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
       "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -5,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "controlled_vehicles": 1,
            "vehicles_count": 3,
            "add_walls": True
        })
        return config
    
    def __init__(
        self,
        *args,
        compute_reward: Callable[["CustomParkingEnv", np.ndarray, np.ndarray, dict, float], float],
        **kwargs
    ):
        self.compute_reward_func = compute_reward
        super().__init__(*args, **kwargs)
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Sparse reward function
        Defined as 1 if the vehicle is essentially at the the goal position
        """
        print('in here')
        weighted_diff = -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)
        if weighted_diff > -self.config["success_goal_reward"]:
            return 1
        else:
            return 0
    
    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        # print(f"inside _reward, obs is : {obs}")
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {}) for agent_obs in obs)
        reward += self.config['collision_reward'] * sum(v.crashed for v in self.controlled_vehicles)
        return reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info={}, p=0.5) == 1

class CustomParkingWithAction(ParkingEnv):
    """
    Similar to CustomParkingEnv but our _reward function is the one being defined
    """
    PARKING_OBS = {"observation": {
            "type": "KinematicsGoal",
            "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": True
        }}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
       "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -5,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "controlled_vehicles": 1,
            "vehicles_count": 3,
            "add_walls": True
        })
        return config
    
    def __init__(
        self,
        *args,
        compute_reward: Callable[["CustomParkingEnv", np.ndarray, np.ndarray, np.ndarray], float],
        **kwargs
    ):
        self.compute_reward_func = compute_reward
        super().__init__(*args, **kwargs)
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, action: np.ndarray) -> float:
        return self.compute_reward_func(self, achieved_goal, desired_goal, action)

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        # print(f"inside _reward, obs is : {obs}")
        # print(f"inside _reward, action is : {action}")
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], action) for agent_obs in obs)
        reward += self.config['collision_reward'] * sum(v.crashed for v in self.controlled_vehicles)
        return reward
    
    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """
        Defining success same as in original environment
        Vehicle is essentailly at the goal position
        """
        weighted_diff = -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), 0.5)
        return weighted_diff > -self.config["success_goal_reward"]     

env_names = ["CustomParking-v0", "CustomSparseParking-v0", "CustomParkingWithAction-v0"]
# for env_name in env_names:
#     if env_name in registry:
#         print(f"Unregistering {env_name}")
#     del registry[env_name]

print("Registering CustomParking-v0")
register(
    id="CustomParking-v0",
    entry_point="custom_parking:CustomParkingEnv",
    # entry_point='highway_env.envs:ParkingEnv',
)

print("Registering CustomSparseParking-v0")
register(
    id="CustomSparseParking-v0",
    entry_point="custom_parking:CustomSparseParkingSparse",
    # entry_point='highway_env.envs:ParkingEnv',
)

print("Registering CustomParkingWithAction-v0")
register(
    id="CustomParkingWithAction-v0",
    entry_point="custom_parking:CustomParkingWithAction",
    # entry_point='highway_env.envs:ParkingEnv',
)
print("Registered new environments")
