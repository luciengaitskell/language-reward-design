import gymnasium as gym
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
import base64
from PIL import Image
import io
import os
from openai import OpenAI
from openaikey import OPENAI_API_KEY
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.logger import configure
import imageio
from utils import record_videos, show_videos


from typing import Callable, Dict, Text

from gymnasium.envs.registration import register, register_envs, registry
from highway_env.envs import RacetrackEnv
from utils import lmap

import numpy as np

def env_render_to_base64(env):
    env.reset()
    rendered = env.render()
    # Convert numpy array to PIL Image
    image = Image.fromarray(rendered)

    # Save the image to a BytesIO object
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')

    # Get the content of the BytesIO object
    image_bytes = buffer.getvalue()

    # Encode the bytes to base64
    image_base64 = base64.b64encode(image_bytes)

    # If you need it as a string
    image_base64_str = image_base64.decode('utf-8')
    return image_base64_str

def prompt_api_vision(openai_client, prompt, image64) -> str:
    response = openai_client.chat.completions.create(
    model="gpt-4-turbo",
    messages = [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt,
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image64}",
                "detail": "low"
            },
            },
        ],
        }
    ],
    max_tokens=600,
    )
    return response.choices[0].message.content

def prompt_api_no_vision(openai_client, prompt1) -> str:
    response = openai_client.chat.completions.create(
    model="gpt-4-turbo",
    messages = [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt1,
            },
        ],
        }
    ],
    max_tokens=600,
    )
    return response.choices[0].message.content


class CustomRaceEnv(RacetrackEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
                "target_speeds": [0, 5, 10]
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 300,
            "collision_reward": -1,
            "lane_centering_cost": 4,
            "lane_centering_reward": 1,
            "action_reward": -0.3,
            "controlled_vehicles": 1,
            "other_vehicles": 1,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.5],
        })
        return config
    
    def __init__(
        self,
        *args,
        reward_func: Callable[["CustomRaceEnv", np.ndarray, dict], float],
        **kwargs
    ):
        self.compute_reward_func = reward_func
        super().__init__(*args, **kwargs)

    # def _reward(self, action: np.ndarray) -> float:
    #     rewards_dict = self._rewards(action)
    #     reward = self.compute_reward_func(self, action, rewards_dict)
    #     return reward

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        return {
            "lane_centering_reward": 1/(1+self.config["lane_centering_cost"]*lateral**2),
            "action_reward": np.linalg.norm(action),
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
        }
    
print("Registering CustomRace-v0")
register(
    id="CustomRace-v0",
    entry_point="custom_race:CustomRaceEnv",
)

# print("Registering CustomSparseParking-v0")
# register(
#     id="CustomSparseParking-v0",
#     entry_point="custom_parking:CustomSparseParkingSparse",
#     # entry_point='highway_env.envs:ParkingEnv',
# )

# print("Registering CustomParkingWithAction-v0")
# register(
#     id="CustomParkingWithAction-v0",
#     entry_point="custom_parking:CustomParkingWithAction",
#     # entry_point='highway_env.envs:ParkingEnv',
# )
# print("Registered new environments")