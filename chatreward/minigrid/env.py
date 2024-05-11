import math

from gymnasium import spaces
from minigrid.wrappers import DictObservationSpaceWrapper

from minigrid.wrappers import (
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
    DictObservationSpaceWrapper,
)
import gymnasium


from gymnasium import ObservationWrapper, RewardWrapper
import numpy as np

from .complete import RewardFuncsDict


COLOR_TO_IDX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "purple": 3,
    "yellow": 4,
    "grey": 5,
}


class Actions:
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    PICKUP = 3
    DROP = 4
    TOGGLE = 5
    DONE = 6


def mission_to_string(mission_encoding):
    indices = [idx - 1 for idx in mission_encoding]  # remove offset
    translation = {
        v: k for k, v in DictObservationSpaceWrapper.get_minigrid_words().items()
    }
    translation[-1] = ""
    return " ".join([translation[idx] for idx in indices])


# a custom wrapper to make the mission vector work with one hot encoding
class MissionEncodingWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.observation_space["mission"] = spaces.MultiDiscrete(
            np.array([n + 1 for n in env.observation_space["mission"].nvec])
        )

    def observation(self, obs):
        return obs


RENDERS = []  # for debugging


class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env, reward_func):
        super().__init__(env)
        self.generate_reward_func = reward_func  # (obs, action) -> reard
        self.counts = {}  # ((x, y), action) => count

    def step(self, action):

        # params
        bonus_factor = 0.01  # the exploration bonus
        intrinsic_factor = 0.9  # how much the intrinsic reward should be weighted
        progress_bonus = 1.5  # the bonus for reaching farther and farther goals

        observation, _, terminated, truncated, info = self.env.step(action)
        reward, goal_number = self.reward_func(observation, action)

        # save render for debugging
        if goal_number >= 3:
            RENDERS.append(self.env.render())

        reward *= (
            progress_bonus**goal_number
        ) * intrinsic_factor  # to incentivize further goals

        pos = tuple(self.agent_pos)
        state_and_action = (pos, action.item())

        # Get the count for this key
        count = self.counts[state_and_action] if state_and_action in self.counts else 0

        # Update the count for this key
        self.counts[state_and_action] = count + 1

        bonus = 1 / math.sqrt(self.counts[state_and_action])
        # reward += bonus_factor + bonus

        # Get the position in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        if (
            action == self.actions.forward
            and fwd_cell is not None
            and fwd_cell.type == "goal"
        ):
            print("REACHED GOAL, EXTRINSIC REWARD: ", self._reward())
            reward += self._reward()

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.reward_func = self.generate_reward_func()
        obs, info = self.env.reset(**kwargs)
        return obs, info


EPSILON = 0.1


def reward_gen_wrapper(reward_funcs: RewardFuncsDict):
    def generate_total_reward():
        goal_number = 0

        def total_reward(obs, action):
            nonlocal goal_number

            # print(obs)
            gamma = 0.0005  # probability to skip to next subgoal

            # preprocess -- ideally this would be done with LLM if we had more compute
            lockedroom_color = COLOR_TO_IDX[obs["mission"].split(" ")[2]]
            keyroom_color = COLOR_TO_IDX[obs["mission"].split(" ")[6]]
            door_color = COLOR_TO_IDX[obs["mission"].split(" ")[10]]

            rewards = []
            for sg_num, sg in reward_funcs.items():
                reward = 0
                for i, r in sg.items():
                    try:
                        reward = max(
                            r(obs, action, lockedroom_color, keyroom_color, door_color),
                            reward,
                        )
                    except:
                        pass
                rewards.append(reward)

            # goals are often related, so weight future goals (on decay)
            decay = 0.4  # lambda
            reward = sum(
                (rewards[i] * (decay**i)) for i in range(len(rewards[goal_number:]))
            )

            # or np.random.rand() < gamma

            if (rewards[goal_number] > 1 - EPSILON) and goal_number < len(
                reward_funcs
            ) - 1:

                print(
                    "switching functions from "
                    + str(goal_number)
                    + " to "
                    + str(goal_number + 1)
                )
                goal_number += 1

            return reward, goal_number

        return total_reward

    return generate_total_reward


def create_env(
    reward_funcs: RewardFuncsDict | None = None, farama_support: bool = True
):
    env = gymnasium.make("MiniGrid-LockedRoom-v0", render_mode="rgb_array")

    if reward_funcs:
        env = CustomRewardWrapper(env, reward_gen_wrapper(reward_funcs))
    env = RGBImgPartialObsWrapper(
        env
    )  # reward is calculated using regular obs, then plugged into model with img obs

    if farama_support:
        env = DictObservationSpaceWrapper(env)
        env = MissionEncodingWrapper(env)

    return env
