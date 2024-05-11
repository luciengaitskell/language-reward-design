from typing import Callable

from PIL import Image
from tqdm import tqdm

from .prompts import prompt1, prompt2
from ..api import complete, vision, image_to_base64


def build_text_subgoals(render: list):
    im = Image.fromarray(render)
    return vision(prompt1, image_to_base64(im))


RewardFuncsDict = dict[int, dict[int, Callable]]


def build_reward_funcs(subgoal_completion, suffixes: list[str]) -> RewardFuncsDict:
    reward_functions = {}

    subgoal_set = []
    for suff in tqdm(suffixes):
        completions = []
        for i, sg in enumerate(
            tqdm(
                subgoal_completion.choices[0].message.content.splitlines(), leave=False
            )
        ):
            completions.append(
                complete(prompt2 + "\nThe textual subgoal is as follows: " + sg)
            )

        completion_funcs = [c.choices[0].message.content for c in completions]
        completion_funcs_execute = [
            "\n".join(c.splitlines()[1:-1]) for c in completion_funcs
        ]
        subgoal_set.append(completion_funcs_execute)
        for i, c in enumerate(completion_funcs_execute):
            if i not in reward_functions:
                reward_functions[i] = {}

            print(f"New reward function: {i}{suff}")
            print(c)
            if "inf')" in c:
                print("Skipping due to infinite value")
                continue

            loc_space = {}
            exec(c, globals(), loc_space)
            reward_functions[i][suff] = loc_space["reward"]

    return reward_functions
