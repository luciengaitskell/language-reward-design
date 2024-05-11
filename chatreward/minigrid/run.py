from .complete import build_reward_funcs, build_text_subgoals
from .env import create_env
from ..ppo import learn

if __name__ == "__main__":
    if True:
        quick_env = create_env()
        quick_env.reset()
        render = quick_env.render()

        suffixes = ["A", "B", "C"]

        reward_funcs = build_reward_funcs(build_text_subgoals(render), suffixes)

        env = create_env(reward_funcs)
        learn(2e5, env, "minigrid_models/minigrid_custom/run_demo", "videos/run_demo")
