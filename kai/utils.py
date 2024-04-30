from pathlib import Path
from easyrl.agents.ppo_agent import PPOAgent
from easyrl.configs import cfg
from easyrl.configs import set_config
from easyrl.utils.common import set_random_seed
import torch
from easyrl.utils.gym_util import make_vec_env
from easyrl.models.mlp import MLP
from torch import nn
import gymnasium as gym
from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.value_net import ValueNet
from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.engine.ppo_engine import PPOEngine

def train_ppo(env_name: str, max_steps: int = 200000) -> str:
    """
      Parameters:
      - env_name (str): The name of the environment.
      - max_steps (int): The maximum number of steps to train the agent.

      Returns:
      - str: The directory where training data and models are saved.
    """
    set_config('ppo')
    
    #cfg is a global variable here (this is a dumb way to code it but alas)
    cfg.alg.num_envs = 1
    cfg.alg.episode_steps = 100
    cfg.alg.max_steps = max_steps
    cfg.alg.deque_size = 20
    cfg.alg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.alg.env_name = env_name
    cfg.alg.save_dir = Path.cwd().absolute().joinpath('data').as_posix()
    cfg.alg.save_dir += '/'
   
    setattr(cfg.alg, 'diff_cfg', dict(save_dir=cfg.alg.save_dir))

    print(f'====================================')
    print(f'      Device:{cfg.alg.device}')
    print(f'      Total number of steps:{cfg.alg.max_steps}')
    print(f'====================================')

    # set_random_seed(cfg.alg.seed)
    env = make_vec_env(cfg.alg.env_name,
                       cfg.alg.num_envs)
    env.reset()
    ob_size = env.observation_space.shape[0]

    actor_body = MLP(input_size=ob_size,
                     hidden_sizes=[64],
                     output_size=64,
                     hidden_act=nn.Tanh,
                     output_act=nn.Tanh)

    critic_body = MLP(input_size=ob_size,
                     hidden_sizes=[64],
                     output_size=64,
                     hidden_act=nn.Tanh,
                     output_act=nn.Tanh)
    
    assert isinstance(env.action_space, gym.spaces.Discrete), "expected Discrete action space in Minigrid environment"

    act_size = env.action_space.n
    actor = CategoricalPolicy(actor_body,
                                in_features=64,
                                action_dim=act_size)

    critic = ValueNet(critic_body, in_features=64)
    agent = PPOAgent(actor=actor, critic=critic, env=env)
    runner = EpisodicRunner(agent=agent, env=env)
    engine = PPOEngine(agent=agent,
                       runner=runner)
    engine.train()
    stat_info, raw_traj_info = engine.eval(render=False, save_eval_traj=True, eval_num=1, sleep_time=0.0)
    print(stat_info)
    return cfg.alg.save_dir
