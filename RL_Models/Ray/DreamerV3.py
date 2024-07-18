import os 
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn
import torch as th
import supersuit as ss
import gymnasium.spaces as spaces
import gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd

from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ContinualSweepingGrid.env.GridEnv import GridEnv

import ray
from ray.rllib.algorithms import dreamerv3
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
ray.init()

gridSize = 8


def env_creator(_):
    env = GridEnv(1, render_mode="human", grid_size=gridSize, num_centers=5, max_timesteps=5000, bound=3, degenerate_case=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    return PettingZooEnv(env)

register_env("custom_grid_env", env_creator)
config = AlgorithmConfig().environment("custom_grid_env")
config.train_batch_size = None

algo = dreamerv3.DreamerV3(config)

while True:
    print(algo.train())
# env = GridEnv(1, render_mode = "human", grid_size = gridSize, num_centers = 5, max_timesteps = 5000, bound = 3, degenerate_case=True)
# env_ = ss.pettingzoo_env_to_vec_env_v1(env)
# env_ = ss.concat_vec_envs_v1(env_, 1, base_class="stable_baselines3")

# algo = dreamerv3.DreamerV3(
#     env = env_,
# )

# while (True):
#     print(algo.train())