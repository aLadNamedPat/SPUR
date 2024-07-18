import os 
import sys
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn
import torch as th
import supersuit as ss
import gymnasium.spaces as spaces
import numpy as np
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd
from typing import Type, Union, Callable, Optional, Dict, Any
from stable_baselines3.common.type_aliases import Schedule
import stable_baselines3.common.callbacks as sb3_cb
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ContinualSweepingGrid.env.GridEnv import GridEnv
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 50):
        # Define the output feature dimension
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Define your custom feature extraction layers here 
        # For example, a simple feed-forward network
        input_channels = 3
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute shape of CNN output
        sample_input = th.zeros(1, input_channels, 8, 8)  # Assuming 8x8 grid
        with th.no_grad():
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        eReward = observations['eReward']
        pGrid = observations['pGrid']
        agent = observations['agent']
        # Stack the observations
        x = th.stack([eReward, pGrid, agent], dim=1)
        return self.linear(self.cnn(x))

class CustomDQNPolicy(DQNPolicy):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule: Union[Schedule, float],
                 features_extractor_class: Type[BaseFeaturesExtractor] = CustomFeatureExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super(CustomDQNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs
        )
        # Ensure q_net and q_net_target are properly initialized
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = 3
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute shape of CNN output
        sample_input = th.zeros(1, n_input_channels, 8, 8)  # Assuming 8x8 grid
        with th.no_grad():
            n_flatten = self.cnn(sample_input).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

        # # Compute the shape of the output of the CNN to define the linear layer
        # with th.no_grad():
        #     print(observation_space["eReward"].sample())
        #     sample_input = th.as_tensor(
        #         np.stack([
        #             observation_space['eReward'].sample(),
        #             observation_space['pGrid'].sample(),
        #             observation_space['agent'].sample()
        #         ], axis=0)
        #     ).float()
        #     n_flatten = self.cnn(sample_input[None]).shape[1]

        # self.linear = nn.Sequential(
        #     nn.Linear(n_flatten, features_dim),
        #     nn.ReLU()
        # )

    def forward(self, observations):
        eReward = observations['eReward']
        pGrid = observations['pGrid']
        agent = observations['agent']
        # Stack the observations
        x = th.stack([eReward, pGrid, agent], dim=1)
        
        return self.linear(self.cnn(x))
    
class CustomCNNPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomCNNPolicy, self).__init__(
            *args,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
            **kwargs
        )

    
gridSize = 8

env = GridEnv(1, render_mode = None, grid_size = gridSize, num_centers = 5, max_timesteps = 5000, bound = 3)
print("Original env reset:")
obs, info = env.reset()
print(obs)

env_ = ss.pettingzoo_env_to_vec_env_v1(env)
env_ = ss.concat_vec_envs_v1(env_, 1, base_class="stable_baselines3")
print("New env reset:")
obs = env_.reset
print(obs)

env = GridEnv(1, render_mode = None, grid_size = gridSize, num_centers = 5, max_timesteps = 250, reset_steps = 100, bound = 3, degenerate_case=True, parallelize = True)
env_ = ss.pettingzoo_env_to_vec_env_v1(env)
env_ = ss.concat_vec_envs_v1(env_, 50, base_class="stable_baselines3")

# for _ in range(10000):
#     print(env_.step(np.array([int(random.random() * 4)]))[0])

# print("Observation Space:", env_.observation_space.shape)
# print("Observation Space:", env_.observation_space)
model = PPO(CustomCNNPolicy, env_, verbose=1)

# model = DQN(CustomDQNPolicy, env_, verbose=1)
model.learn(total_timesteps=100000)
model.save("DQN_Saved")

model.load("DQN_Saved")

# Test the trained model
obs = env_.reset()
for _ in range(10000):
    action, states = model.predict(obs)
    obs, rewards, done, info = env_.step(action)