# from R_Learning_MDP import R_Learning
from R_Learning import R_Learning
import os 
import sys
import torch
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# from ContinualSweepingGrid.env.GridEnv import GridEnv
from ContinualSweepSMDP.env.GridEnv import GridEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gridSize = 8

# env = GridEnv(1, render_mode = None, grid_size = gridSize, num_centers = 5, max_timesteps = 1000, bound = 3, degenerate_case=True)
# m = R_Learning(gridSize, env, input_channels=3, episode_length = 50, reset_steps= 50000, alpha = 0.01, beta = 0.05, tau = 500)

env = GridEnv(1, render_mode = None, grid_size = gridSize, num_centers = 5, 
              max_timesteps = 2048, bound = 3, degenerate_case=True, given_probs = [0.1, 0.2, 0.05, 0.3, 0.2])

# Need to parallelize learning to multiple environments at once
m = R_Learning(gridSize, env, input_channels=2, episode_length = 50, reset_steps= 50000, alpha = 0.01, beta = 0.05, tau = 500)

m.learn(400000)


# env = GridEnv(1, render_mode = "human", grid_size = gridSize, num_centers = 5, max_timesteps = 1000, bound = 3, degenerate_case=True)

# obs, info = env.reset()
# for _ in range(10000):
#     m.agent_positions = np.argwhere(np.array(obs[2]) == 1)
#     obs = np.array(obs)
#     obs = list(obs[0].values())
#     with torch.no_grad():
#         obs = torch.tensor(obs, dtype = torch.float).unsqueeze(0).to(device)
#     q_val, q_index = m.predict(obs)
#     action = m.action_to_env(q_index)
#     obs, rewards, dones, _, info = env.step(action)