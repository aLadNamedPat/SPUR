# from R_Learning_MDP import R_Learning
from R_Learning import R_Learning
import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# from ContinualSweepingGrid.env.GridEnv import GridEnv
from ContinualSweepSMDP.env.GridEnv import GridEnv

gridSize = 8

# env = GridEnv(1, render_mode = None, grid_size = gridSize, num_centers = 5, max_timesteps = 1000, bound = 3, degenerate_case=True)
# m = R_Learning(gridSize, env, input_channels=3, episode_length = 50, reset_steps= 50000, alpha = 0.01, beta = 0.05, tau = 500)

env = GridEnv(1, render_mode = None, grid_size = gridSize, num_centers = 5, max_timesteps = 1000, bound = 3, degenerate_case=True)
m = R_Learning(gridSize, env, input_channels=3, episode_length = 50, reset_steps= 50000, alpha = 0.01, beta = 0.05, tau = 500)

m.learn(200000)