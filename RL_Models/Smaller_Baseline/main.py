from R_Learning import R_Learning
import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ContinualSweepSMDP.env.GridEnv import GridEnv


gridSize = 8

env = GridEnv(1, render_mode = None, grid_size = gridSize, num_centers = 3, max_timesteps = 1000, bound = 3, original_outputs=True)
m = R_Learning(gridSize, env, input_channels=2,episode_length = 50, beta = 0.05)

m.learn(200000)