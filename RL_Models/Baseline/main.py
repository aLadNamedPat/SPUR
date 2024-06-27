from R_Learning import R_Learning
import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ContinualSweepSMDP.env.GridEnv import GridEnv


gridSize = 20

env = GridEnv(1, render_mode = None, grid_size = gridSize, num_centers = 5, max_timesteps = 1000, bound = 10)
m = R_Learning(gridSize, env)

m.learn()
