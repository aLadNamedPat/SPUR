from PPO import PPO
import os
import sys

sys.path.append(os.path.abspath("../../../"))
from ContinualSweepingGrid.env.GridEnv import GridEnv

env = GridEnv(1, render_mode = None, grid_size = 10, num_centers = 5, max_timesteps = 2048, bound = 10)

print("running")
agent = PPO(env)
agent.batch_train(100000)