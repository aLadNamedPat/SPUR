from Grid import GridWorld
from Original_Baseline import OBaseline
from UniformSweep import CoveragePlanner, HeuristicType
import numpy as np

g = GridWorld(5, 5, e_bounds = 10)

#Uniform sweep baseline
grid = np.zeros((5,5))
grid[0][0] = 2
c = CoveragePlanner(grid)
c.set_debug_level(1)
c.start(initial_orientation = 0, cp_heuristic = HeuristicType.MANHATTAN)
c.compute()
traj = c.get_xy_trajectory(c.current_trajectory)
current_time = 0
total_events = 0

while(current_time < 3000):
    traj.reverse()
    events_list = g.step_timesteps(traj)
    for i in range(len(events_list)):
        total_events += events_list[i]
    current_time += len(traj)

print(g.adt())
print(total_events)

Ob = OBaseline(0.05, 5, (0, 0), 0.8, 10)
# g = GridWorld(5, 5, e_bounds = 10)
#Baseline as seen in https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/ICAR05.pdf
Ob._setup__()
current_time = 0
total_events = 0

while (current_time < 3000):
    (location, traj, time) = Ob.take_action()
    events_list = g.step_timesteps(traj)
    for i in range(len(traj)):
        Ob.update_point(traj[i], current_time + i + 1, events_list[i])
        total_events += events_list[i]
    current_time += time

print(g.adt())
print(total_events)