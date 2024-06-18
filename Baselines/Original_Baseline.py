import numpy as np
class OBaseline:

    def __init__(
        self,
        learning_rate : float,
        grid_size : int,
        starting_location : tuple[int, int]
    ) -> None:
        self.lr = learning_rate
        self.gridSize = grid_size
        self.grid = np.ones((grid_size, grid_size))                     #Potential reward of visiting a point
        self.lv = np.zeros((grid_size, grid_size))                      #Last visit time of a grid location

        self.er = np.zeros((grid_size, grid_size))                      #Expected reward overall
        self.trajectories = np.zeros((grid_size, grid_size))            #Trajectories of all lengths from all points
        self.location = starting_location


    def _setup__(self):                                                 #Compute the shortest path and trajectory for all points to each other

        #Initialize the graph of the grid in terms of distances
        self.dist = np.zeros((self.gridSize * self.gridSize, self.gridSize * self.gridSize))
        #Initialize the last location to travel to
        self.l_node = np.zeros((self.gridSize * self.gridSize, self.gridSize * self.gridSize))

        for i in range(self.gridSize):
            for j in range(self.gridSize):
                for k in range(self.gridSize):
                    for l in range(self.gridSize):
                        if (i == k and (j == l -1 or j == l + 1)) or (j == l and (i == k -1 or i == k + 1)):
                            self.dist[i * self.gridSize + j, k * self.gridSize + l] = 1
                        elif (i == k and j == l):
                            self.dist[i * self.gridSize + j, k * self.gridSize + l] = 0
                            self.l_node[i * self.gridSize + j, k * self.gridSize + l] = 0
                        else:
                            self.dist[i * self.gridSize + j, k * self.gridSize + l] = float("inf")
        
        #Floyd-Warshall's algorithm
        for i in range(self.gridSize * self.gridSize):                                  #This is the starting location x, y
            for j in range(self.gridSize * self.gridSize):                              #This is the ending location x, y
                for k in range(self.gridSize * self.gridSize):                          #This is the mid location x, y
                    if (self.dist[i, k] is not float("inf") and self.dist[k, j] is not float("inf")):
                        if (self.dist[i, j] > self.dist[i, k] + self.dist[k, j]):
                            self.dist[i, j] =  self.dist[i, k] + self.dist[k, j]
                            self.l_node[i, j] = k
        
    def update_point(
        self,
        point : tuple[int, int],
        current_time : int,
        e : int
    ) -> None: #This is supposed to update a point's value after visiting it
        
        self.location = point                                           #Update the current location of the agent
        time_spent = current_time - self.lv[point[0], point[1]]         #The time between last visits
        self.lv[point[0], point[1]] = current_time                      #Update the current time as well
        curr_val = self.grid[point[0], point[1]]                        #Save the current value (probability) of the point

        if e == 0:
            self.grid[point[0], point[1]] = curr_val * 0.99
        
        else:
            self.grid[point[0], point[1]] = (1 - self.lr) * curr_val + e * self.lr  / time_spent

        self.er += self.grid                                            #Add the saved value to each point in the grid
        self.er[point[0], point[1]] = 0                                 #Reset the expected reward to 0 throughout


    #Compute the best action for the agent to take at a given timestep
    def take_action(
        self
    ) -> tuple[tuple[int, int], list[tuple[int, int]], int]:
        
        max_reward = -1
        best_traj = None
        for i in range(len(self.grid)):
            for j in range(len(self.grid)):
                (r, traj, t) = self.compute_trajectory((i, j))                #Return the expected reward and time spent to visit the point along the trajectory
                if r / t > max_reward:                                  #Compute whether or not the average reward over time is the max reward possible
                    max_reward = r / t                                  #Redefine the max reward
                    best_action = (i, j)                                #Redefine the optimal action
                    time_spent = t                                      #Time spent to visit a location
                    best_traj = traj
        return (best_action, best_traj, time_spent)

    
    def compute_trajectory(
        self,
        final : tuple[int, int]
    ) -> tuple[float, list[tuple[int, int]], int]:
        total_dist = self.dist[self.location[0] * \
                               self.gridSize + self.location[1],
                               final[0] * self.gridSize +
                               final[1]]
        
        original = self.location[0] * self.gridSize + self.location[1]
        traj = [(final[0], final[1])]
        final = final[0] * self.gridSize + final[1]

        while(final is not None and final != 0):
            final = int(self.l_node[original, final])
            if final != 0:
                traj = [(int(final / self.gridSize), final % self.gridSize)] + traj
        # traj = [(original[0], original[1])] + traj


        expected_reward = 0

        for i, (x, y) in enumerate(traj):
            expected_reward += self.er[x, y] + self.grid[x, y] * i
        return (expected_reward, traj, total_dist) #Returns the expected reward, the trajectory taken, and the total distance of travel
    

# l = OBaseline(0.95, 10, (0,0))
# l._setup__()

# print(l.compute_trajectory((8, 6)))

