import numpy as np
import pygame

class OBaseline:
    def __init__(
        self,
        learning_rate : float,
        grid_size : int,
        starting_location : tuple[int, int],
        forget_rate : float,
        bound : int,
        window_size : int = 256,
        render : bool = False,
    ) -> None:
        self.lr = learning_rate
        self.gridSize = grid_size
        self.bound = bound                                              # The bound as defined by the environment
        self.forget_rate = forget_rate                                  # Define the rate at which the agent forgets bad actions
        self.grid = np.ones((grid_size, grid_size))                     # Potential reward of visiting a point
        self.lv = np.zeros((grid_size, grid_size))                      # Last visit time of a grid location


        self.er = np.zeros((grid_size, grid_size))                      # Expected reward overall
        self.location = starting_location                               # Set current agent location to its starting location

        # # Rendering variables
        # self.window = None
        # self.window_size = window_size
        # self.pix_square_size = self.window_size / self.gridSize
        # self.rendering = render
        # self.clock = pygame.time.Clock()
        
    def _setup__(self):                                                 # Compute the shortest path and trajectory for all points to each other

        # Initialize the graph of the grid in terms of distances
        self.dist = np.zeros((self.gridSize * self.gridSize, self.gridSize * self.gridSize))
        # Initialize the last location to travel to
        self.l_node = np.zeros((self.gridSize * self.gridSize, self.gridSize * self.gridSize))

        # Prepare the adjacency matrix
        for i in range(self.gridSize):
            for j in range(self.gridSize):
                for k in range(self.gridSize):
                    for l in range(self.gridSize):
                        if (i == k and (j == l -1 or j == l + 1)) or (j == l and (i == k -1 or i == k + 1)):
                            self.dist[i * self.gridSize + j, k * self.gridSize + l] = 1
                        elif (i == k and j == l):
                            self.dist[i * self.gridSize + j, k * self.gridSize + l] = 0
                        else:
                            self.dist[i * self.gridSize + j, k * self.gridSize + l] = float("inf")
        
        # Prepare the path matrix
        for i in range(self.gridSize * self.gridSize):
            for j in range(self.gridSize * self.gridSize):
                if i == j:
                    self.l_node[i, j] = -1
                elif self.dist[i, j] != float("inf"):
                    self.l_node[i, j] = i
                else:
                    self.l_node[i, j] = -1


        # Floyd-Warshall's algorithm
        for i in range(self.gridSize * self.gridSize):                                  #This is the starting location x, y
            for j in range(self.gridSize * self.gridSize):                              #This is the ending location x, y
                for k in range(self.gridSize * self.gridSize):                          #This is the mid location x, y
                    if (self.dist[i, k] is not float("inf") and self.dist[k, j] is not float("inf")):
                        if (self.dist[i, j] > self.dist[i, k] + self.dist[k, j]):
                            self.dist[i, j] =  self.dist[i, k] + self.dist[k, j]
                            self.l_node[i, j] = self.l_node[k, j]        
    def update_point(
        self,
        point : tuple[int, int],
        current_time : int,
        e : int
    ) -> None: #This is supposed to update a point's value after visiting it
        
        self.location = point                                           # Update the current location of the agent
        time_spent = current_time - self.lv[point[0], point[1]]         # The time between last visits
        self.lv[point[0], point[1]] = current_time                      # Update the last visited time as well
        curr_val = self.grid[point[0], point[1]]                        # Save the current value (probability) of the point

        if e == 0:
            self.grid[point[0], point[1]] = curr_val * self.forget_rate # Update the potential reward of a point
        
        else:
            self.grid[point[0], point[1]] = (1 - self.lr) * curr_val + e * self.lr  / time_spent # Update the potential reward of a point

        self.er += self.grid                                            # Add the saved value to each point in the grid
        self.er[point[0], point[1]] = 0                                 # Reset the expected reward to 0 throughout
        # if self.rendering:
        #     self.render()

    # Compute the best action for the agent to take at a given timestep
    def take_action(
        self
    ) -> tuple[tuple[int, int], list[tuple[int, int]], int]:
        
        max_reward = -1
        best_traj = None
        for i in range(len(self.grid)):
            for j in range(len(self.grid)):
                if (i, j) == (self.location[0], self.location[1]):
                    continue
                (r, traj, t) = self.compute_trajectory((i, j))          # Return the expected reward and time spent to visit the point along the trajectory
                if r / int(t) > max_reward:                             # Compute whether or not the average reward over time is the max reward possible
                    max_reward = r / int(t)                             # Redefine the max reward
                    best_action = (i, j)                                # Redefine the optimal action
                    time_spent = t                                      # Time spent to visit a location
                    best_traj = traj                                    # Set the best trajectory to be the current trajectory
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

        while(self.l_node[original, final] != -1):
            final = int(self.l_node[original, final])
            if final != -1:
                traj = [(int(final / self.gridSize), final % self.gridSize)] + traj
        traj = traj[1:]
        # traj = [(original[0], original[1])] + traj 
        expected_reward = 0

        for i, (x, y) in enumerate(traj):
            expected_reward += min(self.bound, self.er[x, y] + self.grid[x, y] * i)
        return (expected_reward, traj, total_dist) #Returns the expected reward, the trajectory taken, and the total distance of travel
    
    # Renders the current saved representation of the environment
    def render(
        self,
    ) -> None:
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))

        canvas.fill((255, 255 ,255))

        # Draw the grids that the agent is traveling through
        for x in range(self.gridSize + 1):
            pygame.draw.line(
                canvas,
                0,
                (self.pix_square_size * x, 0),
                (self.pix_square_size * x, self.window_size),
                width = 2
            )
            #Now draw the vertical lines
            pygame.draw.line(
                canvas,
                0,
                (0, self.pix_square_size * x),
                (self.window_size, self.pix_square_size * x)
            )
        
        # Draw the number of events in the grid
        # More red = more events
        for x in range(self.gridSize):
            for y in range(self.gridSize):
                if self.grid[x, y] != 0:
                    r = pygame.Rect(self.pix_square_size * x, self.pix_square_size * y, self.pix_square_size, self.pix_square_size)
                    pygame.draw.rect(
                        canvas, (self.grid[x, y] * 100 , min(self.grid[x, y] * 100, 255) , min(self.grid[x, y] * 100, 255)), r
                    )
                
                if self.location == (x, y):
                    r = pygame.Rect(self.pix_square_size * x, self.pix_square_size * y, self.pix_square_size, self.pix_square_size)
                    pygame.draw.rect(
                        canvas, (0, 0 , 0), r
                    )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(4)       
        
# l = OBaseline(0.95, 10, (0,0))
# l._setup__()

# print(l.compute_trajectory((8, 6)))

