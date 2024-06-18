import numpy as np
import random
import matplotlib.pyplot as plt

# Might need to update the environment so that everything is passed as tensors for the agents.

#Tracking the grid for the agent
class GridTracker:
    def __init__(
        self,
        n : int, # grid size
        bound : int
    ) -> None:
        self.bound = bound
        self.prob_grid = np.ones((n, n))                    #The current calculated probibility of events occurring
        self.tracked_grid = np.zeros((n, n))                #The current grid tracking the expected value of events occurring based on the calculated probability
        self.agent_location = np.zeros((n,n))               #The current location of the agent

        #Used to keep track of probabilities occurring
        self.last_timestep_visited = np.zeros((n, n))       #Keeps track of the timestep that a location was last visited by an agent
        self.total_observed = np.zeros((n, n))              #Keeps track of the number of events that were observed at a location in the past

        self.current_timestep = 0
    
    def adjust_grid(
        self,
        point : tuple[int, int], # point to adjust
        new_prob : float # new probability
    ) -> None:
        
        self.prob_grid[point[0], point[1]] = new_prob
    
    def update(
        self,
        point : tuple[int, int], # point that will have a changing probability
        observed_events : int,
        timestep : int = 0,
    ) -> np.array:
        
        if timestep == 0:
            return self.tracked_grid, self.prob_grid, self.agent_location
    
        if observed_events == self.bound:
            new_prob = self.prob_grid[point[0], point[1]] * 0.5 + \
            (1 + (observed_events / (timestep - \
            self.last_timestep_visited[point[0], point[1]]))) * 0.25  #Change this calculate later on, this is just a filler for now
        
        else:
            new_prob = self.prob_grid[point[0], point[1]] * 0.5 + \
            observed_events / (timestep - \
            self.last_timestep_visited[point[0], point[1]]) * 0.5  

        self.adjust_grid(point,new_prob)
        
        self.last_timestep_visited[point[0], point[1]] = timestep

        self.tracked_grid += self.prob_grid
        self.tracked_grid[point[0], point[1]] = 0
        self.agent_location.fill(0)[point[0],point[1]] = 1

        return self.tracked_grid, self.prob_grid, self.agent_location


class GridWorld:
    def __init__(
        self,
        n : int,
        centers : int,
        central_probs : list[int] = None,
        decrease_rate : list[int] = None,
        p_bounds : float = None,
        e_bounds : int = None,
        seed : int = None
    ) -> None:
        
        if bounds is None:
            bounds = 0.2

        if seed is not None:
            random.seed(seed)

        self.e_bounds = e_bounds
        n_centers = [(random.randint(0, n-1), random.randint(0, n-1)) for _ in range(centers)] 

        if central_probs is not None:
            central_probs = np.array(central_probs)
            decrease_rate = np.array(decrease_rate)
        else:
            central_probs = np.array([random.random() * p_bounds for _ in range(centers)])
            decrease_rate = central_probs / 5

        self.p_grid = np.zeros((n, n))

        for i in range(centers):
            for j in range(n):
                for k in range(n):
                    distance = abs(j - n_centers[i][0]) + abs(k - n_centers[i][1])
                    l = central_probs[i] - (decrease_rate[i] * distance)
                    print(l)
                    if l > 0:
                        self.p_grid[j,k] += l
        
        self.p_grid = self.p_grid.clip(0, 1)
        self.e_grid = np.zeros((n, n))

    def step(
        self,
        point : tuple[int, int], #the location of the agent
    ) -> int:
        #Determines where events have occurred and adds to them        
        random_numbers = np.random.rand(*self.e_grid.shape)
        event_occurrences = random_numbers < self.p_grid
        self.e_grid += event_occurrences.astype(int)
        
        events_found = self.e_grid[point[0], point[1]]
        self.e_grid[point[0], point[1]] = 0
        self.e_grid = self.e_grid.clip(0, self.e_bounds)

        return events_found #Returns the number of events found at the given location