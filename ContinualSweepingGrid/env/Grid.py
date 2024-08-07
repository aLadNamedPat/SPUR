import numpy as np
import random
import matplotlib.pyplot as plt
import copy as copy
import pygame
import time

# Might need to update the environment so that everything is passed as tensors for the agents.

#Tracking the grid for the agents
class GridTracker:
    def __init__(
        self,
        n : int, # grid size
        bound : int,
        unvisited : bool = False,
        original : bool = False,
    ) -> None:
    
        self.bound = bound                                  # Defining the bound in the environment
        self.prob_grid = np.ones((n, n))                    # The current calculated probibility of events occurring
        self.tracked_grid = np.zeros((n, n))                # The current grid tracking the expected value of events occurring based on the calculated probability
        self.agent_locations = np.zeros((n,n))              # The current location of the agents
        self.unvisited = unvisited                          # Mark (Note) squares as unvisited if they haven't been traveled to yet
        self.original = original                            # Decides whether or not to use original costs

        # Used to keep track of probabilities occurring
        self.total_observed = np.zeros((n, n))              # Keeps track of the number of events that were observed at a location in the past

        if original:
            self.last_timestep_visited = np.full((n, n), -float("inf"))   # Keeps track of the timestep last visited if the original tracking is being used
            self.exponent_grid = np.full((n, n), np.e)                   # Exponent grid that will be used for poisson distribution calculations

        else:
            self.last_timestep_visited = np.zeros((n, n))       # Keeps track of the timestep that a location was last visited by an agent

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
        
        if not self.original:
            self.agent_locations[point[0],point[1]] = 1
            if timestep == 0:
                return self.tracked_grid, self.prob_grid, self.agent_locations
        
            if observed_events == self.bound:
                new_prob = self.prob_grid[point[0], point[1]] * 0.8 + (observed_events / (timestep - \
                self.last_timestep_visited[point[0], point[1]])) * 0.2  #Change this calculations later on, this is just a filler for now
            
            else:
                new_prob = self.prob_grid[point[0], point[1]] * 0.9 + \
                observed_events / (timestep - self.last_timestep_visited[point[0], point[1]]) * 0.1
            self.adjust_grid(point, new_prob)

            self.last_timestep_visited[point[0], point[1]] = timestep

            self.tracked_grid += self.prob_grid
            self.tracked_grid[point[0], point[1]] = 0
            self.tracked_grid = self.tracked_grid.clip(0, self.bound)

            if self.unvisited:
                for x in range(len(self.tracked_grid)):
                    for y in range(len(self.tracked_grid[x])):
                        if self.last_timestep_visited[x,y] == 0:
                            self.tracked_grid[x, y] = -1                #Intuitively, this doesn't make any sense at all. Not too sure what to do about it rn

            self.agent_locations.fill(0)
            self.agent_locations[point[0],point[1]] = 1
            return self.tracked_grid, self.prob_grid, self.agent_locations

        else:
            self.agent_locations[point[0], point[1]] = 1

            if timestep == 0:
                exp_grid = np.multiply(self.last_timestep_visited - timestep, self.prob_grid)
                saved_grid = np.power(self.exponent_grid, exp_grid)
                return saved_grid, self.agent_locations
            
            if observed_events[i] == self.bound:
                new_prob = (self.prob_grid[point[i][0], point[i][1]] * 0.8 +
                ((observed_events / (timestep - self.last_timestep_visited[point[i][0], point[i][1] + 1]))) * 0.2)  
                #Change this calculations later on, this is just a filler for now
            else:
                new_prob = (self.prob_grid[point[i][0], point[i][1]] * 0.8 +
                observed_events / (timestep - self.last_timestep_visited[point[i][0], point[i][1]] + 1) * 0.2)

            self.adjust_grid(point[i], new_prob)
            self.last_timestep_visited[point[i][0], point[i][1]] = timestep

            self.agent_locations.fill(0)
            for i in range(len(observed_events)):
                self.agent_locations[point[i][0], point[i][1]] = 1

            exp_grid = np.multiply(self.last_timestep_visited - timestep, self.prob_grid)
            saved_grid = np.power(self.exponent_grid, exp_grid)

            return saved_grid, self.agent_locations

    def multi_update(
        self,
        points : list[tuple[int, int]],
        observed_events : list[int],
        timestep : int = 1,
    ):
        
        if not self.original:
            for point in points:
                self.agent_locations[point[0], point[1]] = 1

            if timestep == 0:
                return self.tracked_grid, self.prob_grid, self.agent_locations
            
            for i in range(len(observed_events)):
                if observed_events[i] == self.bound:
                    new_prob = (self.prob_grid[points[i][0], points[i][1]] * 0.8 +
                    ((observed_events / (timestep - self.last_timestep_visited[points[i][0], points[i][1]]))) * 0.2)  
                    #Change this calculations later on, this is just a filler for now
                else:
                    new_prob = (self.prob_grid[points[i][0], points[i][1]] * 0.8 +
                    observed_events / (timestep - self.last_timestep_visited[points[i][0], points[i][1]]) * 0.2)
                
                # print("Current timestep:", timestep)
                # print("Last visited timestep", self.last_timestep_visited[points[i][0], points[i][1]])
                # print("New probability:", new_prob)

                self.adjust_grid(points[i], new_prob)
                self.last_timestep_visited[points[i][0], points[i][1]] = timestep

            self.tracked_grid += self.prob_grid
            # print(self.tracked_grid)

            self.agent_locations.fill(0)
            for i in range(len(observed_events)):
                self.tracked_grid[points[i][0], points[i][1]] = 0
                self.tracked_grid = self.tracked_grid.clip(0, self.bound)
                self.agent_locations[points[i][0], points[i][1]] = 1
            
            return self.tracked_grid, self.prob_grid, self.agent_locations
        
        else:
            for point in points:
                self.agent_locations[point[0], point[1]] = 1

            if timestep == 0:
                exp_grid = np.multiply(self.last_timestep_visited - timestep, self.prob_grid)
                saved_grid = np.power(self.exponent_grid, exp_grid)
                return saved_grid, self.agent_locations
            
            for i in range(len(observed_events)):
                if observed_events[i] == self.bound:
                    new_prob = (self.prob_grid[points[i][0], points[i][1]] * 0.5 +
                    ((observed_events / (timestep - self.last_timestep_visited[points[i][0], points[i][1]]))) * 0.5)  
                    #Change this calculations later on, this is just a filler for now
                else:
                    new_prob = (self.prob_grid[points[i][0], points[i][1]] * 0.5 +
                    observed_events / (timestep - self.last_timestep_visited[points[i][0], points[i][1]]) * 0.5)

                self.adjust_grid(points[i], new_prob)
                self.last_timestep_visited[points[i][0], points[i][1]] = timestep

            self.agent_locations.fill(0)
            for i in range(len(observed_events)):
                self.agent_locations[points[i][0], points[i][1]] = 1

            exp_grid = np.multiply(self.last_timestep_visited - timestep, self.prob_grid)
            saved_grid = np.power(self.exponent_grid, exp_grid)

            return saved_grid, self.agent_locations

# The actual gridworld where the real number of events and event probabilities are tracked
class GridWorld:
    def __init__(
        self,
        n : int,
        centers : int,
        central_probs : list[int] = None,
        decrease_rate : list[int] = None,
        p_bounds : float = 0.2,
        e_bounds : int = 10,
        chosen_centers : list[tuple[int, int]] = None,
        seed : int = None,
        window_size : int = 256,
        render : bool = False,
    ) -> None:

        self.total_detection_time = 0                           # Keep track of the total detection time taken from appearance to detection
        self.total_events_detected = 0                          # Keep track of the number of events detected
        self.num_events = 0                                     # Keeps track of the number of events that are currently active
        self.window_size = window_size                          # Size of the window when rendering
        self.gridSize = n                                       # Size of the grid that is being tracked
        self.pix_square_size = self.window_size / self.gridSize # Size of each square in the gridworld
        self.rendering = render                                 # Defines if rendering should occur
        self.window = None

        if seed is not None:
            random.seed(seed)

        self.e_bounds = e_bounds

        if chosen_centers is None:
            n_centers = [(random.randint(0, n-1), random.randint(0, n-1)) for _ in range(centers)]  # Fix later on for points spawning on top of each other
        else:
            n_centers = chosen_centers                                                              # Points that are pre-chosen by the user

        if central_probs is not None:
            central_probs = np.array(central_probs)
        else:
            central_probs = np.array([random.random() * p_bounds for _ in range(centers)])

        if decrease_rate is not None:
            decrease_rate = np.array(decrease_rate)
        else:
            decrease_rate = central_probs

        self.p_grid = np.zeros((n, n))

        for i in range(centers):
            for j in range(n):
                for k in range(n):
                    distance = abs(j - n_centers[i][0]) + abs(k - n_centers[i][1])
                    l = central_probs[i] - (decrease_rate[i] * distance)
                    if l > 0:
                        self.p_grid[j,k] += l
        
        self.p_grid = self.p_grid.clip(0, 1)
        self.e_grid = np.zeros((n, n))

    def event_occurence(
        self
    ) -> None:
        random_numbers = np.random.rand(*self.e_grid.shape)                 # Generates random floats from 0 to 1 in the shape of the grid
        event_occurrences = random_numbers < self.p_grid                    # If random_number generated is smaller than the probability of events occurring, then an event has occurred 
        self.old_e_grid = np.copy(self.e_grid)                              # Save the event grid before adding anything or clipping it
        self.e_grid += event_occurrences.astype(int)                        # Add to the event grid data
        self.e_grid = self.e_grid.clip(0, self.e_bounds)                    # Clip the number of events in each grid to the set bounds

    #Single timestep stepping
    def step(
        self,
        point : tuple[int, int], #the location of the agent
    ) -> int:
        self.event_occurence()
        # Determines where events have occurred and adds to them    
        self.location = point                                               # Save the location of the agent for rendering
        self.num_events += int(np.sum(self.e_grid - self.old_e_grid))       # Tracks the number of events that are active on the field

        events_found = self.e_grid[point[0], point[1]]                      # The number of events found by the agent
        self.e_grid[point[0], point[1]] = 0                                 # Sets the number of events at the found grid to 0

        self.num_events -= int(events_found)                                # Subtract the number of events from the total number of current active events
        self.total_events_detected += events_found                          # Add the number of events found to the total number of events detected
        self.total_detection_time += self.num_events                        # Since the active events are still not detected, add their result to the total detection time
        self.clock = pygame.time.Clock()
        if self.rendering:
            self.render()
            time.sleep(0.2)
            
        return events_found                                                 # Returns the number of events found at the given location
    
    #Multi-step single stepping - current issue is that collisions are not handled
    def multistep(
        self,
        points : list[tuple[int, int]],                                     # The locations that the agents are traveling to
    ) -> list[int]:
        self.event_occurence()
        self.num_events += int(np.sum(self.e_grid - self.old_e_grid))
        events_found = []
        for i in range(len(points)):
            event_found = self.e_grid[points[i][0], points[i][1]]
            events_found.append(event_found)
            self.e_grid[points[i][0], points[i][1]] = 0

            self.num_events -= int(event_found)
            self.total_events_detected += event_found
        self.total_detection_time += self.num_events

        return events_found                                                 # Returns the number of events found for multi-robot traversal
    
    #Multi-timestep stepping
    def step_timesteps(
        self,
        traj : list[tuple[int, int]]
    ) -> list[int]:
        
        events_found_list = []                                              # Keeps track of the number of events that have been found per step

        for x, y in traj:                                                   # Uses the trajectory of the agent to calcualate how much the agent accumulates after all its steps
            e = self.step((x,y))
            events_found_list.append(e)                                     # Adds the number of events found to its list
            
        return events_found_list
    
    #Multi-step timestep stepping
    def multistep_timesteps(
        self,
        trajs : list[list[tuple[int, int]]],
    ) -> list[list[int]]:

        events_found_list = []
        for i in range(len(trajs[0])):
            points = [traj[i] for traj in trajs]
            e = self.multistep(points)
            events_found_list.append(e)

        # print(events_found_list) 
        return events_found_list                                            # Returns in the form of [[1, 2, 3], [3, 1, 2], [3, 1, 2]]
    
    #Calculate the expected growth per second
    def gps(
        self,
    ) -> float:
        return np.sum(self.p_grid)
    

    #Find average detection time
    def adt(
        self,
    ) -> float:
        print("total detection time", self.total_detection_time)
        print("total events detected", self.total_events_detected)
        return self.total_detection_time / self.total_events_detected
    
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
                if self.e_grid[x, y] != 0:
                    r = pygame.Rect(self.pix_square_size * x, self.pix_square_size * y, self.pix_square_size, self.pix_square_size)
                    pygame.draw.rect(
                        canvas, (self.e_grid[x, y] , min(self.e_grid[x, y] * 20, 255) , min(self.e_grid[x, y] * 20, 255)), r
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
    
    def get_e_grid(
        self
    ) -> np.array:
        return self.e_grid
    
    def get_p_grid(
        self,
    ) -> np.array:
        return self.p_grid
