from pettingzoo import ParallelEnv
from .Grid import GridTracker, GridWorld
import random
import functools
import numpy as np
import pygame
from gymnasium.spaces import Discrete, MultiDiscrete, Dict, MultiBinary, Box
from copy import copy


class GridEnv(ParallelEnv):
    metadata = {
        "name": "ContinualSweepingGrid_v0",
    }

    def __init__(self, 
        num_agents : int, 
        render_mode : str = None, 
        grid_size : int = 10,
        num_centers : int = 5,
        max_timesteps : int = 4096, 
        reset_steps : int = 10,
        bound : int = 5,
        seed : int = None,
        agent_positions : list[tuple[int, int]] = None,
        degenerate_case : bool = False,
        ):
        self.gridSize = grid_size                                               # Set the grid size of the environment
        self.bound = bound                                                      # Set the bounds to the number of agents in an environment
        self.numAgents = num_agents                                             # Set the number of agents
        self.numCenters = num_centers                                           # Set the number of centers of probability of events occuring
        self.max_timesteps = max_timesteps                                      # Set the max number of timesteps occurring

        self.total_steps = 0
        self.degenerate_case = degenerate_case
        self.rewards = 0
        self.steps_since_last_reset = 0
        self.reset_steps = reset_steps
        
        self.gridTracker = GridTracker(grid_size, bound)                        # Create a grid tracker for tracking events occurring at different locations
        self.envGrid = GridWorld(grid_size, num_centers)                        # Create the actual grid environment

        if agent_positions is not None and len(agent_positions) == num_agents:
            self.agent_positions = agent_positions
        else: 
            self.agent_positions = self.GSL()                                   # Generate the positions of the agents randomly

        self.possible_agents = [f"agent{i}" for i in range(self.numAgents)]     # Set the possible agents for the PettingZoo environment
        self.agents = self.possible_agents[:]                                   # Set the agents for the PettingZoo environment
        self.curr_step = 0                                                      # Current timestep of the environment
        self._setup__()
        self.evaluation_mode = False

        # Rendering variables
        self.pix_square_size = (                                                # Size of the square of the environment
            64
        )
        
        self.render_mode = render_mode
        self.window = None
        self.window_size = (self.pix_square_size) * (self.gridSize)
        self.clock = None

        self.saved_rewards = {f"agent{i}" : 0 for i in range(self.numAgents)}  # Rewards that are saved for each agent when they don't finish traveling their trajectory
        self.last_saved_rewards = {f"agent{i}" : 0 for i in range(self.numAgents)} # Last saved rewards for agents
        self.decision_steps = {f"agent{i}" : 0 for i in range(self.numAgents)} # Save the number of decision steps that each agent is taking
    def _setup__(
        self,
    ) -> None:                                                        # Compute the shortest path and trajectory for all points to each other

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

    # Compute the trajectory that each agent would take to arrive at a point and the number of timesteps it would take to arrive there
    def compute_trajectory(
        self,
        final : list[tuple[int, int]],                                              # List of all the final positions to travel to
    ) -> tuple[float, list[tuple[int, int]], int]:
        
        total_dist = {f"agent{i}" : 0 for i in range(len(self.possible_agents))}    # Compute the total distance that the agents are traveling

        original_positions = {f"agent{i}" : self.agent_positions[i][0] * \
                               self.gridSize + self.agent_positions[i][1] \
                                for i in range(len(self.possible_agents))}          # Save the original positions of the agents
        
        final_positions = {f"agent{i}" : final[i][0] * self.gridSize + final[i][1] for i \
                            in range(len(self.possible_agents))}                    # Save the final positions of the agents
        
        for i in range(len(self.agent_positions)):
            total_dist[f"agent{i}"] = self.dist[                                    
                self.agent_positions[i][0] * \
                self.gridSize + self.agent_positions[i][1],
                final[i][0] * self.gridSize + final[i][1]
            ]                                                                       # Determine the distance that the agent has to travel from beginning to end

        trajectories = {f"agent{i}" : [(final[i][0], final[i][1])] for \
                                       i in range(len(self.possible_agents))}       # Save the trajectories of each agent
        
        for i in range(len(self.possible_agents)):
            while(self.l_node[original_positions[f"agent{i}"], final_positions[f"agent{i}"]] != -1):
                final_positions[f"agent{i}"] = int(self.l_node[original_positions[f"agent{i}"], final_positions[f"agent{i}"]])
                if final_positions[f"agent{i}"] != -1:
                    trajectories[f"agent{i}"] = [(int(final_positions[f"agent{i}"] / self.gridSize), final_positions[f"agent{i}"] % self.gridSize)] + trajectories[f"agent{i}"]
            # if original_positions[f"agent{i}"] == 0:
            #     trajectories[f"agent{i}"] = [(0, 0)] + trajectories[f"agent{i}"]
            trajectories[f"agent{i}"] = trajectories[f"agent{i}"][1:]

        return total_dist, trajectories

    # Generate the random positions of the agents when initiating
    def GSL(
        self
        ) -> None:

        locations = set()
        while len(locations) < self.numAgents:
            x = random.randint(0, self.gridSize - 1)
            y = random.randint(0, self.gridSize - 1)
            locations.add((x, y))
        return list(locations)

    # Reset the environment
    def reset(
        self, 
        seed=None,
        options = None,
        ):
        self.agents = copy(self.possible_agents)
        self.curr_step = 0
        print(sum(sum(self.envGrid.get_p_grid())))
        print(self.rewards)
        self.rewards = 0
        self.agent_positions = self.GSL()
        self.steps_since_last_reset += 1

        # if self.steps_since_last_reset % self.reset_steps == 0:
        #     self.steps_since_last_reset = 0
        #     self.gridTracker = GridTracker(self.gridSize, self.bound)
        #     self.envGrid = GridWorld(self.gridSize, self.numCenters)

        er = [] #Store the expected rewards
        ep = [] #Store the probability grids
        ap = [] #Store the agent positions

        action_mask = {}
        for i in range(len(self.agent_positions)):
            a = np.ones(self.gridSize * self.gridSize)
            a[self.agent_positions[i][0] * self.gridSize + self.agent_positions[i][1]] = 0
            action_mask[f"agent{i}"] = a

        for a in range(len(self.agents)):
            d, e, f = self.gridTracker.update((self.agent_positions[a][0], self.agent_positions[a][1]), 0, 0)
            er.append(d)
            ep.append(e)
            ap.append(f)

        if not self.degenerate_case:
            observations = {
                a : { 
                "eReward" : er[0], # Estimated expected reward of the map
                "pGrid" : ep[0],   # Probabiity grid of the map
                "agent" : ap[0],   # Gridspace of where the agent is located
                "action mask" : action_mask[a]
                } for a in self.agents
            }
        
        else:
            pd = self.envGrid.get_e_grid()
            ad = self.envGrid.get_p_grid()

            observations = {
                a : {
                "eReward" : pd,
                "pGrid" : ad,
                "agent" : ap[0],
                "action mask" : action_mask[a]
                } for a in self.agents
            }

        info = {
            f"agent{a}" : {} for a in range(self.numAgents)
        }        

        return observations, info
    
    def sample(self):
        actions = {}
        for i in range(self.numAgents):
            actions[f"agent{i}"] = random.randint(0, 3)
        observations, rewards, terminations, truncations, info = self.step(actions)
        return observations, rewards, terminations, truncations, info
    
    # Action is the action set generated by the network that is passed to the step function
    def step(self, actions):
        rewards = {f"agent{i}" : 0 for i in range(self.numAgents)}
        self.agent_rewards = {i : 0 for i in range(self.numAgents)}
        actions_list = []
        for key in actions.keys():
            actions_list.append((int(actions[key] / self.gridSize), actions[key] % self.gridSize))

        dist, traj = self.compute_trajectory(actions_list)
        events_list = self.envGrid.multistep_timesteps(list(traj.values()))
        # for i in range(len(self.agents)):
        #     for j in range(len(events_list)):
        #         rewards[f"agent{i}"] += events_list[j][i]               # Keep rewards this way for now. Considering changing them into the weird SMDP form that was used for R-learning
        #     self.agent_positions[i] = traj[f"agent{i}"][-1]
        #     self.rewards += rewards[f"agent{i}"]

        #     if rewards[f"agent{i}"] == 0:
        #         rewards[f"agent{i}"] = -1
        self.curr_step += dist["agent0"]
        self.total_steps += dist["agent0"]

        for i in range(len(self.agents)):
            self.decision_steps[f"agent{i}"] += 1
            if self.evaluation_mode:
                print("AGENT POSITION:", self.agent_positions)
                print("ACTION TAKEN: ", actions)
            self.agent_positions[i] = traj[f"agent{i}"][-1]
            # Reward as implemented here: https://arxiv.org/pdf/2006.00589 (page 4)
            self.saved_rewards[f"agent{i}"] += sum(step[i] for step in events_list)
            self.rewards += sum(step[i] for step in events_list)
            if self.decision_steps[f"agent{i}"] == 1:
                rewards[f"agent{i}"] = 0
            else:
                rewards[f"agent{i}"] = (self.saved_rewards[f"agent{i}"] * (self.decision_steps[f"agent{i}"]) / self.total_steps
                    - self.last_saved_rewards[f"agent{i}"] * (self.decision_steps[f"agent{i}"] - 1) / (self.total_steps - dist[f"agent{i}"]))
            
            self.last_saved_rewards[f"agent{i}"] = self.saved_rewards[f"agent{i}"]

        terminations = {f"agent{a}": False for a in range(self.numAgents)}

        truncations = {f"agent{a}": False for a in range(self.numAgents)}

        if self.curr_step > self.max_timesteps:
            truncations = {f"agent{a}": True for a in range(self.numAgents)}
            self.agents = {}
        
        # self.total_rewards += rewards["agent0"]
        # print(rewards["agent0"])
        er = [] #Store the expected rewards
        ep = [] #Store the probability grids
        ap = [] #Store the agent positions

        for i in range(len(traj.values())): # Doesn't have to be min_idx here
            events_tracked = events_list[i]
            d, e, f = self.gridTracker.multi_update([trimmed_traj[i] for trimmed_traj in list(traj.values())], events_tracked, self.curr_step - dist[f"agent{i}"] + i + 1)
        er = d
        ep = e
        ap.append(f)

        action_mask = {}
        for i in range(len(self.agent_positions)):
            a = np.ones(self.gridSize * self.gridSize)
            a[self.agent_positions[i][0] * self.gridSize + self.agent_positions[i][1]] = 0
            action_mask[f"agent{i}"] = a
        # for a in range(len(self.agents)):
        #     d, e, f = self.gridTracker.update((self.agent_positions[a][0], self.agent_positions[a][1]), self.agent_rewards[a], self.total_steps)
        #     er = d
        #     ep = e
        #     ap.append(f)

        if not self.degenerate_case:
            observations = { #Doesn't save the timestep that events occur intrinsicly
                a : {
                    "eReward" : er / self.bound, #Estimated expected reward of the map
                    "pGrid" : ep / ep.max(), #Probabiity grid of the map
                    "agent" : ap[0], #Gridspace of where the agent is located,
                    "action mask" : action_mask[a]
                } for a in self.agents
            }

        else:
            pd = self.envGrid.get_e_grid()
            ad = self.envGrid.get_p_grid()

            observations = {
                a : {
                    "eReward" : pd,
                    "pGrid" : ad / ad.max(),
                    "agent" : ap[0],
                    "action mask" : action_mask[a]
                } for a in self.agents
            }

        info = {
            f"agent{a}" : {} for a in range(self.numAgents)
        }

        if self.render_mode != None or self.total_steps > 500000:
            self.render_mode = "human"
            self.render()

        return observations, rewards, terminations, truncations, info
    
    # def move_agent(self, actions):
    #     distance_travel, trajectory_taken = self.compute_trajectory(actions)
    #     for i in range(self.num_agents):
    #         action = actions.get(f"agent{i}")
        # action_mask = [np.ones(4) for i in range(self.numAgents)]
        # for i in range(self.numAgents):
        #     x, y = self.agent_positions[i]
        #     action = actions.get(f"agent{i}")
        #     if action == 0 and x < self.gridSize - 1:
        #         x += 1
        #     elif action == 1 and x > 0:
        #         x -= 1  
        #     elif action == 2 and y < self.gridSize - 1:
        #         y += 1
        #     elif action == 3 and y > 0:
        #         y -= 1
        #     self.agent_positions[i] = (x, y)
        #     #Agents move and their locations are recorded
        #     if x == self.gridSize - 1:
        #         action_mask[i][0] = 0
        #     if x == 0:
        #         action_mask[i][1] = 0
        #     if y == self.gridSize - 1:
        #         action_mask[i][2] = 0
        #     if y == 0:
        #         action_mask[i][3] = 0
        # return action_mask
    
    def render(self):
        if self.render_mode=="human":
            self._render_frame(1)
        
        if self.render_mode == "degenerate":
            self._render_frame(2)

    def _render_frame(
        self,
        version = 1):
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

        #Draw the grids that the agent is traveling through
        for x in range(self.gridSize + 1):
            #Draw the horizontal line first
            pygame.draw.line(
                canvas,
                0,
                (self.pix_square_size * x, 0),
                (self.pix_square_size * x, self.window_size),
                width = 2
            )
            #Now draw the vertical line
            pygame.draw.line(
                canvas,
                0,
                (0, self.pix_square_size * x),
                (self.window_size, self.pix_square_size * x)
            )

        if version == 1:
            for x in range(self.gridSize):
                for y in range(self.gridSize):
                    value = self.envGrid.e_grid[x][y]
                    color = (255, 255 - value * 10, 255 - value * 10)

                    pygame.draw.rect(
                        canvas,
                        color,
                        pygame.Rect(self.pix_square_size * x,
                                    self.pix_square_size * y,
                                    self.pix_square_size,
                                    self.pix_square_size)
                    )
        
        elif version == 2:
            for x in range(self.gridSize):
                for y in range(self.gridSize):
                    value = self.gridTracker.tracked_grid[x][y]
                    color = (255, 255 - value * 10, 255 - value * 10)

                    pygame.draw.rect(
                        canvas,
                        color,
                        pygame.Rect(self.pix_square_size * x,
                                    self.pix_square_size * y,
                                    self.pix_square_size,
                                    self.pix_square_size)
                    )

        for a in range(self.numAgents):
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(self.agent_positions[a][0] * self.pix_square_size, 
                            self.agent_positions[a][1] * self.pix_square_size, 
                            self.pix_square_size,
                            self.pix_square_size)
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(4)
        print("timestep rendered")
        #Add some gridlines that we are traveling through
        
    def observation_space(
        self, 
        agent
        ) -> Dict:
        return Dict({
            "eReward" : Box(low = 0, high = 1, shape=(self.gridSize, self.gridSize)),
            "pGrid" : Box(low = 0, high = 1, shape = (self.gridSize, self.gridSize)),
            "agent" : Box(low = 0, high = 1, shape = (self.gridSize, self.gridSize)),
            "action mask" : Box(low=0, high=1, shape=(self.gridSize * self.gridSize,))
        })
        
    def action_space(
        self, 
        agent
    ):
        return Discrete(self.gridSize * self.gridSize)