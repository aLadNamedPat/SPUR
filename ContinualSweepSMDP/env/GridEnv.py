from pettingzoo import ParallelEnv
from .Grid import GridTracker, GridWorld
import random
import functools
import numpy as np
import pygame
from gymnasium.spaces import Discrete, MultiDiscrete, Dict, MultiBinary, Box
from copy import copy

# Implementation of multi-agent version of https://arxiv.org/pdf/2006.00589
class GridEnv(ParallelEnv):
    metadata = {
        "name": "ContinualSweepSMDP_v0",
    }

    def __init__(self, 
        num_agents : int, 
        render_mode : str = None, 
        grid_size : int = 10,
        num_centers : int = 5,
        max_timesteps : int = 1024, 
        bound : int = 5,
        seed : int = None,
        agent_positions : list[tuple[int, int]] = None,
        ):

        self.gridSize = grid_size                                               # Set the grid size of the environment
        self.bound = bound                                                      # Set the bounds to the number of agents in an environment
        self.numAgents = num_agents                                             # Set the number of agents
        self.numCenters = num_centers                                           # Set the number of centers of probability of events occuring
        self.max_timesteps = max_timesteps                                      # Set the max number of timesteps occurring

        self._setup__()                                                         # Calculate all the trajectories

        self.gridTracker = GridTracker(grid_size, bound)                        # Create a grid tracker for tracking events occurring at different locations
        self.envGrid = GridWorld(grid_size, num_centers)                        # Create the actual grid environment

        if agent_positions is not None and len(agent_positions) == num_agents:
            self.agent_positions = agent_positions
        else: 
            self.agent_positions = self.GSL()                                   # Generate the positions of the agents randomly

        self.possible_agents = [f"agent{i}" for i in range(self.numAgents)]     # Set the possible agents for the PettingZoo environment
        self.agents = self.possible_agents[:]                                   # Set the agents for the PettingZoo environment
        self.curr_step = 0                                                      # Current timestep of the environment

        # Rendering variables
        self.pix_square_size = (                                                # Size of the square of the environment
            64
        )
        self.render_mode = render_mode
        self.window = None
        self.window_size = (self.pix_square_size) * (self.gridSize)

        self.saved_rewards = {f"agent{i}" : 0 for i in range(self.numAgents)}  # Rewards that are saved for each agent when they don't finish traveling their trajectory
        self.last_saved_rewards = {f"agent{i}" : 0 for i in range(self.numAgents)} # Last saved rewards for agents
        self.decision_steps = {f"agent{i}" : 0 for i in range(self.numAgents)} # Save the number of decision steps that each agent is taking

    def _setup__(self):                                                        # Compute the shortest path and trajectory for all points to each other

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
                    self.l_node[i, j] = 0
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
                final[0] * self.gridSize + final[1]
            ]                                                                       # Determine the distance that the agent has to travel from beginning to end

        trajectories = {f"agent{i}" : [(final[f"agent{i}"][0], final[f"agent{i}"][1])] for \
                                       i in range(len(self.possible_agents))}       # Save the trajectories of each agent

        for i in range(len(self.possible_agents)):
            while(self.l_node[original_positions[f"agent{i}"], final_positions[f"agent{i}"]]):
                final_positions[f"agent{i}"] = int(self.l_node(original_positions[f"agent{i}"]))
                if final_positions[f"agent{i}"] != 0:
                    trajectories[f"agent{i}"] = [(int(final_positions[f"agent{i}"] / self.gridSize), final_positions[f"agent{i}"] % self.gridSize)] + trajectories[f"agent{i}"]
            
        return (trajectories, total_dist) #Returns the expected reward, the trajectory taken, and the total distance of travel

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
        ):

        self.agents = copy(self.possible_agents)
        self.curr_step = 0
        agent_mask = [None for _ in range(self.num_agents)]
        self.agent_positions = self.GSL()

        self.gridTracker = GridTracker(self.gridSize, self.bound)
        self.envGrid = GridWorld(self.gridSize, self.numCenters)

        ap = [] #Store the other agent positions
        op = [] #Store the agent's own position

        for a in range(len(self.agents)):
            d, e, f = self.gridTracker.update((self.agent_positions[a][0], self.agent_positions[a][1]), 0, 0)
        er = d
        ep = e

        for i in range(self.numAgents):
            ap.append(f)

        # Remove the agent's own position from its own position list
        for i in range(len(ap)):
            ap[i][self.agent_positions[i][0], self.agent_positions[i][1]] = 0
            opo = np.zeros(ap[i].shape)
            opo[self.agent_positions[i][0], self.agent_positions[i][1]] = 1
            op.append(opo)

        observations = {
            f"agent{a}" : { 
                "observation" : {
                    "eReward" : er,         # Estimated expected reward of the map
                    "pGrid" : ep,           # Probabiity grid of the map
                    "agent" : op[a],        # Gridspace of where the agent is located
                    "other_agents" : ap[a], # Gridspace where other agents are located
                },
                "agent_mask" : agent_mask[a]
            }
            for a in range(len(self.agents))
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
    
    # def partial_step(
    #         self,
    #         points : list[tuple[int, int]],
    #     ) -> dict: #A semi-step in the entire step processes
    
    #     curr_rewards = {f"agent{i}" : 0 for i in range(self.numAgents)}
    #     self.curr_step += 1
    #     for i in range(len(points)):
    #         curr_rewards[f"agent{i}"] = self.envGrid.step((self.agent_positions[i][1], ))

    #     return curr_rewards

    # Action is the action set generated by the network that is passed to the step function

    def step(
            self, 
            actions # Actions are going to be a dictionary of points the agents are going to travel to
        ):
        
        rewards = {f"agent{i}" : 0 for i in range(self.numAgents)}
        self.agent_rewards = {i : 0 for i in range(self.numAgents)}

        points = list(actions.values())                             # Actions of all agents (points to travel to)
        trajs, dist = self.compute_trajectory(points)               # Compute the trajectory of all agents
        min_idx = min(enumerate(dist), key = lambda x : x[1])[0]    # Find the index of the minimum distance

        min_idxs = []                                               # Save the index of all agents with minimal travel times
        non_min_idxs = []                                           # Save the index of all agents that are not minimal travel times (these are the masked agents)
        agent_mask = [None for _ in range(self.numAgents)]          # Build a mask for agents that haven't finished their actions
        events = {f"agent{i}" : 0 for i in range(self.numAgents)}

        for i in range(len(dist)):
            if dist[i] == dist[min_idx]:
                min_idxs.append(i)                                  # All movements that are minimal (travel in the same time)
                self.decision_steps[f"agent{i}"] += 1               # Add 1 to the number of decision steps that each agent has taken so far
            else:
                non_min_idxs.append(i)
                agent_mask[i] = points[i]
            
        
        trimmed_trajs = [traj[:dist[min_idx]] for traj in trajs]    # Trim all trajectories to the shortest length

        self.curr_step += dist[min_idx]                             # Update the number of timesteps that are currently taken
        
        events = self.envGrid.multistep_timesteps(trimmed_trajs)    # Update the events that were detected by the agents

        for i in range(len(self.agents)):
            self.agent_positions[i] = trimmed_trajs[i][-1]
            # Reward as implemented here: https://arxiv.org/pdf/2006.00589 (page 4)
            self.saved_rewards[f"agent{i}"] += sum(step[i] for step in events)
            rewards[f"agent{i}"] = self.saved_rewards[f"agent{i}"] / self.curr_step * (self.decision_steps[f"agent{i}"]) \
                  - self.last_saved_rewards[f"agent{i}"] * (self.decision_steps[f"agent{i}"] - 1) / (self.curr_step - dist[min_idx])
            
            if i in min_idxs:
                self.last_saved_rewards[f"agent{i}"] = self.saved_rewards[f"agent{i}"]
                self.saved_rewards[f"agent{i}"] = 0

        terminations = {f"agent{a}": False for a in range(self.numAgents)}

        truncations = {f"agent{a}": False for a in range(self.numAgents)}

        if self.curr_step > self.max_timesteps:
            truncations = {f"agent{a}": True for a in range(self.numAgents)}
            truncate = True
        else:
            truncate = False
        

        op = [] #Store the agent's own positions
        ap = [] #Store the other agent's positions
        for i in range(len(trimmed_trajs[min_idx])):                # Doesn't have to be min_idx here
            events_tracked = [event[i] for event in events]
            d, e, f = self.gridTracker.multi_update((trimmed_traj[i] for trimmed_traj in trimmed_trajs), events_tracked, self.curr_step - dist[min_idx] + i)
        er = d
        ep = e
        
        for i in range(self.numAgents):
            ap.append(f)

        # Remove the agent's own position from its own position list
        for i in range(len(ap)):
            ap[i][self.agent_positions[i][0], self.agent_positions[i][1]] = 0
            opo = np.zeros(ap[i].shape)
            opo[self.agent_positions[i][0], self.agent_positions[i][1]] = 1
            op.append(opo)
        
        observations = { #Doesn't save the timestep that events occur intrinsicly
            f"agent{a}" : {
                "observation" : {
                    "eReward" : er / self.bound, #Estimated expected reward of the map
                    "pGrid" : ep, #Probabiity grid that the agents are tracking
                    "agent" : op[a], #Gridspace of where the agent is located
                    "other_agents" : ap[a], #Gridspace of where other agents are located
                },
                "action_mask" : agent_mask[a] #If agent_mask is not None, then that action must be chosen, otherwise a new action is taken
            }
            for a in range(len(self.agents))
        }

        info = {
            f"agent{a}" : {} for a in range(self.numAgents)
        }

        if self.render_mode != None:
            self.render()

        return observations, rewards, terminations, truncations, info
    
    def render(self):
        if self.render_mode=="human":
            self._render_frame()

    def _render_frame(self):
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
            "agent" : Box(low = 0, high = 1, shape = (self.gridSize, self.gridSize), dtype = np.int16),
            "other_agents" : Box(low = 0, high = 1, shape = (self.gridSize, self.gridSize), dtype = np.int16)
        })
        
    def action_space(self, agent):
        return MultiDiscrete([self.gridSize, self.gridSize])