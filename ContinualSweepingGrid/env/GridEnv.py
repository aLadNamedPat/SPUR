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
        parallelize : bool = False,
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
        self.parallelize = parallelize

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
        self.clock = None

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
        # print(sum(sum(self.envGrid.get_p_grid())))
        # print(self.rewards)
        self.rewards = 0
        self.agent_positions = self.GSL()
        self.steps_since_last_reset += 1

        # if self.steps_since_last_reset % self.reset_steps == 0 or self.parallelize:
        #     self.steps_since_last_reset = 0
        #     self.gridTracker = GridTracker(self.gridSize, self.bound)
        #     self.envGrid = GridWorld(self.gridSize, self.numCenters)
        #     self.parallelize = False

        action_mask = [np.ones(4) for i in range(self.numAgents)]

        for i in range(self.numAgents):
            x, y = self.agent_positions[i]
            if x == self.gridSize - 1:
                action_mask[i][0] = 0
            if x == 0:
                action_mask[i][1] = 0
            if y == self.gridSize - 1:
                action_mask[i][2] = 0
            if y == 0:
                action_mask[i][3] = 0

        er = [] #Store the expected rewards
        ep = [] #Store the probability grids
        ap = [] #Store the agent positions

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
                }
                for a in self.agents
            }
        
        else:
            pd = self.envGrid.get_e_grid()
            ad = self.envGrid.get_p_grid()

            observations = {
                a : {
                    "eReward" : pd,
                    "pGrid" : ad,
                    "agent" : ap[0],
                }
                for a in self.agents
            }

        info = {
            "num_timesteps" : 0,
            "events_detected" : 0,
        }

        return observations, info
    
    def sample(self):
        actions = {}
        for i in range(self.numAgents):
            actions[f"agent{i}"] = random.randint(0, 3)
        observations, rewards, terminations, truncations, info = self.step(actions)
        return observations, rewards, terminations, truncations, info
    
    # Action is the action set generated by the network that is passed to the step function
    def step(
        self,
        actions
        ):

        if self.parallelize:
            self.reset()
        rewards = {f"agent{i}" : 0 for i in range(self.numAgents)}
        self.agent_rewards = {i : 0 for i in range(self.numAgents)}
        self.curr_step += 1
        self.total_steps += 1
        action_mask = self.move_agent(actions)
        self.events_detected = 0
        # print("ACTION TAKEN: ", actions)
        # print("AGENT POSITION: ", self.agent_positions)
        for i in range(len(self.agents)):
            rewards[f"agent{i}"] = self.envGrid.step((self.agent_positions[i][0], self.agent_positions[i][1]))
            self.rewards += rewards[f"agent{i}"]
            self.events_detected += rewards[f"agent{i}"]

            # if rewards[f"agent{i}"] == 0:
            #     rewards[f"agent{i}"] = -1

        terminations = {f"agent{a}": False for a in range(self.numAgents)}

        truncations = {f"agent{a}": False for a in range(self.numAgents)}

        # if self.curr_step > self.max_timesteps:
        #     truncations = {f"agent{a}": True for a in range(self.numAgents)}
        #     self.agents = {}
        
        # self.total_rewards += rewards["agent0"]
        # print(rewards["agent0"])
        er = [] #Store the expected rewards
        ep = [] #Store the probability grids
        ap = [] #Store the agent positions

        for a in range(len(self.agents)):
            d, e, f = self.gridTracker.update((self.agent_positions[a][0], self.agent_positions[a][1]), self.agent_rewards[a], self.total_steps)
            er = d
            ep = e
            ap.append(f)
        
        if not self.degenerate_case:
            observations = { #Doesn't save the timestep that events occur intrinsicly
                a : {
                        "eReward" : er / self.bound, #Estimated expected reward of the map
                        "pGrid" : ep / ep.max(), #Probabiity grid of the map
                        "agent" : ap[0], #Gridspace of where the agent is located
                }
                for a in self.agents
            }
        else:
            pd = self.envGrid.get_e_grid()
            ad = self.envGrid.get_p_grid()

            observations = {
                a : {
                    "eReward" : pd,
                    "pGrid" : ad / ad.max(),
                    "agent" : ap[0],
                }
                for a in self.agents
            }
            # print("OBSERVATIONS: ", observations)

        info = {
            "num_timesteps" : 1,
            "events_detected" : self.events_detected,
        }

        if self.render_mode != None or self.total_steps > 500000:
            self.render_mode = "human"
            self.render()

        return observations, rewards, terminations, truncations, info
    
    def move_agent(self, actions):
        action_mask = [np.ones(4) for i in range(self.numAgents)]
        for i in range(self.numAgents):
            x, y = self.agent_positions[i]
            action = actions.get(f"agent{i}")
            if action == 0 and x < self.gridSize - 1:
                x += 1
            elif action == 1 and x > 0:
                x -= 1  
            elif action == 2 and y < self.gridSize - 1:
                y += 1
            elif action == 3 and y > 0:
                y -= 1
            self.agent_positions[i] = (x, y)
            #Agents move and their locations are recorded
            if x == self.gridSize - 1:
                action_mask[i][0] = 0
            if x == 0:
                action_mask[i][1] = 0
            if y == self.gridSize - 1:
                action_mask[i][2] = 0
            if y == 0:
                action_mask[i][3] = 0
        return action_mask
    
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
            "agent" : Box(low = 0, high = 1, shape = (self.gridSize, self.gridSize))
        })
        
    def action_space(
        self, 
        agent
    ):
        return Discrete(4)