import numpy as np
import math
import random
from copy import deepcopy
import time

def __setup__(
    gridSize
    ):                                                 
    # Compute the shortest path and trajectory for all points to each other

    # Initialize the graph of the grid in terms of distances
    dist = np.zeros((gridSize * gridSize, gridSize * gridSize))
    # Initialize the last location to travel to
    l_node = np.zeros((gridSize * gridSize, gridSize * gridSize))

    # Prepare the adjacency matrix
    for i in range(gridSize):
        for j in range(gridSize):
            for k in range(gridSize):
                for l in range(gridSize):
                    if (i == k and (j == l -1 or j == l + 1)) or (j == l and (i == k -1 or i == k + 1)):
                        dist[i * gridSize + j, k * gridSize + l] = 1
                    elif (i == k and j == l):
                        dist[i * gridSize + j, k * gridSize + l] = 0
                    else:
                        dist[i * gridSize + j, k * gridSize + l] = float("inf")
    
    # Prepare the path matrix
    for i in range(gridSize * gridSize):
        for j in range(gridSize * gridSize):
            if i == j:
                l_node[i, j] = -1
            elif dist[i, j] != float("inf"):
                l_node[i, j] = i
            else:
                l_node[i, j] = -1

    # Floyd-Warshall's algorithm
    for i in range(gridSize * gridSize):                                  #This is the starting location x, y
        for j in range(gridSize * gridSize):                              #This is the ending location x, y
            for k in range(gridSize * gridSize):                          #This is the mid location x, y
                if (dist[i, k] is not float("inf") and dist[k, j] is not float("inf")):
                    if (dist[i, j] > dist[i, k] + dist[k, j]):
                        dist[i, j] =  dist[i, k] + dist[k, j]
                        l_node[i, j] = l_node[k, j]

    return dist, l_node

# Keep track of the environment state that will be used in MCTS
class gridEnv:
    def __init__(
        self,
        learning_rate : float,
        grid_size : int,
        starting_location : tuple[int, int],
        forget_rate : float,
        bound : int,
        dist_matrix,
        l_node_matrix,
    ) -> None:
        self.lr = learning_rate
        self.gridSize = grid_size
        self.bound = bound                                              # The bound as defined by the environment
        self.forget_rate = forget_rate                                  # Define the rate at which the agent forgets bad actions
        self.grid = np.ones((grid_size, grid_size))                     # Potential reward of visiting a point
        self.lv = np.zeros((grid_size, grid_size))                      # Last visit time of a grid location

        self.er = np.zeros((grid_size, grid_size))                      # Expected reward overall
        self.location = starting_location                               # Set current agent location to its starting location
        self.current_position = starting_location[0] * self.gridSize + starting_location[1]

        self.current_time = 0
        self.accumulated_reward = 0
        self.total_dist_traveled = 0

        self.dist = dist_matrix
        self.l_node = l_node_matrix
    # def _setup__(self):                                                 # Compute the shortest path and trajectory for all points to each other
    #     # Initialize the graph of the grid in terms of distances
    #     self.dist = np.zeros((self.gridSize * self.gridSize, self.gridSize * self.gridSize))
    #     # Initialize the last location to travel to
    #     self.l_node = np.zeros((self.gridSize * self.gridSize, self.gridSize * self.gridSize))

    #     # Prepare the adjacency matrix
    #     for i in range(self.gridSize):
    #         for j in range(self.gridSize):
    #             for k in range(self.gridSize):
    #                 for l in range(self.gridSize):
    #                     if (i == k and (j == l -1 or j == l + 1)) or (j == l and (i == k -1 or i == k + 1)):
    #                         self.dist[i * self.gridSize + j, k * self.gridSize + l] = 1
    #                     elif (i == k and j == l):
    #                         self.dist[i * self.gridSize + j, k * self.gridSize + l] = 0
    #                     else:
    #                         self.dist[i * self.gridSize + j, k * self.gridSize + l] = float("inf")
        
    #     # Prepare the path matrix
    #     for i in range(self.gridSize * self.gridSize):
    #         for j in range(self.gridSize * self.gridSize):
    #             if i == j:
    #                 self.l_node[i, j] = 0
    #             elif self.dist[i, j] != float("inf"):
    #                 self.l_node[i, j] = i
    #             else:
    #                 self.l_node[i, j] = -1


    #     # Floyd-Warshall's algorithm
    #     for i in range(self.gridSize * self.gridSize):                                  #This is the starting location x, y
    #         for j in range(self.gridSize * self.gridSize):                              #This is the ending location x, y
    #             for k in range(self.gridSize * self.gridSize):                          #This is the mid location x, y
    #                 if (self.dist[i, k] is not float("inf") and self.dist[k, j] is not float("inf")):
    #                     if (self.dist[i, j] > self.dist[i, k] + self.dist[k, j]):
    #                         self.dist[i, j] =  self.dist[i, k] + self.dist[k, j]
    #                         self.l_node[i, j] = self.l_node[k, j]

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
        self.er[point[0], point[1]] = 0                                 # Reset the expected reward to 0

    def false_take_action(
        self,
        point : tuple[int, int],
    ) -> tuple[float, float]:
        (expected_reward, traj, total_dist) = self.compute_trajectory(point)
        for i in range(len(traj)):
            self.lv[traj[i]] = self.current_time + 1                         # Need to run current_time as the initial first step
            self.current_time += 1
            self.er += self.grid
            self.er[traj[i]] = 0

        self.accumulated_reward += expected_reward
        self.total_dist_traveled += total_dist
        self.current_position = point[0] * self.gridSize + point[1]
        self.location = [point[0], point[1]]
        return self
    
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
    
    def take_known_action(
        self,
        point : tuple[int, int]
    ):
        (rewards, traj, time) = self.compute_trajectory(point)
        self.current_position = point[0] * self.gridSize + point[1]
        self.location = [point[0], point[1]]
        return (traj, time)

    def compute_trajectory(
        self,
        final : tuple[int, int]
    ) -> tuple[float, list[tuple[int, int]], int]:
        total_dist = self.dist[self.location[0] * self.gridSize + self.location[1],
                               final[0] * self.gridSize + final[1]]
        
        original = self.location[0] * self.gridSize + self.location[1]
        traj = [(final[0], final[1])]
        final = final[0] * self.gridSize + final[1]
        running = False

        while(self.l_node[original, final] != -1):
            final = int(self.l_node[original, final])
            if final != -1:
                traj = [(int(final / self.gridSize), final % self.gridSize)] + traj
        
        traj = traj[1:]

        # traj = [(original[0], original[1])] + traj 
        expected_reward = 0

        for i, (x, y) in enumerate(traj):
            expected_reward += min(self.bound, self.er[x, y] + self.grid[x, y] * (i + 1))
        return (expected_reward, traj, total_dist) #Returns the expected reward, the trajectory taken, and the total distance of travel
    
    def get_possible_moves(
        self,
    ):        
        available_moves = np.arange(self.gridSize * self.gridSize)
        position = self.location[0] * self.gridSize + self.location[1]
        return np.delete(available_moves, position) # Returns all the possible moves (where the agents positions is not included)
    
    def is_terminal(
        self,
        horizon_steps,
        steps_taken,
    ):
        return horizon_steps == steps_taken
            
    def get_reward(
        self,
    ):
        saved_reward = self.accumulated_reward
        distance_traveled = self.total_dist_traveled
        self.accumulated_reward = 0
        self.total_dist_traveled = 0
        if distance_traveled != 0:
            return saved_reward / distance_traveled
        else:
            return saved_reward

    # Compute the best action for the agent to take at a given timestep
    def lower_bound(
        self,
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
        return (best_action, max_reward)

# Need a node class to keep track of the next points to travel to
class Node:
    def __init__(
        self, 
        state,
        parent = None,
    ) -> None:
        self.state : gridEnv = state
        self.parent = parent
        self.children : list[Node] = []
        self.times_visited = 0
        self.value = 0
        
    #Fully expanded refers to all of its possible nodes being visited
    def is_fully_expanded(
        self
    ):
        return len(self.children) == len(self.state.get_possible_moves())
    
    def apply_move(
        self,
        action : int,
    ):
        new_env = deepcopy(self.state)
        action = (action // new_env.gridSize, action % new_env.gridSize)
        new_env.false_take_action(action)
        return new_env
    
    # Will return the value of the best child that it has visited
    def best_child(
        self,
        exploration_bias = 1.41,
        final_return = False, # If this is the action that is actually going to be taken
    ) -> float: # 
        choices_weights = [
            child.value / child.times_visited + exploration_bias * math.sqrt(math.log(self.times_visited) / child.times_visited)
            for child in self.children
        ]
        # choices_weights = [
        #     child.value + exploration_bias * math.sqrt(math.log(self.times_visited) / child.times_visited)
        #     for child in self.children
        # ]

        new_env = self.apply_move(self.children[choices_weights.index(max(choices_weights))].state.current_position)
        reward = new_env.get_reward()

        if self.state.lower_bound()[1] > reward and final_return:
            to_return = Node(gridEnv(0, 5, self.state.lower_bound()[0], 0, 0, 0, 0))
        else:
            to_return = self.children[choices_weights.index(max(choices_weights))]

        return to_return, reward
    
class MCTS:
    #Define the MCTS search with the exploration bias that is given to it
    #Using the UCB1 formula which is value + exploration_bias * sqrt(ln(number of visits of parents) / number of times a node has been visited)

    def __init__(
        self, 
        exploration_weight=1.41,
        horizon_steps = 10,
    ) -> None:
        self.exploration_weight = exploration_weight
        self.horizon_steps = horizon_steps

    def search(
        self, 
        initial_state : gridEnv,
        num_iterations : int #Number of total steps downward that the algorithm is going to take
    ) -> Node:
        root = Node(initial_state)

        for _ in range(num_iterations):
            self.layers_deep = 0
            node = self._select(root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward, average_reward=True)

        # for child in root.children:
        #     print("Node: ", child.state.current_position, child.value, child.times_visited)
        # # # print(root.children.index(root.best_child(exploration_bias=0)))
        # print(root.best_child(exploration_bias=0)[0].state.current_position)
        # print(root.best_child(exploration_bias=0)[0])
        # time.sleep(2)
        return root.best_child(exploration_bias=0, final_return=True)[0]

    # Selects a node from the root that is currently unexplored, or chooses the best action otherwise
    def _select(
        self, 
        node : Node,
    ):
        self.saved_reward = 0
        while not node.state.is_terminal(self.horizon_steps, self.layers_deep):
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                self.layers_deep += 1
                node, reward = node.best_child(self.exploration_weight)
                self.saved_reward = reward
                # print("Current reward:", reward)
                # print("Reward saved:", self.saved_reward)
                # print("Layers deep:", self.layers_deep)
                # print("Node current position:", node.state.current_position)
                # time.sleep(0.5)
        return node

    def _expand(
        self, 
        node : Node
    ) -> Node:
        # An action is labeled as a move here instead of a possible state to travel to
        untried_actions = [action for action in node.state.get_possible_moves() if action not in [child.state.current_position for child in node.children]]
        action = random.choice(untried_actions)
        # Make a new grid environment here where the variant is applied
        next_state = node.apply_move(action)
        child_node = Node(next_state, node)
        node.children.append(child_node)
        return child_node

    def _simulate(
        self, 
        state : gridEnv,
    ) -> float:
        current_state = state
        num_steps_taken = self.layers_deep
        while not current_state.is_terminal(self.horizon_steps, num_steps_taken):
            num_steps_taken += 1
            possible_moves = current_state.get_possible_moves()
            action = random.choice(possible_moves)
            action = (action // current_state.gridSize, action % current_state.gridSize)
            current_state = deepcopy(current_state)
            current_state = current_state.false_take_action(action)
        return current_state.get_reward() + self.saved_reward

    def _backpropagate(
        self, 
        node : Node, 
        reward : float,
        average_reward : bool = True
    ):
        while node is not None:
            node.times_visited += 1
            if average_reward:
                node.value += reward # This needs to be weighted somehow to prevent one side from asymmetrically being biased
            else:
                if reward > node.value:
                # This algorithm could work because the reward is already streamed through the entirety of the rollout
                # Therefore, saving just the highest reward could help in determining which algorithm really is best
                    node.value = reward 
            node.value += reward # This needs to be weighted somehow to prevent one side from asymmetrically being biased
            node = node.parent