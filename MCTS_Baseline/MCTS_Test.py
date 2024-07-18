import numpy as np
from typing import Tuple

from MCTS import gridEnv, MCTS, __setup__

def run_mcts_test(
    grid_size: int,
    starting_location: Tuple[int, int],
    learning_rate: float,
    forget_rate: float,
    bound: int,
    num_iterations: int,
    horizon_steps: int
) -> None:
    dist, l_node = __setup__(grid_size)
    env = gridEnv(
        learning_rate=learning_rate,
        grid_size=grid_size,
        starting_location=starting_location,
        forget_rate=forget_rate,
        bound=bound,
        dist_matrix= dist,
        l_node_matrix = l_node
    )
    mcts = MCTS(exploration_weight=15,horizon_steps=horizon_steps)
    best_node = mcts.search(initial_state=env, num_iterations=num_iterations)
    print(f"Best action: {best_node.state.current_position}" ,f"{best_node.state.location}")
    print(f"Expected reward: {best_node.value / best_node.times_visited}")
    print(f"Times visited: {best_node.times_visited}")
    
    # Visualize the grid
    print("\nGrid state:")
    print(env.grid)
    
    print("\nExpected rewards:")
    print(env.er)

if __name__ == "__main__":
    # Set parameters
    GRID_SIZE = 5
    STARTING_LOCATION = (0, 0)
    LEARNING_RATE = 0.1
    FORGET_RATE = 0.8
    BOUND = 10
    NUM_ITERATIONS = 1000
    HORIZON_STEPS = 3
    
    # Run the test
    run_mcts_test(
        grid_size=GRID_SIZE,
        starting_location=STARTING_LOCATION,
        learning_rate=LEARNING_RATE,
        forget_rate=FORGET_RATE,
        bound=BOUND,
        num_iterations=NUM_ITERATIONS,
        horizon_steps=HORIZON_STEPS
    )