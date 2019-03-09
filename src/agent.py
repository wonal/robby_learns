import numpy as np
from src.grid import Grid


class Problem:

    def __init__(self, row_sz, col_sz, policy):
        self.grid = Grid(row_sz, col_sz)
        self.agent = Agent(self.grid, np.random.randint(0,row_sz), np.random.randint(0,col_sz), policy)


class Agent:

    def __init__(self, grid, starting_row, starting_col, policy):
        self.grid = grid
        self.position = (starting_row, starting_col)
        self.policy = policy
        self._total_reward = 0

    def take_action(self):
        percepts = self.grid.retrieve_sensor_inputs(self.position[0], self.position[1])
        state, best_action = self.policy.choose_action(percepts)[1]
        next_position, reward = self.grid.perform_action(best_action, self.position)
        self.position = next_position
        self.policy.update(state, best_action, reward, self.grid.retrieve_sensor_inputs(self.position[0], self.position[1]))


