import numpy as np
import src.environment as env
from src.grid import Grid
from src.policy import Policy


class Problem:

    def __init__(self, row_sz, col_sz, policy):
        self.grid = Grid(row_sz, col_sz)
        self.agent = Agent(self.grid, np.random.randint(0,row_sz), np.random.randint(0,col_sz), policy)

    def train(self,n):
        for i in range(n):
            print(i)
            self.agent.take_action()
        self.agent.take_action()


class Agent:

    def __init__(self, grid, starting_row, starting_col, policy):
        self.grid = grid
        self.position = (starting_row, starting_col)
        self.policy = policy
        self._total_reward = 0

    def take_action(self):
        percepts = self.grid.retrieve_sensor_inputs(self.position[0], self.position[1])
        q_value, state, value, best_action = self.policy.choose_action(percepts)
        next_position, reward = self.grid.perform_action(best_action, self.position)
        self._total_reward += reward
        self.position = next_position
        self.policy.update((state,value), best_action, reward, self.grid.retrieve_sensor_inputs(self.position[0], self.position[1]))


p = Problem(10,10, Policy(env.ACTIONS, 0.2, 0.9, 0.1))
p.train(50)

