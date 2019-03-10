import numpy as np
import src.environment as env
from src.grid import Grid
from src.policy import Policy


class Problem:

    def __init__(self, row_sz, col_sz):
        self.grid = Grid(row_sz, col_sz)
        self.policy = Policy(env.ACTIONS, 0.2, 0.9, 0.1)
        self.robby = Agent(self.grid, np.random.randint(0, row_sz), np.random.randint(0, col_sz), self.policy)
        self.epoch = 1

    def train(self,steps):
        for i in range(steps):
            #print(i)
            self.robby.take_action()
        return self.robby.total_reward

    def run(self, n, m):
        print("Training--------------\n")
        print("Training--------------\n")
        print("Training--------------\n")
        for i in range(n): #episodes
            reward_accumulated = self.train(m)
            self.policy.update_e(self.epoch)
            self.epoch += 1
            self.grid = Grid(10,10)
            self.robby = Agent(self.grid, np.random.randint(0, 10), np.random.randint(0, 10), self.policy)
            print(i, self.policy._e, reward_accumulated)
        print("Test---------------\n")
        print("Test---------------\n")
        print("Test---------------\n")
        self.policy.reset_e()
        for i in range(n):
            self.grid = Grid(10,10)
            self.robby = Agent(self.grid, np.random.randint(0, 10), np.random.randint(0, 10), self.policy)
            reward_accumulated = self.train(m)
            print(i, self.policy._e, reward_accumulated)
        self.grid = Grid(10,10)
        print(self.grid.grid)
        self.robby = Agent(self.grid, np.random.randint(0, 10), np.random.randint(0, 10), self.policy)
        reward_accumulated = self.train(m)
        print(self.policy._e, reward_accumulated)


class Agent:

    def __init__(self, grid, starting_row, starting_col, policy):
        self.grid = grid
        self.position = (starting_row, starting_col)
        self.policy = policy
        self.total_reward = 0

    def take_action(self):
        percepts = self.grid.retrieve_sensor_inputs(self.position[0], self.position[1])
        q_value, state, value, best_action = self.policy.choose_action(percepts)
        next_position, reward = self.grid.perform_action(best_action, self.position)
        self.total_reward += reward
        self.position = next_position
        self.policy.update((state,value), best_action, reward, self.grid.retrieve_sensor_inputs(self.position[0], self.position[1]))


p = Problem(10,10)
#p.train(200)
p.run(5000,200)

