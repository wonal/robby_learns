import numpy as np
import src.environment as env
import matplotlib.pyplot as plt
from src.grid import Grid
from src.policy import Policy


class Problem:

    def __init__(self, row_sz, col_sz):
        self.grid = Grid(row_sz, col_sz)
        self.policy = Policy(env.ACTIONS, 0.2, 0.9, 1.0)
        self.robby = Agent(self.grid, np.random.randint(0, row_sz), np.random.randint(0, col_sz), self.policy)
        self.epoch = 1
        self.rewards_per_episode = []

    def train(self,steps):
        for i in range(steps):
            self.robby.take_action()
        return self.robby.total_reward

    def run(self, n, m):
        self._train(n, m)
        return self._test(n, m)

    def _train(self, episodes, steps):
        print("Beginning Training...\n")
        for i in range(episodes): #episodes
            reward_accumulated = self.train(steps)
            self.policy.update_e(self.epoch)
            self.epoch += 1
            self.grid = Grid(10,10)
            self.robby = Agent(self.grid, np.random.randint(0, 10), np.random.randint(0, 10), self.policy)
            if self.epoch % 100 == 0:
                self.rewards_per_episode.append(reward_accumulated)
            print(i, self.policy._e, reward_accumulated)
        self._create_training_plot()

    def _test(self, episodes, steps):
        print("Beginning Test...\n")
        self.policy.reset_e()
        self.rewards_per_episode = []
        for i in range(episodes):
            self.grid = Grid(10,10)
            self.robby = Agent(self.grid, np.random.randint(0, 10), np.random.randint(0, 10), self.policy)
            reward_accumulated = self.train(steps)
            self.rewards_per_episode.append(reward_accumulated)
            print(i, self.policy._e, reward_accumulated)
        return self._calculate_test_mean_std()

    def _create_training_plot(self):
        episodes = [x*100 for x in range(1,51)]
        plt.plot(episodes, self.rewards_per_episode, 'ro')
        plt.savefig('training_plot.png', bbox_inches='tight')

    def _calculate_test_mean_std(self):
        mean = np.mean(self.rewards_per_episode)
        std = np.std(self.rewards_per_episode)
        return mean, std


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
print(p.run(5000,200))

