import numpy as np
import matplotlib.pyplot as plt
from src.grid import Grid
from src.policy import Policy
from src.types import *


class Problem:

    def __init__(self, row_sz: int, col_sz: int):
        self.grid = Grid(row_sz, col_sz)
        self.policy = Policy(env.ACTIONS, env.ETA, env.GAMMA, env.EPSILON)
        self.robby = Agent(self.grid, np.random.randint(0, row_sz), np.random.randint(0, col_sz), self.policy)
        self.epoch = 1
        self.rewards_per_episode = []

    def run(self, n: int, m: int) -> (float, float):
        """
        Run a training session (runs the robot for m steps, n times) and create a plot, saved in src/training_plot.png.
        Then runs a test session, retuning the mean and standard deviation.
        :param n: The number of episodes
        :param m: The number of steps
        :return: mean, standard deviation
        """
        self._train(n, m)
        return self._test(n, m)

    def _train(self, episodes: int, steps: int):
        """
        Runs the robot for a certain number of steps (steps), and runs this a certain number of times (episodes).
        Each episode, the epsilon value is updated, a new grid is generated along with a starting position for the
        robot. The total reward accumulated per episode is tracked in the rewards_per_episode list. A training plot
        is then created and saved.
        """
        print("Beginning Training...\n")
        for i in range(episodes):
            reward_accumulated = self.run_n_steps(steps)
            self.policy.update_e()
            self.epoch += 1
            self.grid = Grid(env.GRID_BOUND, env.GRID_BOUND)
            self.robby = Agent(self.grid, np.random.randint(0, env.GRID_BOUND), np.random.randint(0, env.GRID_BOUND), self.policy)
            if self.epoch % 100 == 0:
                self.rewards_per_episode.append(reward_accumulated)
            print("Episode: {}, total reward for episode: {}".format(i+1, reward_accumulated))
        self._create_training_plot()

    def _test(self, episodes: int, steps: int) -> (float, float):
        """
        Runs a test session where the epsilon value is fixed at a value, and the robot takes a certain number
        of steps (steps), for a certain number of episodes (episodes).  Total rewards per episode are tracked and
        the mean and standard deviation are returned.
        :param episodes:
        :param steps:
        """
        print("Beginning Test...\n")
        self.policy.reset_e()
        self.rewards_per_episode = []
        for i in range(episodes):
            self.grid = Grid(env.GRID_BOUND, env.GRID_BOUND)
            self.robby = Agent(self.grid, np.random.randint(0, env.GRID_BOUND), np.random.randint(0, env.GRID_BOUND), self.policy)
            reward_accumulated = self.run_n_steps(steps)
            self.rewards_per_episode.append(reward_accumulated)
            print("Episode: {}, total reward for episode: {}".format(i+1, reward_accumulated))
        return self._calculate_test_mean_std()

    def run_n_steps(self, steps: int) -> int:
        """
        Run the robot for a certain number of steps.
        :param steps: number of steps to run the simulation
        :return: the reward accumulated
        """
        for i in range(steps):
            self.robby.take_action()
        return self.robby.total_reward

    def _create_training_plot(self):
        episodes = [x*100 for x in range(1, 51)]
        plt.plot(episodes, self.rewards_per_episode, 'ro')
        plt.savefig('training_plot.png', bbox_inches='tight')

    def _calculate_test_mean_std(self) -> (float, float):
        mean = np.mean(self.rewards_per_episode)
        std = np.std(self.rewards_per_episode)
        return mean, std


class Agent:

    def __init__(self, grid: [[int]], starting_row: int, starting_col: int, policy: Policy):
        self.grid = grid
        self.position = (starting_row, starting_col)
        self.policy = policy
        self.total_reward = 0

    def take_action(self):
        """
        Retrieves inputs from local surroundings based on robot's current position.  Based on those inputs and the
        Q-table, chooses and action using e-greedy selection, performs that action, receives the appropriate reward
        or penalty, updates the robot's position accordingly and then updates the q-table.
        """
        percepts = self.grid.retrieve_sensor_inputs(self.position[0], self.position[1])
        q_value, state, value, best_action = self.policy.choose_action(percepts)
        next_position, reward = self.grid.perform_action(best_action, self.position)
        self.total_reward += reward
        self.position = next_position
        self.policy.update((state, value), best_action, reward, self.grid.retrieve_sensor_inputs(self.position[0], self.position[1]))



