import numpy as np


class Policy:

    def __init__(self, actions, n, y, e):
        self.q_table = np.zeros((15, 5))
        self._actions = actions
        self._n = n
        self._y = y
        self._e = e

    def choose_action(self, state_index):
        best_action = np.random.choice([max(self.q_table[state_index]), None], p=[1-self._e, self._e])
        if not best_action:
            options = [self.q_table[:state_index]]+[self.q_table[state_index+1:]]
            probabilities = [x/sum(options) for x in options]
            if sum(probabilities) == 0:
                probabilities = [1/len(probabilities) for x in probabilities]
            best_action = np.random.choice(options, p=probabilities)
        return best_action

    def update(self, current_state, action, reward, next_state):
        self.q_table[current_state, self._actions[action]] = self.q_table[current_state, self._actions[action]] + self._n * (reward + self._y*(max(self.q_table[next_state])))


