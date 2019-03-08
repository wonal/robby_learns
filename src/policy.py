import numpy as np
import src.environment as env


class Policy:

    def __init__(self, actions, n, y, e):
        self.q_table = np.zeros((15, 5))
        self.input_mapping = self._create_input_index_mapping()
        self._actions = actions
        self._n = n
        self._y = y
        self._e = e

    @staticmethod
    def _create_input_index_mapping():
        """
        Creates a dictionary that maps a sensor position (Current, North, etc.) and value (Empty, Can, Wall) tuple
        to a Q-table index.
        :return:  The dictionary
        """
        d = {}
        for i in range(15):
            for position in env.INPUT:
                for val in env.VALUES:
                    d[(position.value, val.value)] = i
        return d
   #----------------------------------------------------------


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


