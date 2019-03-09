import numpy as np
import src.environment as env


class Policy:

    def __init__(self, actions, n, y, e):
        self.q_table = np.zeros((15, 5))
        self.input_to_index = self._create_input_index_mapping()
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

    def choose_action(self, available_inputs):
        available_actions = self._retrieve_values_actions_from(available_inputs)
        best_index = available_actions.index(max(available_actions, lambda x: x[0]))
        best_action = np.random.choice([available_actions[best_index], None], p=[1-self._e, self._e])
        if not best_action:
            options = available_actions[:best_index] + available_actions[best_index+1:]
            probabilities = [x[0]/sum(n for n,_ in options) for x in options]
            if sum(probabilities) == 0:
                probabilities = [1/len(probabilities) for x in probabilities]
            best_action = np.random.choice(options, p=probabilities)
        return best_action[1] # ((state, val), action enum)

    def _retrieve_values_actions_from(self, inputs):
        available_actions = []
        for sensor_input in inputs:
            for i in range(len(self.q_table[self.input_to_index[sensor_input]])):
                available_actions.append((self.q_table[self.input_to_index[sensor_input]][i], (sensor_input, env.Action(i))))
        return available_actions

    def update(self, position_value, action, reward, next_inputs):
        state_index = self.input_to_index[position_value]
        self.q_table[state_index, action.value] = self.q_table[state_index, action.value] + \
            self._n * (reward + self._y*(max(self._retrieve_values_actions_from(next_inputs), lambda v: v[0])) - self.q_table[state_index, action.value])


