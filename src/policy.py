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
        i = 0
        for position in env.INPUT:
            for val in env.VALUES:
                d[(position.value, val.value)] = i
                i += 1
        return d

    def choose_action(self, available_inputs):
        available_actions = self._retrieve_values_actions_from(available_inputs)
        q_values = [val[0] for val in available_actions]
        if sum(q_values) == 0:
            return self._uniform_random_selection(available_actions)
        else:
            best_value = max(q_values)
            best_index = q_values.index(best_value)
            best_action = np.random.choice([available_actions[best_index], None], p=[1-self._e, self._e])
            if best_action:
                return best_action
            options = available_actions[:best_index] + available_actions[best_index+1:]
            sum_probabilities = sum([x[0] for x in options])
            if sum_probabilities <= 0:
                return self._uniform_random_selection(options)
            return self._weighted_selection(options, sum_probabilities)

    @staticmethod
    def _uniform_random_selection(actions):
        probabilities = [1/len(actions) for x in actions]
        action = np.random.choice(list(range(len(actions))), p=probabilities)
        return actions[action]

    @staticmethod
    def _weighted_selection(actions, total):
        probabilities = [x[0]/total for x in actions]
        best_action = np.random.choice(list(range(len(actions))), p=probabilities)
        return actions[best_action]

    def _retrieve_values_actions_from(self, inputs):
        available_actions = []
        for sensor_input in inputs:
            for i in range(len(self.q_table[self.input_to_index[(sensor_input[0].value, sensor_input[1].value)]])):
                available_actions.append((self.q_table[self.input_to_index[(sensor_input[0].value, sensor_input[1].value)]][i], sensor_input[0], sensor_input[1], env.Action(i)))
        return available_actions # [(q-value,state,value,action)]

    def update(self, state, action, reward, next_inputs):
        state_index = self.input_to_index[(state[0].value, state[1].value)]
        next_q_values = [q[0] for q in self._retrieve_values_actions_from(next_inputs)]
        self.q_table[state_index, action.value] = self.q_table[state_index, action.value] + \
            self._n * (reward + self._y*(max(next_q_values)) - self.q_table[state_index, action.value])

"""
p = Policy(env.ACTIONS, 0.2, 0.9, 0.1)
test = []
print("done")
"""

