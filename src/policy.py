import numpy as np
import decimal as dec
from src.types import *


class Policy:

    def __init__(self, actions: List[Action], n: float, y: float, e: float):
        self.q_table = np.zeros((15, 5))
        self.input_to_index = self._create_input_index_mapping()
        self._actions = actions
        self._n = n
        self._y = y
        self._e = round(dec.Decimal(e), 2)
        self._e_counter = 0

    @staticmethod
    def _create_input_index_mapping() -> Dict[State, int]:
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

    def update_e(self):
        """
        Updates the epsilon value: every 50 episodes, decrements epsilon by 0.01 until it reaches 0.1, and then it
        remains fixed at 0.1
        """
        if self._e != 0.1:
            if self._e_counter % 50 == 0:
                self._e -= round(dec.Decimal(0.01), 2)
            self._e_counter += 1

    def reset_e(self):
        """
        Resets epsilon to 0.1
        """
        self._e = round(dec.Decimal(0.10), 2)

    def choose_action(self, available_inputs: List[State]) -> Tuple[float, Position, Value, Action]:
        """
        Given a list of available inputs (list of (state position, state value) tuples), finds the associated
        q-values for those inputs.  If all of the q-values are 0, selects an action at random uniformly.  Otherwise,
        finds the action with the best q-value and selects that with probability 1-epsilon; else, performs a weighted
        selection.
        :param available_inputs:
        :return: An action captured as a 4-tuple (q-value, state position, state value, action)
        """
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
            sum_probabilities = sum([x[0] for x in options if x[0] >= 0])
            if sum_probabilities <= 0:
                return self._uniform_random_selection(options)
            return self._weighted_selection([x for x in options if x[0] >= 0], sum_probabilities)

    @staticmethod
    def _uniform_random_selection(actions: List[Tuple[float, Position, Value, Action]]) -> Tuple[float, Position, Value, Action]:
        """
        Performs uniform random selection
        :return: a random tuple sample
        """
        probabilities = [1/len(actions) for x in actions]
        action = np.random.choice(list(range(len(actions))), p=probabilities)
        return actions[action]

    @staticmethod
    def _weighted_selection(actions: List[Tuple[float, Position, Value, Action]], total: float) -> Tuple[float, Position, Value, Action]:
        """
        Performs a weighted selection based on q-value.
        :param actions: list of 4-tuples
        :param total: the sum of all positive q-values in the actions list
        :return: a random sample chosen based on q-value weights
        """
        probabilities = [x[0]/total for x in actions]
        best_action = np.random.choice(list(range(len(actions))), p=probabilities)
        return actions[best_action]

    def _retrieve_values_actions_from(self, inputs: List[State]) -> List[Tuple[float, Position, Value, Action]]:
        """
        Given a list of states, looks up the corresponding index in the
        Q-table.  Concatenates all of the results.
        """
        available_actions = []
        for sensor_input in inputs:
            for i in range(len(self.q_table[self.input_to_index[(sensor_input[0].value, sensor_input[1].value)]])):
                available_actions.append((self.q_table[self.input_to_index[(sensor_input[0].value, sensor_input[1].value)]][i], sensor_input[0], sensor_input[1], env.ActionMove(i)))
        return available_actions

    def update(self, state: State, action: Action, reward: int, next_inputs: List[State]):
        """
        Update function
        :param state: Tuple of the form (location, value) -- (env.SensorPosition.North, env.SensorValue.Can), etc.
        :param action: best action
        :param reward: reward associated with the best action
        :param next_inputs: list of states
        """
        state_index = self.input_to_index[(state[0].value, state[1].value)]
        next_q_values = [q[0] for q in self._retrieve_values_actions_from(next_inputs)]
        self.q_table[state_index, action.value] = self.q_table[state_index, action.value] + \
            self._n * (reward + self._y*(max(next_q_values)) - self.q_table[state_index, action.value])


