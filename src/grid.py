import numpy as np
from src.types import *


class Grid:

    def __init__(self, row_sz: int, col_sz: int):
        self.grid = self._initialize_grid(row_sz, col_sz)
        self._row_bound = row_sz
        self._col_bound = col_sz
        self._visited = {}

    @staticmethod
    def _initialize_grid(rows: int, cols: int) -> [[int]]:
        g = np.zeros((rows, cols), dtype=int)
        for i in range(len(g)):
            for j in range(len(g[0])):
                if np.random.randint(0, 2) == 1:
                    g[i][j] = 1
        return g

    def in_bounds(self, row: int, col: int) -> bool:
        """
        Determines if a position is in the bounds of the grid
        :return: True if in bounds; False otherwise
        """
        return 0 <= row < self._row_bound and 0 <= col < self._col_bound

    def retrieve_sensor_inputs(self, row: int, col: int) -> List[State]:
        """
        For a position, retrieves the five inputs corresponding to the five positions (current, north, south,
        east, and west)
        :return: List of tuples (position, value) such as (north, empty)
        """
        values = [(row, col), (row-1, col), (row+1, col), (row, col+1), (row, col-1)]
        options = []
        for i in range(5):
            if not self.in_bounds(values[i][0], values[i][1]):
                options.append((env.SensorPosition(i), env.SensorValue.Wall))
            else:
                if self.grid[values[i][0], values[i][1]] == 0:
                    options.append((env.SensorPosition(i), env.SensorValue.Empty))
                else:
                    options.append((env.SensorPosition(i), env.SensorValue.Can))
        return options

    def perform_action(self, action: List[Action], position: Coordinate) -> Tuple[Coordinate, int]:
        """
        Depending on the action for a given position, makes a move and evaluates the appropriate reward or penalty.
        :return: An updated position if appropriate and the associated reward or penalty
        """
        if action == env.ActionMove.North:
            return self._evaluate_position(position[0] - 1, position[1], position)
        elif action == env.ActionMove.South:
            return self._evaluate_position(position[0] + 1, position[1], position)
        elif action == env.ActionMove.West:
            return self._evaluate_position(position[0], position[1] - 1, position)
        elif action == env.ActionMove.East:
            return self._evaluate_position(position[0], position[1] + 1, position)
        else:
            if self.grid[position[0], position[1]] == 0:
                return position, -1
            else:
                self.grid[position[0], position[1]] = 0
                return position, 10

    def _evaluate_position(self, x: int, y: int, original_position: Coordinate) -> Tuple[Coordinate, int]:
        """
        Determines whether a penalty should be given for crashing into a wall.  If not, calls the function
        to determine whether a penalty should be given for visiting a repeated state.
        :param x: row
        :param y: column
        :param original_position: the original position before the move as a tuple (x,y)
        :return: the updated position if appropriate, and reward/penalty
        """
        if not self.in_bounds(x, y):
            return original_position, -5
        return self._assess_visited((x, y))

    def _assess_visited(self, p: Coordinate) -> Tuple[Coordinate, int]:
        """
        Checks whether the position has been visited already and if so, returns a penalty of -1.
        Else, adds the position to the visited dictionary
        :param p: position as a tuple (x,y)
        :return: position, reward/penalty
        """
        if p in self._visited:
            if self.grid[p[0], p[1]] == 0:
                return p, -1
        self._visited[p] = 1
        return p, 0

