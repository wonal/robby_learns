import numpy as np
from enum import Enum


class Value(Enum):
    Empty = 1
    Can = 2
    Wall = 3


class Action(Enum):
    Move_North = 1
    Move_South = 2
    Move_East = 3
    Move_West = 4
    Pick_Up = 5


class Problem:

    def __init__(self, row_sz, col_sz, agent_actions, policy):
        self.grid = Grid(row_sz, col_sz)
        self.agent = Agent(self.grid, np.random.randint(0,row_sz), np.random.randint(0,col_sz), agent_actions, policy)


class Grid:

    def __init__(self, row_sz, col_sz):
        self.grid = self._initialize_grid(row_sz, col_sz)
        self._row_bound = row_sz
        self._col_bound = col_sz

    @staticmethod
    def _initialize_grid(rows, cols):
        g = np.zeros((rows, cols))
        for row in g:
            for cell in row:
                if np.random.randint(0,1) == 1:
                    g[row,cell] = 1
        return g

    def in_bounds(self, row, col):
        return 0 <= row < self._row_bound and 0 <= col < self._col_bound

    def retrieve_sensor_inputs(self, row, col):
        values = [(row,col),(row-1,col),(row+1,col),(row,col-1),(row,col+1)]
        for r,c in values:
            if not self.in_bounds(r,c):
                values.append((r,c,-1))
            else:
                values.append((r,c,self.grid[r,c]))

    def take_action(self, row, col, clean):
        if clean:
            self.grid[row,col] = 0


class Agent:

    def __init__(self, grid, starting_row, starting_col, actions, policy):
        self.grid = grid
        self.position = (starting_row, starting_col)
        self.actions = actions
        self.policy = policy
        self._total_reward = 0

