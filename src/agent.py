import numpy as np
import src.environment as env


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
        values = [(row,col),(row+1,col),(row-1,col),(row,col+1),(row,col-1)]
        options = []
        for i in range(5):
            if not self.in_bounds(values[i][0], values[i][1]):
                options.append((env.SensorPosition(i), env.SensorValue.Wall))
            else:
                if self.grid[values[i][0],values[i][1]] == 0:
                    options.append((env.SensorPosition(i), env.SensorValue.Empty))
                else:
                    options.append((env.SensorPosition(i), env.SensorValue.Can))
        return options

    def take_action(self, action, position):
        if action == env.Action.Move_North:
            return position[0]+1, position[1]
        elif action == env.Action.Move_South:
            return position[0]-1, position[1]
        elif action == env.Action.Move_West:
            return position[0], position[1]-1
        elif action == env.Action.Move_East:
            return position[0], position[1]+1
        else:
            self.grid[position[0], position[1]] = 0
            return position


class Agent:

    def __init__(self, grid, starting_row, starting_col, actions, policy):
        self.grid = grid
        self.position = (starting_row, starting_col)
        self.actions = actions
        self.policy = policy
        self._total_reward = 0

