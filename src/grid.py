import numpy as np
import environment as env


class Grid:

    def __init__(self, row_sz, col_sz):
        self.grid = self._initialize_grid(row_sz, col_sz)
        self._row_bound = row_sz
        self._col_bound = col_sz

    @staticmethod
    def _initialize_grid(rows, cols):
        g = np.zeros((rows, cols), dtype=int)
        for i in range(len(g)):
            for j in range(len(g[0])):
                if np.random.randint(0,2) == 1:
                    g[i][j] = 1
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

    def perform_action(self, action, position):
        if action == env.Action.Move_North:
            return self.evaluate_position(position[0]+1, position[1], position)
        elif action == env.Action.Move_South:
            return self.evaluate_position(position[0]-1, position[1], position)
        elif action == env.Action.Move_West:
            return self.evaluate_position(position[0], position[1]-1, position)
        elif action == env.Action.Move_East:
            return self.evaluate_position(position[0], position[1]+1, position)
        else:
            if self.grid[position[0], position[1]] == 0:
                return position, -1
            else:
                return position, 0

    def evaluate_position(self, x, y, original_position):
        if not self.in_bounds(x, y):
            return original_position, -5
        return (x,y), 0


#g = Grid(10,10)
#print("done")