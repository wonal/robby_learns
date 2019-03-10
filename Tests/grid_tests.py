import unittest
import src.environment as env
from src.grid import Grid
# python -m unittest discover -s Tests -p "*_tests.py"


class GridTests(unittest.TestCase):

    def setUp(self):
        self.grid = Grid(10, 10)
        self.start = self.grid.retrieve_sensor_inputs(0,0)
        self.end = self.grid.retrieve_sensor_inputs(9,9)

    def test_in_bound_edge(self):
        self.assertEqual(self.grid.in_bounds(0,0), True)

    def test_in_bound_edge2(self):
        self.assertEqual(self.grid.in_bounds(9,9), True)

    def test_out_of_bounds(self):
        self.assertEqual(self.grid.in_bounds(10,9), False)

    def test_negative_bounds(self):
        self.assertEqual(self.grid.in_bounds(0,-1), False)

    def test_num_inputs_received(self):
        inputs = [i[0] for i in self.start]
        self.assertEqual(inputs, env.INPUT)

    def test_north_wall(self):
        self.assertEqual(self.start[1][1], env.SensorValue.Wall)

    def test_west_wall(self):
        self.assertEqual(self.start[4][1], env.SensorValue.Wall)

    def test_south_wall(self):
        self.assertEqual(self.end[2][1], env.SensorValue.Wall)

    def test_east_wall(self):
        self.assertEqual(self.end[3][1], env.SensorValue.Wall)

    def test_invalid_north_move(self):
        self.assertEqual(self.grid.perform_action(env.Action.Move_North, (0,0)),((0,0),-5))

    def test_invalid_south(self):
        self.assertEqual(self.grid.perform_action(env.Action.Move_South, (9,9)),((9,9),-5))

    def test_invalid_east(self):
        self.assertEqual(self.grid.perform_action(env.Action.Move_East, (9,9)),((9,9),-5))

    def test_invalid_west(self):
        self.assertEqual(self.grid.perform_action(env.Action.Move_West, (0,0)),((0,0),-5))

    def test_incorrect_pickup(self):
        can = True
        row = 0
        col = 0
        while can and row < 10:
            if col == 10:
                row += 1
                col = 0
            if self.grid.grid[row, col] == 0:
                can = False
            else:
                col += 1
        self.assertEqual(self.grid.perform_action(env.Action.Pick_Up, (row, col)), ((row, col), -1))

    def test_can_pickup(self):
        empty = True
        row = 0
        col = 0
        while empty and row < 10:
            if col == 10:
                row += 1
                col = 0
            if self.grid.grid[row,col] == 1:
                empty = False
            else:
                col += 1
        self.grid.perform_action(env.Action.Pick_Up, (row, col))
        self.assertTrue(self.grid.grid[row,col] == 0)

    def test_valid_move(self):
        self.grid.grid[0,1] = 0
        self.assertEqual(self.grid.perform_action(env.Action.Move_East, (0,0)), ((0,1),0))

    def test_move_to_same_position(self):
        self.grid.perform_action(env.Action.Move_East, (0,0))
        self.grid.grid[0,1] = 0
        self.assertEqual(self.grid.perform_action(env.Action.Move_East, (0,0)), ((0,1),-1))


if __name__ == '__main__':
    unittest.main()
