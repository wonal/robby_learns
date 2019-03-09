import unittest
from src.grid import Grid


class GridTests(unittest.TestCase):

    def setUp(self):
        self.grid = Grid(10, 10)

    def test_in_bound_edge(self):
        self.assertEqual(self.grid.in_bounds(0,0), True)

    def test_in_bound_edge2(self):
        self.assertEqual(self.grid.in_bounds(9,9), True)

    def test_out_of_bounds(self):
        self.assertEqual(self.grid.in_bounds(10,9), False)

    def test_negative_bounds(self):
        self.assertEqual(self.grid.in_bounds(0,-1), False)


if __name__ == '__main__':
    unittest.main()
