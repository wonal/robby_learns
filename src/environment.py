from enum import Enum


class SensorPosition(Enum):
    Current = 0
    North = 1
    South = 2
    East = 3
    West = 4


class SensorValue(Enum):
    Empty = 0
    Can = 1
    Wall = 2


class ActionMove(Enum):
    North = 0
    South = 1
    East = 2
    West = 3
    Pick_Up = 4


INPUT = [SensorPosition.Current, SensorPosition.North, SensorPosition.South, SensorPosition.East, SensorPosition.West]
VALUES = [SensorValue.Empty, SensorValue.Can, SensorValue.Wall]
ACTIONS = [ActionMove.North, ActionMove.South, ActionMove.East, ActionMove.West, ActionMove.Pick_Up]

ETA = 0.2
GAMMA = 0.9
EPSILON = 1.0
GRID_BOUND = 10
