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


class Action(Enum):
    Move_North = 0
    Move_South = 1
    Move_East = 2
    Move_West = 3
    Pick_Up = 4


INPUT = [SensorPosition.Current, SensorPosition.North, SensorPosition.South, SensorPosition.East, SensorPosition.West]
VALUES = [SensorValue.Empty, SensorValue.Can, SensorValue.Wall]
ACTIONS = [Action.Move_North, Action.Move_South, Action.Move_East, Action.Move_West, Action.Pick_Up]

ETA = 0.2
GAMMA = 0.9
EPSILON = 1.0
GRID_BOUND = 10