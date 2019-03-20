from typing import Tuple, List, Dict
import src.environment as env

Coordinate = Tuple[int, int]
Position = env.SensorPosition
Value = env.SensorValue
Action = env.ActionMove
State = Tuple[Position, Value]
