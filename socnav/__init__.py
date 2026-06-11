"""輕量社交導航模擬 (SocNavBench 風格)。

機器人從起點導航到終點：
  - 以 A* 做全域路徑規劃，避開靜態障礙物
  - 以社交力 (social force) 做區域避讓，和行人保持距離
  - 依 Hall proxemics 與碰撞情形即時計分
"""

from .world import World, Obstacle, Pedestrian, Robot
from .planner import SocialPlanner
from .scoring import Scorer, ScoreConfig
from .simulator import Simulator

__all__ = [
    "World",
    "Obstacle",
    "Pedestrian",
    "Robot",
    "SocialPlanner",
    "Scorer",
    "ScoreConfig",
    "Simulator",
]
