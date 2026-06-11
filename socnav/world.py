"""世界模型：場地、靜態障礙物、行人、機器人。

座標單位為公尺 (m)，原點在左下角，x 向右、y 向上。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class Obstacle:
    """軸對齊矩形障礙物 (x, y 為左下角)。"""

    x: float
    y: float
    w: float
    h: float

    def contains(self, p: np.ndarray, margin: float = 0.0) -> bool:
        return (
            self.x - margin <= p[0] <= self.x + self.w + margin
            and self.y - margin <= p[1] <= self.y + self.h + margin
        )

    def distance(self, p: np.ndarray) -> float:
        """點到矩形的最短距離 (在內部回傳 0)。"""
        dx = max(self.x - p[0], 0.0, p[0] - (self.x + self.w))
        dy = max(self.y - p[1], 0.0, p[1] - (self.y + self.h))
        return float(np.hypot(dx, dy))


@dataclass
class Pedestrian:
    """沿著 waypoints 循環移動的行人。"""

    pos: np.ndarray
    waypoints: List[np.ndarray]
    speed: float = 0.8          # m/s
    radius: float = 0.25        # 身體半徑 (m)
    _idx: int = 0

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.waypoints = [np.asarray(w, dtype=float) for w in self.waypoints]
        self.vel = np.zeros(2)

    def step(self, dt: float) -> None:
        if not self.waypoints:
            self.vel = np.zeros(2)
            return
        target = self.waypoints[self._idx]
        to_target = target - self.pos
        dist = np.linalg.norm(to_target)
        if dist < 0.15:                       # 抵達 waypoint，換下一個 (循環)
            self._idx = (self._idx + 1) % len(self.waypoints)
            target = self.waypoints[self._idx]
            to_target = target - self.pos
            dist = np.linalg.norm(to_target)
        if dist > 1e-6:
            direction = to_target / dist
            self.vel = direction * self.speed
            self.pos = self.pos + self.vel * dt


@dataclass
class Robot:
    pos: np.ndarray
    radius: float = 0.30        # m
    max_speed: float = 1.0      # m/s
    theta: float = 0.0          # 朝向 (rad)

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.vel = np.zeros(2)


@dataclass
class World:
    width: float
    height: float
    start: np.ndarray
    goal: np.ndarray
    obstacles: List[Obstacle] = field(default_factory=list)
    pedestrians: List[Pedestrian] = field(default_factory=list)
    goal_radius: float = 0.4    # 進入此半徑視為抵達終點

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float)
        self.goal = np.asarray(self.goal, dtype=float)

    def in_bounds(self, p: np.ndarray) -> bool:
        return 0 <= p[0] <= self.width and 0 <= p[1] <= self.height

    def obstacle_hit(self, p: np.ndarray, radius: float) -> bool:
        """機器人 (中心 p、半徑 radius) 是否撞到任何障礙物或牆。"""
        if not (
            radius <= p[0] <= self.width - radius
            and radius <= p[1] <= self.height - radius
        ):
            return True
        return any(ob.distance(p) < radius for ob in self.obstacles)

    def nearest_obstacle_distance(self, p: np.ndarray) -> float:
        d = min((ob.distance(p) for ob in self.obstacles), default=1e9)
        # 與四面牆的距離
        d = min(d, p[0], p[1], self.width - p[0], self.height - p[1])
        return float(d)
