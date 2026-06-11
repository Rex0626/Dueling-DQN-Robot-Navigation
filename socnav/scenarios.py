"""預設情境：不同的場地配置、障礙物與行人流。"""
from __future__ import annotations

import numpy as np

from .world import World, Obstacle, Pedestrian


def hallway() -> World:
    """走廊：兩名行人迎面與交叉而來。"""
    return World(
        width=12, height=8,
        start=[1.0, 4.0], goal=[11.0, 4.0],
        obstacles=[
            Obstacle(0, 0, 12, 0.3),      # 下牆
            Obstacle(0, 7.7, 12, 0.3),    # 上牆
            Obstacle(5.5, 0.3, 0.6, 2.2), # 下方柱
            Obstacle(7.5, 5.5, 0.6, 2.2), # 上方柱
        ],
        pedestrians=[
            Pedestrian([10, 4.5], [[2, 4.5], [10, 4.5]], speed=0.9),
            Pedestrian([6, 1.0], [[6, 7.0], [6, 1.0]], speed=0.7),
        ],
    )


def crossing() -> World:
    """開放空間：多名行人交織通過，需保持社交距離。"""
    return World(
        width=10, height=10,
        start=[1.0, 1.0], goal=[9.0, 9.0],
        obstacles=[
            Obstacle(4.5, 4.5, 1.0, 1.0),   # 中央方塊
        ],
        pedestrians=[
            Pedestrian([8, 2], [[2, 8], [8, 2]], speed=0.8),
            Pedestrian([2, 6], [[8, 6], [2, 6]], speed=0.7),
            Pedestrian([6, 9], [[6, 1], [6, 9]], speed=0.6),
        ],
    )


def cluttered() -> World:
    """雜亂場地：多個障礙物 + 一名行人，考驗避障與避人並重。"""
    return World(
        width=12, height=10,
        start=[1.0, 1.0], goal=[11.0, 9.0],
        obstacles=[
            Obstacle(3, 2, 1.5, 1.5),
            Obstacle(6, 5, 2.0, 1.0),
            Obstacle(8.5, 1.5, 1.0, 3.0),
            Obstacle(2.5, 6.5, 3.0, 1.0),
        ],
        pedestrians=[
            Pedestrian([9, 8], [[4, 8], [9, 8], [9, 4], [4, 4]], speed=0.8),
        ],
    )


SCENARIOS = {
    "hallway": hallway,
    "crossing": crossing,
    "cluttered": cluttered,
}
