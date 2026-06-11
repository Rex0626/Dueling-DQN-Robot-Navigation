"""計分：依 Hall proxemics 與碰撞情形評估社交導航品質。

Hall 人際距離分區 (機器人中心 ↔ 行人中心)：
  - 親密區 intimate   < 0.45 m   → 嚴重侵犯，重扣
  - 個人區 personal   0.45–1.20 m → 侵犯個人空間，依深入程度線性扣
  - 社交區 social     > 1.20 m    → 不扣分
碰撞障礙物：每一時步重扣，並計入碰撞次數。
抵達終點：加分；路徑越短、時間越短，分數越高。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ScoreConfig:
    intimate: float = 0.45        # m
    personal: float = 1.20        # m
    w_intimate: float = 12.0      # 親密區每秒扣分
    w_personal: float = 4.0       # 個人區 (邊界) 每秒扣分，向內線性增強
    w_collision: float = 25.0     # 撞障礙物每秒扣分
    goal_bonus: float = 100.0     # 抵達終點加分
    fail_penalty: float = 60.0    # 未抵達終點 (逾時/卡住) 扣分
    w_time: float = 1.0           # 每秒時間成本
    w_path: float = 2.0           # 每公尺路徑成本


@dataclass
class Scorer:
    cfg: ScoreConfig = field(default_factory=ScoreConfig)

    # 累積量
    social_cost: float = 0.0
    collision_cost: float = 0.0
    path_length: float = 0.0
    elapsed: float = 0.0
    min_ped_dist: float = 1e9
    intimate_time: float = 0.0
    personal_time: float = 0.0
    collision_count: int = 0
    _was_colliding: bool = False
    reached: bool = False

    def step(self, robot, world, dt: float, collided: bool, moved: float) -> None:
        self.elapsed += dt
        self.path_length += moved

        # 與最近行人的中心距離
        if world.pedestrians:
            d = min(float(np.linalg.norm(robot.pos - p.pos)) for p in world.pedestrians)
            self.min_ped_dist = min(self.min_ped_dist, d)
            if d < self.cfg.intimate:
                self.social_cost += self.cfg.w_intimate * dt
                self.intimate_time += dt
            elif d < self.cfg.personal:
                # 越深入個人空間，扣越多 (邊界 0 → 親密邊界 1)
                depth = (self.cfg.personal - d) / (self.cfg.personal - self.cfg.intimate)
                self.social_cost += self.cfg.w_personal * depth * dt
                self.personal_time += dt

        if collided:
            self.collision_cost += self.cfg.w_collision * dt
            if not self._was_colliding:
                self.collision_count += 1
        self._was_colliding = collided

    def finalize(self, reached: bool) -> dict:
        self.reached = reached
        cfg = self.cfg
        score = 0.0
        score += cfg.goal_bonus if reached else -cfg.fail_penalty
        score -= self.social_cost
        score -= self.collision_cost
        score -= cfg.w_time * self.elapsed
        score -= cfg.w_path * self.path_length

        return {
            "score": round(score, 2),
            "reached_goal": reached,
            "time_s": round(self.elapsed, 2),
            "path_length_m": round(self.path_length, 2),
            "min_ped_dist_m": round(self.min_ped_dist, 3) if self.min_ped_dist < 1e8 else None,
            "intimate_time_s": round(self.intimate_time, 2),
            "personal_time_s": round(self.personal_time, 2),
            "social_cost": round(self.social_cost, 2),
            "obstacle_collisions": self.collision_count,
            "collision_cost": round(self.collision_cost, 2),
        }
