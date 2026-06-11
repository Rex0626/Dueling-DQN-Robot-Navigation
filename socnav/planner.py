"""路徑規劃：A* 全域規劃 + 社交力區域避讓。

全域層：在膨脹後的佔據網格上跑 A*，得到避開靜態障礙物的路徑。
區域層：以 pure-pursuit 追蹤全域路徑的吸引力，疊加來自行人與障礙物的
        斥力 (social force)，讓機器人和人保持距離。
"""
from __future__ import annotations

import heapq
from typing import List, Optional

import numpy as np

from .world import World


class SocialPlanner:
    def __init__(
        self,
        world: World,
        robot_radius: float,
        resolution: float = 0.1,
        lookahead: float = 0.6,
        ped_influence: float = 2.5,   # 行人斥力作用範圍 (m)
        ped_strength: float = 2.2,    # 行人斥力強度
        obs_influence: float = 0.8,   # 障礙物斥力作用範圍 (m)
        obs_strength: float = 1.5,
        pref_speed: float = 0.9,
    ):
        self.world = world
        self.robot_radius = robot_radius
        self.res = resolution
        self.lookahead = lookahead
        self.ped_influence = ped_influence
        self.ped_strength = ped_strength
        self.obs_influence = obs_influence
        self.obs_strength = obs_strength
        self.pref_speed = pref_speed

        self.cols = int(round(world.width / resolution))
        self.rows = int(round(world.height / resolution))
        self.grid = self._build_occupancy()
        self.global_path = self._astar(world.start, world.goal)
        self._wp_idx = 0

    # ---------- 佔據網格 ----------
    def _build_occupancy(self) -> np.ndarray:
        grid = np.zeros((self.rows, self.cols), dtype=bool)
        inflate = self.robot_radius + 0.05
        for r in range(self.rows):
            for c in range(self.cols):
                p = self._cell_to_world(r, c)
                if self.world.obstacle_hit(p, inflate):
                    grid[r, c] = True
        return grid

    def _cell_to_world(self, r: int, c: int) -> np.ndarray:
        return np.array([(c + 0.5) * self.res, (r + 0.5) * self.res])

    def _world_to_cell(self, p: np.ndarray):
        c = int(np.clip(p[0] / self.res, 0, self.cols - 1))
        r = int(np.clip(p[1] / self.res, 0, self.rows - 1))
        return r, c

    # ---------- A* ----------
    def _astar(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        sr, sc = self._world_to_cell(start)
        gr, gc = self._world_to_cell(goal)
        if self.grid[sr, sc]:
            sr, sc = self._nearest_free(sr, sc)
        if self.grid[gr, gc]:
            gr, gc = self._nearest_free(gr, gc)

        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]

        def h(r, c):
            return np.hypot(r - gr, c - gc)

        open_heap = [(h(sr, sc), 0.0, (sr, sc))]
        came: dict = {}
        gscore = {(sr, sc): 0.0}
        closed = set()

        while open_heap:
            _, g, (r, c) = heapq.heappop(open_heap)
            if (r, c) in closed:
                continue
            closed.add((r, c))
            if (r, c) == (gr, gc):
                break
            for dr, dc in nbrs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr, nc]:
                    continue
                if dr != 0 and dc != 0:           # 禁止穿牆角
                    if self.grid[r, nc] or self.grid[nr, c]:
                        continue
                step = np.hypot(dr, dc)
                ng = g + step
                if ng < gscore.get((nr, nc), 1e18):
                    gscore[(nr, nc)] = ng
                    came[(nr, nc)] = (r, c)
                    heapq.heappush(open_heap, (ng + h(nr, nc), ng, (nr, nc)))

        # 回溯
        path_cells = []
        cur = (gr, gc)
        if cur not in came and cur != (sr, sc):
            # 無解，退回直線
            return [np.asarray(start, float), np.asarray(goal, float)]
        while cur != (sr, sc):
            path_cells.append(cur)
            cur = came[cur]
        path_cells.append((sr, sc))
        path_cells.reverse()
        path = [self._cell_to_world(r, c) for r, c in path_cells]
        path[0] = np.asarray(start, float)
        path[-1] = np.asarray(goal, float)
        return self._simplify(path)

    def _nearest_free(self, r, c):
        for radius in range(1, max(self.rows, self.cols)):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and not self.grid[nr, nc]:
                        return nr, nc
        return r, c

    def _simplify(self, path: List[np.ndarray], keep: int = 3) -> List[np.ndarray]:
        """每隔 keep 個點取樣，保留首尾，減少抖動。"""
        if len(path) <= 2:
            return path
        out = [path[0]]
        out.extend(path[1:-1:keep])
        out.append(path[-1])
        return out

    # ---------- 區域避讓 ----------
    def _lookahead_point(self, pos: np.ndarray) -> np.ndarray:
        path = self.global_path
        # 推進 waypoint 索引
        while (self._wp_idx < len(path) - 1
               and np.linalg.norm(path[self._wp_idx] - pos) < self.lookahead):
            self._wp_idx += 1
        return path[self._wp_idx]

    def compute_velocity(self, robot) -> np.ndarray:
        pos = robot.pos
        target = self._lookahead_point(pos)

        # 吸引力：朝向 lookahead 點
        to_t = target - pos
        n = np.linalg.norm(to_t)
        attract = (to_t / n) * self.pref_speed if n > 1e-6 else np.zeros(2)

        force = attract.copy()

        # 行人斥力 (social force)
        for ped in self.world.pedestrians:
            diff = pos - ped.pos
            d = np.linalg.norm(diff)
            clearance = d - self.robot_radius - ped.radius
            if clearance < self.ped_influence and d > 1e-6:
                direction = diff / d
                mag = self.ped_strength * np.exp(-max(clearance, 0.0) / 0.5)
                # 預測行人前進方向，避免正面對撞
                force = force + direction * mag

        # 障礙物斥力 (輔助，全域路徑已大致避開)
        d_obs = self.world.nearest_obstacle_distance(pos) - self.robot_radius
        if d_obs < self.obs_influence:
            grad = self._obstacle_gradient(pos)
            mag = self.obs_strength * (1.0 - max(d_obs, 0.0) / self.obs_influence)
            force = force + grad * mag

        # 限速
        speed = np.linalg.norm(force)
        if speed > robot.max_speed:
            force = force / speed * robot.max_speed
        return force

    def _obstacle_gradient(self, pos: np.ndarray) -> np.ndarray:
        """以有限差分估計「遠離障礙物」的方向。"""
        eps = 0.05
        d0 = self.world.nearest_obstacle_distance(pos)
        dx = self.world.nearest_obstacle_distance(pos + [eps, 0]) - d0
        dy = self.world.nearest_obstacle_distance(pos + [0, eps]) - d0
        g = np.array([dx, dy])
        n = np.linalg.norm(g)
        return g / n if n > 1e-6 else np.zeros(2)
