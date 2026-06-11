"""模擬主迴圈：整合世界、規劃器、計分，並提供 pygame 即時畫面。

  - render=True ：開 pygame 視窗即時顯示
  - render=False：無畫面快速跑完，回傳評分 (可選擇存軌跡 PNG)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .world import World
from .planner import SocialPlanner
from .scoring import Scorer, ScoreConfig


class Simulator:
    def __init__(
        self,
        world: World,
        dt: float = 0.1,
        max_time: float = 60.0,
        score_cfg: Optional[ScoreConfig] = None,
    ):
        self.world = world
        self.dt = dt
        self.max_time = max_time
        from .world import Robot
        self.robot = Robot(pos=np.array(world.start, dtype=float))
        self.planner = SocialPlanner(world, self.robot.radius)
        self.scorer = Scorer(cfg=score_cfg or ScoreConfig())
        self.trail = [self.robot.pos.copy()]
        self.reached = False
        self.stuck_steps = 0

    # ---------- 單步推進 ----------
    def step(self) -> bool:
        """推進一個時步，回傳是否仍在進行中。"""
        if self.reached or self.scorer.elapsed >= self.max_time:
            return False

        for ped in self.world.pedestrians:
            ped.step(self.dt)

        vel = self.planner.compute_velocity(self.robot)
        old = self.robot.pos.copy()
        new = old + vel * self.dt

        # 撞牆/障礙物：不穿越，但記為碰撞
        collided = self.world.obstacle_hit(new, self.robot.radius)
        if collided:
            # 嘗試只走切線方向，避免完全卡死
            slide = old + np.array([vel[0], 0]) * self.dt
            if not self.world.obstacle_hit(slide, self.robot.radius):
                new = slide
            else:
                slide = old + np.array([0, vel[1]]) * self.dt
                new = slide if not self.world.obstacle_hit(slide, self.robot.radius) else old

        self.robot.pos = new
        self.robot.vel = vel
        if np.linalg.norm(vel) > 1e-3:
            self.robot.theta = np.arctan2(vel[1], vel[0])

        moved = float(np.linalg.norm(new - old))
        self.scorer.step(self.robot, self.world, self.dt, collided, moved)
        self.trail.append(self.robot.pos.copy())

        # 卡住偵測
        self.stuck_steps = self.stuck_steps + 1 if moved < 1e-3 else 0

        if np.linalg.norm(self.robot.pos - self.world.goal) < self.world.goal_radius:
            self.reached = True
            return False
        if self.stuck_steps > int(5.0 / self.dt):   # 卡住 5 秒 → 結束
            return False
        return True

    def run_headless(self) -> dict:
        while self.step():
            pass
        return self.scorer.finalize(self.reached)

    # ---------- pygame 即時畫面 ----------
    def run_render(self, scale: int = 60, fps: int = 30) -> dict:
        import pygame

        pygame.init()
        W = int(self.world.width * scale)
        H = int(self.world.height * scale)
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("SocNav - 社交導航模擬")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 16)

        def to_px(p):  # 世界 → 螢幕 (y 翻轉)
            return int(p[0] * scale), int(H - p[1] * scale)

        COL = dict(
            bg=(245, 245, 248), obs=(90, 95, 105), goal=(60, 200, 120),
            robot=(240, 140, 40), ped=(60, 120, 230), path=(180, 195, 220),
            trail=(245, 175, 90), text=(30, 30, 40),
            intimate=(235, 90, 90), personal=(245, 190, 90),
        )

        running = True
        while running and self.step():
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (
                    ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE
                ):
                    running = False

            screen.fill(COL["bg"])

            # 全域路徑
            if len(self.planner.global_path) > 1:
                pts = [to_px(p) for p in self.planner.global_path]
                pygame.draw.lines(screen, COL["path"], False, pts, 3)

            # 障礙物
            for ob in self.world.obstacles:
                rect = pygame.Rect(
                    int(ob.x * scale), int(H - (ob.y + ob.h) * scale),
                    int(ob.w * scale), int(ob.h * scale),
                )
                pygame.draw.rect(screen, COL["obs"], rect, border_radius=4)

            # 終點
            pygame.draw.circle(screen, COL["goal"], to_px(self.world.goal),
                               int(self.world.goal_radius * scale), 0)

            # 機器人軌跡
            if len(self.trail) > 1:
                pygame.draw.lines(screen, COL["trail"], False,
                                  [to_px(p) for p in self.trail], 2)

            # 行人 + proxemics 圈
            cfg = self.scorer.cfg
            for ped in self.world.pedestrians:
                c = to_px(ped.pos)
                pygame.draw.circle(screen, COL["personal"], c,
                                   int(cfg.personal * scale), 1)
                pygame.draw.circle(screen, COL["intimate"], c,
                                   int(cfg.intimate * scale), 1)
                pygame.draw.circle(screen, COL["ped"], c,
                                   int(ped.radius * scale), 0)

            # 機器人
            rc = to_px(self.robot.pos)
            pygame.draw.circle(screen, COL["robot"], rc,
                               int(self.robot.radius * scale), 0)
            head = self.robot.pos + 0.35 * np.array(
                [np.cos(self.robot.theta), np.sin(self.robot.theta)])
            pygame.draw.line(screen, (255, 255, 255), rc, to_px(head), 3)

            # HUD
            s = self.scorer
            d = s.min_ped_dist if s.min_ped_dist < 1e8 else float("nan")
            lines = [
                f"t={s.elapsed:5.1f}s  path={s.path_length:5.1f}m",
                f"min_ped_dist={d:4.2f}m  collisions={s.collision_count}",
                f"social_cost={s.social_cost:5.1f}  collision_cost={s.collision_cost:5.1f}",
            ]
            for i, ln in enumerate(lines):
                screen.blit(font.render(ln, True, COL["text"]), (8, 6 + i * 18))

            pygame.display.flip()
            clock.tick(fps)

        # 結束畫面停留
        result = self.scorer.finalize(self.reached)
        if running:
            self._draw_end(screen, font, result, W)
            pygame.display.flip()
            self._wait_close(pygame)
        pygame.quit()
        return result

    def _draw_end(self, screen, font, result, W):
        import pygame
        big = pygame.font.SysFont("consolas", 22, bold=True)
        title = "ARRIVED" if result["reached_goal"] else "FAILED"
        col = (60, 170, 100) if result["reached_goal"] else (210, 70, 70)
        box = pygame.Surface((W, 70)); box.set_alpha(230); box.fill((20, 22, 30))
        screen.blit(box, (0, 0))
        screen.blit(big.render(f"{title}   SCORE = {result['score']}", True, col), (10, 8))
        screen.blit(font.render(
            f"time={result['time_s']}s  path={result['path_length_m']}m  "
            f"min_ped={result['min_ped_dist_m']}m  collisions={result['obstacle_collisions']}",
            True, (230, 230, 235)), (10, 40))

    def _wait_close(self, pygame):
        waiting = True
        while waiting:
            for ev in pygame.event.get():
                if ev.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    waiting = False
            pygame.time.wait(30)

    # ---------- matplotlib 軌跡圖 ----------
    def save_plot(self, path: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(0, self.world.width); ax.set_ylim(0, self.world.height)
        ax.set_aspect("equal"); ax.set_title("Social Navigation Trajectory")

        for ob in self.world.obstacles:
            ax.add_patch(Rectangle((ob.x, ob.y), ob.w, ob.h, color="#5a5f69"))
        gp = np.array(self.planner.global_path)
        ax.plot(gp[:, 0], gp[:, 1], "--", color="#b4c3dc", label="global path")
        tr = np.array(self.trail)
        ax.plot(tr[:, 0], tr[:, 1], "-", color="#f5af5a", lw=2, label="robot")
        for ped in self.world.pedestrians:
            ax.add_patch(Circle(ped.pos, self.scorer.cfg.personal, fill=False,
                                 color="#f5be5a", ls=":"))
            ax.add_patch(Circle(ped.pos, ped.radius, color="#3c78e6"))
        ax.add_patch(Circle(self.world.goal, self.world.goal_radius, color="#3cc878"))
        ax.plot(*self.world.start, "ks", ms=8, label="start")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout(); fig.savefig(path, dpi=110)
        plt.close(fig)
