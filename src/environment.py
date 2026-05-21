import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tkinter as tk
import time
import random

class RobotNavigationEnvGUI(gym.Env):

    def __init__(self, render_mode=True):
        super(RobotNavigationEnvGUI, self).__init__()

        # state:
        # [robot_x, robot_y, theta,
        #  dist_to_target, angle_error,
        #  lidar_left, lidar_front, lidar_right]

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(8,),
            dtype=np.float32
        )

        # 0 forward
        # 1 left
        # 2 right
        # 3 brake
        self.action_space = spaces.Discrete(4)

        self.map_size = 400

        self.linear_vel = 8.0
        self.angular_vel = np.deg2rad(15)

        self.robot_pos = np.array([50.0, 350.0])
        self.robot_theta = 0.0

        self.target_pos = np.array([350.0, 50.0])

        # =========================
        # Level3 多障礙物
        # =========================
        self.obstacles = [
            {"pos": np.array([120.0, 150.0]), "radius": 25},
            {"pos": np.array([220.0, 220.0]), "radius": 35},
            {"pos": np.array([320.0, 120.0]), "radius": 25},
            {"pos": np.array([180.0, 80.0]),  "radius": 20},
        ]

        self.max_steps = 300
        self.current_step = 0

        self.render_mode = render_mode

        if self.render_mode:
            self.root = tk.Tk()
            self.root.title("Level3 Multi-Obstacle Navigation")

            self.canvas = tk.Canvas(
                self.root,
                width=self.map_size,
                height=self.map_size,
                bg="white"
            )
            self.canvas.pack()

            self.setup_canvas_objects()

    # ==================================================
    # GUI
    # ==================================================
    def setup_canvas_objects(self):

        # target
        self.canvas.create_oval(
            self.target_pos[0]-10,
            self.target_pos[1]-10,
            self.target_pos[0]+10,
            self.target_pos[1]+10,
            fill="green",
            tags="target"
        )

        # 多障礙物
        for obs in self.obstacles:

            pos = obs["pos"]
            r = obs["radius"]

            self.canvas.create_oval(
                pos[0]-r,
                pos[1]-r,
                pos[0]+r,
                pos[1]+r,
                fill="red",
                tags="obstacle"
            )

        self.robot_marker = self.canvas.create_oval(
            0, 0, 0, 0,
            fill="blue"
        )

        self.robot_line = self.canvas.create_line(
            0, 0, 0, 0,
            fill="black",
            width=2
        )

    # ==================================================
    # RESET
    # ==================================================
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.robot_pos = np.array([
            random.uniform(20, 80),
            random.uniform(300, 380)
        ])

        self.robot_theta = random.uniform(0, 2*np.pi)

        # =========================
        # 隨機終點
        # =========================
        while True:

            candidate_target = np.array([
                random.uniform(30, 370),
                random.uniform(30, 200)
            ])

            dist_to_robot = np.linalg.norm(
                candidate_target - self.robot_pos
            )

            safe = True

            for obs in self.obstacles:

                dist = np.linalg.norm(
                    candidate_target - obs["pos"]
                )

                if dist < (obs["radius"] + 25):
                    safe = False
                    break

            if safe and dist_to_robot > 120:
                self.target_pos = candidate_target
                break

        if self.render_mode:

            self.canvas.coords(
                "target",
                self.target_pos[0]-10,
                self.target_pos[1]-10,
                self.target_pos[0]+10,
                self.target_pos[1]+10
            )

        self.current_step = 0

        return self._get_obs(), {}

    # ==================================================
    # 最近障礙物 LiDAR
    # ==================================================
    def get_nearest_obstacle_distance(self):

        min_dist = 9999

        for obs in self.obstacles:

            dist = np.linalg.norm(
                self.robot_pos - obs["pos"]
            )

            if dist < min_dist:
                min_dist = dist

        return min_dist

    # ==================================================
    # OBSERVATION
    # ==================================================
    def _get_obs(self):

        dist_to_target = np.linalg.norm(
            self.robot_pos - self.target_pos
        )

        nearest_obs_dist = self.get_nearest_obstacle_distance()

        # 簡化版 LiDAR
        lidar_left = nearest_obs_dist
        lidar_front = nearest_obs_dist
        lidar_right = nearest_obs_dist

        angle_to_target = np.arctan2(
            self.robot_pos[1] - self.target_pos[1],
            self.target_pos[0] - self.robot_pos[0]
        )

        if angle_to_target < 0:
            angle_to_target += 2*np.pi

        self.angle_error = (
            angle_to_target - self.robot_theta
        )

        self.angle_error = np.arctan2(
            np.sin(self.angle_error),
            np.cos(self.angle_error)
        )

        return np.array([
            self.robot_pos[0] / 400.0,
            self.robot_pos[1] / 400.0,
            self.robot_theta / (2*np.pi),

            dist_to_target / 400.0,
            self.angle_error / np.pi,

            lidar_left / 400.0,
            lidar_front / 400.0,
            lidar_right / 400.0

        ], dtype=np.float32)

    # ==================================================
    # STEP
    # ==================================================
    def step(self, action):

        self.current_step += 1

        prev_dist = np.linalg.norm(
            self.robot_pos - self.target_pos
        )

        # =========================
        # 動作
        # =========================
        if action == 0:

            self.robot_pos[0] += (
                self.linear_vel * np.cos(self.robot_theta)
            )

            self.robot_pos[1] -= (
                self.linear_vel * np.sin(self.robot_theta)
            )

        elif action == 1:

            self.robot_theta += self.angular_vel

        elif action == 2:

            self.robot_theta -= self.angular_vel

        elif action == 3:
            pass

        self.robot_theta %= (2*np.pi)

        obs = self._get_obs()

        dist_to_target = np.linalg.norm(
            self.robot_pos - self.target_pos
        )

        terminated = False

        truncated = (
            self.current_step >= self.max_steps
        )

        # ==================================================
        # Reward Shaping
        # ==================================================

        # 基礎時間懲罰
        reward = -0.05

        # 越接近目標越好
        reward += (prev_dist - dist_to_target) * 0.2

        # 朝向目標獎勵
        reward -= abs(self.angle_error) * 0.05

        # ==================================================
        # 成功抵達
        # ==================================================
        if dist_to_target < 15:

            reward += 150

            terminated = True

            print("✨ 成功抵達目標")

        # ==================================================
        # 多障礙物碰撞
        # ==================================================
        for obs_item in self.obstacles:

            dist_to_obs = np.linalg.norm(
                self.robot_pos - obs_item["pos"]
            )

            if dist_to_obs < (obs_item["radius"] + 8):

                reward -= 80

                terminated = True

                print("💥 撞上障礙物")

                break

        # ==================================================
        # 撞牆
        # ==================================================
        if (
            self.robot_pos[0] <= 8 or
            self.robot_pos[0] >= self.map_size - 8 or
            self.robot_pos[1] <= 8 or
            self.robot_pos[1] >= self.map_size - 8
        ):

            reward -= 120

            terminated = True

            print("🧱 撞牆")

        if self.render_mode:
            self.render()

        return obs, reward, terminated, truncated, {}

    # ==================================================
    # Render
    # ==================================================
    def render(self):

        r = 8

        self.canvas.coords(
            self.robot_marker,

            self.robot_pos[0]-r,
            self.robot_pos[1]-r,

            self.robot_pos[0]+r,
            self.robot_pos[1]+r
        )

        line_len = 15

        self.canvas.coords(
            self.robot_line,

            self.robot_pos[0],
            self.robot_pos[1],

            self.robot_pos[0] + line_len*np.cos(self.robot_theta),

            self.robot_pos[1] - line_len*np.sin(self.robot_theta)
        )

        self.root.update_idletasks()
        self.root.update()

        time.sleep(0.01)