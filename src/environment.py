import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class RobotNavigationEnvGUI(gym.Env):

    def __init__(self, render_mode=True):
        super(RobotNavigationEnvGUI, self).__init__()

        # 觀察空間 (10維度：機器人角度、目標距離、角度誤差、7根雷達)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.map_size = 400
        self.linear_vel = 8.0
        self.angular_vel = np.deg2rad(15)

        self.robot_pos = np.array([50.0, 350.0])
        self.robot_theta = 0.0
        self.target_pos = np.array([350.0, 50.0])

        self.obstacles = []
        self.max_steps = 300
        self.current_step = 0

        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.map_size, self.map_size))
            pygame.display.set_caption("Level 5: Ultimate Dynamic Obstacles (6 Obstacles)")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ==================================================
        # 1. 隨機生成動態障礙物 (總數提升至 6 個)
        # ==================================================
        self.obstacles = []
        obstacle_configs = [
            {"radius": 25, "speed": 2.5},
            {"radius": 35, "speed": 1.5},
            {"radius": 25, "speed": 3.0},
            {"radius": 20, "speed": 2.0},
            # 👇 新增的兩個障礙物
            {"radius": 15, "speed": 3.5}, # 體積小、速度極快的干擾型障礙物
            {"radius": 30, "speed": 1.8}, # 體積偏大、速度中等的壓迫型障礙物
        ]

        for config in obstacle_configs:
            # 確保生成位置不會一開始就卡在牆壁裡
            pad = config["radius"] + 5
            base_x = random.uniform(pad, self.map_size - pad)
            base_y = random.uniform(pad, self.map_size - pad)
            
            self.obstacles.append({
                "pos": np.array([base_x, base_y]),
                "radius": config["radius"],
                "speed": config["speed"],
                "angle": random.uniform(0, 2*np.pi) # 隨機決定一開始的移動方向
            })

        # ==================================================
        # 2. 隨機起始位置 (避開動態障礙物)
        # ==================================================
        while True:
            candidate_pos = np.array([random.uniform(30, 370), random.uniform(30, 370)])
            safe = True
            for obs in self.obstacles:
                if np.linalg.norm(candidate_pos - obs["pos"]) < (obs["radius"] + 40):
                    safe = False
                    break
            if safe:
                self.robot_pos = candidate_pos
                break

        self.robot_theta = random.uniform(0, 2*np.pi)

        # ==================================================
        # 3. 隨機目標位置 (避開動態障礙物與機器人)
        # ==================================================
        while True:
            candidate_target = np.array([random.uniform(30, 370), random.uniform(30, 200)])
            dist_to_robot = np.linalg.norm(candidate_target - self.robot_pos)
            safe = True
            for obs in self.obstacles:
                if np.linalg.norm(candidate_target - obs["pos"]) < (obs["radius"] + 40):
                    safe = False
                    break
            if safe and dist_to_robot > 120:
                self.target_pos = candidate_target
                break

        self.current_step = 0
        return self._get_obs(), {}

    def cast_lidar_ray(self, angle, max_range=120):
        step_size = 4
        for dist in range(0, max_range, step_size):
            test_x = (self.robot_pos[0] + dist*np.cos(angle))
            test_y = (self.robot_pos[1] - dist*np.sin(angle))
            if (test_x <= 0 or test_x >= self.map_size or test_y <= 0 or test_y >= self.map_size):
                return dist
            for obs in self.obstacles:
                d = np.linalg.norm(np.array([test_x, test_y]) - obs["pos"])
                if d <= obs["radius"]:
                    return dist
        return max_range

    def _get_obs(self):
        dist_to_target = np.linalg.norm(self.robot_pos - self.target_pos)

        lidar_degrees = [-90, -60, -30, 0, 30, 60, 90]
        lidar_values = []
        for deg in lidar_degrees:
            angle = self.robot_theta + np.deg2rad(deg)
            dist = self.cast_lidar_ray(angle)
            lidar_values.append(dist / 400.0)

        angle_to_target = np.arctan2(
            self.robot_pos[1] - self.target_pos[1],
            self.target_pos[0] - self.robot_pos[0]
        )

        if angle_to_target < 0:
            angle_to_target += 2*np.pi

        self.angle_error = (angle_to_target - self.robot_theta)
        self.angle_error = np.arctan2(np.sin(self.angle_error), np.cos(self.angle_error))

        return np.array([
            self.robot_theta / (2*np.pi),
            dist_to_target / 400.0,
            self.angle_error / np.pi,
            *lidar_values
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        prev_dist = np.linalg.norm(self.robot_pos - self.target_pos)
        prev_angle_error = abs(self.angle_error)

        # ==================================================
        # 💡 動態障礙物移動與邊界反彈邏輯
        # ==================================================
        for obs in self.obstacles:
            # 依據當前速度與方向移動
            obs["pos"][0] += obs["speed"] * np.cos(obs["angle"])
            obs["pos"][1] += obs["speed"] * np.sin(obs["angle"])
            
            # 邊界反彈處理 (撞牆後反射)
            if obs["pos"][0] - obs["radius"] <= 0 or obs["pos"][0] + obs["radius"] >= self.map_size:
                obs["angle"] = np.pi - obs["angle"] # X軸反彈
            
            if obs["pos"][1] - obs["radius"] <= 0 or obs["pos"][1] + obs["radius"] >= self.map_size:
                obs["angle"] = -obs["angle"]        # Y軸反彈

        # ==================================================
        # 機器人動作更新
        # ==================================================
        if action == 0:
            self.robot_pos[0] += (self.linear_vel * np.cos(self.robot_theta))
            self.robot_pos[1] -= (self.linear_vel * np.sin(self.robot_theta))
        elif action == 1:
            self.robot_theta += self.angular_vel
        elif action == 2:
            self.robot_theta -= self.angular_vel
        elif action == 3:
            pass

        self.robot_theta %= (2*np.pi)
        obs_state = self._get_obs()

        dist_to_target = np.linalg.norm(self.robot_pos - self.target_pos)
        terminated = False
        truncated = (self.current_step >= self.max_steps)

        # 獎勵函數 (Reward Shaping)
        reward = -0.1
        if action == 0:
            reward += (prev_dist - dist_to_target) * 2.0
        elif action == 3:
            reward -= 1.0  

        angle_improvement = prev_angle_error - abs(self.angle_error)
        reward += angle_improvement * 2.0

        if abs(self.angle_error) < 0.2:
            reward += 0.2

        if dist_to_target < 15:
            reward += 150
            terminated = True
            print("✨ 成功抵達目標")

        for obs_item in self.obstacles:
            dist_to_obs = np.linalg.norm(self.robot_pos - obs_item["pos"])
            if dist_to_obs < (obs_item["radius"] + 8):
                reward -= 80
                terminated = True
                print("💥 撞上動態障礙物")
                break

        if (self.robot_pos[0] <= 8 or self.robot_pos[0] >= self.map_size - 8 or
            self.robot_pos[1] <= 8 or self.robot_pos[1] >= self.map_size - 8):
            reward -= 120
            terminated = True
            print("🧱 撞牆")

        if self.render_mode:
            self.render()

        return obs_state, reward, terminated, truncated, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((255,255,255))

        # 繪製動態障礙物 (加入移動方向指示線增加視覺效果)
        for obs in self.obstacles:
            pygame.draw.circle(self.screen, (255,50,50), obs["pos"].astype(int), obs["radius"])
            # 畫出障礙物的移動向量
            end_x = obs["pos"][0] + obs["radius"] * np.cos(obs["angle"])
            end_y = obs["pos"][1] + obs["radius"] * np.sin(obs["angle"])
            pygame.draw.line(self.screen, (150,0,0), obs["pos"].astype(int), (int(end_x), int(end_y)), 3)

        pygame.draw.circle(self.screen, (0,255,0), self.target_pos.astype(int), 10)
        pygame.draw.circle(self.screen, (0,0,255), self.robot_pos.astype(int), 8)

        line_len = 15
        end_x = (self.robot_pos[0] + line_len*np.cos(self.robot_theta))
        end_y = (self.robot_pos[1] - line_len*np.sin(self.robot_theta))
        pygame.draw.line(self.screen, (0,0,0), self.robot_pos.astype(int), (int(end_x), int(end_y)), 2)

        lidar_degrees = [-90, -60, -30, 0, 30, 60, 90]
        lidar_angles = [self.robot_theta + np.deg2rad(deg) for deg in lidar_degrees]
        lidar_dists = [self.cast_lidar_ray(angle) for angle in lidar_angles]

        for angle, dist in zip(lidar_angles,lidar_dists):
            end_x = (self.robot_pos[0] + dist*np.cos(angle))
            end_y = (self.robot_pos[1] - dist*np.sin(angle))
            pygame.draw.line(self.screen, (255,165,0), self.robot_pos.astype(int), (int(end_x), int(end_y)), 2)

        pygame.display.flip()
        self.clock.tick(120)

    def close(self):
        if self.render_mode:
            pygame.quit()