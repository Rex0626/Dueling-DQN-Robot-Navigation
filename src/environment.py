import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class RobotNavigationEnvGUI(gym.Env):

    def __init__(self, render_mode=True):
        super(RobotNavigationEnvGUI, self).__init__()

        # ==================================================
        # Observation Space (優化：由 12 維降至 10 維，移除絕對座標 X, Y)
        # 1: 機器人朝向角度 (theta)
        # 2: 與目標的相對距離
        # 3: 與目標的相對角度誤差
        # 4-10: 7根雷達的測距值
        # ==================================================
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(10,),
            dtype=np.float32
        )

        # ==================================================
        # Action Space
        # 0 = forward
        # 1 = turn left
        # 2 = turn right
        # 3 = brake
        # ==================================================
        self.action_space = spaces.Discrete(4)

        # ==================================================
        # Map Settings
        # ==================================================
        self.map_size = 400
        self.linear_vel = 8.0
        self.angular_vel = np.deg2rad(15)

        # ==================================================
        # Robot
        # ==================================================
        self.robot_pos = np.array([50.0, 350.0])
        self.robot_theta = 0.0

        # ==================================================
        # Target
        # ==================================================
        self.target_pos = np.array([350.0, 50.0])

        # ==================================================
        # Obstacles
        # ==================================================
        self.obstacles = [
            {"pos": np.array([120.0, 150.0]), "radius": 25},
            {"pos": np.array([220.0, 220.0]), "radius": 35},
            {"pos": np.array([320.0, 120.0]), "radius": 25},
            {"pos": np.array([180.0, 80.0]),  "radius": 20},
        ]

        # ==================================================
        # Episode Settings
        # ==================================================
        self.max_steps = 300
        self.current_step = 0

        # ==================================================
        # Render
        # ==================================================
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.map_size, self.map_size))
            pygame.display.set_caption("Level4 Navigation - Optimized")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 隨機起始位置
        while True:
            candidate_pos = np.array([
                random.uniform(30, 370),
                random.uniform(30, 370)
            ])
            safe = True
            for obs in self.obstacles:
                dist = np.linalg.norm(candidate_pos - obs["pos"])
                if dist < (obs["radius"] + 25):
                    safe = False
                    break
            if safe:
                self.robot_pos = candidate_pos
                break

        self.robot_theta = random.uniform(0, 2*np.pi)

        # 隨機目標位置
        while True:
            candidate_target = np.array([
                random.uniform(30, 370),
                random.uniform(30, 200)
            ])
            dist_to_robot = np.linalg.norm(candidate_target - self.robot_pos)
            safe = True
            for obs in self.obstacles:
                dist = np.linalg.norm(candidate_target - obs["pos"])
                if dist < (obs["radius"] + 25):
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
        self.angle_error = np.arctan2(
            np.sin(self.angle_error),
            np.cos(self.angle_error)
        )

        # 優化：移除絕對座標 robot_pos[0] 與 robot_pos[1]，只回傳相對資訊與雷達
        return np.array([
            self.robot_theta / (2*np.pi),
            dist_to_target / 400.0,
            self.angle_error / np.pi,
            *lidar_values
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        prev_dist = np.linalg.norm(self.robot_pos - self.target_pos)

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
        obs = self._get_obs()

        dist_to_target = np.linalg.norm(self.robot_pos - self.target_pos)
        terminated = False
        truncated = (self.current_step >= self.max_steps)

        reward = -0.1
        
        if action == 0:
            reward += (prev_dist - dist_to_target) * 2.0
            heading_reward = np.cos(self.angle_error)
            if heading_reward > 0: 
                reward += heading_reward * 0.5
        elif action in [1, 2]:
            reward -= 0.25 
        elif action == 3:
            reward -= 2.0  
            
        reward -= abs(self.angle_error) * 0.1

        if dist_to_target < 15:
            reward += 150
            terminated = True
            print("✨ 成功抵達目標")

        for obs_item in self.obstacles:
            dist_to_obs = np.linalg.norm(self.robot_pos - obs_item["pos"])
            if dist_to_obs < (obs_item["radius"] + 8):
                reward -= 80
                terminated = True
                print("💥 撞上障礙物")
                break

        if (self.robot_pos[0] <= 8 or self.robot_pos[0] >= self.map_size - 8 or
            self.robot_pos[1] <= 8 or self.robot_pos[1] >= self.map_size - 8):
            reward -= 120
            terminated = True
            print("🧱 撞牆")

        if self.render_mode:
            self.render()

        return obs, reward, terminated, truncated, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((255,255,255))

        for obs in self.obstacles:
            pygame.draw.circle(self.screen, (255,0,0), obs["pos"].astype(int), obs["radius"])

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