import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tkinter as tk
import time
import random

class RobotNavigationEnvGUI(gym.Env):
    def __init__(self, render_mode=True):
        super(RobotNavigationEnvGUI, self).__init__()
        
        # 觀察空間：[R_x, R_y, R_theta(弧度), T_x, T_y, Lidar_左, Lidar_前, Lidar_右] -> 變成 8 維
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32)
        
        # 動作空間：0: 前進, 1: 左轉彎(原地旋轉), 2: 右轉彎(原地旋轉), 3: 煞車/不動
        self.action_space = spaces.Discrete(4)
        
        # 地圖配置
        self.map_size = 400
        
        # 運動學參數
        self.linear_vel = 8.0  # 前進線速度
        self.angular_vel = np.deg2rad(15) # 旋轉角速度 (每步轉15度)
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_theta = 0.0 # 車頭朝向 (弧度)
        
        self.target_pos = np.array([350.0, 50.0])
        self.obstacle_pos = np.array([200.0, 200.0])
        self.obstacle_radius = 30
        
        self.max_steps = 300
        self.current_step = 0
        self.render_mode = render_mode
        
        if self.render_mode:
            self.root = tk.Tk()
            self.root.title("Dueling-DQN真實差速驅動與撞牆懲罰")
            self.canvas = tk.Canvas(self.root, width=self.map_size, height=self.map_size, bg="white")
            self.canvas.pack()
            self.setup_canvas_objects()

    def setup_canvas_objects(self):
        self.canvas.create_oval(self.target_pos[0]-10, self.target_pos[1]-10, self.target_pos[0]+10, self.target_pos[1]+10, fill="green", tags="target")
        self.canvas.create_oval(self.obstacle_pos[0]-self.obstacle_radius, self.obstacle_pos[1]-self.obstacle_radius, self.obstacle_pos[0]+self.obstacle_radius, self.obstacle_pos[1]+self.obstacle_radius, fill="red", tags="obstacle")
        # 機器人外觀(圓形)
        self.robot_marker = self.canvas.create_oval(0, 0, 0, 0, fill="blue")
        # 新增車頭朝向指標線，方便觀察
        self.robot_line = self.canvas.create_line(0, 0, 0, 0, fill="black", width=2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 🚩目標一：隨機起點，避免卡死在同個角落
        self.robot_pos = np.array([
            random.uniform(20, 100),    # X靠左
            random.uniform(300, 380)    # Y靠下
        ])
        # 隨機初始車頭角度
        self.robot_theta = random.uniform(0, 2 * np.pi)
        
        # 2. 🎯【Level 2 核心】：隨機生成目標點 (綠點)
        # 使用 while 迴圈確保綠點不會與中央紅球重疊，且與機器人保持一定距離
        while True:
            candidate_target = np.array([
                random.uniform(20, 380),
                random.uniform(20, 200)  # 靠近地圖上方，強迫其穿越中央障礙物
            ])
            dist_to_obs = np.linalg.norm(candidate_target - self.obstacle_pos)
            dist_to_robot = np.linalg.norm(candidate_target - self.robot_pos)
            
            # 確保與紅球中心距離大於 (紅球半徑 30 + 安全邊際 20)
            if dist_to_obs > 50 and dist_to_robot > 100:
                self.target_pos = candidate_target
                break
                
        # 3. 如果開啟視覺化，實時將畫布上的綠點移到新座標
        if self.render_mode:
            self.canvas.coords("target", self.target_pos[0]-10, self.target_pos[1]-10, 
                                         self.target_pos[0]+10, self.target_pos[1]+10)

        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # 🚩目標二：Lidar偵測加上"車頭朝向角度"
        dist_to_obs = np.linalg.norm(self.robot_pos - self.obstacle_pos)
        
        # 簡化Lidar，只看絕對距離(這部分在複雜環境會升級為局部雷達)
        lidar_front = dist_to_obs
        
        # 歸一化觀測值
        return np.array([
            self.robot_pos[0]/400.0, self.robot_pos[1]/400.0,
            self.robot_theta / (2 * np.pi), # 角度歸一化
            self.target_pos[0]/400.0, self.target_pos[1]/400.0,
            dist_to_obs/400.0, dist_to_obs/400.0, dist_to_obs/400.0
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        # 🚩目標三：真實差速驅動運動學模型
        if action == 0:   # 前進 (沿著目前theta角前進)
            self.robot_pos[0] += self.linear_vel * np.cos(self.robot_theta)
            self.robot_pos[1] -= self.linear_vel * np.sin(self.robot_theta) # Tkinter Y軸向上為減
        elif action == 1: # 原地左轉彎
            self.robot_theta += self.angular_vel
        elif action == 2: # 原地右轉彎
            self.robot_theta -= self.angular_vel
        elif action == 3: # 煞車/不动
            pass
            
        # 規範化theta在 [0, 2pi] 之間
        self.robot_theta = self.robot_theta % (2 * np.pi)
        
        obs = self._get_obs()
        dist_to_target = np.linalg.norm(self.robot_pos - self.target_pos)
        dist_to_obs = np.linalg.norm(self.robot_pos - self.obstacle_pos)
        
        reward = -dist_to_target * 0.01
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # ✨成功抵達
        if dist_to_target < 15.0:
            reward += 100.0
            terminated = True
            print("✨ 成功抵達目標點！")
            
        # 💥碰撞中央紅色障礙物
        if dist_to_obs < (self.obstacle_radius + 8):
            reward -= 50.0
            terminated = True
            print("💥 發生中央碰撞！")
            
        # 🚩目標四：撞牆懲罰層 (Wall Collision Punishment)
        # 刪除 clip，改為只要觸碰邊界就嚴厲扣分並結束
        if self.robot_pos[0] <= 8 or self.robot_pos[0] >= (self.map_size - 8) or \
           self.robot_pos[1] <= 8 or self.robot_pos[1] >= (self.map_size - 8):
            reward -= 100.0 # 嚴厲懲罰，比撞紅球還重
            terminated = True
            print("🧱 撞擊地圖邊界！")
            
        if self.render_mode:
            self.render()
            
        return obs, reward, terminated, truncated, {}

    def render(self):
        r = 8
        # 更新機器人本體圓形位置
        self.canvas.coords(self.robot_marker, self.robot_pos[0]-r, self.robot_pos[1]-r, 
                                              self.robot_pos[0]+r, self.robot_pos[1]+r)
        # 更新車頭朝向指標線，方便觀察
        line_len = 15
        self.canvas.coords(self.robot_line, self.robot_pos[0], self.robot_pos[1],
                                            self.robot_pos[0] + line_len * np.cos(self.robot_theta),
                                            self.robot_pos[1] - line_len * np.sin(self.robot_theta))
        self.root.update_idletasks()
        self.root.update()
        time.sleep(0.01)