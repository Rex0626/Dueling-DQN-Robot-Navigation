import random
import torch
import torch.optim as optim
import numpy as np
from collections import deque
from model import DuelingQNetwork

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done, dtype=np.uint8)
    def __len__(self): return len(self.buffer)

class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim, enable_safety_layer=True):
        self.action_dim = action_dim
        self.enable_safety_layer = enable_safety_layer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("🚀 Using device:", self.device)
        
        self.policy_net = DuelingQNetwork(state_dim,action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim,action_dim).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(50000)
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.1
        self.gamma = 0.99

    def select_action(self, state):
        # 🔍 【異常響應機制 (Safety Layer)】
        if self.enable_safety_layer:
            # 優化：配合環境維度縮減，更新讀取索引
            target_dist = state[1] * 400.0  # 索引 1: 機器人與目標的當前距離
            lidar_front = state[6] * 400.0  # 索引 6: 正前方雷達的距離 (-90度為3, 0度為6, 90度為9)
            
            if lidar_front < 45.0 and lidar_front < target_dist:
                lidar_left = state[3] * 400.0   # -90 度雷達
                lidar_right = state[9] * 400.0  # 90 度雷達
                return 1 if lidar_left > lidar_right else 2
                
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = (torch.FloatTensor(state).unsqueeze(0).to(self.device))
        with torch.no_grad():
            return self.policy_net(state_t).argmax().item()

    def train(self, batch_size=64):
        if len(self.memory) < batch_size: return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        
        with torch.no_grad():
            # 優化：導入 Double DQN (DDQN) 邏輯，分離動作選擇與價值評估
            # 1. 透過 Policy Net 選擇下一步最佳動作
            next_actions = self.policy_net(next_states_t).max(1)[1].unsqueeze(1)
            # 2. 透過 Target Net 評估該動作的價值
            next_q_values = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            # 3. 計算目標 Q 值
            target_q_values = rewards_t + (1 - dones_t) * self.gamma * next_q_values
            
        # 優化：將 MSE Loss 更改為 Huber Loss (SmoothL1Loss) 以抵抗極端獎勵的梯度爆炸
        loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath='robot_model.pth'):
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"💾 成功將模型權重儲存至：{filepath}")

    def load_model(self, filepath='robot_model.pth'):
        import os
        if os.path.exists(filepath):
            self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"📖 成功載入先前的模型權重：{filepath}")
            return True
        else:
            print(f"⚠️ 找不到權重檔案 {filepath}，將從頭開始訓練。")
            return False