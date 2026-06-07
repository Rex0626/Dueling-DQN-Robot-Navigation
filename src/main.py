import argparse
import csv
from environment import RobotNavigationEnvGUI
from agent import DuelingDQNAgent

def main():
    parser = argparse.ArgumentParser(description="Dueling-DQN 機器人訓練參數設定")
    parser.add_argument('--episodes', type=int, default=600, help='設定訓練的總回合數 (預設: 600)')
    parser.add_argument('--lr', type=float, default=1e-3, help='設定神經網路學習率 (預設: 0.001)')
    parser.add_argument('--render', action='store_true', help='是否開啟 pygame GUI')
    args = parser.parse_args()

    # 優化：配合環境，將 state_dim 更改為 10
    env = RobotNavigationEnvGUI(render_mode=args.render)
    agent = DuelingDQNAgent(state_dim=10, action_dim=4, enable_safety_layer=True)
    
    load_model_filename = 'robot_model_level5.pth'
    save_model_filename = 'robot_model_level5.pth'
    has_old_model = agent.load_model(load_model_filename)
    
    if has_old_model:
        # 優化：降低繼承模型後的隨機探索率，防止「災難性遺忘」破壞原有權重
        agent.epsilon = 0.2  
        print("💡 已成功繼承經驗，將 Epsilon 設為 0.2 進行微調探索。") 
    
    episodes = args.episodes
    batch_size = 64
    history_data = []

    log_filename = 'training_log_level5.csv'
    print(f"🚀 開始訓練 Level 5：隨機起終點導航，總計執行 {episodes} 個回合...")
    
    global_steps = 0

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        collision_occurred = 0
        success_occurred = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if terminated and reward < 0:  collision_occurred = 1
            if terminated and reward > 20: success_occurred = 1
                
            agent.memory.push(state, action, reward, next_state, terminated)
            
            # 確保記憶庫有足夠的資料才開始訓練
            if global_steps % 4 == 0 and len(agent.memory) > 1000:
                agent.train(batch_size)
          
            state = next_state
            episode_reward += reward
            step_count += 1
            global_steps += 1
            
            if global_steps % 2000 == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                print(f"🔄 [Global Step {global_steps}] 同步目標網路 (Target Network) 權重")
            
        agent.decay_epsilon() 
            
        print(f"Episode {episode + 1}/{episodes} | 得分: {episode_reward:.2f} | 步數: {step_count} | Epsilon: {agent.epsilon:.2f}")
        
        history_data.append({
            'episode': episode + 1,
            'reward': round(episode_reward, 2),
            'steps': step_count,
            'epsilon': round(agent.epsilon, 4),
            'collision': collision_occurred,
            'success': success_occurred
        })

    print("\n🏁 訓練完成！正在處理數據持久化...")
    agent.save_model(save_model_filename)
    
    with open(log_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'reward', 'steps', 'epsilon', 'collision', 'success'])
        writer.writeheader()
        writer.writerows(history_data)
    print(f"📊 成功將訓練過程數據儲存至：{log_filename}")

if __name__ == "__main__":
    main()