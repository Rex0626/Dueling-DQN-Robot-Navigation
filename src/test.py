import argparse
import time
from environment import RobotNavigationEnvGUI
from agent import DuelingDQNAgent

def main():
    parser = argparse.ArgumentParser(description="Level 5 動態環境機器人導航 - 測試與展示模式")
    parser.add_argument('--episodes', type=int, default=10, help='展示的回合數 (預設: 10)')
    parser.add_argument('--fps', type=int, default=60, help='畫面更新率，降低可讓展示變慢以方便解說 (預設: 60)')
    args = parser.parse_args()

    print("🚀 初始化測試環境 (Level 5)...")
    env = RobotNavigationEnvGUI(render_mode=True)
    
    # 初始化 Agent (大腦)
    agent = DuelingDQNAgent(state_dim=10, action_dim=4, enable_safety_layer=True)
    
    # 載入我們訓練好的最強權重
    model_filename = 'robot_model_level5.pth'
    if not agent.load_model(model_filename):
        print("❌ 找不到訓練好的模型權重！請先執行 main.py 進行訓練。")
        return

    # 💡 測試模式的最關鍵設定：關閉探索，火力全開！
    agent.epsilon = 0.0  # 0% 隨機探索，100% 依賴神經網路的最佳決策
    
    episodes = args.episodes
    
    # 統計數據 (報告時可以拿出來講)
    success_count = 0
    collision_count = 0
    timeout_count = 0

    print(f"\n🎬 開始展示！預計執行 {episodes} 回合...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            # 💡 純推論模式：只選動作，不存記憶，不更新權重
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # 控制畫面速度，方便錄影或台上解說
            env.clock.tick(args.fps)
            
            if terminated:
                if reward > 20: 
                    success_count += 1
                    result_text = "✅ 成功抵達"
                else: 
                    collision_count += 1
                    result_text = "💥 發生碰撞"
            elif truncated:
                timeout_count += 1
                result_text = "⏳ 時間耗盡"
                
        print(f"回合 {episode + 1}/{episodes} | 結果: {result_text} | 步數: {step_count} | 總得分: {episode_reward:.2f}")
        time.sleep(1) # 回合結束時暫停 1 秒，讓觀眾看清楚結果

    # 顯示最終統計報表
    print("\n" + "="*40)
    print("📊 展示階段最終統計報告")
    print("="*40)
    print(f"總回合數: {episodes}")
    print(f"成功次數: {success_count} ({(success_count/episodes)*100:.1f}%)")
    print(f"碰撞次數: {collision_count} ({(collision_count/episodes)*100:.1f}%)")
    print(f"超時次數: {timeout_count} ({(timeout_count/episodes)*100:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    main()