import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(log_filename='training_log.csv'):
    """
    讀取訓練日誌 CSV 檔案，並繪製符合學術論文標準的收斂曲線與效率圖表。
    """
    episodes = []
    rewards = []
    steps = []
    collisions = []
    successes = []

    # 1. 讀取 CSV 數據
    try:
        with open(log_filename, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row['episode']))
                rewards.append(float(row['reward']))
                steps.append(int(row['steps']))
                collisions.append(int(row['collision']))
                successes.append(int(row['success']))
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到數據檔案 '{log_filename}'，請確認是否已執行 main.py 並完成訓練。")
        return

    # 2. 計算滑動平均 (Moving Average)
    # 強化學習的原始獎勵訊號通常高度震盪，學術論文標準作法是使用滑動平均來呈現平滑的趨勢趨向。
    def moving_average(data, window_size=5):
        if len(data) < window_size:
            return np.array(data)
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window = 5 # 視窗大小設定為 5 回合
    smoothed_rewards = moving_average(rewards, window_size=window)
    smoothed_steps = moving_average(steps, window_size=window)
    
    # 3. 建立畫布配置 (12x5 英吋，包含兩個並排的子圖)
    plt.figure(figsize=(12, 5))

    # --- 左子圖：累積獎勵收斂曲線 ---
    plt.subplot(1, 2, 1)
    # 繪製半透明的原始數據，保留真實細節
    plt.plot(episodes, rewards, alpha=0.25, color='blue', label='Raw Reward')
    # 繪製加粗的平滑曲線，呈現整體學習趨勢
    plt.plot(episodes[window-1:], smoothed_rewards, color='blue', linewidth=2, label=f'Smoothed (MA-{window})')
    plt.title('Dueling-DQN Learning Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Total Reward', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    # --- 右子圖：每回合執行步數變化 (導航效率) ---
    plt.subplot(1, 2, 2)
    # 繪製半透明的原始步數
    plt.plot(episodes, steps, alpha=0.25, color='darkorange', label='Raw Steps')
    # 繪製加粗的平滑步數曲線
    plt.plot(episodes[window-1:], smoothed_steps, color='darkorange', linewidth=2, label=f'Smoothed (MA-{window})')
    plt.title('Navigation Efficiency Analysis', fontsize=12, fontweight='bold')
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Steps to Target / Terminate', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    # 4. 優化版面配置並儲存高解析度圖表
    plt.tight_layout()
    output_fig = 'training_results_plot2.png'
    plt.savefig(output_fig, dpi=300) # 設定 300 DPI 以符合印刷與論文提交標準
    
    print("\n==========================================")
    print(f"🖼️  圖表已成功生成並儲存至：{output_fig}")
    print("==========================================")
    
    # 5. 顯示互動式視窗
    plt.show()

if __name__ == '__main__':
    plot_training_results()