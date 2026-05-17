# 基於 Dueling-DQN 於動態障礙環境下之機器人避障策略與異常響應機制研究

本專案實作了一個基於深度強化學習（Deep Reinforcement Learning, DRL）的 2D 機器人導航與避障系統。透過將 **Dueling-DQN 演算法** 與傳統工程的 **硬性安全防護層（Fail-safe Safety Layer）** 結合，使機器人不僅能透過自主學習學會最優導航路徑，更能在面對極端危險時觸發異常響應，確保系統的硬性安全。

---

## 📌 系統架構與運作原理

本專案遵循標準的強化學習架構（Agent-Environment Interaction）：

1. **智慧體 (Agent)**：由 Dueling-DQN 神經網路構成，負責觀察環境狀態並決定移動動作。
2. **環境 (Environment)**：基於 Gymnasium 標準介面封裝的 2D 導航地圖，內置 Lidar 雷達模擬、碰撞偵測與視覺化 GUI。
3. **異常響應 (Anomaly Response)**：獨立於 AI 大腦之外的安全防護層。當雷達偵測距離小於安全臨界值時，強制中斷 DQN 輸出，執行緊急煞車，結合了 AI 的靈活性與傳統控制的確定性。

---

## 📂 檔案結構說明

專案採用模組化設計，各檔案職責分工明確，方便後續擴充與進行論文對比實驗：

### 核心程式碼檔案：
* **`environment.py` (環境與畫面渲染)**
    * 負責定義機器人的物理運動模型（前進、左轉、右轉、煞車）。
    * 模擬 2D 射線雷達（Lidar）數據與目標相對距離。
    * 使用 Python 內建的 Tkinter 庫實作 100% 輕量不崩潰的 2D 視覺化實時訓練畫面。
* **`model.py` (Dueling-DQN 神經網路)**
    * 使用 PyTorch 實作類神經網路。
    * 核心技術：將 Q 值拆解為「狀態價值流 V(s)」與「動作優勢流 A(s, a)」，提升機器人在空曠處與危險邊緣時的決策效率。
* **`agent.py` (智慧體核心與記憶庫)**
    * 實作經驗回放池（Replay Buffer）以打破數據相關性。
    * 實作 $\epsilon$-greedy 探索策略與模型權重儲存/載入介面。
    * 內置 **異常響應機制 (Safety Layer)** 門檻觸發演算法。
* **`main.py` (專案主程式)**
    * 調度環境與智慧體，配置超參數並啟動、管理整個訓練流程。
* **`plot_results.py` (數據可視化工具)**
    * 獨立的數據分析腳本，負責將訓練日誌繪製成符合學術論文標準的高解析度圖表。

### 自動生成之數據檔案（訓練後產生）：
* **`robot_model_level1.pth`**：儲存訓練完成的神經網路大腦權重，可用於後續繼承訓練或獨立測試。
* **`training_log.csv`**：詳細記錄每回合的累積獎勵、執行步數、是否碰撞及成功率的科學數據表。

---

## 💻 系統環境需求

本專案設計之初即考量到 Windows 環境的相容性，避開了所有需要編譯 C++ 的複雜套件，僅需純 Python 環境即可執行。

### 必要依賴套件：
* Python 3.8+
* Gymnasium (標準強化學習環境介面)
* NumPy (矩陣運算)
* PyTorch (深度學習核心)
* Matplotlib (學術圖表繪製)
* Tkinter (Python 內建，無需額外安裝)

### 一鍵安裝指令：
請在 VS Code 終端機（Terminal）中輸入以下指令：
```bash
pip install gymnasium numpy torch matplotlib
```