# SocNav — 輕量社交導航模擬

用 Python 實作的社交導航 (social navigation) 模擬，風格參考 [SocNavBench](https://github.com/CMU-TBD/SocNavBench)，但**不需要 Linux / OpenGL / 大型資料集**，在 Windows + Python 即可直接執行。本模組獨立於本 repo 的 Dueling-DQN 專案，互不影響。

機器人從**起點導航到終點**，過程中：
- **避開靜態障礙物**（撞到會扣分）
- **和行人保持社交距離**（依 Hall proxemics 分區扣分）
- 路徑越短、時間越短分數越高，抵達終點加分

## 安裝需求
```
pip install numpy pygame matplotlib
```

## 執行（在 repo 根目錄）
```bash
python run.py                       # 預設 crossing 情境，開即時畫面
python run.py --scenario hallway    # 指定情境 (hallway / crossing / cluttered)
python run.py --no-render           # 無畫面快速跑，印出評分
python run.py --plot traj.png       # 另存軌跡圖
python run.py --all --no-render     # 跑完所有情境並比較分數
```
即時畫面中：橘色=機器人，藍色=行人（外圈為 proxemics 個人/親密區），
綠色=終點，灰色=障礙物，虛線=全域路徑。按 ESC 關閉。

## 運作原理
| 層級 | 做法 | 檔案 |
|------|------|------|
| 全域規劃 | 在膨脹後的佔據網格上跑 **A***，避開靜態障礙物 | `socnav/planner.py` |
| 區域避讓 | **pure-pursuit 吸引力** + 行人/障礙物 **社交斥力 (social force)** | `socnav/planner.py` |
| 計分 | Hall proxemics 分區 + 碰撞 + 時間/路徑成本 | `socnav/scoring.py` |
| 模擬/畫面 | 主迴圈、pygame 即時渲染、matplotlib 軌跡圖 | `socnav/simulator.py` |

## 計分規則 (Hall proxemics)
中心對中心距離 (機器人 ↔ 行人)：

| 分區 | 距離 | 處置 |
|------|------|------|
| 親密區 intimate | < 0.45 m | 嚴重侵犯，每秒重扣 12 |
| 個人區 personal | 0.45–1.20 m | 依深入程度線性扣 (邊界→內 0→4/s) |
| 社交區 social | > 1.20 m | 不扣分 |

- 撞障礙物：每秒扣 25，並計入碰撞次數
- 抵達終點：+100；未抵達 (逾時/卡住)：−60
- 時間成本 1/s、路徑成本 2/m

最終分數 = 終點加分 − 社交成本 − 碰撞成本 − 時間成本 − 路徑成本。
所有權重可在 `socnav/scoring.py` 的 `ScoreConfig` 調整。

## 自訂情境
編輯 `socnav/scenarios.py`，用 `World / Obstacle / Pedestrian` 組出場地：
```python
World(
    width=10, height=10,
    start=[1, 1], goal=[9, 9],
    obstacles=[Obstacle(4.5, 4.5, 1, 1)],           # x, y, w, h
    pedestrians=[Pedestrian([8, 2], [[2, 8], [8, 2]], speed=0.8)],  # 起點, 巡迴 waypoints
)
```
