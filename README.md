# 基於近端策略優化 (PPO) 演算法之智慧代理人開發專案

<div align="center">
  <video src="https://github.com/user-attachments/assets/4e9feaed-a5d9-4ac8-a080-a78335911135" width="80%" controls muted autoplay loop></video>
  <p><i>影片：PPO 代理人與人類玩家之即時對戰側錄</i></p>
</div>

## 1. 專案概述
本專案實現了一個基於離散狀態空間的強化學習 (Reinforcement Learning) 代理人，主要應用於井字遊戲 (Tic-Tac-Toe) 之策略訓練。核心技術採用近端策略優化 (PPO) 演算法，並結合啟發式演算法 (Minimax) 作為教練機制，解決強化學習在稀疏獎勵 (Sparse Reward) 環境下的收斂問題。

專案設計強調軟體工程實務，包含環境隔離 (Python Virtual Environment)、自動化單元測試 (Pytest) 以及持續整合 (GitHub Actions) 的完整流程。

## 2. 技術核心與演算法設計

### 2.1 神經網路架構
採用多層感知器 (Multi-Layer Perceptron, MLP) 作為 Policy 與 Value 網路的基礎架構：
- **輸入層**：9 個神經元（對應 3x3 棋盤狀態）。
- **隱藏層**：包含三層全連接層 (256, 256, 128)，並導入 Dropout (0.1) 以增強泛化能力並防止過擬合。
- **輸出層**：
    - **Actor**：9 個動作機率分布。
    - **Critic**：1 個狀態價值評估 (State Value Estimation)。

### 2.2 訓練策略與獎勵塑造 (Reward Shaping)
為引導 AI 學習戰略要地並降低探索成本，設計了以下獎勵機制：
- **終局獎勵**：贏球獲取 +10.0，平手獲取 +1.0，輸球或違規移動則重罰 -10.0。
- **戰略獎勵 (Reward Shaping)**：佔領中心位 (Center position) 提供 +0.5 額外獎勵，引導模型建立初期的戰略優勢。
- **對手課程學習 (Curriculum Learning)**：採用混合對手模式，包含 30% Minimax 完美防守、30% 隨機移動以及 40% 自我博弈 (Self-play)，確保代理人具備防守穩定度與攻擊侵略性。

## 3. 專案架構
```text
DRL_pr/
├── .github/workflows/   # CI/CD 自動化腳本
├── src/                 # 核心原始碼
│   ├── tic_tac_train.py # PPO 訓練邏輯實現
│   └── PPO_play.py      # 基於 CustomTkinter 之使用者互動介面
├── tests/               # 自動化測試案例
│   ├── test_game_logic.py # 遊戲規則與勝負判定測試
│   └── test_model.py      # 模型維度與權重載入測試
├── models/              # 模型權重存檔 (.pt)
├── results/             # 訓練統計圖表 (SVG/PNG)
├── .gitignore           # Git 版本控制排除清單
├── requirements.txt     # 環境相依套件清單
└── README.md            # 技術說明文件
