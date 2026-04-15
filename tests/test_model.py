import sys
import os
import torch
import pytest
import numpy as np

# 將 src 加入路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# 引用你寫的神經網路大腦
from PPO_play import PPOAgentUltra

class TestPPOModel:
    
    def test_model_initialization(self):
        # 測試 1：模型是否能成功實體化，沒有語法錯誤？
        agent = PPOAgentUltra()
        assert isinstance(agent, torch.nn.Module)

    def test_model_forward_pass_shape(self):
        # 測試 2：大腦的「輸入」與「輸出」矩陣維度是否符合預期？
        agent = PPOAgentUltra()
        
        # 模擬一個空白的井字棋盤狀態 (9個格子)
        dummy_state = np.zeros(9, dtype=float)
        
        # 讓大腦進行一次推論
        logits, value = agent(dummy_state)
        
        # 驗證 1：Actor (策略網路) 應該輸出 9 個動作的權重
        assert logits.shape == torch.Size([9]), f"預期 Actor 輸出維度為 [9]，但得到 {logits.shape}"
        
        # 驗證 2：Critic (價值網路) 應該輸出 1 個勝率評估值
        assert value.shape == torch.Size([1]), f"預期 Critic 輸出維度為 [1]，但得到 {value.shape}"

    def test_model_can_load_weights(self):
        # 測試 3：確保我們辛苦訓練出來的 .pt 檔案真的存在，而且能被讀取
        agent = PPOAgentUltra()
        model_path = os.path.join(os.path.dirname(__file__), '../models/ppo_brain_final.pt')
        
        # 如果檔案存在，測試它是否能成功載入而不會報錯
        if os.path.exists(model_path):
            agent.load_state_dict(torch.load(model_path, weights_only=True))
            assert True # 成功載入即通過
        else:
            pytest.skip("找不到模型權重檔，跳過此測試。")
