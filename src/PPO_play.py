import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import customtkinter as ctk

# ==========================================
# 神經網路大腦架構 (需與 train.py 完全一致)
# ==========================================
class PPOAgentUltra(nn.Module):
    def __init__(self):
        super(PPOAgentUltra, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, 9)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        state = torch.FloatTensor(state)
        x = self.network(state)
        return self.actor(x), self.critic(x)

# ==========================================
# 具備「在線學習」與「戰況統計」的淺色主題 UI
# ==========================================
class PPO_Learning_GUI(ctk.CTk):
    def __init__(self, agent, model_path):
        super().__init__()
        self.title("PPO Learning Mode")
        self.geometry("300x500") # 高度拉長至 500 以容納計分板
        
        # --- 設定淺色模式與視窗背景色 ---
        ctk.set_appearance_mode("Light") 
        ctk.set_default_color_theme("blue") 
        self.configure(fg_color="#FFFFFF") # 視窗背景為純白
        
        self.agent = agent
        self.model_path = model_path
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.00005) 
        
        self.board = np.zeros(9, dtype=float)
        self.game_history = [] 
        self.buttons = []
        self.done = False
        
        # --- 計分板與步數變數 ---
        self.ai_wins = 0
        self.human_wins = 0
        self.draws = 0
        self.moves_count = 0
        
        self.turn = random.choice(['AI', 'Human'])
        self.ai_symbol, self.human_symbol = (1, -1) if self.turn == 'AI' else (-1, 1)
        
        self._setup_ui()
        if self.turn == 'AI': self.after(500, self.ai_move)

    def _setup_ui(self):
        # 狀態列設定為淺灰背景、黑字
        self.status_label = ctk.CTkLabel(self, text="準備對戰...", 
                                        font=('Segoe UI', 14, 'bold'), 
                                        fg_color="#F0F2F5",  
                                        text_color="#222222", 
                                        corner_radius=8, height=40)
        self.status_label.pack(fill="x", padx=15, pady=15)

        # 計分板 UI
        self.stats_label = ctk.CTkLabel(self, text="AI 勝: 0 | 人類 勝: 0 | 平手: 0\n目前步數: 0", 
                                        font=('Segoe UI', 13), 
                                        text_color="#555555") 
        self.stats_label.pack(pady=(0, 10))

        # 網格框架設定為透明，直接顯示白底
        self.grid_frame = ctk.CTkFrame(self, fg_color="transparent") 
        self.grid_frame.pack(padx=10, pady=5)

        # 設定按鈕為近白色、深色粗邊框
        for i in range(9):
            btn = ctk.CTkButton(self.grid_frame, text="", 
                                font=('Segoe UI', 22, 'bold'), 
                                width=75, height=75,
                                fg_color="#FDFDFD",       
                                text_color="#222222",      
                                border_width=3, 
                                border_color="#222222",    
                                corner_radius=10,
                                hover_color="#EAECEF",    
                                command=lambda i=i: self.human_click(i))
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
            self.buttons.append(btn)

    def update_stats_ui(self):
        """刷新計分板與步數畫面"""
        self.stats_label.configure(
            text=f"AI 勝: {self.ai_wins} | 人類 勝: {self.human_wins} | 平手: {self.draws}\n目前步數: {self.moves_count}"
        )

    def human_click(self, i):
        if self.board[i] == 0 and self.turn == 'Human' and not self.done:
            self.game_history.append((self.board.copy(), self.human_symbol, i))
            self.make_move(i, self.human_symbol)
            if not self.check_game_over():
                self.turn = 'AI'
                self.after(500, self.ai_move)

    def ai_move(self):
        if self.done: return
        action = self.get_intelligent_move_logic()
        self.game_history.append((self.board.copy(), self.ai_symbol, action))
        self.make_move(action, self.ai_symbol)
        if not self.check_game_over():
            self.turn = 'Human'

    def get_intelligent_move_logic(self):
        empty_spots = np.where(self.board == 0)[0]
        
        # 硬規則防護：贏棋優先與防守優先
        for move in empty_spots: 
            temp = self.board.copy(); temp[move] = self.ai_symbol
            if self.check_win_static(temp, self.ai_symbol): return move
        for move in empty_spots: 
            temp = self.board.copy(); temp[move] = self.human_symbol
            if self.check_win_static(temp, self.human_symbol): return move
            
        # AI 推論階段：切換為 eval 模式，關閉 Dropout
        self.agent.eval()
        with torch.no_grad():
            canonical_state = self.board * self.ai_symbol
            logits, _ = self.agent(canonical_state)
            mask = torch.tensor(canonical_state != 0, dtype=torch.bool)
            logits = torch.where(mask, torch.tensor(-1e9), logits)
            return torch.argmax(logits).item()

    def make_move(self, i, symbol):
        self.board[i] = symbol
        
        # 步數 +1 並刷新 UI
        self.moves_count += 1
        self.update_stats_ui()
        
        char, color = ("X", "#E74C3C") if symbol == 1 else ("O", "#27AE60")
        
        # 下棋後按鈕變為淺灰色背景，並稍微減淡粗邊框
        self.buttons[i].configure(text=char, state="disabled", 
                                   text_color_disabled=color, 
                                   fg_color="#F0F0F0", 
                                   border_width=1,
                                   border_color="#C0C0C0")

    def check_win_static(self, board, p):
        win_cond = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        return any(board[i]==board[j]==board[k]==p for i,j,k in win_cond)

    def check_game_over(self):
        winner = None
        if self.check_win_static(self.board, self.ai_symbol): winner = "AI"
        elif self.check_win_static(self.board, self.human_symbol): winner = "Human"
        elif 0 not in self.board: winner = "Draw"
            
        if winner:
            self.done = True
            
            # 記錄勝負局數並刷新
            if winner == "AI": self.ai_wins += 1
            elif winner == "Human": self.human_wins += 1
            else: self.draws += 1
            self.update_stats_ui()
            
            if winner == "Human":
                # 人類獲勝時狀態列顯示淺綠色
                self.status_label.configure(text="人類獲勝！AI 正在檢討學習...", 
                                           fg_color="#A9DFBF", 
                                           text_color="#222222") 
                self.update_idletasks()
                self.learn_from_human_victory()
            else:
                # AI獲勝或平手時狀態列顯示淺紅或淺黃
                final_bg = "#F1948A" if winner=="AI" else "#F7DC6F" 
                self.status_label.configure(text=f"遊戲結束：{winner}", 
                                           fg_color=final_bg, 
                                           text_color="#222222") 
            
            self.after(2000, self.reset_game)
            return True
        return False

    def learn_from_human_victory(self):
        print("觸發在線微調機制：學習人類專家路徑...")
        # 學習階段：必須切換回 train 模式，重啟 Dropout
        self.agent.train()
        
        for state, symbol, action in self.game_history:
            if symbol == self.human_symbol:
                canonical_state = state * symbol
                target_action = torch.tensor([action])
                
                logits, _ = self.agent(canonical_state)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(logits.unsqueeze(0), target_action)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        torch.save(self.agent.state_dict(), self.model_path)
        print("學習完成，權重已同步更新。")

    def reset_game(self):
        self.board = np.zeros(9, dtype=float)
        self.game_history = []
        self.done = False
        
        # 新局開始，步數歸零並刷新 UI
        self.moves_count = 0
        self.update_stats_ui()
        
        # 重置按鈕樣式和邊框顏色回到淺色
        for btn in self.buttons: 
            btn.configure(text="", state="normal", 
                          fg_color="#FDFDFD",       
                          border_width=3, 
                          border_color="#222222") 
        self.turn = random.choice(['AI', 'Human'])
        self.ai_symbol, self.human_symbol = (1, -1) if self.turn == 'AI' else (-1, 1)
        
        # 重置狀態列為預設淺灰背景
        self.status_label.configure(text=f"新局：{self.turn}", 
                                   fg_color="#F0F2F5", 
                                   text_color="#222222") 
        if self.turn == 'AI': self.after(500, self.ai_move)

if __name__ == "__main__":
    # 使用安全的 .pt 副檔名
    path = "models/ppo_brain_final.pt"
    agent = PPOAgentUltra()
    if os.path.exists(path):
        agent.load_state_dict(torch.load(path, weights_only=True))
        print("已載入模型權重。")
    else:
        print("尚未找到 models/ppo_brain_final.pt 模型，AI 將使用隨機初始權重下棋")
        
    app = PPO_Learning_GUI(agent, path)
    app.mainloop()