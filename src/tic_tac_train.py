import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import math
import sys

print("啟動 嗜血刺客版 PPO 訓練 (強化必勝陷阱學習)...")

# ==========================================
# 1. Minimax 快取教練
# ==========================================
minimax_cache = {}

def get_minimax_move(board, player):
    board_tuple = tuple(board)
    if (board_tuple, player) in minimax_cache:
        return minimax_cache[(board_tuple, player)]

    def check_win(b):
        win_cond = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for (i, j, k) in win_cond:
            if b[i] == b[j] == b[k] != 0: return b[i]
        return 0 if 0 not in b else None

    def solve(b, depth, is_max, p):
        state_key = (tuple(b), is_max, p)
        if state_key in minimax_cache: return minimax_cache[state_key]

        res = check_win(b)
        if res == p: return 10 - depth
        if res == -p: return -10 + depth
        if res == 0: return 0
        
        scores = []
        for i in range(9):
            if b[i] == 0:
                b[i] = p if is_max else -p
                scores.append(solve(b, depth+1, not is_max, p))
                b[i] = 0
        
        result = (max(scores) if is_max else min(scores)) if scores else 0
        minimax_cache[state_key] = result
        return result

    best_score, best_act = -math.inf, None
    for i in range(9):
        if board[i] == 0:
            board[i] = player
            score = solve(board, 0, False, player)
            board[i] = 0
            if score > best_score:
                best_score, best_act = score, i
    
    minimax_cache[(board_tuple, player)] = best_act
    return best_act

# ==========================================
# 2. 訓練環境 (重構獎勵：極度渴望贏球)
# ==========================================
class TicTacToeUltra:
    def __init__(self):
        self.board = np.zeros(9, dtype=float)
        self.done = False

    def reset(self):
        self.board = np.zeros(9, dtype=float)
        self.done = False
        return self.board

    def step(self, action, player):
        if self.board[action] != 0: 
            return self.board, -10.0, True # 違規重罰
        
        self.board[action] = player
        
       
        center_bonus = 0.8 if action == 4 else 0.0

        win_cond = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for (i, j, k) in win_cond:
            if self.board[i] == self.board[j] == self.board[k] == player:
                return self.board, 10.0 + center_bonus, True # 【贏球】：+10 分 (巨大誘因)
                
        if 0 not in self.board: 
            return self.board, 1.0 + center_bonus, True # 【平手】：只給 +1 分 (逼迫它去想辦法贏)
            
        return self.board, 0.0 + center_bonus, False

# ==========================================
# 3. 神經網路架構
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
# 4. PPO 訓練主迴圈
# ==========================================
def train_ultra(episodes=30000): # 拉高到 3 萬局，徹底消除對稱性盲點
    agent = PPOAgentUltra()
    optimizer = optim.Adam(agent.parameters(), lr=0.0003)
    env = TicTacToeUltra()
    
    batch_size = 64
    history_rewards = []
    
    gamma = 0.99 
    eps_clip = 0.2 
    entropy_coef = 0.01 
    
    batch_states, batch_actions, batch_old_logprobs = [], [], []
    batch_rewards, batch_values = [], []

    print(f"開始訓練 {episodes} 局...")
    print("模式：30% 完美防守 + 30% 隨機找漏洞 + 40% 自我博弈進化")
    print("-" * 50)
    sys.stdout.flush()

    # 封裝對手邏輯：讓 AI 有機會屠殺弱者以學習「陷阱必勝法」
    def get_opp_act(s, sym, r_val):
        if r_val < 0.3:
            return get_minimax_move(s, sym) # 完美防守
        elif r_val < 0.6:
            return np.random.choice(np.where(s == 0)[0]) # 隨機犯錯，讓 AI 抓
        else:
            with torch.no_grad():
                o_logits, _ = agent(s * sym) # 自我博弈
                o_mask = torch.tensor(s != 0, dtype=torch.bool)
                o_logits = torch.where(o_mask, torch.tensor(-1e9), o_logits)
                return torch.argmax(o_logits).item()

    for ep in range(episodes):
        state = env.reset()
        done = False
        ai_sym = random.choice([1, -1])
        opp_sym = -ai_sym
        
        # 決定這一局對手的性格
        rand_val = random.random()
        ep_rewards = []
        
        if ai_sym == -1:
            opp_act = get_opp_act(state, opp_sym, rand_val)
            state, _, _ = env.step(opp_act, opp_sym)

        while not done:
            canonical_state = state * ai_sym
            logits, val = agent(canonical_state)
            
            mask = torch.tensor(canonical_state != 0, dtype=torch.bool)
            logits = torch.where(mask, torch.tensor(-1e9), logits)
            
            prob = torch.softmax(logits, dim=-1)
            m = Categorical(prob)
            act = m.sample()
            
            batch_states.append(canonical_state.copy())
            batch_actions.append(act)
            batch_old_logprobs.append(m.log_prob(act).detach()) 
            batch_values.append(val)
            
            state, reward, done = env.step(act.item(), ai_sym)
            
            if not done:
                opp_act = get_opp_act(state, opp_sym, rand_val)
                state, opp_rew, done = env.step(opp_act, opp_sym)
                if opp_rew >= 10: reward = -10.0 # 對手贏了，代表自己被殺，重罰
            
            ep_rewards.append(reward)

        R = 0
        returns = []
        for r in ep_rewards[::-1]: 
            R = r + gamma * R
            returns.insert(0, R)
        batch_rewards.extend(returns)
        
        history_rewards.append(sum(ep_rewards))

        if (ep + 1) % batch_size == 0:
            returns_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
            values_tensor = torch.cat(batch_values).squeeze(-1)
            old_logprobs_tensor = torch.stack(batch_old_logprobs)
            states_tensor = torch.tensor(np.array(batch_states), dtype=torch.float32)
            actions_tensor = torch.stack(batch_actions)

            advantages = returns_tensor - values_tensor.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            logits, values = agent(states_tensor)
            mask = torch.tensor(states_tensor != 0, dtype=torch.bool)
            logits = torch.where(mask, torch.tensor(-1e9), logits)
            
            probs = torch.softmax(logits, dim=-1)
            m = Categorical(probs)
            
            new_logprobs = m.log_prob(actions_tensor)
            ratios = torch.exp(new_logprobs - old_logprobs_tensor)
            
            entropy = m.entropy().mean()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() 

            critic_loss = nn.MSELoss()(values.squeeze(-1), returns_tensor)

            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_states, batch_actions, batch_old_logprobs = [], [], []
            batch_rewards, batch_values = [], []

        if (ep + 1) % 100 == 0:
            avg_rew = np.mean(history_rewards[-100:])
            print(f"局數: {ep+1}/{episodes} | 近100局平均回報: {avg_rew:.2f}", flush=True)

    torch.save(agent.state_dict(), "models/ppo_brain_final.pt")
    print("-" * 50)
    print("訓練結束，大腦已保存為 models/ppo_brain_final.pt")

if __name__ == "__main__":
    train_ultra()