import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import os

# 設置隨機種子
np.random.seed(42)
torch.manual_seed(42)

# 抓取 2330.TW 股價數據
def fetch_0050_data(start_date, end_date):
    ticker = "SPMO" 
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    if data.empty:
        raise ValueError("無法抓取 2330.TW 數據，請檢查網路或日期範圍")
    # 處理缺失：前向填充
    data = data.fillna(method='ffill')
    prices = np.array(data['Close'].values, dtype=np.float64).flatten()
    volumes = np.array(data['Volume'].values, dtype=np.float64).flatten()
    dates = pd.to_datetime(data.index).values
    mask = ~np.isnan(prices) & ~np.isnan(volumes)
    valid_idx = np.where(mask)[0]
    return prices[valid_idx], volumes[valid_idx], dates[valid_idx]

# 技術指標：5 日均線、RSI、成交量
def calculate_technical_indicators(prices, volumes, window=5):
    prices = np.asarray(prices, dtype=np.float64).flatten()
    volumes = np.asarray(volumes, dtype=np.float64).flatten()
    ma = np.convolve(prices, np.ones(window)/window, mode='valid')
    ma = np.concatenate([np.full(window-1, ma[0]), ma])
    diff = np.diff(prices, prepend=prices[0])
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)
    avg_gain = np.convolve(gain, np.ones(14)/14, mode='valid')
    avg_loss = np.convolve(loss, np.ones(14)/14, mode='valid')
    avg_gain = np.concatenate([np.full(13, avg_gain[0]), avg_gain])
    avg_loss = np.concatenate([np.full(13, avg_loss[0]), avg_loss])
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    volume_norm = volumes / np.max(volumes)
    return ma, rsi / 100, volume_norm

# 股票交易環境
class StockTradingEnv:
    def __init__(self, prices, volumes, dates,
                 window_size=5, initial_balance=100000, trading_cost=0.001):
        self.prices = np.array(prices, dtype=np.float64).flatten()
        self.ma, self.rsi, self.volume = calculate_technical_indicators(
            prices, volumes, window_size)
        self.dates = dates
        self.window_size = window_size
        self.n = len(prices)
        self.current_step = 0
        self.balance = initial_balance
        self.shares = 0
        self.max_steps = self.n - window_size - 1
        self.action_log = []
        self.hold_streak = 0
        self.last_buy_price = None
        self.trading_cost = trading_cost
        self.trade_cooldown = 0

    def reset(self):
        self.current_step = 0
        self.balance = 4500
        self.shares = 0
        self.action_log = []
        self.hold_streak = 0
        self.last_buy_price = None
        self.trade_cooldown = 0
        return self._get_state()

    def step(self, action):
        self.current_step += 1
        idx = self.current_step + self.window_size - 1
        current_price = self.prices[idx]
        current_ma = self.ma[idx]
        current_date = self.dates[idx]

        reward = 0
        self.trade_cooldown = max(0, self.trade_cooldown - 1)
        sold_shares = 0
        trade_profit = 0

        if action == 1 and self.trade_cooldown == 0:
            if self.balance >= current_price:
                shares_to_buy = int(self.balance // current_price) // 2
                cost = shares_to_buy * current_price * (1 + self.trading_cost)
                if self.balance >= cost:
                    self.shares += shares_to_buy
                    self.balance -= cost
                    self.last_buy_price = current_price
                    self.last_buy_step = self.current_step  # ← 新增紀錄買入步驟
                    self.action_log.append(f"Buy {shares_to_buy} shares at {current_price:.2f}")
                    reward += 0.5
                    if current_price > current_ma:
                        reward += 0.3
                    self.hold_streak = 0
                    self.trade_cooldown = 2  # 冷卻期調成10天
                else:
                    reward = -0.2
                    self.action_log.append("Failed buy: insufficient funds")
                    self.hold_streak = 0
            else:
                reward = -0.2
                self.action_log.append("Failed buy: insufficient funds")
                self.hold_streak = 0

        elif action == 2 and self.shares > 0 and self.trade_cooldown == 0:
            sold_shares = self.shares / 2
            buy_price = self.last_buy_price
            holding_days = self.current_step - getattr(self, 'last_buy_step', self.current_step)
            revenue = self.shares * current_price * (1 - self.trading_cost)
            self.balance += revenue
            if buy_price and current_price > buy_price:
                trade_profit = (current_price - buy_price) * sold_shares
                reward += 5.0
                reward += trade_profit / (buy_price * sold_shares) * 200  # 根據持倉利潤額外加分
            if holding_days >= 10:
                reward += 2.0  # 持有超過10天賣出，長線獎勵
            if current_price < current_ma:
                reward += 0.3
            self.action_log.append(f"Sell {self.shares} shares at {current_price:.2f}")
            self.shares = 0
            self.last_buy_price = None
            self.hold_streak = 0
            self.trade_cooldown = 2

        else:
            self.action_log.append("Hold")
            self.hold_streak += 1
            if self.hold_streak >= 5:
                reward -= 0.05
            else:
                reward += 0.1

        next_price = (self.prices[self.current_step + self.window_size]
                    if self.current_step < self.max_steps else current_price)
        pv_now = self.balance + self.shares * current_price
        pv_next = self.balance + self.shares * next_price
        reward += (pv_next - pv_now) / pv_now * 500

        done = self.current_step >= self.max_steps
        if done:
            total_return = (pv_now - 100000) / 100000  # 基於初始資金的總回報率
            reward += total_return * 300  # 給整個 episode 的總回報 bonus

        info = {
            'date': current_date,
            'action': self.action_log[-1],
            'price': current_price,
            'shares': self.shares,
            'balance': self.balance,
            'portfolio_value': pv_now,
            'sold_shares': sold_shares,
            'trade_profit': trade_profit
        }
        return self._get_state(), reward, done, info


    def _get_state(self):
        s = self.current_step
        e = s + self.window_size
        price_window = self.prices[s:e] / np.max(self.prices)
        ma_window = self.ma[s:e] / np.max(self.ma)
        rsi_window = self.rsi[s:e]
        volume_window = self.volume[s:e]
        return np.concatenate([price_window, ma_window,
                               rsi_window, volume_window,
                               [self.shares / 1000,
                                self.balance / 100000]])

# --- DQN 網絡 ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        x = self.feature(x)
        v = self.value(x)
        a = self.advantage(x)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q


# --- DQN 代理 ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.95
        self.temperature = 1.0
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q = self.model(state_t)
            prob = torch.softmax(q / self.temperature, dim=1).cpu().numpy()[0]
            return np.random.choice(self.action_size, p=prob)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state_t)
        return np.argmax(act_values.cpu().numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.temperature = max(0.1, self.temperature * 0.9995)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())

# --- 訓練與測試 ---
def train_and_test_dqn():
    end_date = datetime(2024, 5, 9)
    train_start = (end_date - timedelta(days=8*365)).strftime('%Y-%m-%d')
    test_start = (end_date - timedelta(days=365)).strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    prices, volumes, dates = fetch_0050_data(train_start, end_str)
    train_mask = dates < np.datetime64(test_start)
    test_mask = ~train_mask

    train_env = StockTradingEnv(prices[train_mask], volumes[train_mask],
                                dates[train_mask], window_size=5)
    state_size = train_env.window_size * 4 + 2
    action_size = 3
    agent = DQNAgent(state_size, action_size)

    episodes = 100
    batch = 32
    best_pv = 0
    model_path = "2330_dqn_best.pth"

    for e in range(episodes):
        state = train_env.reset()
        total_reward = 0
        profit = 0
        action_cnt = [0, 0, 0]

        for _ in range(train_env.max_steps):
            action = agent.act(state)
            action_cnt[action] += 1
            next_state, reward, done, info = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            # 若此回合有成交利潤，直接累加
            profit += info.get('trade_profit', 0)
            if done:
                break

        agent.replay(batch)
        agent.update_target_model()

        pv = train_env.balance + train_env.shares * train_env.prices[-1]
        if pv > best_pv:
            best_pv = pv
            agent.save_model(model_path)

        print(f"Ep {e+1}/{episodes} | Reward {total_reward:.2f} | PV {pv:.2f} | "
              f"Profit {profit:.2f} | ε={agent.epsilon:.2f}")
        print(f"Cnt: Hold={action_cnt[0]}, Buy={action_cnt[1]}, Sell={action_cnt[2]}")
        if e % 10 == 0:
            print("Recent actions:", train_env.action_log[-5:])

    # ---- 測試 ----
    test_env = StockTradingEnv(prices[test_mask], volumes[test_mask],
                               dates[test_mask], window_size=5)
    agent.load_model(model_path)
    state = test_env.reset()
    results = []

    for _ in range(test_env.max_steps):
        action = agent.act(state)
        next_state, reward, done, info = test_env.step(action)
        results.append({
            'Date': info['date'], 'Action': info['action'],
            'Price': info['price'], 'Shares': info['shares'],
            'Balance': info['balance'], 'Portfolio_Value': info['portfolio_value']
        })
        state = next_state
        if done:
            break

    df = pd.DataFrame(results)
    df['Date'] = pd.to_datetime(df['Date'])
    out_path = "C:/Users/user/Desktop/2330_test_results.csv"
    df.to_csv(out_path, index=False)
    print(f"測試結果已儲存：{out_path}")
    print(f"最終資產：{results[-1]['Portfolio_Value']:.2f} TWD")
    print(f"動作統計：Hold={sum(r['Action']=='Hold' for r in results)}, "
          f"Buy={sum(r['Action'].startswith('Buy') for r in results)}, "
          f"Sell={sum(r['Action'].startswith('Sell') for r in results)}")



def predict_today_action():
    model_path = "2330_dqn_best.pth"
    agent = DQNAgent(state_size=22, action_size=3)
    agent.load_model(model_path)

    today = datetime.today()
    lookback_days = 30
    start_date = (today - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    prices, volumes, dates = fetch_0050_data(start_date, end_date)
    env = StockTradingEnv(prices, volumes, dates, window_size=5)
    state = env.reset()
    
    # 滾動到最後一天（今天）
    for _ in range(env.max_steps):
        action = agent.act(state)
        state, _, done, info = env.step(action)
        if done:
            break

    action_str = ["Hold", "Buy", "Sell"][action]
    print(f"[{info['date']}] 建議動作：{action_str}")
    print(f"價格：{info['price']:.2f} | 持股：{info['shares']} | 現金：{info['balance']:.2f} | 資產：{info['portfolio_value']:.2f}")
    
    # 存檔紀錄
    daily = {
        'Date': info['date'],
        'Action': action_str,
        'Price': info['price'],
        'Shares': info['shares'],
        'Balance': info['balance'],
        'Portfolio_Value': info['portfolio_value']
    }
    path = "daily_signal_log.csv"
    df = pd.DataFrame([daily])
    if os.path.exists(path):
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)



if __name__ == "__main__":
    train_and_test_dqn()
    predict_today_action()