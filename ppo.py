import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from datetime import datetime, timedelta
from collections import deque

# ------------------ 資料處理 ------------------
def fetch_2330(start_date, end_date):
    df = yf.download("2330.TW", start=start_date, end=end_date, interval="1d")
    if df.empty:
        raise RuntimeError("無法抓取 2330.TW 股價")
    df = df.fillna(method="ffill")
    prices  = df["Close"].values.astype(np.float32)
    volumes = df["Volume"].values.astype(np.float32)
    dates   = pd.to_datetime(df.index).values
    mask = ~np.isnan(prices) & ~np.isnan(volumes)
    idx  = np.where(mask)[0]
    return prices[idx].flatten(), volumes[idx].flatten(), dates[idx]

def calc_indicators(prices, volumes, win=5):
    prices  = np.asarray(prices,  dtype=np.float32).flatten()
    volumes = np.asarray(volumes, dtype=np.float32).flatten()

    ma = np.convolve(prices, np.ones(win)/win, mode="valid")
    ma = np.concatenate([np.full(win-1, ma[0]), ma])

    diff = np.diff(prices, prepend=prices[0])
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)
    ag = np.convolve(gain, np.ones(14)/14, mode="valid")
    al = np.convolve(loss, np.ones(14)/14, mode="valid")
    ag = np.concatenate([np.full(13, ag[0]), ag])
    al = np.concatenate([np.full(13, al[0]), al])
    rs  = np.where(al != 0, ag/al, 100)
    rsi = 100 - 100/(1+rs)

    v_norm = volumes / volumes.max()
    return ma.astype(np.float32), (rsi/100).astype(np.float32), v_norm.astype(np.float32)

# ------------------ 環境 ------------------
class StockEnv:
    def __init__(self, prices, vols, dates, win=5, init_balance=100000, fee=0.001):
        self.prices = prices
        self.ma, self.rsi, self.vol = calc_indicators(prices, vols, win)
        self.dates = dates
        self.win   = win
        self.n     = len(prices)
        self.max_step = self.n - win - 1
        self.fee   = fee
        self.init_balance = init_balance
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.balance  = float(self.init_balance)
        self.shares   = 0
        self.last_buy_price = None
        self.last_buy_step  = None
        self.hold_streak = 0
        return self._get_state()

    def _get_state(self):
        s = self.step_idx
        e = s + self.win
        state = np.concatenate([
            self.prices[s:e]/self.prices.max(),
            self.ma[s:e]    /self.ma.max(),
            self.rsi[s:e],
            self.vol[s:e],
            [self.shares/1000, self.balance/100000]
        ]).astype(np.float32)
        return state

    def step(self, action):
        # 0=Hold, 1=Buy, 2=Sell
        self.step_idx += 1
        idx = self.step_idx + self.win - 1
        price = self.prices[idx]
        date  = self.dates[idx]

        reward = 0.0
        done   = self.step_idx >= self.max_step

        # Buy
        if action == 1 and self.balance > price:
            qty = int(self.balance // price)
            cost = qty * price * (1 + self.fee)
            self.balance -= cost
            self.shares  += qty
            self.last_buy_price = price
            self.last_buy_step  = self.step_idx
            reward += 0.3
            self.hold_streak = 0

        # Sell
        elif action == 2 and self.shares > 0:
            rev    = self.shares * price * (1 - self.fee)
            profit = rev - (self.shares * self.last_buy_price)
            self.balance += rev
            self.shares   = 0
            hold_days = self.step_idx - self.last_buy_step
            reward += profit / self.init_balance * 50
            if hold_days >= 10:
                reward += 2.0
            self.hold_streak = 0

        # Hold
        else:
            self.hold_streak += 1
            reward += 0.05 if self.hold_streak < 5 else -0.05

        # Portfolio delta
        next_price = self.prices[idx+1] if not done else price
        pv_now  = self.balance + self.shares * price
        pv_next = self.balance + self.shares * next_price
        reward += (pv_next - pv_now) / pv_now * 300

        if done:
            total_ret = (pv_now - self.init_balance)/self.init_balance
            reward += total_ret * 50

        return self._get_state(), reward, done, {
            "date": date, "price": price,
            "shares": self.shares, "balance": self.balance,
            "pv": pv_now
        }

# ------------------ PPO ------------------
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim,256), nn.ReLU(),
            nn.Linear(256,128),   nn.ReLU(),
            nn.Linear(128,a_dim), nn.Softmax(dim=-1)
        )
    def forward(self,x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, s_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim,256), nn.ReLU(),
            nn.Linear(256,128),   nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self,x): return self.net(x).squeeze(-1)

class PPO:
    def __init__(self, s_dim, a_dim, clip=0.2, lr=3e-4, gamma=0.99, lam=0.95, device="cpu"):
        self.actor  = Actor(s_dim,a_dim).to(device)
        self.critic = Critic(s_dim).to(device)
        self.optimA = optim.Adam(self.actor.parameters(),  lr=lr)
        self.optimC = optim.Adam(self.critic.parameters(), lr=lr)
        self.clip   = clip
        self.gamma  = gamma
        self.lam    = lam
        self.device = device

    def select(self, s):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.actor(s).cpu().numpy()[0]
        return np.random.choice(len(prob), p=prob), prob

    # --- Generalized Advantage Estimation ---
    def compute_gae(self, rewards, values, dones, next_value):
        adv, gae = np.zeros_like(rewards, dtype=np.float32), 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae   = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
            next_value = values[t]
        returns = adv + values
        return adv, returns

    def update(self, batch, epochs=4):
        states, actions, old_probs, returns, adv = batch
        states  = torch.tensor(states,  dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        old_probs = torch.tensor(old_probs, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        adv     = torch.tensor(adv,     dtype=torch.float32).to(self.device)
        adv     = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(epochs):
            probs = self.actor(states)
            dist  = probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            ratio = dist / (old_probs + 1e-8)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * adv
            lossA = -torch.min(surr1, surr2).mean()

            v = self.critic(states)
            lossC = nn.functional.mse_loss(v, returns)

            self.optimA.zero_grad(); lossA.backward(); self.optimA.step()
            self.optimC.zero_grad(); lossC.backward(); self.optimC.step()

# ------------------ 訓練 + 測試 ------------------
def train_test_ppo():
    end = datetime(2025,5,9)
    train_start = (end - timedelta(days=5*365)).strftime("%Y-%m-%d")
    test_start  = (end - timedelta(days=90)).strftime("%Y-%m-%d")

    prices, vols, dates = fetch_2330(train_start, end.strftime("%Y-%m-%d"))
    train_mask = dates < np.datetime64(test_start)
    test_mask  = ~train_mask

    env_train = StockEnv(prices[train_mask], vols[train_mask], dates[train_mask])
    env_test  = StockEnv(prices[test_mask],  vols[test_mask],  dates[test_mask])

    s_dim = env_train._get_state().shape[0]
    a_dim = 3
    agent = PPO(s_dim, a_dim, device="cpu")

    buffer_s, buffer_a, buffer_p, buffer_r, buffer_v, buffer_d = [],[],[],[],[],[]
    max_ep = 400
    batch_size = 2048
    ep_rewards = deque(maxlen=10)

    for ep in range(max_ep):
        s = env_train.reset()
        ep_reward = 0
        while True:
            a, prob = agent.select(s)
            v = agent.critic(torch.tensor(s, dtype=torch.float32)).item()
            next_s, r, done, _ = env_train.step(a)

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_p.append(prob[a])
            buffer_r.append(r)
            buffer_v.append(v)
            buffer_d.append(done)

            s = next_s
            ep_reward += r

            reach_batch = len(buffer_s) >= batch_size
            if reach_batch or done:
                # 取得下一狀態的 V(s')
                next_v = agent.critic(torch.tensor(next_s, dtype=torch.float32)).item()
                adv, ret = agent.compute_gae(np.array(buffer_r, dtype=np.float32),
                                             np.array(buffer_v, dtype=np.float32),
                                             np.array(buffer_d, dtype=np.float32),
                                             next_v)
                agent.update((buffer_s, buffer_a, buffer_p, ret, adv))
                buffer_s, buffer_a, buffer_p, buffer_r, buffer_v, buffer_d = [],[],[],[],[],[]
            if done: break

        ep_rewards.append(ep_reward)
        if ep % 10 == 0:
            print(f"Episode {ep}/{max_ep} | AvgReward {np.mean(ep_rewards):.2f}")

    # ------------ 測試 ------------
    s = env_test.reset()
    records = []
    while True:
        a, _ = agent.select(s)
        s, r, done, info = env_test.step(a)
        records.append({
            "Date": info["date"], "PV": info["pv"],
            "Action": a, "Price": info["price"],
            "Shares": info["shares"], "Balance": info["balance"]
        })
        if done: break

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    out = "C:/Users/user/Desktop/2330_PPO_test.csv"
    df.to_csv(out, index=False)
    print(f"測試結果輸出至 {out}")
    print("最終資產：", df.iloc[-1]["PV"])

if __name__ == "__main__":
    train_test_ppo()
