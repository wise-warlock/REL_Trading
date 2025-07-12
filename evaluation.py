import yfinance as yf
import numpy as np
import pandas as pd
import gym
from gym import spaces
from sb3_contrib import RecurrentPPO
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# ------------------------------
# 1. Load and Process Data (Same as training)
# ------------------------------
df = pd.read_csv("stock_best_1y.csv")
df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
df.columns.name = None

# Add technical indicators
df.ta.rsi(length=14, append=True)
df.ta.ema(length=14, append=True)
df.ta.macd(append=True)
df.ta.bbands(length=20, append=True)

# Rename columns for simplicity
df.rename(columns={
    'RSI_14': 'RSI', 'EMA_14': 'EMA',
    'MACD_12_26_9': 'MACD', 'MACDh_12_26_9': 'MACD_hist', 'MACDs_12_26_9': 'MACD_signal',
    'BBL_20_2.0': 'BB_lower', 'BBU_20_2.0': 'BB_upper', 'BBB_20_2.0': 'BB_bandwidth', 'BBP_20_2.0': 'BB_percent'
}, inplace=True)

df['Close_raw'] = df['Close'].copy()
df = df.dropna().reset_index(drop=True)

# Scale the features
scaler = StandardScaler()
scaled_features = ['Close', 'Open', 'High', 'Low', 'Volume', 'RSI', 'EMA', 'MACD', 'BB_percent']
df[scaled_features] = scaler.fit_transform(df[scaled_features])

# Split data to get the same test set
train_size = int(len(df) * 0.8)
train_df = df[:train_size].reset_index(drop=True)
test_df = df[train_size:].reset_index(drop=True)


# ------------------------------
# 2. Define the Environment Class
# --- CHANGE 1: This environment definition now matches stock_trader_v17 ---
# ------------------------------
class StockTradingEnv(gym.Env):
    def __init__(self, df, scaler):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.scaler = scaler
        self.max_steps = len(df)
        
        self.action_space = spaces.Discrete(16) 
        
        # The v17 model expects an observation space of shape (10,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        self.buy_pcts = np.linspace(0.24, 0.53, 5) 
        self.sell_pcts = np.linspace(0.22, 0.73, 5) 
        self.leverage_levels = np.linspace(1.5, 3.0, 5)
        
        self.init_cash = 100_000 
        self.transaction_fee = 0.03 
        self.total_asset_min = 1000 

    def reset(self):
        self.step_idx = 0
        self.cash = self.init_cash
        self.stock_owned = 0 
        self.cost_basis = 0
        self.prev_total_asset = self.init_cash
        self.peak_equity = self.init_cash
        return self._get_obs()

    def _get_obs(self):
        obs_idx = min(self.step_idx, self.max_steps - 1)
        row = self.df.iloc[obs_idx]
        
        # This now returns 10 pieces of information, matching the v17 model
        return np.array([
            self.cash / self.init_cash,
            self.stock_owned,
            row['Close'],
            row['Volume'],
            row['RSI'],
            row['EMA'],
            row['MACD'],
            row['BB_percent'],
            self.step_idx / self.max_steps,
            self.peak_equity / self.init_cash
        ], dtype=np.float32)

    def step(self, action):
        done = False
        price_raw = self.df['Close_raw'].iloc[self.step_idx]
        
        if action == 0: pass
        elif 1 <= action <= 5:
            trade_amount = self.cash * self.buy_pcts[action - 1] if self.cash > 0 else self.init_cash * self.buy_pcts[action - 1]
            if trade_amount > 1:
                quantity = (trade_amount / price_raw) * (1 - self.transaction_fee)
                self.cash -= quantity * price_raw * (1 + self.transaction_fee)
                self.stock_owned += quantity
        elif 6 <= action <= 10:
            equity = self.cash + self.stock_owned * price_raw
            trade_amount = equity * self.sell_pcts[action - 6]
            quantity = (trade_amount / price_raw) * (1 - self.transaction_fee)
            self.cash += quantity * price_raw * (1 - self.transaction_fee)
            self.stock_owned -= quantity
        elif 11 <= action <= 15:
            leverage = self.leverage_levels[action - 11]
            equity = self.cash + self.stock_owned * price_raw
            buying_power = equity * leverage
            trade_amount = buying_power * self.buy_pcts[action - 11]
            quantity = (trade_amount / price_raw) * (1 - self.transaction_fee)
            self.cash -= quantity * price_raw * (1 + self.transaction_fee)
            self.stock_owned += quantity

        equity = self.cash + self.stock_owned * price_raw
        
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity
        reward = (equity - self.prev_total_asset) / self.init_cash
        reward -= drawdown * 0.5
        
        self.prev_total_asset = equity
        if self.stock_owned != 0 and self.cost_basis == 0:
            self.cost_basis = price_raw

        self.step_idx += 1
        if self.step_idx >= self.max_steps:
            done = True
        if equity < self.total_asset_min:
            done = True
            reward = -1 
        
        return self._get_obs(), reward, done, {}

# Function to plot the investment backtest
def plot_investment_backtest(df, dates, portfolio_values, buy_signals, sell_signals, model_name):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(dates, df['Close_raw'], label='Stock Price', color='skyblue', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(dates, portfolio_values, label='Portfolio Value', color='blue', alpha=0.6)
    ax.scatter(dates, buy_signals, label='Buy Signal', marker='^', color='green', s=100, alpha=1)
    ax.scatter(dates, sell_signals, label='Sell Signal', marker='v', color='red', s=100, alpha=1)
    ax.set_title(f'Investment Backtest: {model_name}', fontsize=20)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel('Stock Price ($)', fontsize=15, color='skyblue')
    ax2.set_ylabel('Portfolio Value ($)', fontsize=15, color='blue')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    
    plot_filename = f'investment_backtest_{model_name}.png'
    plt.savefig(plot_filename)
    print(f"\n✅ Investment backtest plot saved to '{plot_filename}'")
    plt.close(fig)

# ------------------------------
# 4. Main Evaluation Logic
# ------------------------------
# --- CHANGE 2: Set path to the v17 model ---
MODEL_PATH = "ppo_recurrent_stock_trader_v17.zip" 

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    print("Please make sure the MODEL_PATH variable is set correctly.")
else:
    print(f"\n🧪 Starting evaluation for model: {MODEL_PATH}")

    model = RecurrentPPO.load(MODEL_PATH)
    env_test = StockTradingEnv(test_df, scaler)

    obs = env_test.reset()
    lstm_states = None
    done = False
    last_stock_owned = 0
    dates, portfolio_values, buy_signals, sell_signals = [], [], [], []

    while not done:
        # --- CHANGE 3: Correctly handle the action from model.predict ---
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, reward, done, info = env_test.step(action) # Pass the scalar action directly

        step_num = env_test.step_idx - 1
        date = test_df['Date'].iloc[step_num]
        price = test_df['Close_raw'].iloc[step_num]
        
        dates.append(pd.to_datetime(date))
        portfolio_values.append(env_test.cash + env_test.stock_owned * price)
        
        action_idx = action
        current_buy_signal, current_sell_signal = np.nan, np.nan
        
        if 1 <= action_idx <= 5 or 11 <= action_idx <= 15:
            current_buy_signal = price
        elif 6 <= action_idx <= 10:
            current_sell_signal = price
            
        buy_signals.append(current_buy_signal)
        sell_signals.append(current_sell_signal)
        
        last_stock_owned = env_test.stock_owned
        if done:
            break

    model_name = os.path.basename(MODEL_PATH).replace('.zip', '')
    plot_investment_backtest(test_df, dates, portfolio_values, buy_signals, sell_signals, model_name)

    final_price = test_df['Close_raw'].iloc[-1]
    final_portfolio_value = env_test.cash + env_test.stock_owned * final_price
    profit = final_portfolio_value - env_test.init_cash
    profit_pct = (profit / env_test.init_cash) * 100

    print("\n--- FINAL RESULTS ---")
    print(f"Model: {model_name}")
    print(f"Initial Portfolio: ${env_test.init_cash:,.2f}")
    print(f"Final Equity:      ${final_portfolio_value:,.2f}")
    print(f"Profit/Loss:       ${profit:,.2f} ({profit_pct:.2f}%)")
    print("-----------------------")
