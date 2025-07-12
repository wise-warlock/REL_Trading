import yfinance as yf
import numpy as np
import pandas as pd
import gym
from gym import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# ------------------------------
# 1. Load and Process Data
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


# ------------------------------
# 2. Split Data into Train and Test Sets
# ------------------------------
train_size = int(len(df) * 0.8)
train_df = df[:train_size].reset_index(drop=True)
test_df = df[train_size:].reset_index(drop=True)


# ------------------------------
# 3. Custom Reinforcement Learning Environment
# ------------------------------
class StockTradingEnv(gym.Env):
    def __init__(self, df, scaler):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.scaler = scaler
        self.max_steps = len(df)
        
        self.action_space = spaces.Discrete(16) 
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        # --- Parameters, including the restored inactivity penalty ---
        self.buy_pcts = np.linspace(0.24, 0.53, 5) 
        self.sell_pcts = np.linspace(0.22, 0.73, 5) 
        self.leverage_levels = np.linspace(1.5, 3.0, 5)
        
        self.init_cash = 100_000 
        self.transaction_fee = 0.03 
        self.total_asset_min = 1000 
        # --- CHANGE 1: Re-add inactivity penalty parameters ---
        self.penalty_sessions = 25
        self.penalty_fee = 3981

    def reset(self):
        self.step_idx = 0
        self.cash = self.init_cash
        self.stock_owned = 0 
        self.cost_basis = 0
        self.prev_total_asset = self.init_cash
        self.peak_equity = self.init_cash
        self.buy_and_hold_value = self.init_cash
        self.prev_buy_and_hold_value = self.init_cash
        # --- CHANGE 2: Add tracking for last trade ---
        self.last_trade_step = 0
        return self._get_obs()

    def _get_obs(self):
        obs_idx = min(self.step_idx, self.max_steps - 1)
        row = self.df.iloc[obs_idx]
        equity = self.cash + self.stock_owned * row['Close_raw']
        
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
            self.peak_equity / self.init_cash,
            equity / self.init_cash
        ], dtype=np.float32)

    def step(self, action):
        done = False
        traded = False
        price_raw = self.df['Close_raw'].iloc[self.step_idx]
        equity = self.cash + self.stock_owned * price_raw
        
        current_stage = max(1, int(equity // 100_000))

        # --- ACTION LOGIC ---
        if action == 0: pass
        elif 1 <= action <= 5:
            trade_amount = self.cash * self.buy_pcts[action - 1] if self.cash > 0 else 0
            if trade_amount > 1:
                quantity = (trade_amount / price_raw) * (1 - self.transaction_fee)
                self.cash -= quantity * price_raw * (1 + self.transaction_fee)
                self.stock_owned += quantity
                traded = True
        elif 6 <= action <= 10:
            trade_amount = equity * self.sell_pcts[action - 6]
            quantity = (trade_amount / price_raw) * (1 - self.transaction_fee)
            self.cash += quantity * price_raw * (1 - self.transaction_fee)
            self.stock_owned -= quantity
            traded = True
        elif 11 <= action <= 15:
            leverage = self.leverage_levels[action - 11]
            max_leverage = max(1.0, 3.5 - (current_stage * 0.5))
            leverage = min(leverage, max_leverage)
            buying_power = equity * leverage
            trade_amount = buying_power * self.buy_pcts[action - 11]
            quantity = (trade_amount / price_raw) * (1 - self.transaction_fee)
            self.cash -= quantity * price_raw * (1 + self.transaction_fee)
            self.stock_owned += quantity
            traded = True

        new_equity = self.cash + self.stock_owned * price_raw
        
        # --- REWARD SYSTEM BASED ON STAGE ---
        if current_stage <= 3:
            initial_shares = self.init_cash / self.df['Close_raw'].iloc[0]
            self.buy_and_hold_value = initial_shares * price_raw
            agent_return = (new_equity / self.prev_total_asset) if self.prev_total_asset > 0 else 0
            benchmark_return = (self.buy_and_hold_value / self.prev_buy_and_hold_value) if self.prev_buy_and_hold_value > 0 else 0
            reward = agent_return - benchmark_return
            self.prev_buy_and_hold_value = self.buy_and_hold_value
        else:
            self.peak_equity = max(self.peak_equity, new_equity)
            drawdown = (self.peak_equity - new_equity) / self.peak_equity
            reward = (new_equity - self.prev_total_asset) / self.init_cash
            reward -= drawdown * (0.1 * (current_stage - 2))

        if traded:
            reward -= self.transaction_fee * 0.1 
            self.last_trade_step = self.step_idx
        # --- CHANGE 3: Apply inactivity penalty ---
        elif self.step_idx - self.last_trade_step >= self.penalty_sessions:
            self.cash -= self.penalty_fee
            reward -= self.penalty_fee / self.init_cash

        self.prev_total_asset = new_equity
        if self.stock_owned != 0 and self.cost_basis == 0:
            self.cost_basis = price_raw

        self.step_idx += 1
        if self.step_idx >= self.max_steps: done = True
        if new_equity < self.total_asset_min:
            done = True
            reward = -1 
        
        return self._get_obs(), reward, done, {}

# Callback for logging and plotting
class LoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.last_step = 0
        self.steps = []
        self.portfolio_values = []
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 7))

    def _on_step(self):
        if self.num_timesteps - self.last_step >= 5000:
            self.last_step = self.num_timesteps
            actual_env = self.training_env.envs[0].env
            cash = actual_env.cash
            stock_owned = actual_env.stock_owned
            step_idx = min(actual_env.step_idx, len(actual_env.df) - 1)
            price_raw = actual_env.df['Close_raw'].iloc[step_idx]
            total_asset = cash + stock_owned * price_raw
            
            self.steps.append(self.num_timesteps)
            self.portfolio_values.append(total_asset)
            
            self.ax.cla()
            self.ax.plot(self.steps, self.portfolio_values, 'b-')
            self.ax.set_xlabel("Training Timesteps")
            self.ax.set_ylabel("Portfolio Value ($)")
            self.ax.set_title("Live Training Equity Curve")
            self.ax.grid(True)
            self.ax.ticklabel_format(style='plain', axis='y')
            self.fig.tight_layout()
            plt.draw()
            plt.pause(0.01)

            print(f"\n📊 Step {self.num_timesteps} | Cash: ${cash:,.2f} | Position: {stock_owned:,.2f} shares | Equity: ${total_asset:,.2f}")
        return True

    def _on_training_end(self):
        plt.ioff()
        print("\nFinal training plot displayed. Close the plot window to continue.")
        plt.show()

# Function to plot the investment backtest
def plot_investment_backtest(df, dates, portfolio_values, buy_signals, sell_signals):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(dates, df['Close_raw'], label='Stock Price', color='skyblue', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(dates, portfolio_values, label='Portfolio Value', color='blue', alpha=0.6)
    ax.scatter(dates, buy_signals, label='Buy Signal', marker='^', color='green', s=100, alpha=1)
    ax.scatter(dates, sell_signals, label='Sell Signal', marker='v', color='red', s=100, alpha=1)
    ax.set_title('Investment Backtest Visualization', fontsize=20)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel('Stock Price ($)', fontsize=15, color='skyblue')
    ax2.set_ylabel('Portfolio Value ($)', fontsize=15, color='blue')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    plt.savefig('investment_backtest.png')
    print("\n✅ Investment backtest plot saved to 'investment_backtest.png'")
    plt.close(fig)

# ------------------------------
# 4. Train PPO Agent
# ------------------------------
CHECKPOINT_DIR = './train_checkpoints/'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'ppo_recurrent_stock_trader_checkpoint.zip')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

env_train = StockTradingEnv(train_df, scaler)

logger_callback = LoggerCallback()
checkpoint_callback = CheckpointCallback(
  save_freq=50000,
  save_path=CHECKPOINT_DIR,
  name_prefix='ppo_recurrent_stock_trader_checkpoint'
)
callback_list = CallbackList([logger_callback, checkpoint_callback])

if os.path.exists(CHECKPOINT_PATH):
    print("✅ Checkpoint found! Resuming training...")
    model = RecurrentPPO.load(CHECKPOINT_PATH, env=env_train)
else:
    print("🚀 No checkpoint found. Starting a new training session...")
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env_train,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=2e-5,
        batch_size=256,
        n_steps=4096,
        gamma=0.99,
        ent_coef=0.02,
        tensorboard_log="./tensorboard_logs"
    )

print("🏋️ Starting model training... A plot window will open.")
model.learn(total_timesteps=1_000_000, progress_bar=False, callback=callback_list, reset_num_timesteps=False)

FINAL_MODEL_PATH = "ppo_recurrent_stock_trader_v24_final.zip"
model.save(FINAL_MODEL_PATH)
print(f"✅ Training complete! Final model saved to {FINAL_MODEL_PATH}")


# ------------------------------
# 5. Evaluate the Trained Agent
# ------------------------------
print("\n🧪 Starting evaluation on test data...")

model = RecurrentPPO.load(FINAL_MODEL_PATH)
env_test = StockTradingEnv(test_df, scaler)

obs = env_test.reset()
lstm_states = None
done = False
last_stock_owned = 0
dates, portfolio_values, buy_signals, sell_signals = [], [], [], []

print("\n--- STEP-BY-STEP BACKTEST ---")
print(f"{'Step':<5}{'Date':<12}{'Action':<20}{'Price':<10}{'Cash':<15}{'Position':<15}{'Portfolio':<15}")
print("-" * 103)

while not done:
    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
    obs, reward, done, info = env_test.step(action)

    step_num = env_test.step_idx - 1
    date = test_df['Date'].iloc[step_num]
    price = test_df['Close_raw'].iloc[step_num]
    cash = env_test.cash
    stock_owned = env_test.stock_owned
    portfolio_value = cash + stock_owned * price

    dates.append(pd.to_datetime(date))
    portfolio_values.append(portfolio_value)
    
    action_idx = action
    action_str = "HOLD"
    current_buy_signal, current_sell_signal = np.nan, np.nan
    
    if 1 <= action_idx <= 5 or 11 <= action_idx <= 15:
        action_str = "BUY"
        current_buy_signal = price
    elif 6 <= action_idx <= 10:
        action_str = "SELL"
        current_sell_signal = price
        
    buy_signals.append(current_buy_signal)
    sell_signals.append(current_sell_signal)

    print(f"{step_num:<5}{date:<12}{action_str:<20}${price:<9.2f}${cash: <14,.2f}{stock_owned:<14.2f} ${portfolio_value: <14,.2f}")
    
    last_stock_owned = stock_owned
    if done:
        print("--- Backtest finished ---")
        break

plot_investment_backtest(test_df, dates, portfolio_values, buy_signals, sell_signals)

final_price = test_df['Close_raw'].iloc[-1]
final_portfolio_value = env_test.cash + env_test.stock_owned * final_price
profit = final_portfolio_value - env_test.init_cash
profit_pct = (profit / env_test.init_cash) * 100

print("\n--- FINAL RESULTS ---")
print(f"Initial Portfolio: ${env_test.init_cash:,.2f}")
print(f"Final Equity:      ${final_portfolio_value:,.2f}")
print(f"Profit/Loss:       ${profit:,.2f} ({profit_pct:.2f}%)")
print("-----------------------")
