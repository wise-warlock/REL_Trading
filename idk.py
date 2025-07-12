import yfinance as yf
import numpy as np
import pandas as pd
import gym
from gym import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
        
        self.action_space = spaces.Discrete(11) # Hold, 5xBuy, 5xSell
        self.buy_pcts = np.linspace(0.24, 0.53, 5)
        self.sell_pcts = np.linspace(0.22, 0.73, 5)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        self.init_cash = 100_000
        self.transaction_fee = 0.03
        self.total_asset_min = 1000

    def reset(self):
        self.step_idx = 0
        self.cash = self.init_cash
        self.stock_owned = 0
        self.cost_basis = 0
        self.prev_total_asset = self.init_cash
        
        # --- CHANGE 1: Add Buy and Hold benchmark tracking ---
        self.buy_and_hold_value = self.init_cash
        self.prev_buy_and_hold_value = self.init_cash
        
        return self._get_obs()

    def _get_obs(self):
        obs_idx = min(self.step_idx, self.max_steps - 1)
        row = self.df.iloc[obs_idx]
        
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
            self.buy_and_hold_value / self.init_cash # Add benchmark to observation
        ], dtype=np.float32)

    def step(self, action):
        done = False
        price_raw = self.df['Close_raw'].iloc[self.step_idx]
        
        # Action 0: HOLD
        if action == 0:
            pass
        # Actions 1-5: BUY
        elif action <= 5:
            pct = self.buy_pcts[action - 1]
            budget = pct * self.cash
            if budget > 1:
                quantity = (budget / price_raw) * (1 - self.transaction_fee)
                cost = quantity * price_raw * (1 + self.transaction_fee)
                new_total_cost = self.cost_basis * self.stock_owned + cost
                self.stock_owned += quantity
                self.cost_basis = new_total_cost / self.stock_owned if self.stock_owned > 0 else 0
                self.cash -= cost
        # Actions 6-10: SELL
        elif action > 5 and self.stock_owned > 0:
            pct = self.sell_pcts[action - 6]
            quantity_to_sell = pct * self.stock_owned
            if quantity_to_sell > 0:
                revenue = quantity_to_sell * price_raw * (1 - self.transaction_fee)
                self.cash += revenue
                self.stock_owned -= quantity_to_sell
                if self.stock_owned < 1e-6:
                    self.stock_owned = 0
                    self.cost_basis = 0
        
        total_asset = self.cash + self.stock_owned * price_raw
        
        # --- CHANGE 2: Implement reward relative to Buy and Hold ---
        # Calculate the benchmark's value at this step
        initial_shares = self.init_cash / self.df['Close_raw'].iloc[0]
        self.buy_and_hold_value = initial_shares * price_raw

        # Calculate the returns for both strategies
        agent_return = (total_asset / self.prev_total_asset) - 1
        benchmark_return = (self.buy_and_hold_value / self.prev_buy_and_hold_value) - 1
        
        # The reward is the agent's outperformance over the benchmark
        reward = agent_return - benchmark_return
        
        self.prev_total_asset = total_asset
        self.prev_buy_and_hold_value = self.buy_and_hold_value

        # --- End Conditions ---
        self.step_idx += 1
        if self.step_idx >= self.max_steps:
            done = True
        
        if total_asset < self.total_asset_min:
            done = True
            reward = -1 # Severe penalty for bankruptcy
        
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
            stock_val = stock_owned * price_raw
            total_asset = cash + stock_val
            
            self.steps.append(self.num_timesteps)
            self.portfolio_values.append(total_asset)
            
            self.ax.cla()
            self.ax.plot(self.steps, self.portfolio_values, 'b-')
            self.ax.set_xlabel("Training Timesteps")
            self.ax.set_ylabel("Portfolio Value ($)")
            self.ax.set_title("Real-Time Training Performance")
            self.ax.grid(True)
            self.ax.ticklabel_format(style='plain', axis='y')
            self.fig.tight_layout()
            plt.draw()
            plt.pause(0.01)

            print(f"\n📊 Step {self.num_timesteps} | Cash: ${cash:,.2f} | Stocks: ${stock_val:,.2f} | Total: ${total_asset:,.2f}")
        return True

    def _on_training_end(self):
        plt.ioff()
        print("\nFinal training plot displayed. Close the plot window to continue.")
        plt.show()

# ------------------------------
# 4. Train PPO Agent
# ------------------------------
env_train = StockTradingEnv(train_df, scaler)

# --- CHANGE 3: Increase network capacity ---
policy_kwargs = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128]) # Wider network
)

model = RecurrentPPO(
    "MlpLstmPolicy",
    env_train,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=3e-5,
    batch_size=128,
    n_steps=4096,
    gamma=0.99,
    tensorboard_log="./tensorboard_logs"
)

callback = LoggerCallback()

print("🚀 Starting model training... A plot window will open.")
model.learn(total_timesteps=1_000_000, progress_bar=False, callback=callback)
model.save("ppo_recurrent_stock_trader_v9")
print("✅ Training complete!")


# ------------------------------
# 5. Evaluate the Trained Agent
# ------------------------------
print("\n🧪 Starting evaluation on test data...")

model = RecurrentPPO.load("ppo_recurrent_stock_trader_v9")
env_test = StockTradingEnv(test_df, scaler)

obs = env_test.reset()
lstm_states = None
done = False

print("\n--- STEP-BY-STEP BACKTEST ---")
print(f"{'Step':<5}{'Date':<12}{'Action':<12}{'Price':<10}{'Cash':<15}{'Stocks':<12}{'Portfolio':<15}")
print("-" * 83)

while not done:
    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
    obs, reward, done, info = env_test.step(action)

    step_num = env_test.step_idx - 1
    date = test_df['Date'].iloc[step_num]
    price = test_df['Close_raw'].iloc[step_num]
    cash = env_test.cash
    stocks_owned = env_test.stock_owned
    portfolio_value = cash + stocks_owned * price

    action_idx = action
    if action_idx == 0:
        action_str = "HOLD"
    elif action_idx <= 5:
        action_str = f"BUY ({env_test.buy_pcts[action_idx-1]*100:.0f}%)"
    else:
        action_str = f"SELL ({env_test.sell_pcts[action_idx-6]*100:.0f}%)"
    
    print(f"{step_num:<5}{date:<12}{action_str:<12}${price:<9.2f}${cash: <14,.2f}{stocks_owned:<12.2f}${portfolio_value: <14,.2f}")
    
    if done:
        print("--- Backtest finished ---")
        break


final_price = test_df['Close_raw'].iloc[-1]
final_portfolio_value = env_test.cash + env_test.stock_owned * final_price
profit = final_portfolio_value - env_test.init_cash
profit_pct = (profit / env_test.init_cash) * 100

print("\n--- FINAL RESULTS ---")
print(f"Initial Portfolio: ${env_test.init_cash:,.2f}")
print(f"Final Portfolio:   ${final_portfolio_value:,.2f}")
print(f"Profit/Loss:       ${profit:,.2f} ({profit_pct:.2f}%)")
print("-----------------------")
