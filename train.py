# file: train.py
import pandas as pd
import pandas_ta as ta
from stock_trading_env import StockTradingEnv
from dddqn_agent import DDDQNAgent

# --- 1. Load Data and Add Features ---
try:
    # This code works best with a large dataset (e.g., 5-10 years)
    df = pd.read_csv('stock_best_1y.csv') 
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
except FileNotFoundError:
    print("Error: Data file not found. Please provide a large (5-10 year) stock data CSV.")
    exit()

df.ta.sma(length=10, append=True)
df.ta.sma(length=30, append=True)
df.ta.rsi(length=14, append=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# --- 2. Split Data with Sanity Check ---
split_point = int(len(df) * 0.80)
train_df = df.iloc[:split_point].copy()
test_df = df.iloc[split_point:].copy()

if len(train_df) < 50: # Need enough data for lookback + training
    print("Error: Not enough training data after split. Please use a larger source data file.")
    exit()

# --- 3. Initialize Environment and Agent ---
env = StockTradingEnv(train_df)
agent = DDDQNAgent(state_size=env.state_size, action_space_size=env.action_space_size)

num_episodes = 500
batch_size = 128

# --- 4. Training Loop ---
print("--- Starting DDDQN Training ---")
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    
    for time_step in range(len(train_df) - env.lookback_window): # Run episode for the length of the data
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        if done:
            break
            
    agent.update_target_model()
    total_assets = env.cash + (env.shares_held * train_df['Close'].iloc[env.current_step])
    print(f"Episode: {episode+1}/{num_episodes}, Total Assets: ${total_assets:,.2f}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    if len(agent.memory) > batch_size:
        agent.learn(batch_size)

# --- 5. Save the Final Model ---
agent.model.save("dddqn_stock_trader.h5")
print("\n--- Training Complete. Model Saved. ---")