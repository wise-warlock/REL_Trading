# import pandas as pd
# from stock_trading_env import StockTradingEnv
# from model import model

# df = pd.read_csv("stock.csv")
# env = StockTradingEnv(df)

# obs = env.reset()
# done = False

# # while not done:
# #     action = env.action_space.sample()  # Replace with model prediction later
# #     obs, reward, done, _ = env.step(action)
# #     print(f"Day {env.day} | Action {action} | Reward: {reward:.2f} | Cash: {env.cash:.2f} | Stock: {env.stock:.2f}")

# while not done:
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     print(f"Reward: {reward}, Cash: {env.get_attr('cash')[0]}, Stock: {env.get_attr('stock')[0]}")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_trading_env import StockTradingEnv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load stock data
df = pd.read_csv("stock.csv")

# Wrap the environment
env = DummyVecEnv([lambda: StockTradingEnv(df)])

# Load the trained model
model = PPO.load("ppo_stock_trading")

# Run evaluation
obs = env.reset()
done = False

asset_log = []
cash_log = []
action_log = []

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    r = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
    cash = env.get_attr("cash")[0]
    stock = env.get_attr("stock")[0]
    price = obs[0][2]
    total_assets = cash + stock * price

    print(f"Action: {action}, Reward: {r:.5f}, Cash: {cash:.2f}, Stock: {stock:.2f}, Total Assets: {total_assets:.2f}")

    asset_log.append(total_assets)
    cash_log.append(cash)
    action_log.append(action)

# Plot total assets over time
plt.plot(asset_log, label="Total Assets")
plt.plot(cash_log, label="Cash")
plt.legend()
plt.title("Asset & Cash Over Time")
plt.grid()
plt.show()

# plt.hist(action_log, bins=3, rwidth=0.7)
# plt.xticks([0, 1, 2], ["Hold", "Buy", "Sell"])
# plt.title("Action Distribution")
# plt.show()