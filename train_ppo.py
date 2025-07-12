from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_trading_env import StockTradingEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd

class RandomActionWarmup(BaseCallback):
    def __init__(self, warmup_steps=10000, verbose=1):
        super().__init__(verbose)
        self.warmup_steps = warmup_steps

    def _on_step(self) -> bool:
        if self.num_timesteps < self.warmup_steps:
            self.locals["actions"][:] = [self.training_env.action_space.sample()]
        return True

# Load your stock data
df = pd.read_csv("stock.csv")

# Wrap the environment in DummyVecEnv (required for stable-baselines3)
env = DummyVecEnv([lambda: StockTradingEnv(df)])

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard_logs")

# Train the model
model.learn(total_timesteps=100_000, callback=RandomActionWarmup(10_000))

# Save the model
model.save("ppo_stock_trading")
print("Training complete and model saved.")
