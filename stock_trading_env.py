# file: stock_trading_env.py
import gymnasium as gym
import numpy as np

class StockTradingEnv(gym.Env):
    """A stock trading environment for Gymnasium, based on the assignment rules."""
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.lookback_window = 30

        # --- Parameters from the assignment PDF ---
        self.initial_balance = 100000.0 # [cite: 1]
        self.buy_min_pct = 0.24 # [cite: 2]
        self.buy_max_pct = 0.53 # [cite: 2]
        self.sell_min_pct = 0.22 # [cite: 3]
        self.sell_max_pct = 0.73 # [cite: 3]
        self.transaction_fee_pct = 0.03 # [cite: 5]
        self.inactive_session_limit = 25 # [cite: 5]
        self.inactive_penalty = 3981.0 # [cite: 5]
        self.win_asset_threshold = 1000000.0 # [cite: 7]
        self.loss_asset_threshold = 1000.0 # [cite: 6]
        self.loss_cash_threshold = -5000.0 # [cite: 7]

        # Define feature columns and action/observation spaces
        self.feature_columns = ['Open', 'High', 'Low', 'Close', 'SMA_10', 'SMA_30', 'RSI_14']
        self.num_features = len(self.feature_columns)
        self.action_space = gym.spaces.Discrete(3) # 0:Sell, 1:Hold, 2:Buy
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lookback_window * self.num_features + 2,), # +2 for cash, shares
            dtype=np.float64
        )

    def _get_observation(self):
        frame = self.df.iloc[self.current_step - self.lookback_window + 1 : self.current_step + 1]
        features = frame[self.feature_columns].values
        
        # Normalize features
        current_price = features[-1, 3] # Current closing price
        features = features / current_price if current_price != 0 else features
        
        obs = np.concatenate((features.flatten(), [self.cash / self.initial_balance, self.shares_held / 1000]))
        return obs

    def _take_action(self, action):
        reward = 0
        # "Perfect Execution" logic: buy at day's Low, sell at day's High [cite: 13]
        buy_price = self.df['Low'].iloc[self.current_step]
        sell_price = self.df['High'].iloc[self.current_step]

        if action == 2: # Buy
            buy_value = self.cash * self.buy_max_pct
            if buy_value >= (self.cash * self.buy_min_pct): # [cite: 2]
                fee = buy_value * self.transaction_fee_pct
                cash_for_shares = buy_value - fee
                if cash_for_shares > 0 and buy_price > 0:
                    shares_to_buy = cash_for_shares / buy_price
                    self.cash -= buy_value
                    self.shares_held += shares_to_buy
                    self.last_trade_step = self.current_step

        elif action == 0: # Sell
            if self.shares_held > 0:
                shares_to_sell = self.shares_held * self.sell_max_pct
                if shares_to_sell >= (self.shares_held * self.sell_min_pct): # [cite: 3]
                    sell_value = shares_to_sell * sell_price
                    fee = sell_value * self.transaction_fee_pct # [cite: 5]
                    # For simplicity, we define reward as change in cash from selling
                    reward = sell_value - fee
                    self.cash += sell_value - fee
                    self.shares_held -= shares_to_sell
                    self.last_trade_step = self.current_step
        
        # Apply inactivity penalty
        if (self.current_step - self.last_trade_step) >= self.inactive_session_limit: # [cite: 5]
            self.cash -= self.inactive_penalty # [cite: 5]
            reward -= self.inactive_penalty
            self.last_trade_step = self.current_step

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = self.initial_balance
        self.shares_held = 0
        self.current_step = self.lookback_window
        self.last_trade_step = self.current_step
        return self._get_observation(), {}

    def step(self, action):
        # The transaction is completed immediately at the time the order is executed [cite: 4]
        reward = self._take_action(action)
        self.current_step += 1
        
        # Check termination conditions
        total_assets = self.cash + (self.shares_held * self.df['Close'].iloc[self.current_step])
        terminated = False
        if total_assets >= self.win_asset_threshold: # [cite: 7]
            terminated = True
        elif total_assets < self.loss_asset_threshold or self.cash < self.loss_cash_threshold: # [cite: 6, 7]
            terminated = True
        elif self.current_step >= len(self.df) - 1:
            terminated = True
            
        observation = self._get_observation()
        # Gymnasium's step returns 5 values
        return observation, reward, terminated, False, {}

    def render(self, mode='human'):
        total_assets = self.cash + (self.shares_held * self.df['Close'].iloc[self.current_step])
        print(f"Step: {self.current_step}, Assets: ${total_assets:,.2f}, Cash: ${self.cash:,.2f}, Shares: {self.shares_held:,.2f}")