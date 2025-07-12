# file: stock_trading_env.py
import numpy as np

class StockTradingEnv:
    """A stock trading environment that simulates the market based on the assignment rules."""
    def __init__(self, df):
        self.df = df
        self.lookback_window = 30
        self.current_step = 0

        # Parameters from the assignment PDF
        self.initial_balance = 100000.0 #
        self.buy_min_pct = 0.24 #
        self.buy_max_pct = 0.53 #
        self.sell_min_pct = 0.22 #
        self.sell_max_pct = 0.73 #
        self.transaction_fee_pct = 0.03 #
        self.inactive_session_limit = 25 #
        self.inactive_penalty = 3981.0 #
        self.win_asset_threshold = 1000000.0 #
        self.loss_asset_threshold = 1000.0 #
        self.loss_cash_threshold = -5000.0 #

        self.feature_columns = ['Open', 'High', 'Low', 'Close', 'SMA_10', 'SMA_30', 'RSI_14']
        self.num_features = len(self.feature_columns)
        
        # Observation space shape: (window * features) + portfolio state
        self.state_size = self.lookback_window * self.num_features + 2 # +2 for cash, shares
        self.action_space_size = 3 # 0:Sell, 1:Hold, 2:Buy

    def _get_state(self):
        # Get the historical data for the lookback window
        frame = self.df.iloc[self.current_step - self.lookback_window + 1 : self.current_step + 1]
        features = frame[self.feature_columns].values
        
        # Normalize features for better network performance
        current_price = features[-1, 3] # Current closing price
        features = features / current_price # Normalize by current price

        # Add portfolio status to the state
        state = np.concatenate((features.flatten(), [self.cash / self.initial_balance, self.shares_held / 1000]))
        return state

    def _take_action(self, action):
        action_type = action
        current_price = self.df['Close'].iloc[self.current_step]
        reward = 0

        if action_type == 2: # Buy
            buy_value = self.cash * self.buy_max_pct
            if buy_value >= (self.cash * self.buy_min_pct):
                fee = buy_value * self.transaction_fee_pct
                cash_for_shares = buy_value - fee
                if cash_for_shares > 0 and current_price > 0:
                    shares_to_buy = cash_for_shares / current_price
                    # Update average buy price
                    total_cost = (self.avg_buy_price * self.shares_held) + (current_price * shares_to_buy)
                    self.shares_held += shares_to_buy
                    self.avg_buy_price = total_cost / self.shares_held
                    # Update portfolio
                    self.cash -= buy_value
                    self.last_trade_step = self.current_step

        elif action_type == 0: # Sell
            if self.shares_held > 0:
                shares_to_sell = self.shares_held * self.sell_max_pct
                if shares_to_sell >= (self.shares_held * self.sell_min_pct):
                    sell_value = shares_to_sell * current_price
                    fee = sell_value * self.transaction_fee_pct
                    # Calculate profit for the reward
                    reward = (current_price - self.avg_buy_price) * shares_to_sell - fee
                    # Update portfolio
                    self.cash += sell_value - fee
                    self.shares_held -= shares_to_sell
                    self.avg_buy_price = 0 if self.shares_held == 0 else self.avg_buy_price
                    self.last_trade_step = self.current_step
        
        # Apply inactivity penalty
        if (self.current_step - self.last_trade_step) > self.inactive_session_limit:
            self.cash -= self.inactive_penalty
            reward -= self.inactive_penalty # Make penalty a direct negative reward
            self.last_trade_step = self.current_step

        return reward

    def reset(self):
        self.cash = self.initial_balance
        self.shares_held = 0
        self.avg_buy_price = 0
        self.current_step = self.lookback_window
        self.last_trade_step = self.current_step
        return self._get_state()

    def step(self, action):
        reward = self._take_action(action)
        self.current_step += 1
        
        # Check for termination conditions
        total_assets = self.cash + (self.shares_held * self.df['Close'].iloc[self.current_step])
        done = False
        if total_assets >= self.win_asset_threshold or total_assets < self.loss_asset_threshold or self.cash < self.loss_cash_threshold:
            done = True
        if self.current_step >= len(self.df) - 1:
            done = True
            
        next_state = self._get_state()
        return next_state, reward, done, {}