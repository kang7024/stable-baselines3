"""
ETF Portfolio Trading Environment for Stable-Baselines3.

An agent manages a portfolio of 11 ETFs + cash using PPO.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class EtfPortfolioEnv(gym.Env):
    """
    Custom Gymnasium environment for ETF portfolio allocation.

    Observation
    -----------
    A flat vector consisting of:
      - Market features: technical indicators, log returns, volatilities,
        and macro indicators for each ETF at the current timestep.
      - Current portfolio weights: 11 ETF weights + 1 cash weight (12 values).
      - Current cumulative P&L ratio (1 value).

    Action
    ------
    Continuous vector of 11 values in [0, 0.3] representing target weights
    for each ETF. Cash weight = 1 - sum(ETF weights).

    Reward
    ------
    Portfolio return from day t+1 close to day t+2 close, minus transaction
    cost of 5bp on absolute weight changes.

    Episode Termination
    -------------------
    - Cumulative P&L drops below -7% → terminated with -0.07 penalty.
    - Reached end of data → truncated.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Pre-computed feature matrix (from features.compute_features).
        Index is dates, columns are feature names.
    close_df : pd.DataFrame
        Close prices for all 11 ETFs. Columns are ticker names.
        Must share the same date index as feature_df.
    etf_tickers : list[str]
        List of 11 ETF ticker names (must match close_df columns).
    transaction_cost_bp : float
        Transaction cost in basis points (default: 5).
    max_loss_pct : float
        Maximum cumulative loss before early termination (default: 0.07).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        feature_df: pd.DataFrame,
        close_df: pd.DataFrame,
        etf_tickers: list[str],
        transaction_cost_bp: float = 5.0,
        max_loss_pct: float = 0.07,
    ):
        super().__init__()

        assert len(etf_tickers) == 11, "Must have exactly 11 ETFs"
        assert set(etf_tickers).issubset(set(close_df.columns)), \
            "All ETF tickers must be in close_df columns"

        self.etf_tickers = etf_tickers
        self.n_etfs = len(etf_tickers)
        self.tc_rate = transaction_cost_bp / 10_000.0  # Convert bp to decimal
        self.max_loss_pct = max_loss_pct

        # Align dates between features and close prices
        common_dates = feature_df.index.intersection(close_df.index)
        common_dates = common_dates.sort_values()
        self.feature_df = feature_df.loc[common_dates]
        self.close_df = close_df.loc[common_dates][etf_tickers]

        # We need close at t+1 and t+2 for reward, so usable range is [0, T-3]
        self.n_steps_total = len(common_dates) - 2
        assert self.n_steps_total > 0, "Not enough data (need at least 3 dates)"

        # Pre-compute close price arrays for fast access
        self.close_array = self.close_df.values.astype(np.float64)  # (T, 11)
        self.feature_array = self.feature_df.values.astype(np.float32)  # (T, n_features)

        self.n_market_features = self.feature_array.shape[1]
        # Observation: market features + portfolio weights (12) + cumulative PnL (1)
        self.n_obs = self.n_market_features + self.n_etfs + 1 + 1

        # --- Spaces ---
        # Action: 11 ETF weights, each in [0, 0.3]
        self.action_space = spaces.Box(
            low=0.0, high=0.3, shape=(self.n_etfs,), dtype=np.float32
        )

        # Observation: all values are roughly normalized, but use generous bounds
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_obs,), dtype=np.float32
        )

        # Internal state
        self._current_step = 0
        self._weights = np.zeros(self.n_etfs + 1, dtype=np.float64)  # 11 ETFs + cash
        self._weights[-1] = 1.0  # Start fully in cash
        self._cumulative_pnl = 0.0
        self._portfolio_value = 1.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._current_step = 0
        self._weights = np.zeros(self.n_etfs + 1, dtype=np.float64)
        self._weights[-1] = 1.0  # Fully in cash
        self._cumulative_pnl = 0.0
        self._portfolio_value = 1.0

        obs = self._get_obs()
        info = {"portfolio_value": self._portfolio_value, "weights": self._weights.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action.astype(np.float64), 0.0, 0.3)

        # Ensure total ETF weight doesn't exceed 1
        total_etf_weight = action.sum()
        if total_etf_weight > 1.0:
            action = action / total_etf_weight  # Scale down proportionally
            total_etf_weight = 1.0
        cash_weight = 1.0 - total_etf_weight

        new_weights = np.append(action, cash_weight)

        # --- Transaction cost ---
        weight_change = np.abs(new_weights - self._weights)
        turnover = weight_change.sum()
        tc = turnover * self.tc_rate

        # Update weights to new allocation
        old_weights = self._weights.copy()
        self._weights = new_weights

        # --- Compute portfolio return ---
        # Action decided at step t → uses close[t+1] to close[t+2]
        t = self._current_step
        close_t1 = self.close_array[t + 1]  # Close at t+1
        close_t2 = self.close_array[t + 2]  # Close at t+2

        # Per-ETF returns from t+1 to t+2
        etf_returns = (close_t2 - close_t1) / close_t1  # (11,)
        # Cash return is 0
        asset_returns = np.append(etf_returns, 0.0)  # (12,)

        # Portfolio return
        portfolio_return = np.dot(self._weights, asset_returns)

        # --- Reward ---
        reward = portfolio_return - tc

        # Update cumulative PnL
        self._portfolio_value *= (1.0 + portfolio_return - tc)
        self._cumulative_pnl = self._portfolio_value - 1.0

        # --- Update weights after market moves (drift) ---
        # Weights change due to relative price changes
        drifted_values = self._weights * (1.0 + asset_returns)
        total_value = drifted_values.sum()
        if total_value > 0:
            self._weights = drifted_values / total_value

        # --- Check termination ---
        terminated = False
        if self._cumulative_pnl < -self.max_loss_pct:
            terminated = True
            reward -= self.max_loss_pct  # Additional penalty

        # --- Check truncation (end of data) ---
        self._current_step += 1
        truncated = self._current_step >= self.n_steps_total

        obs = self._get_obs()
        info = {
            "portfolio_value": self._portfolio_value,
            "cumulative_pnl": self._cumulative_pnl,
            "weights": self._weights.copy(),
            "turnover": turnover,
            "transaction_cost": tc,
            "portfolio_return": portfolio_return,
        }

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Build observation vector for the current step."""
        t = self._current_step
        market_features = self.feature_array[t]  # (n_market_features,)
        portfolio_weights = self._weights.astype(np.float32)  # (12,)
        pnl = np.array([self._cumulative_pnl], dtype=np.float32)  # (1,)

        obs = np.concatenate([market_features, portfolio_weights, pnl])
        return obs

    def render(self, mode="human"):
        print(
            f"Step {self._current_step}/{self.n_steps_total} | "
            f"PnL: {self._cumulative_pnl:+.4f} | "
            f"Value: {self._portfolio_value:.4f} | "
            f"Cash: {self._weights[-1]:.2%}"
        )
