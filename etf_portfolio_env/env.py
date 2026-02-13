"""
ETF Portfolio Trading Environment for Stable-Baselines3.

An agent manages a portfolio of ETFs + cash.
Supports dynamic asset universes: ETFs that are not yet listed (NaN close price)
are automatically masked out — the agent can only trade currently listed ETFs.
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
      - Tradable mask: binary vector (1=tradable, 0=not listed) per ETF.
      - Current portfolio weights: N ETF weights + 1 cash weight.
      - Current cumulative P&L ratio (1 value).

    Action
    ------
    Continuous vector of (N+1) logits in [-1, 1]: N ETFs + 1 cash.
    softmax(logits / temperature) converts them to weights summing to 1.
    Weights for untradable ETFs are forced to 0 and redistributed to cash.

    Reward
    ------
    Portfolio return from day t+1 close to day t+2 close, minus transaction
    cost on absolute weight changes.

    Episode Termination
    -------------------
    - Cumulative P&L drops below max_loss_pct → terminated with penalty.
    - Reached end of data → truncated.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Pre-computed feature matrix (from features.compute_features).
    close_df : pd.DataFrame
        Close prices for all ETFs. Columns are ticker names.
        NaN values indicate the ETF is not listed on that date.
    etf_tickers : list[str]
        List of ETF ticker names (must match close_df columns).
    tradable_mask : pd.DataFrame or None
        Boolean DataFrame (dates x tickers). True = ETF is tradable.
        If None, inferred from non-NaN close prices in close_df.
    transaction_cost_bp : float
        Transaction cost in basis points (default: 5).
    max_loss_pct : float
        Maximum cumulative loss before early termination (default: 0.07).
    softmax_temperature : float
        Temperature for softmax action mapping (default: 1.0).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        feature_df: pd.DataFrame,
        close_df: pd.DataFrame,
        etf_tickers: list[str],
        tradable_mask: pd.DataFrame | None = None,
        transaction_cost_bp: float = 5.0,
        max_loss_pct: float = 0.07,
        softmax_temperature: float = 1.0,
    ):
        super().__init__()

        assert set(etf_tickers).issubset(set(close_df.columns)), \
            "All ETF tickers must be in close_df columns"

        self.etf_tickers = etf_tickers
        self.n_etfs = len(etf_tickers)
        self.tc_rate = transaction_cost_bp / 10_000.0
        self.max_loss_pct = max_loss_pct
        self.softmax_temperature = softmax_temperature

        # Align dates between features and close prices
        common_dates = feature_df.index.intersection(close_df.index)
        common_dates = common_dates.sort_values()
        self.feature_df = feature_df.loc[common_dates]
        self.close_df = close_df.loc[common_dates][etf_tickers]

        # Build tradable mask
        if tradable_mask is not None:
            self.tradable_mask_df = tradable_mask.loc[common_dates][etf_tickers]
        else:
            # Infer from non-NaN close prices
            self.tradable_mask_df = self.close_df.notna()

        # Forward-fill close prices AFTER building mask
        # (so NaN before listing stays NaN in mask, but we have values for computation)
        self.close_df = self.close_df.ffill().fillna(0.0)

        # We need close at t+1 and t+2 for reward, so usable range is [0, T-3]
        self.n_steps_total = len(common_dates) - 2
        assert self.n_steps_total > 0, "Not enough data (need at least 3 dates)"

        # Pre-compute arrays for fast access
        self.close_array = self.close_df.values.astype(np.float64)     # (T, N)
        self.feature_array = self.feature_df.values.astype(np.float32) # (T, n_features)
        self.tradable_array = self.tradable_mask_df.values.astype(np.float32)  # (T, N)

        self.n_market_features = self.feature_array.shape[1]
        # Observation: market features + tradable mask (N) + portfolio weights (N+1) + PnL (1)
        self.n_obs = self.n_market_features + self.n_etfs + (self.n_etfs + 1) + 1

        # --- Spaces ---
        self.n_assets = self.n_etfs + 1  # N ETFs + cash
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_obs,), dtype=np.float32
        )

        # Internal state
        self._current_step = 0
        self._weights = np.zeros(self.n_assets, dtype=np.float64)
        self._weights[-1] = 1.0  # Start fully in cash
        self._cumulative_pnl = 0.0
        self._portfolio_value = 1.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._current_step = 0
        self._weights = np.zeros(self.n_assets, dtype=np.float64)
        self._weights[-1] = 1.0
        self._cumulative_pnl = 0.0
        self._portfolio_value = 1.0

        obs = self._get_obs()
        info = {
            "portfolio_value": self._portfolio_value,
            "weights": self._weights.copy(),
            "tradable_mask": self._get_tradable_mask(),
        }
        return obs, info

    def _get_tradable_mask(self) -> np.ndarray:
        """Get the tradable mask for the current timestep. Shape: (N_etfs,)"""
        return self.tradable_array[self._current_step]

    def _map_action(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Map raw logits to portfolio weights via masked tempered softmax.

        Untradable ETF logits are set to -inf BEFORE softmax, so they
        naturally get weight 0. This keeps the mapping differentiable
        and avoids confusing the agent with post-hoc weight redistribution.

        Cash logit is never masked (always tradable).

        Returns array of length (N+1), summing to 1.
        """
        tau = self.softmax_temperature
        mask = self._get_tradable_mask()  # (N_etfs,)

        # Set untradable ETF logits to -inf before softmax
        # Cash (last element) is always tradable
        logits = raw_action.copy()
        logits[:self.n_etfs] = np.where(mask > 0, logits[:self.n_etfs], -np.inf)

        # Tempered softmax (subtract max for numerical stability)
        scaled = logits / tau
        scaled -= np.max(scaled[np.isfinite(scaled)])  # max over finite values only
        exp_vals = np.where(np.isfinite(scaled), np.exp(scaled), 0.0)
        weights = exp_vals / exp_vals.sum()

        return weights

    def step(self, action: np.ndarray):
        new_weights = self._map_action(action.astype(np.float64))

        # --- Transaction cost ---
        weight_change = np.abs(new_weights - self._weights)
        turnover = weight_change.sum()
        tc = turnover * self.tc_rate

        # Update weights to new allocation
        self._weights = new_weights

        # --- Compute portfolio return ---
        t = self._current_step
        close_t1 = self.close_array[t + 1]
        close_t2 = self.close_array[t + 2]
        mask = self._get_tradable_mask()

        # Per-ETF returns: only for tradable ETFs (others contribute 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            etf_returns = np.where(
                (mask > 0) & (close_t1 > 0),
                (close_t2 - close_t1) / close_t1,
                0.0,
            )
        asset_returns = np.append(etf_returns, 0.0)  # cash return = 0

        # Portfolio return
        portfolio_return = np.dot(self._weights, asset_returns)

        # --- Reward ---
        reward = portfolio_return - tc

        # Update cumulative PnL
        self._portfolio_value *= (1.0 + portfolio_return - tc)
        self._cumulative_pnl = self._portfolio_value - 1.0

        # --- Update weights after market moves (drift) ---
        drifted_values = self._weights * (1.0 + asset_returns)
        total_value = drifted_values.sum()
        if total_value > 0:
            self._weights = drifted_values / total_value

        # --- Check termination ---
        terminated = False
        if self._cumulative_pnl < -self.max_loss_pct:
            terminated = True
            reward -= self.max_loss_pct

        # --- Check truncation (end of data) ---
        self._current_step += 1
        truncated = self._current_step >= self.n_steps_total

        # After step advance, if ETFs become untradable, force-sell to cash
        if not (terminated or truncated):
            self._enforce_mask_on_drift()

        obs = self._get_obs()
        info = {
            "portfolio_value": self._portfolio_value,
            "cumulative_pnl": self._cumulative_pnl,
            "weights": self._weights.copy(),
            "turnover": turnover,
            "transaction_cost": tc,
            "portfolio_return": portfolio_return,
            "tradable_mask": self._get_tradable_mask(),
            "n_tradable": int(self._get_tradable_mask().sum()),
        }

        return obs, float(reward), terminated, truncated, info

    def _enforce_mask_on_drift(self):
        """
        After stepping to a new date, check if any held ETFs became untradable.
        If so, force their weight to 0 and move to cash (no transaction cost
        since this is a forced liquidation due to delisting/suspension).
        """
        mask = self._get_tradable_mask()
        etf_weights = self._weights[:self.n_etfs]
        forced_to_cash = (etf_weights * (1.0 - mask)).sum()

        if forced_to_cash > 1e-10:
            self._weights[:self.n_etfs] = etf_weights * mask
            self._weights[-1] += forced_to_cash

    def _get_obs(self) -> np.ndarray:
        """Build observation vector for the current step."""
        t = self._current_step
        market_features = self.feature_array[t]              # (n_market_features,)
        tradable_mask = self.tradable_array[t]               # (n_etfs,)
        portfolio_weights = self._weights.astype(np.float32) # (n_assets,)
        pnl = np.array([self._cumulative_pnl], dtype=np.float32)

        obs = np.concatenate([market_features, tradable_mask, portfolio_weights, pnl])
        return obs

    def render(self, mode="human"):
        mask = self._get_tradable_mask()
        n_tradable = int(mask.sum())
        print(
            f"Step {self._current_step}/{self.n_steps_total} | "
            f"PnL: {self._cumulative_pnl:+.4f} | "
            f"Value: {self._portfolio_value:.4f} | "
            f"Cash: {self._weights[-1]:.2%} | "
            f"Tradable: {n_tradable}/{self.n_etfs}"
        )
