"""
Training script for ETF Portfolio environment with PPO.

Usage:
    python train_etf_portfolio.py

This script:
  1. Generates synthetic OHLCV + macro data (replace with real data)
  2. Computes technical indicators and features
  3. Creates the EtfPortfolioEnv
  4. Trains a PPO agent
  5. Evaluates the trained agent
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from etf_portfolio_env.env import EtfPortfolioEnv
from etf_portfolio_env.features import compute_features
from etf_portfolio_env.sample_data import (
    DEFAULT_ETF_TICKERS,
    generate_synthetic_macro,
    generate_synthetic_ohlcv,
)


class PortfolioLogCallback(BaseCallback):
    """Logs portfolio metrics during training."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            infos = self.locals.get("infos", [])
            if infos:
                latest = infos[-1]
                pv = latest.get("portfolio_value", 0)
                pnl = latest.get("cumulative_pnl", 0)
                tc = latest.get("transaction_cost", 0)
                self.logger.record("portfolio/value", pv)
                self.logger.record("portfolio/cumulative_pnl", pnl)
                self.logger.record("portfolio/transaction_cost", tc)
        return True


def build_env(
    etf_data: dict[str, pd.DataFrame],
    macro_data: pd.DataFrame,
    tickers: list[str],
) -> EtfPortfolioEnv:
    """Build the environment from raw data."""
    # Compute features
    feature_df = compute_features(etf_data, macro_data)

    # Build close price DataFrame aligned to feature dates
    close_frames = []
    for ticker in tickers:
        close_frames.append(etf_data[ticker]["Close"].rename(ticker))
    close_df = pd.concat(close_frames, axis=1)
    close_df = close_df.loc[feature_df.index]

    env = EtfPortfolioEnv(
        feature_df=feature_df,
        close_df=close_df,
        etf_tickers=tickers,
        transaction_cost_bp=5.0,
        max_loss_pct=0.07,
    )
    return env


def main():
    print("=" * 60)
    print("ETF Portfolio PPO Training")
    print("=" * 60)

    # --- 1. Generate data ---
    print("\n[1/5] Generating synthetic data...")
    tickers = DEFAULT_ETF_TICKERS
    etf_data = generate_synthetic_ohlcv(tickers, n_days=500)
    macro_data = generate_synthetic_macro(n_days=500)

    # --- 2. Build environment ---
    print("[2/5] Building environment...")
    env = build_env(etf_data, macro_data, tickers)

    # --- 3. Validate environment ---
    print("[3/5] Validating environment with SB3 check_env...")
    check_env(env, warn=True, skip_render_check=True)
    print("  Environment validation passed!")

    print(f"\n  Observation space: {env.observation_space}")
    print(f"  Observation dim:   {env.observation_space.shape[0]}")
    print(f"    - Market features: {env.n_market_features}")
    print(f"    - Portfolio weights: {env.n_etfs + 1}")
    print(f"    - Cumulative PnL: 1")
    print(f"  Action space:      {env.action_space}")
    print(f"  Usable timesteps:  {env.n_steps_total}")

    # --- 4. Train PPO ---
    print("\n[4/5] Training PPO agent...")
    vec_env = DummyVecEnv([lambda: build_env(etf_data, macro_data, tickers)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=1,
        seed=42,
    )

    callback = PortfolioLogCallback(log_freq=500)
    model.learn(total_timesteps=50_000, callback=callback)
    model.save("ppo_etf_portfolio")
    print("  Model saved to ppo_etf_portfolio.zip")

    # --- 5. Evaluate ---
    print("\n[5/5] Evaluating trained agent...")
    eval_env = build_env(etf_data, macro_data, tickers)
    obs, info = eval_env.reset()

    total_reward = 0.0
    step_count = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

        if step_count % 50 == 0 or done:
            eval_env.render()

    print(f"\n{'=' * 60}")
    print(f"Evaluation Results:")
    print(f"  Total steps:      {step_count}")
    print(f"  Total reward:     {total_reward:+.6f}")
    print(f"  Final PnL:        {info['cumulative_pnl']:+.4%}")
    print(f"  Final value:      {info['portfolio_value']:.4f}")
    print(f"  Final weights:")
    for i, ticker in enumerate(tickers):
        w = info['weights'][i]
        if w > 0.001:
            print(f"    {ticker:20s}: {w:.2%}")
    print(f"    {'CASH':20s}: {info['weights'][-1]:.2%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
