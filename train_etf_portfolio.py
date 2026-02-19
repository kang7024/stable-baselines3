"""
Training script for ETF Portfolio environment with RL algorithms.

Usage:
    python train_etf_portfolio.py                  # use default config.yaml
    python train_etf_portfolio.py --config my.yaml  # use custom config

This script:
  1. Loads configuration from YAML file
  2. Generates synthetic OHLCV + macro data (replace with real data)
  3. Computes technical indicators and features
  4. Creates the EtfPortfolioEnv
  5. Trains an RL agent (PPO/A2C/SAC)
  6. Evaluates the trained agent
  7. Saves model, logs, config, and evaluation results to a run directory
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from etf_portfolio_env.env import EtfPortfolioEnv
from etf_portfolio_env.features import compute_features
from etf_portfolio_env.sample_data import (
    DEFAULT_ETF_TICKERS,
    generate_synthetic_macro,
    generate_synthetic_ohlcv,
)

ALGO_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
}


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_run_dir(base: str, algo_name: str) -> str:
    """Create a timestamped run directory: runs/<timestamp>_<algo>/"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base, f"{timestamp}_{algo_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


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
    transaction_cost_bp: float = 5.0,
    max_loss_pct: float = 0.07,
    feature_cfg: dict[str, bool] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> EtfPortfolioEnv:
    """Build the environment from raw data, optionally sliced to [start_date, end_date]."""
    feature_df = compute_features(etf_data, macro_data, feature_cfg=feature_cfg)

    close_frames = []
    for ticker in tickers:
        close_frames.append(etf_data[ticker]["Close"].rename(ticker))
    close_df = pd.concat(close_frames, axis=1)
    close_df = close_df.loc[feature_df.index]

    if start_date is not None:
        mask = feature_df.index >= start_date
        feature_df = feature_df.loc[mask]
        close_df = close_df.loc[mask]
    if end_date is not None:
        mask = feature_df.index <= end_date
        feature_df = feature_df.loc[mask]
        close_df = close_df.loc[mask]

    env = EtfPortfolioEnv(
        feature_df=feature_df,
        close_df=close_df,
        etf_tickers=tickers,
        transaction_cost_bp=transaction_cost_bp,
        max_loss_pct=max_loss_pct,
    )
    return env


def make_model(algo_name: str, env, cfg_training: dict, tb_log_dir: str):
    """Create an RL model from config."""
    algo_cls = ALGO_MAP[algo_name]

    # Common params for all algorithms
    common = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg_training["learning_rate"],
        gamma=cfg_training["gamma"],
        verbose=1,
        seed=cfg_training.get("seed", 42),
        tensorboard_log=tb_log_dir,
    )

    if algo_name == "PPO":
        return algo_cls(
            **common,
            n_steps=cfg_training["n_steps"],
            batch_size=cfg_training["batch_size"],
            n_epochs=cfg_training["n_epochs"],
            gae_lambda=cfg_training["gae_lambda"],
            clip_range=cfg_training["clip_range"],
            ent_coef=cfg_training["ent_coef"],
        )
    elif algo_name == "A2C":
        return algo_cls(
            **common,
            n_steps=cfg_training["n_steps"],
            gae_lambda=cfg_training.get("gae_lambda", 0.95),
            ent_coef=cfg_training.get("ent_coef", 0.01),
        )
    elif algo_name == "SAC":
        return algo_cls(
            **common,
            batch_size=cfg_training.get("batch_size", 256),
            learning_starts=cfg_training.get("learning_starts", 1000),
            ent_coef=cfg_training.get("ent_coef", "auto"),
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")


def main():
    parser = argparse.ArgumentParser(description="Train ETF Portfolio RL Agent")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to YAML config file"
    )
    args = parser.parse_args()

    # --- Load config ---
    cfg = load_config(args.config)
    algo_name = cfg["algorithm"].upper()
    cfg_data = cfg["data"]
    cfg_env = cfg["env"]
    cfg_training = cfg["training"]
    cfg_features = cfg.get("features")

    if algo_name not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Choose from: {list(ALGO_MAP.keys())}")

    # Parse date split params
    start_date = cfg_data["start_date"]
    train_end = cfg_data["train_end_date"]
    val_end = cfg_data["val_end_date"]
    test_end = cfg_data["test_end_date"]

    # Val/test start from the next business day after the boundary (no overlap)
    val_start = str((pd.Timestamp(train_end) + pd.offsets.BDay(1)).date())
    test_start = str((pd.Timestamp(val_end) + pd.offsets.BDay(1)).date())

    # --- Create run directory ---
    run_dir = create_run_dir("runs", algo_name)

    # Save config copy to run directory
    shutil.copy2(args.config, os.path.join(run_dir, "config.yaml"))

    tb_log_dir = os.path.join(run_dir, "tensorboard")

    print("=" * 60)
    print(f"ETF Portfolio Training — {algo_name}")
    print(f"Run directory: {run_dir}")
    print(f"Date splits:   train [{start_date} ~ {train_end}]")
    print(f"               val   [{val_start} ~ {val_end}]")
    print(f"               test  [{test_start} ~ {test_end}]")
    print("=" * 60)

    # --- 1. Generate data ---
    print("\n[1/6] Generating synthetic data...")
    tickers = DEFAULT_ETF_TICKERS
    n_days = len(pd.bdate_range(start=start_date, end=test_end))
    etf_data = generate_synthetic_ohlcv(tickers, n_days=n_days, start_date=start_date, seed=cfg_data.get("seed", 42))
    macro_data = generate_synthetic_macro(n_days=n_days, start_date=start_date, seed=cfg_data.get("seed", 123))
    print(f"  Generated {n_days} business days: {start_date} ~ {test_end}")

    # Shared helper to build env for a date window
    def _make_env(sd: str | None, ed: str | None) -> EtfPortfolioEnv:
        return build_env(
            etf_data, macro_data, tickers,
            transaction_cost_bp=cfg_env["transaction_cost_bp"],
            max_loss_pct=cfg_env["max_loss_pct"],
            feature_cfg=cfg_features,
            start_date=sd,
            end_date=ed,
        )

    # --- 2. Build train / val / test environments ---
    print("[2/6] Building train / val / test environments...")
    train_env = _make_env(start_date, train_end)
    val_env = _make_env(val_start, val_end)
    test_env = _make_env(test_start, test_end)
    print(f"  train steps: {train_env.n_steps_total}")
    print(f"  val   steps: {val_env.n_steps_total}")
    print(f"  test  steps: {test_env.n_steps_total}")

    # --- 3. Validate train environment ---
    print("[3/6] Validating train environment with SB3 check_env...")
    check_env(train_env, warn=True, skip_render_check=True)
    print("  Environment validation passed!")

    print(f"\n  Observation space: {train_env.observation_space}")
    print(f"  Observation dim:   {train_env.observation_space.shape[0]}")
    print(f"    - Market features: {train_env.n_market_features}")
    print(f"    - Portfolio weights: {train_env.n_etfs + 1}")
    print(f"    - Cumulative PnL: 1")
    print(f"  Action space:      {train_env.action_space}")

    # --- 4. Train with EvalCallback on val set ---
    print(f"\n[4/6] Training {algo_name} agent (EvalCallback on val set)...")
    vec_train_env = DummyVecEnv([lambda: _make_env(start_date, train_end)])
    vec_val_env = DummyVecEnv([lambda: _make_env(val_start, val_end)])

    eval_callback = EvalCallback(
        vec_val_env,
        best_model_save_path=run_dir,
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=cfg_training.get("eval_freq", 5000),
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )

    model = make_model(algo_name, vec_train_env, cfg_training, tb_log_dir)

    log_callback = PortfolioLogCallback(log_freq=cfg_training.get("log_freq", 500))
    model.learn(
        total_timesteps=cfg_training["total_timesteps"],
        callback=[log_callback, eval_callback],
    )

    # Save final (last) model in addition to best_model saved by EvalCallback
    final_model_path = os.path.join(run_dir, "model")
    model.save(final_model_path)
    print(f"  Final model saved to {final_model_path}.zip")

    # --- 5. Load best model (saved by EvalCallback on val reward) ---
    print("\n[5/6] Loading best model from validation...")
    best_model_path = os.path.join(run_dir, "best_model")
    if os.path.exists(best_model_path + ".zip"):
        model = ALGO_MAP[algo_name].load(best_model_path)
        print(f"  Loaded best_model.zip")
    else:
        print("  best_model.zip not found — using final model")

    # --- 6. Evaluate on test set ---
    print("\n[6/6] Evaluating best model on test set...")
    obs, info = test_env.reset()

    total_reward = 0.0
    step_count = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

        if step_count % 50 == 0 or done:
            test_env.render()

    # Build evaluation results
    eval_results = {
        "algorithm": algo_name,
        "split": {
            "start_date": start_date,
            "train_end_date": train_end,
            "val_end_date": val_end,
            "test_end_date": test_end,
        },
        "total_steps": step_count,
        "total_reward": float(total_reward),
        "final_pnl": float(info["cumulative_pnl"]),
        "final_value": float(info["portfolio_value"]),
        "final_weights": {
            ticker: float(info["weights"][i])
            for i, ticker in enumerate(tickers)
        },
        "final_cash": float(info["weights"][-1]),
        "config": cfg,
    }

    # Save evaluation results
    eval_path = os.path.join(run_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Test Set Evaluation Results:")
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
    print(f"\n  Results saved to: {eval_path}")
    print(f"  TensorBoard:      tensorboard --logdir {tb_log_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
