"""
Sample / synthetic data generator for testing the ETF portfolio environment.

Replace this with real data loading in production.
"""

import numpy as np
import pandas as pd


# 11 Korean ETF tickers (example names)
DEFAULT_ETF_TICKERS = [
    "KODEX200",        # KOSPI 200
    "KODEX_INV",       # KOSPI 200 Inverse
    "TIGER_US500",     # S&P 500
    "KODEX_BOND",      # Korea Treasury Bond
    "TIGER_GOLD",      # Gold
    "KODEX_SEMI",      # Semiconductor
    "TIGER_SECONDARY", # Secondary Battery
    "KODEX_BANK",      # Banking
    "TIGER_PHARMA",    # Pharmaceutical/Bio
    "KODEX_ENERGY",    # Energy/Chemical
    "TIGER_REITS",     # REITs
]


def generate_synthetic_ohlcv(
    tickers: list[str] | None = None,
    n_days: int = 500,
    start_date: str = "2022-01-03",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Generate synthetic OHLCV data for testing.

    Parameters
    ----------
    tickers : list[str]
        ETF ticker names. Defaults to DEFAULT_ETF_TICKERS.
    n_days : int
        Number of trading days to generate.
    start_date : str
        Start date string.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker -> DataFrame with OHLCV columns.
    """
    if tickers is None:
        tickers = DEFAULT_ETF_TICKERS

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)

    etf_data = {}
    for i, ticker in enumerate(tickers):
        # Generate price via geometric Brownian motion
        base_price = 10_000 + i * 5_000  # Different starting prices
        mu = rng.uniform(-0.0001, 0.0003)  # Daily drift
        sigma = rng.uniform(0.008, 0.025)  # Daily volatility

        log_returns = rng.normal(mu, sigma, size=n_days)
        close_prices = base_price * np.exp(np.cumsum(log_returns))

        # Generate OHLV from close
        high_pct = rng.uniform(0.001, 0.02, size=n_days)
        low_pct = rng.uniform(0.001, 0.02, size=n_days)
        open_pct = rng.uniform(-0.005, 0.005, size=n_days)

        high = close_prices * (1 + high_pct)
        low = close_prices * (1 - low_pct)
        open_prices = close_prices * (1 + open_pct)

        # Ensure OHLC consistency
        high = np.maximum(high, np.maximum(open_prices, close_prices))
        low = np.minimum(low, np.minimum(open_prices, close_prices))

        volume = rng.integers(100_000, 10_000_000, size=n_days).astype(float)

        etf_data[ticker] = pd.DataFrame(
            {
                "Open": open_prices,
                "High": high,
                "Low": low,
                "Close": close_prices,
                "Volume": volume,
            },
            index=dates,
        )

    return etf_data


def generate_synthetic_macro(
    n_days: int = 500,
    start_date: str = "2022-01-03",
    seed: int = 123,
) -> pd.DataFrame:
    """
    Generate synthetic macro indicator data for testing.

    Columns:
      - put_call_ratio: Simulated put-call ratio
      - usd_krw: Simulated USD/KRW exchange rate
      - vix: Simulated VIX index
      - us10y_yield: Simulated US 10Y treasury yield

    Parameters
    ----------
    n_days : int
        Number of trading days.
    start_date : str
        Start date string.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Macro indicator DataFrame with DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)

    # Put-call ratio: mean-reverting around 1.0
    pcr = np.cumsum(rng.normal(0, 0.02, n_days)) + 1.0
    pcr = np.clip(pcr, 0.3, 2.5)

    # USD/KRW exchange rate: random walk around 1300
    usd_krw = 1300 + np.cumsum(rng.normal(0, 3, n_days))
    usd_krw = np.clip(usd_krw, 1100, 1500)

    # VIX: mean-reverting around 20
    vix = 20 + np.cumsum(rng.normal(0, 0.5, n_days))
    vix = np.clip(vix, 10, 80)

    # US 10Y yield: random walk around 3.5%
    us10y = 3.5 + np.cumsum(rng.normal(0, 0.02, n_days))
    us10y = np.clip(us10y, 0.5, 7.0)

    macro = pd.DataFrame(
        {
            "put_call_ratio": pcr,
            "usd_krw": usd_krw / 1000.0,  # Normalize to ~1.3
            "vix": vix / 100.0,  # Normalize to ~0.2
            "us10y_yield": us10y / 100.0,  # Normalize to ~0.035
        },
        index=dates,
    )

    return macro
