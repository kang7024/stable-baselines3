"""
Feature engineering for ETF portfolio environment.

Computes technical indicators, log returns, and volatilities
from raw OHLCV data for each ETF.
"""

import numpy as np
import pandas as pd


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=window, adjust=False).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line and signal line."""
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line


def _bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger Band %B indicator (position within bands)."""
    sma = _sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pct_b


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (normalized by close)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, min_periods=window).mean()
    return atr / close  # Normalize by close price


def compute_features(etf_data: dict[str, pd.DataFrame],
                     macro_data: pd.DataFrame | None = None,
                     fill_unlisted: bool = True) -> pd.DataFrame:
    """
    Compute all features from raw ETF OHLCV data and optional macro data.

    Supports ETFs with different listing dates. Features for unlisted periods
    are filled with 0.0 (the tradable_mask in the env tells the agent which
    ETFs are real vs padded).

    Parameters
    ----------
    etf_data : dict[str, pd.DataFrame]
        Dictionary mapping ETF ticker -> DataFrame with columns
        ['Open', 'High', 'Low', 'Close', 'Volume'] and a DatetimeIndex.
    macro_data : pd.DataFrame or None
        DataFrame with macro indicators (e.g., put-call ratio, exchange rate)
        indexed by date. Columns are used as-is.
    fill_unlisted : bool
        If True, fill NaN features (from unlisted/warmup periods) with 0.0
        and only drop rows where ALL ETF features are NaN.
        If False, drop all rows with any NaN (original behavior, requires
        all ETFs to share the same date range).

    Returns
    -------
    pd.DataFrame
        Combined feature DataFrame indexed by date.
    """
    all_features = []

    for ticker, df in etf_data.items():
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        feat = pd.DataFrame(index=df.index)

        # --- Log returns ---
        for period in [1, 5, 20, 60]:
            feat[f"{ticker}_logret_{period}d"] = np.log(close / close.shift(period))

        # --- Realized volatility (std of 1-day log returns over window) ---
        log_ret_1d = np.log(close / close.shift(1))
        for period in [1, 5, 20, 60]:
            if period == 1:
                # For 1-day "volatility", use absolute return as proxy
                feat[f"{ticker}_vol_1d"] = log_ret_1d.abs()
            else:
                feat[f"{ticker}_vol_{period}d"] = log_ret_1d.rolling(
                    window=period, min_periods=period
                ).std() * np.sqrt(252)  # Annualized

        # --- Technical indicators ---
        # RSI-14
        feat[f"{ticker}_rsi_14"] = _rsi(close, 14) / 100.0  # Normalize to [0,1]

        # MACD histogram (normalized by close)
        macd_line, signal_line = _macd(close)
        feat[f"{ticker}_macd_hist"] = (macd_line - signal_line) / close

        # Bollinger %B
        feat[f"{ticker}_bb_pctb"] = _bollinger_bands(close, 20, 2.0)

        # ATR (already normalized)
        feat[f"{ticker}_atr_14"] = _atr(high, low, close, 14)

        # Volume ratio (current / 20-day avg)
        vol_ma20 = _sma(volume.astype(float), 20)
        feat[f"{ticker}_vol_ratio"] = volume / vol_ma20.replace(0, np.nan)

        # SMA cross signal: (close - SMA20) / close
        feat[f"{ticker}_sma20_dist"] = (close - _sma(close, 20)) / close

        all_features.append(feat)

    features = pd.concat(all_features, axis=1)

    # Add macro data if provided
    if macro_data is not None:
        features = features.join(macro_data, how="left")
        # Forward-fill macro data (may have different frequency)
        for col in macro_data.columns:
            features[col] = features[col].ffill()

    if fill_unlisted:
        # Drop rows where ALL features are NaN (no ETF has data at all)
        features = features.dropna(how="all")
        # Drop initial rows where macro data hasn't started yet
        if macro_data is not None:
            macro_cols = [c for c in macro_data.columns if c in features.columns]
            if macro_cols:
                features = features.dropna(subset=macro_cols)
        # Fill remaining NaN with 0.0 (unlisted ETF features + rolling warmup)
        features = features.fillna(0.0)
    else:
        # Original behavior: drop all rows with any NaN
        features = features.dropna()

    return features
