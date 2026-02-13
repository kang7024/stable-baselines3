"""
ETF data preprocessing module.

Applies cleaning rules to handle missing data from different listing dates,
non-trading days, and other data quality issues.

Rules
-----
1. OHLC all NaN → ffill Close, set O/H/L = filled Close, Volume = 0
   (No trading that day, so no price movement)
2. ticker_Bloomberg and name columns → ffill within same parquet
3. O == H == L == C and Volume is NaN → Volume = 0
   (Price unchanged = likely no trading, so volume is 0)
4. H/L/C all NaN (IPO period with no price formation) → drop those rows
   (Simplified condition for detecting unlisted/pre-trading period)
"""

import numpy as np
import pandas as pd


def preprocess_etf(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """
    Apply all preprocessing rules to a single ETF DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
        May also contain 'ticker_Bloomberg' and 'name' columns.
    ticker : str or None
        Ticker name for logging purposes.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame. Rows matching Rule 4 are dropped.
        NaN prices are forward-filled per Rule 1.
    """
    df = df.copy()

    # --- Rule 4: Drop rows where High/Low/Close are ALL NaN ---
    # These are pre-listing rows with no price formation at all.
    hlc_all_nan = df["High"].isna() & df["Low"].isna() & df["Close"].isna()
    n_dropped = hlc_all_nan.sum()
    if n_dropped > 0 and ticker:
        first_valid = df.index[~hlc_all_nan][0] if (~hlc_all_nan).any() else None
        print(f"  [{ticker}] Rule 4: Dropped {n_dropped} pre-listing rows "
              f"(first valid date: {first_valid})")
    df = df[~hlc_all_nan].copy()

    if df.empty:
        return df

    # --- Rule 2: ffill ticker_Bloomberg and name ---
    for col in ["ticker_Bloomberg", "name"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    # --- Rule 1: OHLC all NaN → ffill Close, O/H/L = Close, Volume = 0 ---
    ohlc_cols = ["Open", "High", "Low", "Close"]
    ohlc_all_nan = df[ohlc_cols].isna().all(axis=1)

    if ohlc_all_nan.any():
        # Forward-fill Close first
        df["Close"] = df["Close"].ffill()
        # Where all OHLC were NaN, set O/H/L to the filled Close
        df.loc[ohlc_all_nan, "Open"] = df.loc[ohlc_all_nan, "Close"]
        df.loc[ohlc_all_nan, "High"] = df.loc[ohlc_all_nan, "Close"]
        df.loc[ohlc_all_nan, "Low"] = df.loc[ohlc_all_nan, "Close"]
        df.loc[ohlc_all_nan, "Volume"] = 0.0

    # --- Rule 3: O == H == L == C and Volume is NaN → Volume = 0 ---
    price_unchanged = (
        (df["Open"] == df["High"])
        & (df["High"] == df["Low"])
        & (df["Low"] == df["Close"])
        & df["Volume"].isna()
    )
    df.loc[price_unchanged, "Volume"] = 0.0

    return df


def preprocess_all_etfs(
    etf_data: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Preprocess all ETF DataFrames and build a tradable mask.

    Parameters
    ----------
    etf_data : dict[str, pd.DataFrame]
        Raw ETF data, ticker -> DataFrame with OHLCV.
    common_dates : pd.DatetimeIndex or None
        If provided, reindex all ETFs to these dates.
        If None, use the union of all ETF dates.

    Returns
    -------
    cleaned_data : dict[str, pd.DataFrame]
        Cleaned ETF data.
    tradable_mask : pd.DataFrame
        Boolean DataFrame (dates x tickers). True = ETF is tradable on that date.
        An ETF is tradable if it has valid Close price data after preprocessing.
    """
    tickers = list(etf_data.keys())

    # Determine date range
    if common_dates is None:
        all_dates = set()
        for df in etf_data.values():
            all_dates.update(df.index)
        common_dates = pd.DatetimeIndex(sorted(all_dates))

    # Preprocess each ETF
    cleaned_data = {}
    for ticker in tickers:
        df = etf_data[ticker]
        # Reindex to common dates (introduces NaN for missing dates)
        df = df.reindex(common_dates)
        cleaned = preprocess_etf(df, ticker=ticker)
        cleaned_data[ticker] = cleaned

    # Build tradable mask: True if Close is not NaN after preprocessing
    tradable_mask = pd.DataFrame(index=common_dates, columns=tickers, dtype=bool)
    for ticker in tickers:
        cleaned = cleaned_data[ticker]
        # An ETF is tradable on dates that exist in its cleaned data
        tradable_mask[ticker] = common_dates.isin(cleaned.index)

    # Re-reindex cleaned data to common_dates and forward-fill
    # (so we have price values even for non-trading days — needed for env)
    for ticker in tickers:
        cleaned = cleaned_data[ticker].reindex(common_dates)
        # Only ffill Close for dates AFTER the ETF's first valid date
        first_valid_idx = tradable_mask[ticker].idxmax()
        mask_after_listing = common_dates >= first_valid_idx
        cleaned.loc[mask_after_listing, "Close"] = (
            cleaned.loc[mask_after_listing, "Close"].ffill()
        )
        cleaned.loc[mask_after_listing, "Open"] = (
            cleaned.loc[mask_after_listing, "Open"].ffill()
        )
        cleaned.loc[mask_after_listing, "High"] = (
            cleaned.loc[mask_after_listing, "High"].ffill()
        )
        cleaned.loc[mask_after_listing, "Low"] = (
            cleaned.loc[mask_after_listing, "Low"].ffill()
        )
        cleaned.loc[mask_after_listing, "Volume"] = (
            cleaned.loc[mask_after_listing, "Volume"].fillna(0)
        )
        cleaned_data[ticker] = cleaned

    return cleaned_data, tradable_mask
