"""
ETF data preprocessing module.

Applies cleaning rules to handle missing data from non-trading days
and other data quality issues in Bloomberg-sourced OHLCV data.

Pre-listing rows are NOT dropped here — they are kept so that the
feature layer can encode them as (0, 1) flag vectors.

Rules (applied in order)
------------------------
1. OHLC all NaN → ffill Close (targeted via temp column), O/H/L = Close, Volume = 0
2. Close exists but O/H/L NaN → O/H/L = Close
3. O == H == L == C and Volume is NaN → Volume = 0
"""

import numpy as np
import pandas as pd


def preprocess_etf(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """
    Apply all preprocessing rules to a single ETF DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame. Expected columns: Open, High, Low, Close, Volume.
        Handles cases where some columns may be missing.
    ticker : str or None
        Ticker name for logging purposes.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame. Pre-listing rows are kept (not dropped).
        NaN prices are filled per the rules above where applicable.
    """
    df = df.copy()
    ohlc_cols = ["Open", "High", "Low", "Close"]

    if df.empty:
        return df

    # --- Prepare: temp column for targeted Close ffill ---
    if "Close" in df.columns:
        df["__Close_ffilled_temp__"] = df["Close"].ffill()

    # --- Rule 1: OHLC all NaN → ffill Close, O/H/L = Close, Volume = 0 ---
    actual_ohlc_cols = [c for c in ohlc_cols if c in df.columns]
    target_rows_mask = pd.Series(False, index=df.index)

    if actual_ohlc_cols:
        target_rows_mask = df[actual_ohlc_cols].isnull().all(axis=1)

    if target_rows_mask.any():
        # 1. Fill Close with ffilled temp value
        if "Close" in df.columns:
            df.loc[target_rows_mask, "Close"] = df.loc[
                target_rows_mask, "__Close_ffilled_temp__"
            ]
        # 2. Set O/H/L to the filled Close
        if "Close" in df.columns:
            for col in ["Open", "High", "Low"]:
                if col in df.columns:
                    df.loc[target_rows_mask, col] = df.loc[
                        target_rows_mask, "Close"
                    ]
        # 3. Volume = 0 (regardless of existing value)
        if "Volume" in df.columns:
            df.loc[target_rows_mask, "Volume"] = 0.0

    # Clean up temp column
    if "__Close_ffilled_temp__" in df.columns:
        df.drop(columns=["__Close_ffilled_temp__"], inplace=True)

    # --- Rule 2: Close exists but O/H/L NaN → O/H/L = Close ---
    if "Close" in df.columns:
        for col in ["Open", "High", "Low"]:
            if col in df.columns:
                mask = df[col].isnull() & df["Close"].notnull()
                df.loc[mask, col] = df.loc[mask, "Close"]

    # --- Rule 3: O == H == L == C and Volume is NaN → Volume = 0 ---
    required_cols = ohlc_cols + ["Volume"]
    if all(col in df.columns for col in required_cols):
        cond_ohlc_equal = (
            (df["Open"] == df["High"])
            & (df["High"] == df["Low"])
            & (df["Low"] == df["Close"])
        )
        cond_volume_null = df["Volume"].isnull()
        cond_rule3 = cond_ohlc_equal & cond_volume_null
        if cond_rule3.any():
            df.loc[cond_rule3, "Volume"] = 0.0

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

    # Build tradable mask: True if Close is not NaN after rule-based cleaning
    # (before the blanket ffill below — so pre-listing NaN → tradable=False)
    tradable_mask = pd.DataFrame(index=common_dates, columns=tickers, dtype=bool)
    for ticker in tickers:
        cleaned = cleaned_data[ticker]
        tradable_mask[ticker] = cleaned["Close"].notna()

    # Re-reindex cleaned data to common_dates and forward-fill
    # (so we have price values even for non-trading days — needed for env)
    for ticker in tickers:
        cleaned = cleaned_data[ticker].reindex(common_dates)
        # Only ffill for dates AFTER the ETF's first valid date
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
