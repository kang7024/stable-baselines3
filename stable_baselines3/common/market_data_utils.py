"""Utilities for preprocessing market data (e.g., from yfinance)."""

from typing import Dict, List, Optional

import pandas as pd


def multiindex_to_dict(
    df: pd.DataFrame,
    ohlc_columns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Convert a MultiIndex-column DataFrame into a dict of per-ticker DataFrames.

    Typical input shape (e.g. from ``yf.download(['AAPL', 'GOOG'], ...)``):

    .. code-block:: text

                   Open              High             Close
                   AAPL    GOOG      AAPL    GOOG     AAPL    GOOG
        Date
        2024-01-02  ...    ...       ...     ...      ...     ...

    Output::

        {
            "AAPL": pd.DataFrame(columns=["Open", "High", "Close", ...]),
            "GOOG": pd.DataFrame(columns=["Open", "High", "Close", ...]),
        }

    :param df: DataFrame with 2-level MultiIndex columns.
        Level 0 = price fields (e.g. Open, High, Low, Close, Volume).
        Level 1 = ticker symbols.
    :param ohlc_columns: If provided, only keep these level-0 columns.
        Defaults to ``None`` (keep all columns).
    :return: ``{ticker: DataFrame}`` dictionary.
    :raises ValueError: If the DataFrame columns are not a 2-level MultiIndex.
    """
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
        raise ValueError(
            "Expected a DataFrame with 2-level MultiIndex columns "
            f"(got {type(df.columns).__name__}, nlevels={getattr(df.columns, 'nlevels', 1)})."
        )

    if ohlc_columns is not None:
        # Filter to requested price fields only
        available = df.columns.get_level_values(0).unique()
        missing = set(ohlc_columns) - set(available)
        if missing:
            raise ValueError(f"Columns not found in level 0: {sorted(missing)}. Available: {list(available)}")
        df = df[ohlc_columns]

    tickers = df.columns.get_level_values(1).unique().tolist()

    result: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        ticker_df = df.xs(ticker, level=1, axis=1).copy()
        # Drop rows where all OHLC values are NaN (e.g. non-trading days for that ticker)
        ticker_df.dropna(how="all", inplace=True)
        result[ticker] = ticker_df

    return result
