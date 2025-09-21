"""
Alpha factor creation functionality.

This module provides functions for:
- Momentum factors
- Mean reversion factors
- Overnight sentiment factors
- 10K sentiment factors
- Custom factor implementations
"""

import os

import numpy as np
import pandas as pd
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (
    CustomFactor,
    Returns,
    SimpleMovingAverage,
)


class CTO(Returns):
    """
    Computes the overnight return, per hypothesis from
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554010
    """

    inputs = [USEquityPricing.open, USEquityPricing.close]

    def compute(self, today, assets, out, opens, closes):
        """
        The opens and closes matrix is 2 rows x N assets, with the most recent at the bottom.
        As such, opens[-1] is the most recent open, and closes[0] is the earlier close
        """
        out[:] = (opens[-1] - closes[0]) / closes[0]


class TrailingOvernightReturns(Returns):
    """
    Sum of trailing 1m O/N returns
    """

    window_safe = True

    def compute(self, today, asset_ids, out, cto):
        out[:] = np.nansum(cto, axis=0)


def momentum_1yr(window_length, universe, sector):
    """
    Generate momentum 1 year factor.

    Parameters
    ----------
    window_length : int
        Window length for returns calculation
    universe : Zipline Filter
        Universe of stocks filter
    sector : Zipline Classifier
        Sector classifier

    Returns
    -------
    factor : Zipline Factor
        Momentum 1 year factor
    """
    return (
        Returns(window_length=window_length, mask=universe)
        .demean(groupby=sector)
        .rank()
        .zscore()
    )


def mean_reversion_5day_sector_neutral(window_length, universe, sector):
    """
    Generate the mean reversion 5 day sector neutral factor

    Parameters
    ----------
    window_length : int
        Returns window length
    universe : Zipline Filter
        Universe of stocks filter
    sector : Zipline Classifier
        Sector classifier

    Returns
    -------
    factor : Zipline Factor
        Mean reversion 5 day sector neutral factor
    """
    return (
        -Returns(window_length=window_length, mask=universe)
        .demean(groupby=sector)
        .rank()
        .zscore()
    )


def mean_reversion_5day_sector_neutral_smoothed(
    window_length, universe, sector
):
    """
    Generate the mean reversion 5 day sector neutral smoothed factor

    Parameters
    ----------
    window_length : int
        Returns window length
    universe : Zipline Filter
        Universe of stocks filter
    sector : Zipline Classifier
        Sector classifier

    Returns
    -------
    factor : Zipline Factor
        Mean reversion 5 day sector neutral smoothed factor
    """
    factor = mean_reversion_5day_sector_neutral(
        window_length, universe, sector
    )
    factor_smoothed = (
        SimpleMovingAverage(inputs=[factor], window_length=window_length)
        .rank()
        .zscore()
    )

    return factor_smoothed


def overnight_sentiment(
    cto_window_length, trail_overnight_returns_window_length, universe
):
    """
    Generate overnight sentiment factor.

    Parameters
    ----------
    cto_window_length : int
        Window length for CTO calculation
    trail_overnight_returns_window_length : int
        Window length for trailing overnight returns
    universe : Zipline Filter
        Universe of stocks filter

    Returns
    -------
    factor : Zipline Factor
        Overnight sentiment factor
    """
    cto_out = CTO(mask=universe, window_length=cto_window_length)
    return (
        TrailingOvernightReturns(
            inputs=[cto_out],
            window_length=trail_overnight_returns_window_length,
        )
        .rank()
        .zscore()
    )


def overnight_sentiment_smoothed(
    cto_window_length, trail_overnight_returns_window_length, universe
):
    """
    Generate smoothed overnight sentiment factor.

    Parameters
    ----------
    cto_window_length : int
        Window length for CTO calculation
    trail_overnight_returns_window_length : int
        Window length for trailing overnight returns
    universe : Zipline Filter
        Universe of stocks filter

    Returns
    -------
    factor : Zipline Factor
        Smoothed overnight sentiment factor
    """
    unsmoothed_factor = overnight_sentiment(
        cto_window_length, trail_overnight_returns_window_length, universe
    )
    return (
        SimpleMovingAverage(
            inputs=[unsmoothed_factor],
            window_length=trail_overnight_returns_window_length,
        )
        .rank()
        .zscore()
    )


class SentimentFactor(CustomFactor):
    """
    Base class for 10K sentiment-based factors.

    This factor uses pre-computed sentiment data from 10K filings
    to generate alpha signals based on sentiment changes over time.
    """

    def __init__(
        self,
        sentiment_data_path=None,
        sentiment_type="cosine_similarity",
        **kwargs,
    ):
        """
        Initialize the sentiment factor.

        Parameters
        ----------
        sentiment_data_path : str, optional
            Path to the sentiment data file
        sentiment_type : str, default 'cosine_similarity'
            Type of sentiment measure to use
        **kwargs
            Additional parameters for CustomFactor
        """
        self.sentiment_data_path = sentiment_data_path
        self.sentiment_type = sentiment_type
        self.sentiment_data = None
        super().__init__(**kwargs)

    def _load_sentiment_data(self):
        """Load sentiment data from file."""
        if self.sentiment_data is None and self.sentiment_data_path:
            if os.path.exists(self.sentiment_data_path):
                self.sentiment_data = pd.read_csv(
                    self.sentiment_data_path, index_col=0, parse_dates=True
                )
            else:
                print(
                    f"Warning: Sentiment data file not found at "
                    f"{self.sentiment_data_path}"
                )
                self.sentiment_data = pd.DataFrame()

    def compute(self, today, assets, out, *args):
        """
        Compute sentiment factor values.

        Parameters
        ----------
        today : datetime
            Current date
        assets : array
            Asset identifiers
        out : array
            Output array to fill with factor values
        *args
            Additional inputs (not used for sentiment factors)
        """
        self._load_sentiment_data()

        if self.sentiment_data.empty:
            out[:] = np.nan
            return

        # Get sentiment values for current date
        try:
            sentiment_values = self.sentiment_data.loc[today]
            # Map sentiment values to assets
            for i, asset in enumerate(assets):
                if asset in sentiment_values.index:
                    out[i] = sentiment_values[asset]
                else:
                    out[i] = np.nan
        except KeyError:
            # No sentiment data for this date
            out[:] = np.nan


def create_sentiment_factors(sentiment_data_path, universe):
    """
    Create sentiment-based alpha factors from 10K analysis.

    Parameters
    ----------
    sentiment_data_path : str
        Path to the sentiment data file
    universe : Zipline Filter
        Universe of stocks filter

    Returns
    -------
    dict
        Dictionary of sentiment factors
    """
    sentiment_factors = {}

    # Define sentiment types from the 10K analysis
    sentiment_types = [
        "negative",
        "positive",
        "uncertainty",
        "litigious",
        "constraining",
        "interesting",
    ]

    for sentiment_type in sentiment_types:
        factor_name = f"sentiment_{sentiment_type}"
        sentiment_factors[factor_name] = SentimentFactor(
            sentiment_data_path=sentiment_data_path,
            sentiment_type=sentiment_type,
            mask=universe,
            window_length=1,
        )

    return sentiment_factors


def load_and_process_10k_sentiment_data(raw_fillings_path, output_path):
    """
    Load and process 10K sentiment data from raw filings.

    This function processes the sentiment analysis from the 10K notebook
    and creates a standardized format for use in the pipeline.

    Parameters
    ----------
    raw_fillings_path : str
        Path to the raw 10K filings data
    output_path : str
        Path to save the processed sentiment data

    Returns
    -------
    pd.DataFrame
        Processed sentiment data with dates as index and tickers as columns
    """
    from .sentiment_processor import SentimentProcessor

    # Create sentiment processor
    processor = SentimentProcessor()

    # Process sentiment data
    sentiment_data = processor.process_sentiment_data(
        raw_fillings_path, output_path
    )

    return sentiment_data


def scrape_and_process_10k_sentiment_data(
    tickers, output_path, years_back=5, delay=0.1, cache_dir=None
):
    """
    Scrape 10-K filings and process them for sentiment analysis.

    This function scrapes 10-K filings from SEC EDGAR and processes them
    for sentiment analysis, creating a standardized format for the pipeline.

    Parameters
    ----------
    tickers : list
        List of stock ticker symbols to scrape
    output_path : str
        Path to save the processed sentiment data
    years_back : int, default 5
        Number of years back to scrape
    delay : float, default 0.1
        Delay between requests in seconds
    cache_dir : str, optional
        Directory to cache scraped filings

    Returns
    -------
    pd.DataFrame
        Processed sentiment data with dates as index and tickers as columns
    """
    from .sentiment_processor import SentimentProcessor

    # Create sentiment processor with cache directory
    processor = SentimentProcessor(cache_dir=cache_dir)

    # Scrape and process sentiment data
    sentiment_data = processor.scrape_and_process_sentiment(
        tickers=tickers,
        years_back=years_back,
        delay=delay,
        output_path=output_path,
    )

    return sentiment_data
