"""
Alpha factor creation functionality.

This module provides functions for:
- Momentum factors
- Mean reversion factors
- Overnight sentiment factors
- Custom factor implementations
"""

import numpy as np
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import Returns, SimpleMovingAverage


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
