"""
Data loading and pipeline engine setup functionality.

This module provides functions for:
- Registering data bundles
- Building pipeline engines
- Creating data portals
- Getting pricing data
"""

import os
import pandas as pd
from zipline.data import bundles
from zipline.data.data_portal import DataPortal
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.data import USEquityPricing
from zipline.utils.calendars import get_calendar

# Constants
EOD_BUNDLE_NAME = 'eod-quotemedia'


class PricingLoader(object):
    """Custom pricing loader for the pipeline engine."""
    
    def __init__(self, bundle_data):
        self.loader = USEquityPricingLoader(
            bundle_data.equity_daily_bar_reader,
            bundle_data.adjustment_reader)

    def get_loader(self, column):
        if column not in USEquityPricing.columns:
            raise Exception('Column not in USEquityPricing')
        return self.loader


def register_data_bundle(data_path=None):
    """
    Register the EOD data bundle in zipline.
    
    Parameters
    ----------
    data_path : str, optional
        Path to the data directory. If None, uses default project structure.
    
    Returns
    -------
    None
    """
    if data_path is None:
        data_path = os.path.join(os.getcwd(), 'data', 'project_4_eod')
    
    os.environ['ZIPLINE_ROOT'] = data_path
    
    ingest_func = bundles.csvdir.csvdir_equities(['daily'], EOD_BUNDLE_NAME)
    bundles.register(EOD_BUNDLE_NAME, ingest_func)
    
    print('Data Registered')


def build_pipeline_engine(bundle_data, trading_calendar):
    """
    Build a pipeline engine for data processing.
    
    Parameters
    ----------
    bundle_data : BundleData
        The registered bundle data
    trading_calendar : TradingCalendar
        The trading calendar to use
        
    Returns
    -------
    SimplePipelineEngine
        Configured pipeline engine
    """
    pricing_loader = PricingLoader(bundle_data)

    engine = SimplePipelineEngine(
        get_loader=pricing_loader.get_loader,
        calendar=trading_calendar.all_sessions,
        asset_finder=bundle_data.asset_finder)

    return engine


def create_data_portal(bundle_data, trading_calendar):
    """
    Create a data portal for accessing historical data.
    
    Parameters
    ----------
    bundle_data : BundleData
        The registered bundle data
    trading_calendar : TradingCalendar
        The trading calendar to use
        
    Returns
    -------
    DataPortal
        Configured data portal
    """
    data_portal = DataPortal(
        bundle_data.asset_finder,
        trading_calendar=trading_calendar,
        first_trading_day=bundle_data.equity_daily_bar_reader.first_trading_day,
        equity_minute_reader=None,
        equity_daily_reader=bundle_data.equity_daily_bar_reader,
        adjustment_reader=bundle_data.adjustment_reader)
    
    return data_portal


def get_pricing(data_portal, trading_calendar, assets, start_date, end_date, field='close'):
    """
    Get pricing data for specified assets and date range.
    
    Parameters
    ----------
    data_portal : DataPortal
        The data portal to use
    trading_calendar : TradingCalendar
        The trading calendar
    assets : list
        List of asset identifiers
    start_date : datetime
        Start date for the data
    end_date : datetime
        End date for the data
    field : str, default 'close'
        The price field to retrieve
        
    Returns
    -------
    DataFrame
        Pricing data for the specified assets and date range
    """
    end_dt = pd.Timestamp(end_date.strftime('%Y-%m-%d'), tz='UTC', offset='C')
    start_dt = pd.Timestamp(start_date.strftime('%Y-%m-%d'), tz='UTC', offset='C')

    end_loc = trading_calendar.closes.index.get_loc(end_dt)
    start_loc = trading_calendar.closes.index.get_loc(start_dt)

    return data_portal.get_history_window(
        assets=assets,
        end_dt=end_dt,
        bar_count=end_loc - start_loc,
        frequency='1d',
        field=field,
        data_frequency='daily')
