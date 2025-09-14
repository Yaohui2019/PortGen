"""
Main module for quantitative portfolio generation.

This module provides a high-level interface for running the complete
portfolio generation pipeline including:
- Data loading and setup
- Risk model creation
- Alpha factor generation
- Machine learning model training
- Portfolio optimization
"""

import os

import numpy as np
import pandas as pd
from zipline.data import bundles
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import (
    AnnualizedVolatility,
    AverageDollarVolume,
    CustomFactor,
    DailyReturns,
    Returns,
    SimpleMovingAverage,
)
from zipline.utils.calendars import get_calendar

from .data import (
    build_pipeline_engine,
    create_data_portal,
    get_pricing,
    register_data_bundle,
)
from .factors import (
    mean_reversion_5day_sector_neutral,
    mean_reversion_5day_sector_neutral_smoothed,
    momentum_1yr,
    overnight_sentiment,
    overnight_sentiment_smoothed,
)
from .ml import NoOverlapVoter, sharpe_ratio, train_valid_test_split
from .optimization import (
    OptimalHoldings,
    OptimalHoldingsRegualization,
    OptimalHoldingsStrictFactor,
)
from .risk import (
    factor_betas,
    factor_cov_matrix,
    factor_returns,
    fit_pca,
    idiosyncratic_var_matrix,
    idiosyncratic_var_vector,
    predict_portfolio_risk,
)
from .utils import (
    Sector,
    get_alpha_vector,
    get_factor_exposures,
    show_sample_results,
)


class MarketDispersion(CustomFactor):
    """Market dispersion factor."""

    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, returns):
        # returns are days in rows, assets across columns
        out[:] = np.sqrt(np.nanmean((returns - np.nanmean(returns)) ** 2))


class MarketVolatility(CustomFactor):
    """Market volatility factor."""

    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, returns):
        mkt_returns = np.nanmean(returns, axis=1)
        out[:] = np.sqrt(
            260.0
            * np.nanmean((mkt_returns - np.nanmean(mkt_returns)) ** 2)
        )


class PortfolioGenerator:
    """Main class for quantitative portfolio generation."""

    def __init__(self, data_path=None):
        """
        Initialize the portfolio generator.

        Parameters
        ----------
        data_path : str, optional
            Path to the data directory
        """
        self.data_path = data_path
        self.bundle_data = None
        self.trading_calendar = None
        self.engine = None
        self.data_portal = None
        self.universe = None
        self.sector = None
        self.risk_model = None
        self.all_factors = None
        self.classifier = None

    def setup_data(self):
        """Setup data bundle, calendar, and pipeline engine."""
        print("Setting up data...")

        # Register data bundle
        register_data_bundle(self.data_path)

        # Get trading calendar and bundle data
        self.trading_calendar = get_calendar("NYSE")
        self.bundle_data = bundles.load("eod-quotemedia")

        # Build pipeline engine and data portal
        self.engine = build_pipeline_engine(
            self.bundle_data, self.trading_calendar
        )
        self.data_portal = create_data_portal(
            self.bundle_data, self.trading_calendar
        )

        # Setup universe and sector
        self.universe = AverageDollarVolume(window_length=120).top(500)
        self.sector = Sector()

        print("Data setup complete.")

    def build_risk_model(
        self, universe_end_date, num_factor_exposures=20, ann_factor=252
    ):
        """
        Build the statistical risk model using PCA.

        Parameters
        ----------
        universe_end_date : datetime
            End date for universe selection
        num_factor_exposures : int, default 20
            Number of factors for PCA
        ann_factor : int, default 252
            Annualization factor
        """
        print("Building risk model...")

        # Get universe tickers
        universe_tickers = (
            self.engine.run_pipeline(
                Pipeline(screen=self.universe),
                universe_end_date,
                universe_end_date,
            )
            .index.get_level_values(1)
            .values.tolist()
        )

        # Get 5-year returns data
        five_year_returns = (
            get_pricing(
                self.data_portal,
                self.trading_calendar,
                universe_tickers,
                universe_end_date - pd.DateOffset(years=5),
                universe_end_date,
            )
            .pct_change()[1:]
            .fillna(0)
        )

        # Fit PCA model
        pca = fit_pca(five_year_returns, num_factor_exposures, "full")

        # Build risk model components
        self.risk_model = {
            "factor_betas": factor_betas(
                pca,
                five_year_returns.columns.values,
                np.arange(num_factor_exposures),
            ),
            "factor_returns": factor_returns(
                pca,
                five_year_returns,
                five_year_returns.index,
                np.arange(num_factor_exposures),
            ),
            "factor_cov_matrix": factor_cov_matrix(
                factor_returns(
                    pca,
                    five_year_returns,
                    five_year_returns.index,
                    np.arange(num_factor_exposures),
                ),
                ann_factor,
            ),
            "idiosyncratic_var_matrix": idiosyncratic_var_matrix(
                five_year_returns,
                factor_returns(
                    pca,
                    five_year_returns,
                    five_year_returns.index,
                    np.arange(num_factor_exposures),
                ),
                factor_betas(
                    pca,
                    five_year_returns.columns.values,
                    np.arange(num_factor_exposures),
                ),
                ann_factor,
            ),
            "idiosyncratic_var_vector": idiosyncratic_var_vector(
                five_year_returns,
                idiosyncratic_var_matrix(
                    five_year_returns,
                    factor_returns(
                        pca,
                        five_year_returns,
                        five_year_returns.index,
                        np.arange(num_factor_exposures),
                    ),
                    factor_betas(
                        pca,
                        five_year_returns.columns.values,
                        np.arange(num_factor_exposures),
                    ),
                    ann_factor,
                ),
            ),
        }

        print("Risk model complete.")

    def generate_alpha_factors(self, factor_start_date, universe_end_date):
        """
        Generate alpha factors using the pipeline.

        Parameters
        ----------
        factor_start_date : datetime
            Start date for factor generation
        universe_end_date : datetime
            End date for factor generation
        """
        print("Generating alpha factors...")

        # Create pipeline with all factors
        pipeline = Pipeline(screen=self.universe)

        # Add alpha factors
        pipeline.add(
            momentum_1yr(252, self.universe, self.sector), "Momentum_1YR"
        )
        pipeline.add(
            mean_reversion_5day_sector_neutral(
                5, self.universe, self.sector
            ),
            "Mean_Reversion_5Day_Sector_Neutral",
        )
        pipeline.add(
            mean_reversion_5day_sector_neutral_smoothed(
                5, self.universe, self.sector
            ),
            "Mean_Reversion_5Day_Sector_Neutral_Smoothed",
        )
        pipeline.add(
            overnight_sentiment(2, 5, self.universe), "Overnight_Sentiment"
        )
        pipeline.add(
            overnight_sentiment_smoothed(2, 5, self.universe),
            "Overnight_Sentiment_Smoothed",
        )

        # Add universal quant features
        pipeline.add(
            AnnualizedVolatility(window_length=20, mask=self.universe)
            .rank()
            .zscore(),
            "volatility_20d",
        )
        pipeline.add(
            AnnualizedVolatility(window_length=120, mask=self.universe)
            .rank()
            .zscore(),
            "volatility_120d",
        )
        pipeline.add(
            AverageDollarVolume(window_length=20, mask=self.universe)
            .rank()
            .zscore(),
            "adv_20d",
        )
        pipeline.add(
            AverageDollarVolume(window_length=120, mask=self.universe)
            .rank()
            .zscore(),
            "adv_120d",
        )
        pipeline.add(self.sector, "sector_code")

        # Add regime features
        pipeline.add(
            SimpleMovingAverage(
                inputs=[MarketDispersion(mask=self.universe)],
                window_length=20,
            ),
            "dispersion_20d",
        )
        pipeline.add(
            SimpleMovingAverage(
                inputs=[MarketDispersion(mask=self.universe)],
                window_length=120,
            ),
            "dispersion_120d",
        )
        pipeline.add(MarketVolatility(window_length=20), "market_vol_20d")
        pipeline.add(
            MarketVolatility(window_length=120), "market_vol_120d"
        )

        # Add target
        pipeline.add(
            Returns(window_length=5, mask=self.universe).quantiles(2),
            "return_5d",
        )
        pipeline.add(
            Returns(window_length=5, mask=self.universe).quantiles(25),
            "return_5d_p",
        )

        # Run pipeline
        self.all_factors = self.engine.run_pipeline(
            pipeline, factor_start_date, universe_end_date
        )

        # Add date features
        self._add_date_features(factor_start_date, universe_end_date)

        # Add sector one-hot encoding
        self._add_sector_features()

        # Add target with shift
        self.all_factors["target"] = self.all_factors.groupby(level=1)[
            "return_5d"
        ].shift(-5)

        print("Alpha factors generated.")

    def _add_date_features(self, factor_start_date, universe_end_date):
        """Add date-based features."""
        self.all_factors["is_January"] = (
            self.all_factors.index.get_level_values(0).month == 1
        )
        self.all_factors["is_December"] = (
            self.all_factors.index.get_level_values(0).month == 12
        )
        self.all_factors["weekday"] = (
            self.all_factors.index.get_level_values(0).weekday
        )
        self.all_factors["quarter"] = (
            self.all_factors.index.get_level_values(0).quarter
        )
        self.all_factors["qtr_yr"] = (
            self.all_factors.quarter.astype("str")
            + "_"
            + self.all_factors.index.get_level_values(0).year.astype("str")
        )
        self.all_factors[
            "month_end"
        ] = self.all_factors.index.get_level_values(0).isin(
            pd.date_range(
                start=factor_start_date, end=universe_end_date, freq="BM"
            )
        )
        self.all_factors[
            "month_start"
        ] = self.all_factors.index.get_level_values(0).isin(
            pd.date_range(
                start=factor_start_date, end=universe_end_date, freq="BMS"
            )
        )
        self.all_factors[
            "qtr_end"
        ] = self.all_factors.index.get_level_values(0).isin(
            pd.date_range(
                start=factor_start_date, end=universe_end_date, freq="BQ"
            )
        )
        self.all_factors[
            "qtr_start"
        ] = self.all_factors.index.get_level_values(0).isin(
            pd.date_range(
                start=factor_start_date, end=universe_end_date, freq="BQS"
            )
        )

    def _add_sector_features(self):
        """Add sector one-hot encoding."""
        sector_lookup = pd.read_csv(
            os.path.join(
                os.getcwd(), "data", "project_7_sector", "labels.csv"
            ),
            index_col="Sector_i",
        )["Sector"].to_dict()

        sector_columns = []
        for sector_i, sector_name in sector_lookup.items():
            sector_column = "sector_{}".format(sector_name)
            sector_columns.append(sector_column)
            self.all_factors[sector_column] = (
                self.all_factors["sector_code"] == sector_i
            )

    def train_ml_model(self, n_trees=500, clf_random_state=0):
        """
        Train the machine learning model.

        Parameters
        ----------
        n_trees : int, default 500
            Number of trees in the ensemble
        clf_random_state : int, default 0
            Random state for reproducibility
        """
        print("Training ML model...")

        # Define features and target
        features = [
            "Mean_Reversion_5Day_Sector_Neutral_Smoothed",
            "Momentum_1YR",
            "Overnight_Sentiment_Smoothed",
            "adv_120d",
            "adv_20d",
            "dispersion_120d",
            "dispersion_20d",
            "market_vol_120d",
            "market_vol_20d",
            "volatility_20d",
            "is_January",
            "is_December",
            "weekday",
            "month_end",
            "month_start",
            "qtr_end",
            "qtr_start",
        ]

        # Add sector columns
        sector_columns = [
            col
            for col in self.all_factors.columns
            if col.startswith("sector_")
        ]
        features.extend(sector_columns)

        target_label = "target"

        # Prepare data
        temp = self.all_factors.dropna().copy()
        X = temp[features]
        y = temp[target_label]

        # Split data
        X_train, X_valid, X_test, y_train, y_valid, y_test = (
            train_valid_test_split(X, y, 0.6, 0.2, 0.2)
        )

        # Train model
        from sklearn.ensemble import RandomForestClassifier

        clf_parameters = {
            "criterion": "entropy",
            "min_samples_leaf": 500 * 10,  # n_stocks * n_days
            "oob_score": True,
            "n_jobs": -1,
            "random_state": clf_random_state,
        }

        clf = RandomForestClassifier(n_trees, **clf_parameters)
        self.classifier = NoOverlapVoter(clf)
        self.classifier.fit(
            pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid])
        )

        print("ML model training complete.")

    def optimize_portfolio(
        self, universe_end_date, optimization_type="regularized", **kwargs
    ):
        """
        Optimize portfolio weights.

        Parameters
        ----------
        universe_end_date : datetime
            End date for optimization
        optimization_type : str, default 'regularized'
            Type of optimization ('basic', 'regularized', 'strict_factor')
        **kwargs
            Additional parameters for optimization

        Returns
        -------
        DataFrame
            Optimal portfolio weights
        """
        print(f"Optimizing portfolio using {optimization_type} method...")

        # Get alpha vector
        factor_names = [
            "Mean_Reversion_5Day_Sector_Neutral_Smoothed",
            "Momentum_1YR",
            "Overnight_Sentiment_Smoothed",
            "adv_120d",
            "volatility_20d",
        ]

        # Get test data for alpha vector
        features = [
            "Mean_Reversion_5Day_Sector_Neutral_Smoothed",
            "Momentum_1YR",
            "Overnight_Sentiment_Smoothed",
            "adv_120d",
            "adv_20d",
            "dispersion_120d",
            "dispersion_20d",
            "market_vol_120d",
            "market_vol_20d",
            "volatility_20d",
            "is_January",
            "is_December",
            "weekday",
            "month_end",
            "month_start",
            "qtr_end",
            "qtr_start",
        ]

        sector_columns = [
            col
            for col in self.all_factors.columns
            if col.startswith("sector_")
        ]
        features.extend(sector_columns)

        temp = self.all_factors.dropna().copy()
        X = temp[features]
        y = temp["target"]

        _, _, X_test, _, _, _ = train_valid_test_split(X, y, 0.6, 0.2, 0.2)

        # Get pricing data
        all_assets = self.all_factors.index.levels[1].values.tolist()
        all_pricing = get_pricing(
            self.data_portal,
            self.trading_calendar,
            all_assets,
            universe_end_date - pd.DateOffset(years=2, days=2),
            universe_end_date,
        )

        alpha_vector = get_alpha_vector(
            self.all_factors,
            X_test,
            self.classifier,
            factor_names,
            all_pricing,
        )

        # Choose optimization method
        if optimization_type == "basic":
            optimizer = OptimalHoldings(**kwargs)
        elif optimization_type == "regularized":
            optimizer = OptimalHoldingsRegualization(**kwargs)
        elif optimization_type == "strict_factor":
            optimizer = OptimalHoldingsStrictFactor(**kwargs)
        else:
            raise ValueError(
                f"Unknown optimization type: {optimization_type}"
            )

        # Find optimal weights
        optimal_weights = optimizer.find(
            alpha_vector,
            self.risk_model["factor_betas"],
            self.risk_model["factor_cov_matrix"],
            self.risk_model["idiosyncratic_var_vector"],
        )

        print("Portfolio optimization complete.")
        return optimal_weights

    def run_full_pipeline(self, universe_end_date, **kwargs):
        """
        Run the complete portfolio generation pipeline.

        Parameters
        ----------
        universe_end_date : datetime
            End date for the pipeline
        **kwargs
            Additional parameters for various pipeline components

        Returns
        -------
        dict
            Results including optimal weights and model performance
        """
        print("Starting full portfolio generation pipeline...")

        # Setup
        self.setup_data()

        # Build risk model
        self.build_risk_model(universe_end_date)

        # Generate alpha factors
        factor_start_date = universe_end_date - pd.DateOffset(
            years=2, days=2
        )
        self.generate_alpha_factors(factor_start_date, universe_end_date)

        # Train ML model
        self.train_ml_model()

        # Optimize portfolio
        optimal_weights = self.optimize_portfolio(
            universe_end_date, **kwargs
        )

        print("Full pipeline complete!")

        return {
            "optimal_weights": optimal_weights,
            "risk_model": self.risk_model,
            "classifier": self.classifier,
            "all_factors": self.all_factors,
        }
