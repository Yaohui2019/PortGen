"""
Utility functions and helpers for portfolio generation.

This module provides functionality for:
- Plotting and visualization
- Feature importance ranking
- Factor analysis utilities
- General helper functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alphalens as al
import graphviz
from IPython.display import Image
from sklearn.tree import export_graphviz
from zipline.assets._assets import Equity  # Required for USEquityPricing
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.classifiers import Classifier
from zipline.utils.numpy_utils import int64_dtype

# Constants
EOD_BUNDLE_NAME = 'eod-quotemedia'


class Sector(Classifier):
    """Sector classifier for pipeline."""
    dtype = int64_dtype
    window_length = 0
    inputs = ()
    missing_value = -1

    def __init__(self):
        self.data = np.load('../../data/project_4_sector/data.npy')

    def _compute(self, arrays, dates, assets, mask):
        return np.where(
            mask,
            self.data[assets],
            self.missing_value,
        )


def plot_tree_classifier(clf, feature_names=None):
    """Plot a decision tree classifier."""
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        rotate=True)

    return Image(graphviz.Source(dot_data).pipe(format='png'))


def plot(xs, ys, labels, title='', x_label='', y_label=''):
    """Plot multiple lines with labels."""
    for x, y, label in zip(xs, ys, labels):
        plt.ylim((0.5, 0.55))
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.show()


def rank_features_by_importance(importances, feature_names):
    """Rank features by their importance scores."""
    indices = np.argsort(importances)[::-1]
    max_feature_name_length = max([len(feature) for feature in feature_names])

    print('      Feature{space: <{padding}}      Importance'.format(
        padding=max_feature_name_length - 8, space=' '))

    for x_train_i in range(len(importances)):
        print('{number:>2}. {feature: <{padding}} ({importance})'.format(
            number=x_train_i + 1,
            padding=max_feature_name_length,
            feature=feature_names[indices[x_train_i]],
            importance=importances[indices[x_train_i]]))


def get_factor_exposures(factor_betas, weights):
    """Calculate factor exposures for given weights."""
    return factor_betas.loc[weights.index].T.dot(weights)


def get_factor_returns(factor_data):
    """Calculate factor returns from factor data."""
    ls_factor_returns = pd.DataFrame()

    for factor, factor_data in factor_data.items():
        ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[:, 0]

    return ls_factor_returns


def plot_factor_returns(factor_returns):
    """Plot cumulative factor returns."""
    (1 + factor_returns).cumprod().plot(ylim=(0.8, 1.2))


def plot_factor_rank_autocorrelation(factor_data):
    """Plot factor rank autocorrelation."""
    ls_FRA = pd.DataFrame()

    unixt_factor_data = {
        factor: factor_data.set_index(pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in factor_data.index.values],
            names=['date', 'asset']))
        for factor, factor_data in factor_data.items()}

    for factor, factor_data in unixt_factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)

    ls_FRA.plot(title="Factor Rank Autocorrelation", ylim=(0.8, 1.0))


def build_factor_data(factor_data, pricing):
    """Build factor data for alphalens analysis."""
    return {
        factor_name: al.utils.get_clean_factor_and_forward_returns(factor=data, prices=pricing, periods=[1])
        for factor_name, data in factor_data.iteritems()
    }


def show_sample_results(data, samples, classifier, factors, pricing=None):
    """Show sample results for a classifier."""
    # Calculate the Alpha Score
    prob_array = [-1, 1]
    alpha_score = classifier.predict_proba(samples).dot(np.array(prob_array))
    
    # Add Alpha Score to rest of the factors
    alpha_score_label = 'AI_ALPHA'
    factors_with_alpha = data.loc[samples.index].copy()
    factors_with_alpha[alpha_score_label] = alpha_score
    
    # Setup data for AlphaLens
    print('Cleaning Data...\n')
    factor_data = build_factor_data(factors_with_alpha[factors + [alpha_score_label]], pricing)
    print('\n-----------------------\n')
    
    # Calculate Factor Returns and Sharpe Ratio
    factor_returns = get_factor_returns(factor_data)
    sharpe_ratio = factor_returns.mean() / factor_returns.std() * np.sqrt(252)
    
    # Show Results
    print('             Sharpe Ratios')
    print(sharpe_ratio.round(2))
    plot_factor_returns(factor_returns)
    plot_factor_rank_autocorrelation(factor_data)


def get_alpha_vector(data, samples, classifier, factors, pricing=None):
    """Get alpha vector for the last date."""
    # Calculate the Alpha Score
    prob_array = [-1, 1]
    alpha_score = classifier.predict_proba(samples).dot(np.array(prob_array))
    
    # Add Alpha Score to rest of the factors
    alpha_score_label = 'AI_ALPHA'
    all_factors = data.loc[samples.index].copy()
    all_factors[alpha_score_label] = alpha_score
    alphas = all_factors[[alpha_score_label]]
    # Get the last date
    alpha_vector = alphas.loc[all_factors.index.get_level_values(0)[-1]]
    return alpha_vector

