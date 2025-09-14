"""
Machine learning models and ensemble methods.

This module provides functionality for:
- Data splitting for time series
- Non-overlapping sample handling
- Bagging classifiers
- Custom ensemble methods
- Model evaluation
"""

import abc
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch


def train_valid_test_split(all_x, all_y, train_size, valid_size, test_size):
    """
    Generate the train, validation, and test dataset.

    Parameters
    ----------
    all_x : DataFrame
        All the input samples
    all_y : Pandas Series
        All the target values
    train_size : float
        The proportion of the data used for the training dataset
    valid_size : float
        The proportion of the data used for the validation dataset
    test_size : float
        The proportion of the data used for the test dataset

    Returns
    -------
    x_train : DataFrame
        The train input samples
    x_valid : DataFrame
        The validation input samples
    x_test : DataFrame
        The test input samples
    y_train : Pandas Series
        The train target values
    y_valid : Pandas Series
        The validation target values
    y_test : Pandas Series
        The test target values
    """
    assert train_size >= 0 and train_size <= 1.0
    assert valid_size >= 0 and valid_size <= 1.0
    assert test_size >= 0 and test_size <= 1.0
    assert train_size + valid_size + test_size == 1.0
    
    indexes = all_x.index.levels[0].tolist()
    train_end = int(len(indexes) * train_size)
    valid_end = train_end + int(len(indexes) * valid_size)
    
    X_train = all_x.loc[indexes[:train_end]]
    X_valid = all_x.loc[indexes[train_end:valid_end]]
    X_test = all_x.loc[indexes[valid_end:]]
    
    y_train = all_y.loc[indexes[:train_end]]
    y_valid = all_y.loc[indexes[train_end:valid_end]]
    y_test = all_y.loc[indexes[valid_end:]]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def non_overlapping_samples(x, y, n_skip_samples, start_i=0):
    """
    Get the non overlapping samples.

    Parameters
    ----------
    x : DataFrame
        The input samples
    y : Pandas Series
        The target values
    n_skip_samples : int
        The number of samples to skip
    start_i : int
        The starting index to use for the data
    
    Returns
    -------
    non_overlapping_x : 2 dimensional Ndarray
        The non overlapping input samples
    non_overlapping_y : 1 dimensional Ndarray
        The non overlapping target values
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    
    idx = x.index.get_level_values(level=0).unique().tolist()[start_i::n_skip_samples+1]
    non_overlapping_x = x.loc[idx]
    non_overlapping_y = y.loc[idx]
    return non_overlapping_x, non_overlapping_y


def bagging_classifier(n_estimators, max_samples, max_features, parameters):
    """
    Build the bagging classifier.

    Parameters
    ----------
    n_estimators : int 
        The number of base estimators in the ensemble
    max_samples : float 
        The proportion of input samples drawn from when training each base estimator
    max_features : float 
        The proportion of input sample features drawn from when training each base estimator
    parameters : dict
        Parameters to use in building the bagging classifier
        It should contain the following parameters:
            criterion
            min_samples_leaf
            oob_score
            n_jobs
            random_state
    
    Returns
    -------
    bagging_clf : Scikit-Learn BaggingClassifier
        The bagging classifier
    """
    required_parameters = {'criterion', 'min_samples_leaf', 'oob_score', 'n_jobs', 'random_state'}
    assert not required_parameters - set(parameters.keys())
    
    base_clf = DecisionTreeClassifier(
        criterion=parameters['criterion'],
        max_features=max_features,
        min_samples_leaf=parameters['min_samples_leaf']
    )
    bagging_clf = BaggingClassifier(
        base_estimator=base_clf,
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=True,
        oob_score=parameters['oob_score'],
        n_jobs=parameters['n_jobs'],
        verbose=0,
        random_state=parameters['random_state']
    )
    return bagging_clf


def calculate_oob_score(classifiers):
    """
    Calculate the mean out-of-bag score from the classifiers.

    Parameters
    ----------
    classifiers : list of Scikit-Learn Classifiers
        The classifiers used to calculate the mean out-of-bag score
    
    Returns
    -------
    oob_score : float
        The mean out-of-bag score
    """
    oob_scores = [classifier.oob_score_ for classifier in classifiers]
    return np.mean(oob_scores)


def non_overlapping_estimators(x, y, classifiers, n_skip_samples):
    """
    Fit the classifiers to non overlapping data.

    Parameters
    ----------
    x : DataFrame
        The input samples
    y : Pandas Series
        The target values
    classifiers : list of Scikit-Learn Classifiers
        The classifiers used to fit on the non overlapping data
    n_skip_samples : int
        The number of samples to skip
    
    Returns
    -------
    fit_classifiers : list of Scikit-Learn Classifiers
        The classifiers fit to the the non overlapping data
    """
    assert len(classifiers) <= n_skip_samples + 1
    model_datas = [
        [classifiers[i], non_overlapping_samples(x, y, n_skip_samples, i)] 
        for i in range(len(classifiers))
    ]
    fit_classifiers = [
        model_data[0].fit(model_data[1][0], model_data[1][1]) 
        for model_data in model_datas
    ]
    return fit_classifiers


class NoOverlapVoterAbstract(VotingClassifier):
    """Abstract base class for non-overlapping voter ensemble."""
    
    @abc.abstractmethod
    def _calculate_oob_score(self, classifiers):
        raise NotImplementedError
        
    @abc.abstractmethod
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        raise NotImplementedError
    
    def __init__(self, estimator, voting='soft', n_skip_samples=4):
        # List of estimators for all the subsets of data
        estimators = [('clf'+str(i), estimator) for i in range(n_skip_samples + 1)]
        
        self.n_skip_samples = n_skip_samples
        super().__init__(estimators, voting)
    
    def fit(self, X, y, sample_weight=None):
        estimator_names, clfs = zip(*self.estimators)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        
        clone_clfs = [clone(clf) for clf in clfs]
        self.estimators_ = self._non_overlapping_estimators(X, y, clone_clfs, self.n_skip_samples)
        self.named_estimators_ = Bunch(**dict(zip(estimator_names, self.estimators_)))
        self.oob_score_ = self._calculate_oob_score(self.estimators_)
        
        return self


class NoOverlapVoter(NoOverlapVoterAbstract):
    """Non-overlapping voter ensemble implementation."""
    
    def _calculate_oob_score(self, classifiers):
        return calculate_oob_score(classifiers)
        
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        return non_overlapping_estimators(x, y, classifiers, n_skip_samples)


def sharpe_ratio(factor_returns, annualization_factor):
    """
    Get the sharpe ratio for each factor for the entire period

    Parameters
    ----------
    factor_returns : DataFrame
        Factor returns for each factor and date
    annualization_factor: float
        Annualization Factor

    Returns
    -------
    sharpe_ratio : Pandas Series of floats
        Sharpe ratio
    """
    df_sharpe = pd.Series(annualization_factor * factor_returns.mean() / factor_returns.std())
    return df_sharpe

