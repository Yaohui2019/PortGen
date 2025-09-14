"""
Machine learning module for portfolio generation.

This module provides functionality for:
- Random forest models
- Ensemble methods
- Non-overlapping sample handling
- Model evaluation and validation
"""

from .models import (
    train_valid_test_split,
    non_overlapping_samples,
    bagging_classifier,
    calculate_oob_score,
    non_overlapping_estimators,
    NoOverlapVoter,
    NoOverlapVoterAbstract,
    sharpe_ratio
)

__all__ = [
    'train_valid_test_split',
    'non_overlapping_samples',
    'bagging_classifier',
    'calculate_oob_score',
    'non_overlapping_estimators',
    'NoOverlapVoter',
    'NoOverlapVoterAbstract',
    'sharpe_ratio'
]

