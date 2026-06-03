#!/usr/bin/env python3
"""
This module provides a function to standardize tabular data
using Scikit-learn's preprocessing tools.
"""
from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes a tabular dataset.

    Arguments:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)

    Returns:
        numpy.ndarray: The standardized version of the input data.
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
