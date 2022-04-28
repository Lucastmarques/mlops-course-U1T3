"""Auxiliary functions to manipulate a ML Model

This module assists the main module "mission155solutions_refactored.py"
with functions dedicated to create and test models.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def fit_knn(train_dataframe: pd.DataFrame, train_cols: list, target_cols: list,
            n_neighbors=5):
    """ Create and fit a KNN Regressor model
    Args:
        train_dataframe (pd.DataFrame): Train dataset.
        train_cols (list): Columns of train dataset that will be used as
        knn model input.
        target_cols (list): Columns of train dataset that will be used as
        knn model output.
        n_neighbors (Optional[int]): Number of KNN model neighbors.
    Returns:
        (KNeighborsRegressor): Trained model.
    """
    new_knn = KNeighborsRegressor(n_neighbors= n_neighbors)

    # Fit a KNN model using default k value.
    new_knn.fit(train_dataframe[train_cols], train_dataframe[target_cols])

    return new_knn

def rmse(model, test_dataframe: pd.DataFrame, train_cols: list, target_cols: list):
    """ Get the RMSE from a ML model
    Args:
        model (Regressor): Can be any regressor model from Keras or Sklearn.
        test_df (pd.DataFrame): Test dataset.
        train_cols (list): Columns of train dataset that will be used as
        knn model input.
        target_cols (list): Columns of train dataset that will be used as
        knn model output.
    Returns:
        (float): RMSE.
    """
    # Make predictions using model.
    predicted_labels = model.predict(test_dataframe[train_cols])

    # Calculate and return RMSE.
    mse = mean_squared_error(test_dataframe[target_cols], predicted_labels)
    rmse_value = np.sqrt(mse)
    return rmse_value

def plot_results(results: dict, xlabel: str, ylabel: str):
    """ Plot the results of a given model
    Args:
        results (dict(dict)): Stores the results of a given model.
        xlabel (str): X axis label.
        ylabel (str): Y axis label.
    """
    for _, results_dict in results.items():
        keys = list(results_dict.keys())
        values = list(results_dict.values())
        plt.plot(keys, values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.show()
