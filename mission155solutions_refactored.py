"""Explore many KNN models for Cars Pricing Regression

Using Scikit-learn, pandas and numpy, this script explores
many KNN models for cars pricing regression, using the dataset
named imports-85.data, where 14 numeric features are used for
predictions.
"""
import pandas as pd
import numpy as np
from utils.dataset_func import preprocess_dataset, split_dataset
from utils.model_func import fit_knn, rmse, plot_results

pd.options.display.max_columns = 99

cols = [
        'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
        'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
        'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
        'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg',
        'price']

cars = pd.read_csv('imports-85.data', names=cols)
print(cars.head())

# Select only the columns with continuous values from -
#  https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width',
                          'height', 'curb-weight', 'engine-size', 'bore',
                          'stroke', 'compression-rate', 'horsepower', 'peak-rpm',
                          'city-mpg', 'highway-mpg', 'price']
numeric_cars = cars[continuous_values_cols]

## Data Cleaning
target_col_list = ['price']
numeric_cars = preprocess_dataset(numeric_cars, target_col_list)

## Univariate Model

# Split datasets into train and test data
train_df, test_df = split_dataset(numeric_cars)

rmse_results = {}
train_col_list = list(numeric_cars.columns.drop(target_col_list))

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.
for train_col in train_col_list:
    knn = fit_knn(train_df, [train_col], target_col_list)
    rmse_results[train_col] = rmse(knn, test_df, [train_col], target_col_list)

# Create a Series object from the dictionary so
# we can easily view the results, sort, etc
rmse_results_series = pd.Series(rmse_results)
print(rmse_results_series.sort_values())

k_values = [1,3,5,7,9]
k_rmse_results = {}

for train_col in train_col_list:
    k_rmses = {}

    for k_neighbors in k_values:
        # Fit model using k nearest neighbors.
        knn = fit_knn(train_df, [train_col], target_col_list, k_neighbors)
        k_rmses[k_neighbors] = rmse(knn, test_df, [train_col], target_col_list)

    k_rmse_results[train_col] = k_rmses

print(k_rmse_results)

plot_results(k_rmse_results, 'k value', 'RMSE')

# Compute average RMSE across different `k` values for each feature.
feature_avg_rmse = {}
for k,v in k_rmse_results.items():
    avg_rmse = np.mean(list(v.values()))
    feature_avg_rmse[k] = avg_rmse

series_avg_rmse = pd.Series(feature_avg_rmse)
sorted_series_avg_rmse = series_avg_rmse.sort_values()

print(sorted_series_avg_rmse)

sorted_features = sorted_series_avg_rmse.index

k_rmse_results = {}

for nr_best_feats in range(2,7):
    train_col_list = sorted_features[:nr_best_feats].tolist()
    knn = fit_knn(train_df, train_col_list, target_col_list, 5)
    k_rmse_results[f"{nr_best_feats} best features"] = rmse(
        knn, test_df, train_col_list, target_col_list
    )

print(k_rmse_results)

k_rmse_results = {}

for nr_best_feats in range(2,6):
    # Split train/test dataset
    train_col_list = sorted_features[:nr_best_feats].tolist()

    k_rmses = {}
    for k_neighbors in range(1, 25):
        # Fit model using k nearest neighbors.
        knn = fit_knn(train_df, train_col_list, target_col_list, k_neighbors)
        k_rmses[k_neighbors] = rmse(knn, test_df, train_col_list, target_col_list)

    k_rmse_results[f"{nr_best_feats} best features"] = k_rmses

print(k_rmse_results)

plot_results(k_rmse_results, "k value", "RMSE")
