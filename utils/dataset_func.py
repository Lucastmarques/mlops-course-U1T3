""" Auxiliary functions to manipulate the Dataset

This module assists the main modle "mission155solutions_refactored.py"
with functions dedicated to process the dataset.
"""
import numpy as np
import pandas as pd

def preprocess_dataset(data: pd.DataFrame, target_col_list: list) -> pd.DataFrame:
    """Pre-process a given dataset
    Args:
        data (pd.DataFrame): Dataset to be pre-processed.
        target_col_list (list): Target column labels list.
    Returns:
        (pd.DataFrame): Pre-processed dataset.
    """
    aux_df = data.copy()
    aux_df = aux_df.replace('?', np.nan)

    aux_df = aux_df.astype('float')
    print(aux_df.isnull().sum())

    # Because `price` is the column we want to predict,
    # let's remove any rows with missing `price` values.
    aux_df = aux_df.dropna(subset=target_col_list)
    print(aux_df.isnull().sum())

    # Replace missing values in other columns using column means.
    aux_df = aux_df.fillna(aux_df.mean())

    # Confirm that there's no more missing values!
    print(aux_df.isnull().sum())

    # Normalize all columnns to range from 0 to 1 except the target column.
    target_col_values = aux_df[target_col_list]
    aux_df = (aux_df - aux_df.min())/(aux_df.max() - aux_df.min())
    aux_df[target_col_list] = target_col_values

    return aux_df


def split_dataset(data: pd.DataFrame) -> tuple(pd.DataFrame, pd.DataFrame):
    """ Split dataset into train and test
    Args:
        data (pd.DataFrame): Data set to split into train and test.
    Returns:
        train, test (tuple): The train and test dataset respectively.
    """
    np.random.seed(1)

    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(data.index)
    rand_df = data.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)

    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train = rand_df.iloc[0:last_train_row]
    test = rand_df.iloc[last_train_row:]

    return train, test
