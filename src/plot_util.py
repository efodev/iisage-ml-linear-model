import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import zlib


def _checksum_dataset(row):
    """
    Helper function uses zlib function to hash a row in dataset into a unique number
    
    Args: 
    ----
    row: The row in a dataset

    Returns:
    ------
    Returns the unique hashed representation of the row
    """
    row_str = str(row).encode('utf-8')
    hashed =  zlib.adler32(row_str)
    return np.array(hashed)

def _hash_dataset(data, target=None):
    """
    Uses numpy's vectorized function to call _check_sum_dataset to has a dataset rows

    Args:
    -----
    data: dataset to be hashed
    target: the name of the target column
    """
    if target:
        y = data[target] # tar
        x = data.drop(columns=[target]).to_numpy()
        x_values = np.apply_along_axis(_checksum_dataset, axis=1, arr=x) # apply hashing function on  the rows
        return x_values, y
    x = data.to_numpy()
    return np.apply_along_axis(_checksum_dataset, axis=1, arr=x) # apply hashing function on  the rows
    



def _standard_scale_hash(hashed_features):
    # scale the hashed values between 0 (min) and 1 (max)
    mean_ = hashed_features.mean()
    std_ = hashed_features.std()
    x_norm = (hashed_features - mean_) / (std_)
    return x_norm

def scatter_plot(data, target_str, scaled=False):
    x, y = _hash_dataset(data, target_str) # obtain the x and y values

    if scaled:
        x = _standard_scale_hash(x)

    x_min, x_max = x.min(), x.max()
    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_box_aspect(1)
    ax.set_xlabel("Hashed feature values")
    ax.set_ylabel(f'{target_str}')
    ax.set_title(f"Rough estimate of the distribution of {target_str} with features")
    ax.plot(x, y, 'ob')

    plt.show()


def plot_reg(x_test, y_test, y_pred, scaled=False):
    x = []
    if scaled:
        x = _standard_scale_hash(_hash_dataset(x_test))
    else:
        x = _hash_dataset(x_test)
    x_min, x_max = x.min(), x.max()
    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_box_aspect(1)
    ax.set_xlabel("Hashed feature values")
    ax.set_ylabel(f'target values')
    ax.plot(x, y_test, "-b", label="actual values")
    ax.plot(x, y_pred, "-r", label="predicted values")
    plt.legend(loc="upper left")
    plt.show()
   
    

    
