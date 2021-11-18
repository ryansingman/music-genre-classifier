from typing import Tuple

import numpy as np


def split_dataset(full_ds: np.ndarray, test_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Splits dataset into train, test datsets.

    Parameters
    ----------
    full_ds : tf.data.Dataset
        full dataset to split
    test_split : float, optional
        ratio of samples to use for test set, by default 0.2

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        tuple of train, test datasets
    """
    split_idx: int = int(test_split * full_ds.shape[0])
    np.random.shuffle(full_ds)
    return full_ds[:-split_idx], full_ds[-split_idx:]


def split_features_and_labels(ds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Splits features and labels in dataset.

    Parameters
    ----------
    ds : np.ndarray
        dataset to split into features and labels

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        tuple of features, labels arrays
    """
    return ds[:, :-1], ds[:, -1]
