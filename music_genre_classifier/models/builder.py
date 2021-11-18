from typing import Dict

import numpy as np

from .base import ModelTrainable
from .neural_net import NeuralNet


def build_from_config(
    model_params: Dict, train_ds: np.ndarray, test_ds: np.ndarray,
) -> ModelTrainable:
    """Instantiates model trainable from config.

    Parameters
    ----------
    model_params : Dict
        model parameters
    train_ds : np.ndarray
        train dataset
    test_ds : np.ndarray
        test dataset

    Returns
    -------
    ModelTrainable
        instantiated model trainable
    """
    model_name = model_params.pop("model")
    if model_name == "neural_net":
        return NeuralNet(train_ds, test_ds, **model_params)
    else:
        raise ValueError(f"Invalid model name {model_name}.")
