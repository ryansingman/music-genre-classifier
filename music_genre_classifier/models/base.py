import abc
from typing import Dict
from typing import Optional
from typing import Tuple

import keras
import keras_tuner as kt
import numpy as np

from music_genre_classifier import dataset


class ModelTrainable(abc.ABC):
    """Trainable model base class, defines interface for trainable models."""

    def __init__(self, train_ds: np.ndarray, test_ds: np.ndarray):
        """Initializes trainable with train dataset.

        Parameters
        ----------
        train_ds : np.ndarray
            dataset to train model with
        test_ds : np.ndarray
            dataset to test model with
        """
        self._train_ds = train_ds
        self._test_ds = test_ds

        # trainable status flags
        self._best_hyperparams: Optional[Dict] = None
        self._model: Optional[keras.Model] = None

    def tune(self):
        """Performs hyperparameter tuning for model."""
        self._best_hyperparams = self._tune()

    def train(self, hyperparams: Optional[Dict] = None):
        """Trains model using optimal (or provided) hyperparameters."""
        if hyperparams is None:
            if self._best_hyperparams is None:
                raise RuntimeError(
                    "Cannot train model without tuning or providing hyperparameters.",
                )
            else:
                hyperparams = self._best_hyperparams

        self._trained_model = self._train(hyperparams)

    @abc.abstractmethod
    def test(self) -> Tuple[float, float]:
        """Evaluates model on test dataset.

        Returns
        -------
        Tuple[float, float]
            tuple of model loss and accuracy
        """
        ...

    @staticmethod
    def _train_val_split(
        full_ds: np.ndarray, val_split: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Splits dataset into training and validation datasets.

        Parameters
        ----------
        full_ds : np.ndarray
            dataset to split
        val_split : float, optional
            ratio of validation to full dataset, by default 0.2

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            train, validation datasets
        """
        return dataset.split_dataset(full_ds, val_split)

    @abc.abstractmethod
    def _tune(self) -> Dict:
        """Performs hyperparameter tuning for model and returns best hyperparams.

        Returns
        -------
        Dict
            best hyperparameters found in search
        """
        ...

    @abc.abstractmethod
    def _train(self, hyperparams: Dict) -> keras.Model:
        """Trains model on hyperparameters and returns trained model.

        Parameters
        ----------
        hyperparams : Dict
            hyperparameter dictionary for model

        Returns
        -------
        keras.Model
            trained model
        """
        ...

    @abc.abstractmethod
    def _model_builder(self, hp: kt.HyperParameters) -> keras.Model:
        """Model builder function, builds model to tune and train.

        Parameters
        ----------
        hp : kt.HyperParameters
            hyperparameter space container

        Returns
        -------
        keras.Model
            keras model to tune and train
        """
        ...
