import abc
from typing import Dict
from typing import Optional
from typing import Tuple

import keras_tuner as kt
import tensorflow as tf


class ModelTrainable(abc.ABC):
    """Trainable model base class, defines interface for trainable models."""

    def __init__(self, train_ds: tf.data.Dataset, test_ds: tf.data.Dataset):
        """Initializes trainable with train dataset.

        Parameters
        ----------
        train_ds : tf.data.Dataset
            dataset to train model with
        test_ds : tf.data.Dataset
            dataset to test model with
        """
        self._train_ds = train_ds
        self._test_ds = test_ds

        # trainable status flags
        self._best_hyperparams: Optional[Dict] = None
        self._model: Optional[tf.keras.Model] = None

    def tune(self):
        """Performs hyperparameter tuning for model."""
        self._best_hyperparams = self._tune()

    def train(self, hyperparams: Optional[Dict]):
        """Trains model using optimal (or provided) hyperparameters."""
        if hyperparams is None:
            if self._best_hyperparams is None:
                raise RuntimeError(
                    "Cannot train model without tuning or providing hyperparameters.",
                )
            else:
                hyperparams = self._best_hyperparams

        self._trained_model = self._train(hyperparams)

    def test(self) -> Tuple[float, float]:
        """Evaluates model on test dataset.

        Returns
        -------
        Tuple[float, float]
            tuple of model loss and accuracy
        """
        return self._trained_model.evaluate(self._test_ds)

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
    def _train(self, hyperparams: Dict) -> tf.keras.Model:
        """Trains model on hyperparameters and returns trained model.

        Parameters
        ----------
        hyperparams : Dict
            hyperparameter dictionary for model

        Returns
        -------
        tf.keras.Model
            trained model
        """
        ...

    @abc.abstractstaticmethod
    def _model_builder(hp: kt.HyperParameters) -> tf.keras.Model:
        """Model builder function, builds model to tune and train.

        Parameters
        ----------
        hp : kt.HyperParameters
            hyperparameter space container

        Returns
        -------
        tf.keras.Model
            keras model to tune and train
        """
        ...
