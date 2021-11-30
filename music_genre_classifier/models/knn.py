from typing import Dict

import keras
import keras_tuner as kt

from .base import ModelTrainable


class KNN(ModelTrainable):
    """Implements trainable KNN class."""
    
    def __init__(self, train_ds: np.ndarray, test_ds: np.ndarray):
        """Initializes neural network trainable.
        Parameters
        ----------
        train_ds : np.ndarray
            dataset to train model with
        test_ds : np.ndarray
            dataset to test model with
        """
        super().__init__(train_ds, test_ds)

    def _tune(self) -> Dict:
        """Performs hyperparameter tuning for KNN model and returns best hyperparams.

        Returns
        -------
        Dict
            best hyperparameters found in search
        """
        
        ...

    def _train(self, hyperparams: Dict) -> keras.Model:
        """Trains KNN model on hyperparameters and returns trained model.

        Parameters
        ----------
        hyperparams : Dict
            hyperparameter dictionary for KNN model

        Returns
        -------
        keras.Model
            trained KNN model
        """
        ...

    def _model_builder(self, hp: kt.HyperParameters) -> keras.Model:
        """Builds hyperparameter tunable KNN model.

        Parameters
        ----------
        hp : kt.HyperParameters
            hyperparameter space container

        Returns
        -------
        keras.Model
            keras KNN model to tune and train
        """
        ...
