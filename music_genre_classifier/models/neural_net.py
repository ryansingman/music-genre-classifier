from typing import Dict

import keras
import keras_tuner as kt
import numpy as np
import tensorflow as tf

from .base import ModelTrainable
from music_genre_classifier import dataset


class NeuralNet(ModelTrainable):
    """Implements trainable Neural Net class."""

    def __init__(self, train_ds: np.ndarray, test_ds: np.ndarray, train_epochs: int):
        """Initializes neural network trainable.

        Parameters
        ----------
        train_ds : np.ndarray
            dataset to train model with
        test_ds : np.ndarray
            dataset to test model with
        train_epochs : int
            number of epochs to train model with
        """
        super().__init__(train_ds, test_ds)
        self._train_epochs = train_epochs

    def _tune(self) -> Dict:
        """Performs hyperparameter tuning for Neural Net model and returns best hyperparams.

        Returns
        -------
        Dict
            best hyperparameters found in search
        """
        # create tuner and create early stopping callback
        self._tuner = kt.Hyperband(
            self._model_builder,
            objective="val_accuracy",
            max_epochs=100,
            directory="trained_models",
            project_name="neural_net_mgc",
        )

        stop_early = keras.callbacks.EarlyStopping(monitor="val_loss")

        # perform hyperparameter search and get best hyperparameters
        self._tuner.search(
            *dataset.split_features_and_labels(self._train_ds),
            validation_split=0.2, callbacks=[stop_early],
        )
        return self._tuner.get_best_hyperparameters()[0]

    def _train(self, hyperparams: Dict) -> keras.Model:
        """Trains NN model on hyperparameters and returns trained model.

        Parameters
        ----------
        hyperparams : Dict
            hyperparameter dictionary for NN model

        Returns
        -------
        keras.Model
            trained NN model
        """
        # build model with optimal hyperparams and fit to data
        model = self._tuner.hypermodel.build(hyperparams)
        history = model.fit(
            *dataset.split_features_and_labels(self._train_ds),
            epochs=self._train_epochs, validation_split=0.2,
        )

        # find best epoch
        best_epoch = np.argmax(history.history["val_accuracy"])

        # re-train to best epoch
        model = self._tuner.hypermodel.build(hyperparams)
        model.fit(
            *dataset.split_features_and_labels(self._train_ds),
            epochs=best_epoch, validation_split=0.2,
        )

        # return fit model
        return model

    def _model_builder(self, hp: kt.HyperParameters) -> keras.Model:
        """Builds hyperparameter tunable NN model.

        Parameters
        ----------
        hp : kt.HyperParameters
            hyperparameter space container

        Returns
        -------
        keras.Model
            keras NN model to tune and train
        """
        # create sequential model and add input layer
        model = keras.Sequential()
        model.add(keras.layers.BatchNormalization())
        model.add(keras.Input(shape=(self._train_ds.shape[-1] - 1,)))

        # add tunable dense layers
        dense_layer_units_1 = hp.Int("dense_layer_units_1", min_value=16, max_value=256, step=32)
        model.add(keras.layers.Dense(units=dense_layer_units_1, activation="relu"))
        dense_layer_units_2 = hp.Int("dense_layer_units_2", min_value=16, max_value=256, step=32)
        model.add(keras.layers.Dense(units=dense_layer_units_2, activation="relu"))

        # add output layer
        model.add(keras.layers.Dense(10))

        # add tunable learning rate
        learning_rate = hp.Choice("learning_rate", values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4])

        # compile and return model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model
