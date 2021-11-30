from typing import Dict

import numpy as np
import keras
import keras_tuner as kt
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from .base import ModelTrainable
from music_genre_classifier import dataset


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

    def normalized_dataset(self, array: np.array):
        """Normalizes dataset in order to maximize knn accuracy.

        Returns
        -------
        np.array
            Normalized array of features and labels
        """
        labels = array[:,-1]
        scaled_ds = scale(array[:,:-1])
        scaled_ds = np.concatenate((scaled_ds, np.array(labels)[:,None]),axis=1)
        scaled_features, scaled_labels = dataset.split_features_and_labels(scaled_ds)
        return scaled_features, scaled_labels

    def _tune(self) -> Dict:
        """Performs hyperparameter tuning for KNN model and returns best hyperparams.

        Returns
        -------
        Dict
            best hyperparameters found in search
        """
        #Normalize train dataset to fit knn
        scaled_train_features, scaled_train_labels = self.normalized_dataset(self._train_ds)
        
        # perform hyperparameter search and get best hyperparameters
        hyperparameters = { 'n_neighbors' : list(range(1,20)),
                       'weights' : ['uniform','distance'],
                       'metric' : ['euclidean','minkowski','manhattan']}
        grid_search = GridSearchCV(KNeighborsClassifier(), hyperparameters, verbose = 1, cv=3, n_jobs = -1)
        fit = grid_search.fit(scaled_train_features, scaled_train_labels)

        return fit.best_params_

    def _train(self, hyperparams: Dict):
        """Trains KNN model on hyperparameters and returns trained model.

        Parameters
        ----------
        hyperparams : Dict
            hyperparameter dictionary for KNN model

        Returns
        -------
        sklearn.Model
            trained sklearn model
        """
        #Normalize train dataset to fit knn
        scaled_train_features, scaled_train_labels = self.normalized_dataset(self._train_ds)
        
        #Build knn model with hyperparameters
        knn = KNeighborsClassifier(weights= hyperparams['weights'], 
                                     n_neighbors= hyperparams['n_neighbors'],
                                     metric= hyperparams['metric'])
        
        #Create trained model
        trained_model = knn.fit(scaled_train_features,scaled_train_labels)
        return trained_model
        

    def test(self):
        """Test function to see accuracy of knn.

        Returns
        -------
        float
            An Accuracy score for knn
        """
        #Normalize test dataset to fit knn
        scaled_test_features, scaled_test_labels = self.normalized_dataset(self._test_ds)
        
        #Testing model and determining accuracy
        prediction = self._train(self._best_hyperparams).predict(scaled_test_features)
        return None, metrics.accuracy_score(scaled_test_labels, prediction)

    def _model_builder(self, hp: kt.HyperParameters) -> keras.Model:
        pass
