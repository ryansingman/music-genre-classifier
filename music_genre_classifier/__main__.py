from typing import List
from typing import Tuple

import numpy as np
import yaml

from music_genre_classifier import dataset
from music_genre_classifier import models


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(prog="Music Genre Classifier")
    parser.add_argument("classifier_conf_path", help="path to yaml config for classifier")
    parser.add_argument(
        "--display_results",
        help="if should display results of training",
        action="store_true",
    )

    args = parser.parse_args()
    with open(args.classifier_conf_path) as classifier_conf_file:
        classifier_conf = yaml.load(classifier_conf_file, Loader=yaml.Loader)

    # create dataset
    full_ds: np.ndarray = dataset.create_gtzan_dataset(**classifier_conf["dataset"])

    # create train, test, validation split
    train_ds, test_ds = dataset.split_dataset(full_ds)

    # create models from config
    model_trainables: List[models.ModelTrainable] = [
        models.build_from_config(model_conf, train_ds, test_ds)
        for model_conf in classifier_conf["models"]
    ]

    # find hyperparameters (TODO: parallelize this later)
    for model in model_trainables:
        model.tune()

    # train models (TODO: parallelize this later)
    for model in model_trainables:
        model.train()

    # evaluate models (TODO: parallelize this later)
    results: List[Tuple[float, float]] = []
    for model in model_trainables:
        results.append(model.test())

    # display results
    if args.display_results:
        for model, result in zip(model_trainables, results):
            print(str(model), model._best_hyperparams.values, result)   # type: ignore
