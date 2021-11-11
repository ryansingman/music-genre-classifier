from typing import List

import tensorflow as tf
import yaml

import dataset
import evaluation
import models


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(prog="Music Genre Classifier")
    parser.add_argument("classifier_conf_path", help="path to yaml config for classifier")
    parser.add_argument(
        "--display_results",
        help="if should display results of training",
        action="store_true"
    )

    args = parser.parse_args()
    with open(args.classifier_conf_path, "r") as classifier_conf_file:
        classifier_conf = yaml.load(classifier_conf_file, Loader=yaml.Loader)

    # create dataset 
    full_ds: tf.data.Dataset = dataset.create_gtzan_dataset(**classifier_conf["dataset"])

    # create train, test, validation split
    train_ds, test_ds, validate_ds = dataset.split_dataset(full_ds)

    # create models from config
    model_trainables: List[models.Model] = [
        models.Model.from_config(model_conf) for model_conf in classifier_conf.models
    ]

    # find hyperparameters (TODO: parallelize this later)
    for model in model_trainables:
        model.tune(train_ds, validate_ds)

    # train models (TODO: parallelize this later)
    for model in model_trainables:
        model.train(train_ds)

    # evaluate models (TODO: parallelize this later)
    results: List[evaluation.ModelResult] = []
    for model in model_trainables:
        results.append(evaluation.eval_model(model, test_ds))

    # save model weights
    for model in model_trainables:
        model.save()

    # display results
    if args.display_results:
        for result in results:
            print(result)
