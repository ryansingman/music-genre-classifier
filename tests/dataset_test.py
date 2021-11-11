import yaml

from music_genre_classifier import dataset


def test_create_gtzan_dataset():
    """Tests that create gtzan dataset function returns a valid dataset."""
    # load default dataset configuration from yaml
    with open("configs/default.yaml") as conf:
        dataset_conf = yaml.load(conf, Loader=yaml.Loader)["dataset"]

    # run dataset creator with config
    ds = dataset.create_gtzan_dataset(**dataset_conf)

    # assert that we can iterate a few times
    for _ in zip(range(10), ds.as_numpy_iterator()):
        pass
