import pathlib
import shutil
from typing import List

import kaggle
import tensorflow as tf


def _get_csv_file_path(data_dir: str, three_sec_song: bool) -> pathlib.Path:
    """Gets CSV file path from data directory.

    Parameters
    ----------
    data_dir : str
        directory where datafiles are stored
    three_sec_song : bool
        if should use the three second song csv (as opposed to thirty second songs)

    Returns
    -------
    pathlib.Path
        path to csv file
    """
    if three_sec_song:
        return pathlib.Path(data_dir).joinpath(pathlib.Path("Data/features_3_sec.csv"))
    else:
        return pathlib.Path(data_dir).joinpath(pathlib.Path("Data/features_30_sec.csv"))


def _download_dataset(gtzan_url: str, data_dir: str) -> None:
    """Downloads dataset from kaggle, deleting contents of data_dir.

    Parameters
    ----------
    gtzan_url : str
        url for gtzan kaggle dataset
    data_dir : str
        directory to download gtzan dataset to
    """
    # delete data directory
    try:
        shutil.rmtree(pathlib.Path(data_dir).joinpath("Data"))
    except FileNotFoundError:
        pass

    # download dataset
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(gtzan_url, data_dir, unzip=True)


def create_gtzan_dataset(
    gtzan_url: str,
    data_dir: str,
    features: List[str],
    batch_size: int,
    three_sec_songs: bool = False
) -> tf.data.Dataset:
    """Creates GTZAN tensorflow dataset.

    Parameters
    ----------
    gtzan_url : str
        url for gtzan kaggle dataset
    data_dir : str
        directory to download gtzan dataset to
    features : List[str]
        list of dataset features to include
    batch_size: int
        size of batch for dataset
    three_sec_songs : bool, optional
        if should use the three second songs (instead of 30 second songs), by default False

    Returns
    -------
    tf.data.Dataset
        tensorflow dataset composed of the requested features
    """
    # check if dataset is not yet downloaded
    if not _get_csv_file_path(data_dir, three_sec_songs).exists():
        # download dataset
        _download_dataset(gtzan_url, data_dir)

    # load csv into dataset and return
    return tf.data.experimental.make_csv_dataset(
        str(_get_csv_file_path(data_dir, three_sec_songs)),
        batch_size=batch_size,
        select_columns=features + ["label"],
        label_name="label",
    )
