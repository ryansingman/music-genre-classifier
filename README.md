# music-genre-classifier
Classifies music genres based on extracted features from the GTZAN Dataset. Uses KNN and Deep NN for classification.

## Build Instructions
You can install the `music_genre_classifier` package as follows:

```
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip install -e .
```

## Development Instructions
Install pre-commit to ensure code quality.
```
pre-commit install
```

This will run checks on your code every time you commit.

### Kaggle API
Follow the API credential steps [here]{https://github.com/Kaggle/kaggle-api#api-credentials} to download the dataset from Kaggle.

## Training Instructions
To train a model, run the following command:

```
python -m music_genre_classifier <path_to_config> --display_results
```

For example, to train only the KNN model, you would run:
```
python -m music_genre_classifier configs/knn_only.yaml --display_results
```
