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

## Test Instructions
You can run the test suite with the following command:

```
tox
```
