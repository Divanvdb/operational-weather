# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased](https://github.com/mllam/neural-lam/tree/main)

### Fixed

- `all_gather_cat` function in `ar_model.py` removed since it messes with the shapes of the my losses
- Added an `use_numpy` variable to the `create_dataarray_from_tensor` functions for both `weather_dataset.py`, `train_model.py` and `ar_model.py` to be able to use this funtion in Notebooks to generate data_arrays


<!-- ## [v0.1.0]() -->