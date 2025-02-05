# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased](https://github.com/mllam/neural-lam/tree/main)

### Fixed

- `all_gather_cat` function in `ar_model.py` removed since it messes with the shapes of the my losses
- Added an `use_numpy` variable to the `create_dataarray_from_tensor` functions for both `weather_dataset.py`, `train_model.py` and `ar_model.py` to be able to use this funtion in Notebooks to generate data_arrays


## [v0.1.0](https://github.com/Divanvdb/dk-neural-lam)

- Initialised the ERA5 2020 testing data and config files `test_config.yaml` and `test.datastore.yaml`
- Added images to the `README.md` file
- Created a new `validate_model.py` file for the evaluation phase
- Updated the Lightning Trainer in `train_model.py` to include multi-GPU training and a profiler

## [v0.1.1](https://github.com/Divanvdb/dk-neural-lam)

- Remade the test datastore with `label = train` in the `test.datastore.yaml` file
- Added a train option into the `setup` function of `weather_dataset` to create a testing datastore with unshuffled dataloader
- Redid `validate_model.py` to produce and `output_20x7x49x69` file for weather evaluation using untrained model

## [v0.1.2](https://github.com/Divanvdb/dk-neural-lam)

- Ensured the **standardization** of the weather data is done correctly
- Reduced `n_boundary_points=2` to `n_boundary_points=0` in `mdp.py` to ensure the model doesn't use future forcings with `--num_future_forcing_steps 0` 
- Used: `saved_models\train-graph_lam-4x64-01_23_13-3100\min_val_loss.ckpt`
- Reordered the output file coordinates in `validate_model.py`
- Added a `total` variable to the `validate_model.py` script

## TODO:

- Add the `plot_data.py` to the project

