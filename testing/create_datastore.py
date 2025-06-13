import os
import shutil
from typing import Any
from datetime import timedelta

import pandas as pd
import xarray as xr

from datetime import datetime

from parameters import InferenceParameters

import mllam_data_prep as mdp

#subtask
def get_targets(inference_parameters: InferenceParameters, start_time: datetime, steps: int = 0) -> xr.Dataset:

    ds = xr.open_dataset(inference_parameters.target_file)

    end_time = start_time + timedelta(hours= inference_parameters.intervals * steps)

    # Select using datetime bounds
    ds_new = ds.sel(time=slice(start_time, end_time))

    return ds_new


#subtask
def load_current(inference_parameters: InferenceParameters) -> xr.Dataset:

    with xr.open_dataset(inference_parameters.current_file) as ds:
        ds = ds.load()

    return ds


#subtask
def concatenate_data(ds: xr.Dataset, ds_new: xr.Dataset) -> xr.Dataset:

    # Concatenate datasets
    ds_combined = xr.merge([ds, ds_new]).sortby(["time", "x", "y", "pressure_level"])


    return ds_combined


#task
def update_use_dataset_dataset(parameters: InferenceParameters) -> None:
    init_data = load_current(inference_parameters=parameters)

    seed_time = pd.to_datetime(init_data.time.values[-1]) + pd.Timedelta(hours=parameters.intervals)

    targets = get_targets(inference_parameters=parameters, start_time=seed_time, steps=parameters.steps)

    combined_data = concatenate_data(init_data, targets)

    combined_data.to_netcdf(parameters.use_file)


#task
def add_step_to_current_dataset(parameters: InferenceParameters) -> None:

    init_data = load_current(inference_parameters=parameters)

    seed_time = pd.to_datetime(init_data.time.values[-1]) + pd.Timedelta(hours=parameters.intervals)

    targets = get_targets(inference_parameters=parameters, start_time=seed_time)

    combined_data = concatenate_data(init_data, targets)

    combined_data.to_netcdf(parameters.current_file)


#task
def revert_to_initial_dataset(parameters: InferenceParameters) -> None:
    ds = xr.open_dataset(parameters.og_file)
    ds.to_netcdf(parameters.current_file)


#task
def get_dates(parameters: InferenceParameters) -> tuple[Any, Any]:
    ds = xr.open_dataset(parameters.use_file)
    print(f"Opening file with {len(ds.time.values)} entries")

    training_dates = ds.time.values[[0, -parameters.steps - 4]]
    target_dates = ds.time.values[[-parameters.steps - 3, -1]]

    training_dates_str = [
        pd.Timestamp(dt).strftime("%Y-%m-%dT%H:%M") for dt in training_dates
    ]
    target_dates_str = [
        pd.Timestamp(dt).strftime("%Y-%m-%dT%H:%M") for dt in target_dates
    ]

    print("Training dates: ", training_dates_str)
    print("Target dates: ", target_dates_str)

    return training_dates_str, target_dates_str


#task
def remove_current_datastore(parameters: InferenceParameters) -> None:
    if os.path.isdir(parameters.datastore_path):
        shutil.rmtree(parameters.datastore_path)
        print(f"Deleted folder: {parameters.datastore_path}")
    else:
        print(f"Folder not found: {parameters.datastore_path}")


#task
def update_config(parameters: InferenceParameters, ds_config, training_dates_str, target_dates_str):
    ds_config.output.coord_ranges["time"].start = training_dates_str[0]
    ds_config.output.coord_ranges["time"].end = target_dates_str[1]

    ds_config.output.splitting.splits["train"].start = training_dates_str[0]
    ds_config.output.splitting.splits["train"].end = training_dates_str[1]

    ds_config.output.splitting.splits["test"].start = target_dates_str[0]
    ds_config.output.splitting.splits["test"].end = target_dates_str[1]

    ds_config.inputs["era_variables"].path = parameters.use_file
    ds_config.inputs["era_times"].path = parameters.use_file
    ds_config.inputs["era_lsm"].path = parameters.use_file

    return ds_config


#model
def create_datastore(parameters: InferenceParameters()):

    if parameters.add_step:
        add_step_to_current_dataset(parameters)

    if parameters.revert_original_datasets:
        revert_to_initial_dataset(parameters)

    update_use_dataset_dataset(parameters)
    print('Updated Dataset')


    training_dates_str, target_dates_str = get_dates(parameters)


    remove_current_datastore(parameters)


    datastore_config = mdp.Config.from_yaml_file(parameters.datastore_config_path)


    datastore_config = update_config(parameters, datastore_config, training_dates_str, target_dates_str)


    ds = mdp.create_dataset(config=datastore_config)

    print('Datastore created')

    ds.to_zarr(parameters.datastore_path)

    print('Datastore added successfully')


if __name__ == "__main__":
    params = InferenceParameters()
    create_datastore(params)