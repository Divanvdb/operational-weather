import os
import shutil

import pandas as pd

import mllam_data_prep as mdp

import xarray as xr

from dataset_utils import update_use_dataset_dataset, add_step_to_current_dataset, revert_to_initial_dataset

from parameters import InferenceParameters


def create_datastore(parameters: InferenceParameters()):


    if parameters.add_day:
        add_step_to_current_dataset(parameters)

    if parameters.restore_original_datasets:
        revert_to_initial_dataset(parameters)

    update_use_dataset_dataset(parameters)
    print('Updated Dataset')


    ds = xr.open_dataset(parameters.use_file)
    print(f'Opening file with {len(ds.time.values)} entries')


    training_dates = ds.time.values[[0, -parameters.steps - 4]]
    target_dates = ds.time.values[[-parameters.steps - 3, -1]]


    config_path = parameters.datastore_config_path


    if os.path.isdir(parameters.datastore_path):
        shutil.rmtree(parameters.datastore_path)
        print(f"Deleted folder: {parameters.datastore_path}")
    else:
        print(f"Folder not found: {parameters.datastore_path}")


    ds_config = mdp.Config.from_yaml_file(config_path)

    training_dates_str = [
        pd.Timestamp(dt).strftime("%Y-%m-%dT%H:%M") for dt in training_dates
    ]
    target_dates_str = [
        pd.Timestamp(dt).strftime("%Y-%m-%dT%H:%M") for dt in target_dates
    ]

    print("Training dates: ", training_dates_str)
    print("Target dates: ", target_dates_str)

    # Assign to config
    ds_config.output.coord_ranges["time"].start = training_dates_str[0]
    ds_config.output.coord_ranges["time"].end = target_dates_str[1]

    ds_config.output.splitting.splits["train"].start = training_dates_str[0]
    ds_config.output.splitting.splits["train"].end = training_dates_str[1]

    ds_config.output.splitting.splits["test"].start = target_dates_str[0]
    ds_config.output.splitting.splits["test"].end = target_dates_str[1]

    ds_config.inputs["era_variables"].path = parameters.use_file
    ds_config.inputs["era_times"].path = parameters.use_file
    ds_config.inputs["era_lsm"].path = parameters.use_file



    ds = mdp.create_dataset(config=ds_config)

    print('Datastore created')

    ds.to_zarr(parameters.datastore_path)

    print('Datastore added successfully')


if __name__ == "__main__":
    params = InferenceParameters()
    create_datastore(params)