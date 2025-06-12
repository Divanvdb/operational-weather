from datetime import timedelta

import pandas as pd
import xarray as xr

from datetime import datetime

from parameters import InferenceParameters

#task
def get_targets(inference_parameters: InferenceParameters, start_time: datetime, steps: int = 0) -> xr.Dataset:

    ds = xr.open_dataset(inference_parameters.target_file)

    end_time = start_time + timedelta(hours= inference_parameters.intervals * steps)

    # Select using datetime bounds
    ds_new = ds.sel(time=slice(start_time, end_time))

    return ds_new

#task
def load_current(inference_parameters: InferenceParameters) -> xr.Dataset:

    with xr.open_dataset(inference_parameters.current_file) as ds:
        ds = ds.load()

    return ds

#task
def concatenate_data(ds: xr.Dataset, ds_new: xr.Dataset) -> xr.Dataset:

    # Concatenate datasets
    ds_combined = xr.merge([ds, ds_new]).sortby(["time", "x", "y", "pressure_level"])


    return ds_combined

#model
def update_use_dataset_dataset(parameters: InferenceParameters) -> None:
    init_data = load_current(inference_parameters=parameters)

    seed_time = pd.to_datetime(init_data.time.values[-1]) + pd.Timedelta(hours=parameters.intervals)

    targets = get_targets(inference_parameters=parameters, start_time=seed_time, steps=parameters.steps)

    combined_data = concatenate_data(init_data, targets)

    combined_data.to_netcdf(parameters.use_file)

#model
def add_step_to_current_dataset(parameters: InferenceParameters) -> None:

    init_data = load_current(inference_parameters=parameters)

    seed_time = pd.to_datetime(init_data.time.values[-1]) + pd.Timedelta(hours=parameters.intervals)

    targets = get_targets(inference_parameters=parameters, start_time=seed_time)

    combined_data = concatenate_data(init_data, targets)

    combined_data.to_netcdf(parameters.current_file)


#model
def revert_to_initial_dataset(parameters: InferenceParameters) -> None:
    ds = xr.open_dataset(parameters.og_file)
    ds.to_netcdf(parameters.current_file)


if __name__ == "__main__":
    params = InferenceParameters()

    update_use_dataset_dataset(params)

    use_set = xr.open_dataset(params.use_file)

    print(use_set.time.values)

