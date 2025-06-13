
class InferenceParameters:

    target_file: str = 'C:/github/operational-weather/testing/era_nov_dec.nc'
    current_file: str = 'C:/github/operational-weather/era_test/era_current.nc'
    use_file: str = 'C:/github/operational-weather/era_test/era_use.nc'
    og_file: str = "C:/github/operational-weather/testing/era_jan_oct_og.nc"
    
    datastore_config_path: str = "C:\github\operational-weather\era_test\era_test.datastore.yaml"
    datastore_path: str = 'C:\github\operational-weather\era_test\era_test.datastore.zarr'

    args_path: str = 'C:/github/operational-weather/testing/training_params.yaml'

    add_day: bool = True

    restore_original_datasets: bool = False

    steps: int = 12
    intervals: int = 3

