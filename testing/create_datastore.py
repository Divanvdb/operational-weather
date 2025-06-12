
import shutil
import os

import mllam_data_prep as mdp

if __name__ == "__main__":

    config_path = "/era_test/era_test.datastore.yaml"

    if os.path.isdir(config_path):
        shutil.rmtree(config_path)
        print(f"Deleted folder: {config_path}")
    else:
        print(f"Folder not found: {config_path}")


    config = mdp.Config.from_yaml_file(config_path)
    ds = mdp.create_dataset(config=config)
    ds.to_zarr('C:\github\operational-weather\era_test\era_test.datastore.zarr')

    print('Datastore added successfully')

