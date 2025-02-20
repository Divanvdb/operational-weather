# Standard library
from argparse import ArgumentParser
import xarray as xr
import numpy as np

# Third-party
from lightning_fabric.utilities import seed
from loguru import logger

# Local
from .config import load_config_and_datastore
from .models import GraphLAM
from .weather_dataset import WeatherDataModule

@logger.catch
def main(input_args=None, total = 10):
    """Main function for training and evaluating models."""
    parser = ArgumentParser(description="Train or evaluate NeurWP models for LAM")
    
    # Configuration and model arguments
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration for neural-lam")
    parser.add_argument("--model", type=str, default="graph_lam", help="Model architecture to train/evaluate (default: graph_lam)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--load", type=str, help="Path to load model parameters from (default: None)")
    parser.add_argument("--graph", type=str, default="multiscale", help="Graph to load and use in graph-based model (default: multiscale)")
    parser.add_argument("--ar_steps_train", type=int, default=1, help="Number of steps to unroll prediction for during training (default: 10)")
    parser.add_argument("--ar_steps_eval", type=int, default=7, help="Number of steps to unroll prediction for during evaluation (default: 10)")
    parser.add_argument("--num_past_forcing_steps", type=int, default=1, help="Number of past forcing steps (default: 4)")
    parser.add_argument("--num_future_forcing_steps", type=int, default=0, help="Number of future forcing steps (default: 4)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading (default: 1)")
    parser.add_argument("--eval", type=str, choices=[None, "val", "test"], default="test", help="Evaluation mode (default: None)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args(input_args)

    # Set seed
    seed.seed_everything(args.seed)

    # Load neural-lam configuration and datastore
    config, datastore = load_config_and_datastore(config_path=args.config_path)
    logger.info("Datastore loaded.")

    # Load model
    model = GraphLAM.load_from_checkpoint(args.load, config=config, datastore=datastore)
    logger.info("Model loaded.")

    # Create datamodule
    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=args.ar_steps_train,
        ar_steps_eval=args.ar_steps_eval,
        standardize=True,
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Setup data module for evaluation
    data_module.setup('train')
    train_loader = data_module.train_dataloader()

    logger.info("Data Module Created.")
    
    catch = True
    count = 0

    logger.info("Forecast generation started.")

    # Iterate through the test loader
    for i, batch in enumerate(train_loader):
        logger.info(f"Processing batch {i}/{len(train_loader)}")
        prediction, target, _, times = model.common_step(batch)

        # Convert predictions and targets to DataArrays
        predictions = model._create_dataarray_from_tensor(
            tensor=prediction[0], time=times[0], split='train', category='state', use_numpy=False
        )
        targets = model._create_dataarray_from_tensor(
            tensor=target[0], time=times[0], split='train', category='state', use_numpy=False
        )

        # Extract wind speed at 850 hPa
        predictions = predictions.sel(state_feature='wind_speed850.0hPa').drop_vars('state_feature')
        targets = targets.sel(state_feature='wind_speed850.0hPa').drop_vars('state_feature')

        # Unstack grid coordinates
        predictions = datastore.unstack_grid_coords(predictions)
        targets = datastore.unstack_grid_coords(targets)

        # Unstandardise the predictions and targets
        wspd_meand = data_module.train_dataset.da_state_mean.sel(state_feature='wind_speed850.0hPa').values
        wspd_stdd = data_module.train_dataset.da_state_std.sel(state_feature='wind_speed850.0hPa').values

        predictions = predictions * wspd_stdd + wspd_meand
        targets = targets * wspd_stdd + wspd_meand


        if catch:
            # Initialize an empty Dataset
            output = xr.Dataset()

            # Assign coordinates
            output.coords['time'] = predictions['time'][0].values  # Copy the time coordinate from the input
            output.coords['prediction_timedelta'] = ('prediction_timedelta', np.arange(0, 7))  # Correct dim name
            output.coords['latitude'] = ('latitude', predictions['y'].values)  # Explicitly assign dimension
            output.coords['longitude'] = ('longitude', predictions['x'].values)  # Explicitly assign dimension

            output['wind_speed'] = (('prediction_timedelta', 'longitude', 'latitude'), predictions.values)
            output['target'] = (('prediction_timedelta', 'longitude', 'latitude'), targets.values)

            catch = False

        else:
            # Init temp dataset
            temp = xr.Dataset()

            # Assign coordinates
            temp.coords['time'] = predictions['time'][0].values  # Copy the time coordinate from the input
            temp.coords['prediction_timedelta'] = ('prediction_timedelta', np.arange(0, 7))  # Correct dim name
            temp.coords['latitude'] = ('latitude', predictions['y'].values)  # Explicitly assign dimension
            temp.coords['longitude'] = ('longitude', predictions['x'].values)  # Explicitly assign dimension
            
            temp['wind_speed'] = (('prediction_timedelta', 'longitude', 'latitude'), predictions.values)
            temp['target'] = (('prediction_timedelta', 'longitude', 'latitude'), targets.values)

            # Concatenate the temp dataset to the output dataset
            output = xr.concat([output, temp], dim='time')

        count += 1
        if count == total:
            break


    # Add metadata to the output DataArray
    output.attrs['description'] = 'Wind speed at 850 hPa predictions and targets'
    output.attrs['units'] = 'm/s'

    logger.info("Forecast generation completed.")
    return output

if __name__ == "__main__":
    total = 10
    output = main(total = total)

    output.to_netcdf(f'output_{total}x7x49x69.nc')