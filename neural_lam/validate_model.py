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
def main(input_args=None):
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
    parser.add_argument("--num_future_forcing_steps", type=int, default=1, help="Number of future forcing steps (default: 4)")
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
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    logger.info("Data Module Created.")


    # Initialize the DataArray with the required dimensions and coordinates
    output = xr.DataArray(
        data=np.nan,  # Initialize with NaN values
        dims=['time', 'lon', 'lat', 'prediction_timedelta'],
        coords={
            'time': [],  # Will be populated dynamically
            'lon': datastore.x.values,  # Assuming lon and lat are available in pred_wind
            'lat': datastore.y.values,
            'prediction_timedelta': np.arange(args.ar_steps_eval),  # Lead times for predictions
        },
    )

    logger.info("Forecast generation started.")

    # Iterate through the test loader
    for i, batch in enumerate(test_loader):
        logger.info(f"Processing batch {i}/{len(test_loader)}")
        prediction, target, _, times = model.common_step(batch)

        # Convert predictions and targets to DataArrays
        predictions = model._create_dataarray_from_tensor(
            tensor=prediction[0], time=times[0], split='test', category='state', use_numpy=False
        )
        targets = model._create_dataarray_from_tensor(
            tensor=target[0], time=times[0], split='test', category='state', use_numpy=False
        )

        # Unstack grid coordinates
        unstacked_pred = datastore.unstack_grid_coords(predictions)
        unstacked_target = datastore.unstack_grid_coords(targets)

        # Extract wind speed at 850 hPa
        pred_wind = unstacked_pred.sel(state_feature='wind_speed850.0hPa').drop('state_feature')
        target_wind = unstacked_target.sel(state_feature='wind_speed850.0hPa').drop('state_feature')

        # Create a temporary DataArray for the current batch
        temp = xr.DataArray(
            data=np.stack([pred_wind.values, target_wind.values], axis=0),  # Stack prediction and target
            dims=['variable', 'time', 'lon', 'lat'],
            coords={
                'variable': ['prediction', 'target'],
                'time': pred_wind.time,
                'lon': pred_wind.lon,
                'lat': pred_wind.lat,
            },
        )

        # Append the temporary DataArray to the output DataArray
        output = xr.concat([output, temp], dim='time')

    # Add metadata to the output DataArray
    output.attrs['description'] = 'Wind speed at 850 hPa predictions and targets'
    output.attrs['units'] = 'm/s'

    logger.info("Forecast generation completed.")
    return output

if __name__ == "__main__":
    main()