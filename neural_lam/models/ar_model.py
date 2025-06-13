# Standard library
import os
from typing import List, Union

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import xarray as xr

# Local
from .. import metrics, vis
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..loss_weighting import get_state_feature_weighting
from ..weather_dataset import WeatherDataset


class ARModel(pl.LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["datastore"])
        self.args = args
        self._datastore = datastore
        num_state_vars = datastore.get_num_data_vars(category="state")
        num_forcing_vars = datastore.get_num_data_vars(category="forcing")
        da_static_features = datastore.get_dataarray(
            category="static", split=None
        )
        da_state_stats = datastore.get_standardization_dataarray(
            category="state"
        )
        da_boundary_mask = datastore.boundary_mask
        num_past_forcing_steps = args.num_past_forcing_steps
        num_future_forcing_steps = args.num_future_forcing_steps

        # Load static features for grid/data, NB: self.predict_step assumes
        # dimension order to be (grid_index, static_feature)
        arr_static = da_static_features.transpose(
            "grid_index", "static_feature"
        ).values
        self.register_buffer(
            "grid_static_features",
            torch.tensor(arr_static, dtype=torch.float32),
            persistent=False,
        )

        state_stats = {
            "state_mean": torch.tensor(
                da_state_stats.state_mean.values, dtype=torch.float32
            ),
            "state_std": torch.tensor(
                da_state_stats.state_std.values, dtype=torch.float32
            ),
            "diff_mean": torch.tensor(
                da_state_stats.state_diff_mean.values, dtype=torch.float32
            ),
            "diff_std": torch.tensor(
                da_state_stats.state_diff_std.values, dtype=torch.float32
            ),
        }

        for key, val in state_stats.items():
            self.register_buffer(key, val, persistent=False)

        state_feature_weights = get_state_feature_weighting(
            config=config, datastore=datastore
        )
        self.feature_weights = torch.tensor(
            state_feature_weights, dtype=torch.float32
        )

        # Double grid output dim. to also output std.-dev.
        self.output_std = bool(args.output_std)
        if self.output_std:
            # Pred. dim. in grid cell
            self.grid_output_dim = 2 * num_state_vars
        else:
            # Pred. dim. in grid cell
            self.grid_output_dim = num_state_vars
            # Store constant per-variable std.-dev. weighting
            # NOTE that this is the inverse of the multiplicative weighting
            # in wMSE/wMAE
            self.register_buffer(
                "per_var_std",
                self.diff_std / torch.sqrt(self.feature_weights),
                persistent=False,
            )

        # Standardization data for state
        self.ds_state_stats = self._datastore.get_standardization_dataarray(
                category="state"
            )

        self.da_state_mean = self.ds_state_stats.state_mean
        self.da_state_std = self.ds_state_stats.state_std

        self.count = 0
        self.catch = True
        # grid_dim from data + static
        (
            self.num_grid_nodes,
            grid_static_dim,
        ) = self.grid_static_features.shape

        self.grid_dim = (
            2 * self.grid_output_dim
            + grid_static_dim
            + num_forcing_vars
            * (num_past_forcing_steps + num_future_forcing_steps + 1)
        )

        # Instantiate loss function
        self.loss = metrics.get_metric(args.loss)

        boundary_mask = torch.tensor(
            da_boundary_mask.values, dtype=torch.float32, device=self.device
        ).unsqueeze(
            1
        )  # add feature dim

        self.register_buffer("boundary_mask", boundary_mask, persistent=False)
        # Pre-compute interior mask for use in loss function
        self.register_buffer(
            "interior_mask", 1.0 - self.boundary_mask, persistent=False
        )  # (num_grid_nodes, 1), 1 for non-border

        self.val_metrics = {
            "mse": [],
        }
        self.test_metrics = {
            "mse": [],
            "mae": [],
        }
        if self.output_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # For making restoring of optimizer state optional
        self.restore_opt = args.restore_opt

        # For example plotting
        self.n_example_pred = args.n_example_pred
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

    def _create_dataarray_from_tensor(
        self,
        tensor: torch.Tensor,
        time: Union[int, List[int]],
        split: str,
        category: str,
        use_numpy: bool = True,
    ) -> xr.DataArray:
        """
        Create an `xr.DataArray` from a tensor, with the correct dimensions and
        coordinates to match the datastore used by the model. This function in
        in effect is the inverse of what is returned by
        `WeatherDataset.__getitem__`.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to convert to a `xr.DataArray` with dimensions [time,
            grid_index, feature]. The tensor will be copied to the CPU if it is
            not already there.
        time : Union[int,List[int]]
            The time index or indices for the data, given as integers or a list
            of integers representing epoch time in nanoseconds. The ints will be
            copied to the CPU memory if they are not already there.
        split : str
            The split of the data, either 'train', 'val', or 'test'
        category : str
            The category of the data, either 'state' or 'forcing'
        """
        # TODO: creating an instance of WeatherDataset here on every call is
        # not how this should be done but whether WeatherDataset should be
        # provided to ARModel or where to put plotting still needs discussion
        weather_dataset = WeatherDataset(datastore=self._datastore, split=split)
        time = np.array(time.cpu(), dtype="datetime64[ns]")
        if use_numpy:
            da = weather_dataset.create_dataarray_from_tensor(
                tensor=tensor.cpu().numpy(),
                time=time,
                category=category,
                use_numpy=use_numpy,
            )
        else:
            da = weather_dataset.create_dataarray_from_tensor(
                tensor=tensor.detach().numpy(),
                time=time,
                category=category,
                use_numpy=use_numpy,
            )
        return da

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, betas=(0.9, 0.95)
        )
        return opt

    @property
    def interior_mask_bool(self):
        """
        Get the interior mask as a boolean (N,) mask.
        """
        return self.interior_mask[:, 0].to(torch.bool)

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)

    def predict_step(self, prev_state, prev_prev_state, forcing):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t prev_prev_state: (B,
        num_grid_nodes, feature_dim), X_{t-1} forcing: (B, num_grid_nodes,
        forcing_dim)
        """
        raise NotImplementedError("No prediction step implemented")

    def unroll_prediction(self, init_states, forcing_features, true_states):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_grid_nodes, d_f) forcing_features: (B,
        pred_steps, num_grid_nodes, d_static_f) true_states: (B, pred_steps,
        num_grid_nodes, d_f)
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing_features.shape[1]

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            border_state = true_states[:, i]

            pred_state, pred_std = self.predict_step(
                prev_state, prev_prev_state, forcing
            )
            # state: (B, num_grid_nodes, d_f) pred_std: (B, num_grid_nodes,
            # d_f) or None

            # Overwrite border with true state
            new_state = (
                self.boundary_mask * border_state + self.interior_mask * pred_state
            )

            prediction_list.append(new_state)
            if self.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std

    def common_step(self, batch):
        """
        Predict on single batch batch consists of: init_states: (B, 2,
        num_grid_nodes, d_features) target_states: (B, pred_steps,
        num_grid_nodes, d_features) forcing_features: (B, pred_steps,
        num_grid_nodes, d_forcing),
            where index 0 corresponds to index 1 of init_states
        """
        (init_states, target_states, forcing_features, batch_times) = batch

        prediction, pred_std = self.unroll_prediction(
            init_states, forcing_features, target_states
        )  # (B, pred_steps, num_grid_nodes, d_f)
        # prediction: (B, pred_steps, num_grid_nodes, d_f) pred_std: (B,
        # pred_steps, num_grid_nodes, d_f) or (d_f,)

        return prediction, target_states, pred_std, batch_times

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target, pred_std, _ = self.common_step(batch)

        # Compute loss
        batch_loss = torch.mean(
            self.loss(prediction, target, pred_std, mask=self.interior_mask_bool)
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return batch_loss

    def all_gather_cat(self, tensor_to_gather):
        """
        Gather tensors across all ranks, and concatenate across dim. 0 (instead
        of stacking in new dim. 0)

        tensor_to_gather: (d1, d2, ...), distributed over K ranks

        returns: (K*d1, d2, ...)
        """
        # Skipping this step due to wrong error shapes beign returned
        # return self.all_gather(tensor_to_gather).flatten(0, 1)
        return tensor_to_gather

    # newer lightning versions requires batch_idx argument, even if unused
    # pylint: disable-next=unused-argument
    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target, pred_std, _ = self.common_step(batch)

        time_step_loss = torch.mean(
            self.loss(prediction, target, pred_std, mask=self.interior_mask_bool),
            dim=0,
        )  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
            if step <= len(time_step_loss)
        }
        val_log_dict["val_mean_loss"] = mean_loss
        self.log_dict(
            val_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        # Store MSEs
        entry_mses = metrics.mse(
            prediction,
            target,
            pred_std,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["mse"].append(entry_mses)  # Shape of [batch, steps, variables]

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")

        # Clear lists with validation metrics values
        for metric_list in self.val_metrics.values():
            metric_list.clear()

    def test_step(self, batch, batch_idx):
        prediction, target, _, batch_times = self.common_step(batch)

        predictions = self._create_dataarray_from_tensor(
            tensor=prediction[0],
            time=batch_times[0],
            split="train",
            category="state",
            use_numpy=self.args.use_numpy,
        )
        targets = self._create_dataarray_from_tensor(
            tensor=target[0],
            time=batch_times[0],
            split="train",
            category="state",
            use_numpy=self.args.use_numpy,
        )

        # predictions = predictions.sel(state_feature='wind_speed850.0hPa').drop_vars('state_feature')
        # targets = targets.sel(state_feature='wind_speed850.0hPa').drop_vars('state_feature')
        predictions = predictions.sel(state_feature=["u850.0hPa", "v850.0hPa"])
        targets = targets.sel(state_feature=["u850.0hPa", "v850.0hPa"])

        # Unstack grid coordinates
        predictions = self._datastore.unstack_grid_coords(predictions)
        targets = self._datastore.unstack_grid_coords(targets)

        # Unstandardize the u and v components
        u_mean = self.da_state_mean.sel(state_feature="u850.0hPa").values
        v_mean = self.da_state_mean.sel(state_feature="v850.0hPa").values
        u_std = self.da_state_std.sel(state_feature="u850.0hPa").values
        v_std = self.da_state_std.sel(state_feature="v850.0hPa").values

        predictions_u = (
            predictions.sel(state_feature="u850.0hPa").drop_vars("state_feature")
            * u_std
            + u_mean
        )
        predictions_v = (
            predictions.sel(state_feature="v850.0hPa").drop_vars("state_feature")
            * v_std
            + v_mean
        )
        targets_u = (
            targets.sel(state_feature="u850.0hPa").drop_vars("state_feature") * u_std
            + u_mean
        )
        targets_v = (
            targets.sel(state_feature="v850.0hPa").drop_vars("state_feature") * v_std
            + v_mean
        )

        # Compute wind speed: sqrt(u^2 + v^2)
        pred_wind_speed = np.sqrt(predictions_u**2 + predictions_v**2)
        target_wind_speed = np.sqrt(targets_u**2 + targets_v**2)

        self.count += 1

        if self.catch:
            # Initialize an empty Dataset
            self.output = xr.Dataset()

            # Assign coordinates
            self.output.coords["time"] = predictions["time"][
                0
            ].values  # Copy the time coordinate from input
            self.output.coords["prediction_timedelta"] = (
                "prediction_timedelta",
                np.arange(0, 12),
            )
            self.output.coords["latitude"] = ("latitude", predictions["y"].values)
            self.output.coords["longitude"] = ("longitude", predictions["x"].values)

            # Assign wind speed output
            self.output["wind_speed"] = (
                ("prediction_timedelta", "longitude", "latitude"),
                pred_wind_speed.values,
            )
            self.output["target"] = (
                ("prediction_timedelta", "longitude", "latitude"),
                target_wind_speed.values,
            )

            self.catch = False

        else:
            # Init temp dataset
            temp = xr.Dataset()

            # Assign coordinates
            temp.coords["time"] = predictions["time"][0].values
            temp.coords["prediction_timedelta"] = (
                "prediction_timedelta",
                np.arange(0, 12),
            )
            temp.coords["latitude"] = ("latitude", predictions["y"].values)
            temp.coords["longitude"] = ("longitude", predictions["x"].values)

            # Assign wind speed output
            temp["wind_speed"] = (
                ("prediction_timedelta", "longitude", "latitude"),
                pred_wind_speed.values,
            )
            temp["target"] = (
                ("prediction_timedelta", "longitude", "latitude"),
                target_wind_speed.values,
            )

            # Concatenate the temp dataset to the output dataset
            self.output = xr.concat([self.output, temp], dim="time")


    def plot_examples(self, batch, n_examples, split, prediction=None):
        """
        Plot the first n_examples forecasts from batch

        batch: batch with data to plot corresponding forecasts for n_examples:
        number of forecasts to plot prediction: (B, pred_steps, num_grid_nodes,
        d_f), existing prediction.
            Generate if None.
        """
        if prediction is None:
            prediction, target, _, _ = self.common_step(batch)

        target = batch[1]
        time = batch[3]

        # Rescale to original data scale
        prediction_rescaled = prediction * self.state_std + self.state_mean
        target_rescaled = target * self.state_std + self.state_mean

        # Iterate over the examples
        for pred_slice, target_slice, time_slice in zip(
            prediction_rescaled[:n_examples],
            target_rescaled[:n_examples],
            time[:n_examples],
        ):
            # Each slice is (pred_steps, num_grid_nodes, d_f)
            self.plotted_examples += 1  # Increment already here

            da_prediction = self._create_dataarray_from_tensor(
                tensor=pred_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")
            da_target = self._create_dataarray_from_tensor(
                tensor=target_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")

            var_vmin = (
                torch.minimum(
                    pred_slice.flatten(0, 1).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    pred_slice.flatten(0, 1).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # Iterate over prediction horizon time steps
            for t_i, _ in enumerate(zip(pred_slice, target_slice), start=1):
                # Create one figure per variable at this time step
                var_figs = [
                    vis.plot_prediction(
                        datastore=self._datastore,
                        title=f"{var_name} ({var_unit}), "
                        f"t={t_i} ({self._datastore.step_length * t_i} h)",
                        vrange=var_vrange,
                        da_prediction=da_prediction.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                        da_target=da_target.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                    )
                    for var_i, (var_name, var_unit, var_vrange) in enumerate(
                        zip(
                            self._datastore.get_vars_names("state"),
                            self._datastore.get_vars_units("state"),
                            var_vranges,
                        )
                    )
                ]

                example_i = self.plotted_examples

                wandb.log(
                    {
                        f"{var_name}_example_{example_i}": wandb.Image(fig)
                        for var_name, fig in zip(
                            self._datastore.get_vars_names("state"), var_figs
                        )
                    }
                )
                plt.close("all")  # Close all figs for this time step, saves memory

            # Save pred and target as .pt files
            torch.save(
                pred_slice.cpu(),
                os.path.join(wandb.run.dir, f"example_pred_{self.plotted_examples}.pt"),
            )
            torch.save(
                target_slice.cpu(),
                os.path.join(
                    wandb.run.dir, f"example_target_{self.plotted_examples}.pt"
                ),
            )

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        Put together a dict with everything to log for one metric. Also saves
        plots as pdf and csv if using test prefix.

        metric_tensor: (pred_steps, d_f), metric values per time and variable
        prefix: string, prefix to use for logging metric_name: string, name of
        the metric

        Return: log_dict: dict with everything to log for given metric
        """
        log_dict = {}
        metric_fig = vis.plot_error_map(
            errors=metric_tensor,
            datastore=self._datastore,
        )
        full_log_name = f"{prefix}_{metric_name}"
        log_dict[full_log_name] = wandb.Image(metric_fig)

        if prefix == "test":
            # Save pdf
            metric_fig.savefig(os.path.join(wandb.run.dir, f"{full_log_name}.pdf"))
            # Save errors also as csv
            np.savetxt(
                os.path.join(wandb.run.dir, f"{full_log_name}.csv"),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        # Check if metrics are watched, log exact values for specific vars
        var_names = self._datastore.get_vars_names(category="state")
        if full_log_name in self.args.metrics_watch:
            for var_i, timesteps in self.args.var_leads_metrics_watch.items():
                var_name = var_names[var_i]
                for step in timesteps:
                    key = f"{full_log_name}_{var_name}_step_{step}"
                    log_dict[key] = metric_tensor[step - 1, var_i]

        return log_dict

    def aggregate_and_plot_metrics(self, metrics_dict, prefix):
        """
        Aggregate and create error map plots for all metrics in metrics_dict

        metrics_dict: dictionary with metric_names and list of tensors
            with step-evals.
        prefix: string, prefix to use for logging
        """
        # TODO: Errors due to shape mismathes in the metrics_dict tensors
        log_dict = {}
        for metric_name, metric_val_list in metrics_dict.items():
            # print(f"{prefix} {metric_name} list length: {len(metric_val_list)}")
            # print(f"{prefix} {metric_name} tensor shape: {metric_val_list[0].shape}")
            # print(f"{prefix} {metric_name} tensor shape: {metric_val_list[1].shape}")
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )  # (N_eval, pred_steps, d_f)
            # print(f"{prefix} {metric_name} tensor shape: {metric_tensor.shape}")

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)
                # (pred_steps, d_f)

                # print(f"{prefix} {metric_name} tensor averaged shape: {metric_tensor_averaged.shape}")

                # Take square root after all averaging to change MSE to RMSE
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")

                # NOTE: we here assume rescaling for all metrics is linear
                metric_rescaled = metric_tensor_averaged * self.state_std
                # (pred_steps, d_f)
                log_dict.update(
                    self.create_metric_log_dict(metric_rescaled, prefix, metric_name)
                )

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            wandb.log(log_dict)  # Log all
            plt.close("all")  # Close all figs

    def on_test_epoch_end(self):
        print('Outputting full output')
        self.output.to_netcdf("output/final_output.nc")

    def on_load_checkpoint(self, checkpoint):
        """
        Perform any changes to state dict before loading checkpoint
        """
        loaded_state_dict = checkpoint["state_dict"]

        # Fix for loading older models after IneractionNet refactoring, where
        # the grid MLP was moved outside the encoder InteractionNet class
        if "g2m_gnn.grid_mlp.0.weight" in loaded_state_dict:
            replace_keys = list(
                filter(
                    lambda key: key.startswith("g2m_gnn.grid_mlp"),
                    loaded_state_dict.keys(),
                )
            )
            for old_key in replace_keys:
                new_key = old_key.replace(
                    "g2m_gnn.grid_mlp", "encoding_grid_mlp"
                )
                loaded_state_dict[new_key] = loaded_state_dict[old_key]
                del loaded_state_dict[old_key]
        if not self.restore_opt:
            opt = self.configure_optimizers()
            checkpoint["optimizer_states"] = [opt.state_dict()]
