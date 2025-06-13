# Standard library
import json

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed

from parameters import InferenceParameters

from types import SimpleNamespace

import yaml

# Local
from neural_lam.config import load_config_and_datastore
from neural_lam.models import GraphLAM, HiLAM, HiLAMParallel
from neural_lam.weather_dataset import WeatherDataModule

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}


def dict_to_namespace(d):
    """Recursively converts a dict to a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def forecast_weatehr(parameters: InferenceParameters):
    # Load YAML config
    with open(parameters.args_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # If 'var_leads_metrics_watch' is given as a string (like '{"1": [1, 2]}'), convert to dict
    if isinstance(
        config_dict.get("logger", {}).get("var_leads_metrics_watch", {}), str
    ):
        config_dict["logger"]["var_leads_metrics_watch"] = json.loads(
            config_dict["logger"]["var_leads_metrics_watch"]
        )


    config_ns = dict_to_namespace(config_dict)


    print("Loaded config:")
    print(config_ns)


    # Set seed
    seed.seed_everything(config_ns.seed)


    # Load neural-lam configuration and datastore to use
    config, datastore = load_config_and_datastore(config_path=config_ns.config_path)

    # Create datamodule
    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=config_ns.ar_steps_train,
        ar_steps_eval=config_ns.ar_steps_eval,
        standardize=True,
        num_past_forcing_steps=config_ns.num_past_forcing_steps,
        num_future_forcing_steps=config_ns.num_future_forcing_steps,
        batch_size=config_ns.batch_size,
        num_workers=config_ns.num_workers,
    )

    print('Data Module Created')


    if torch.cuda.is_available() and not config_ns.use_cpu:
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )
    else:
        device_name = "cpu"

    print(f"Using device: {device_name}")


    ModelClass = MODELS[config_ns.model]
    model = ModelClass(config_ns, config=config, datastore=datastore)

    print("Model created!")


    trainer = pl.Trainer(
        max_epochs=config_ns.epochs,
        deterministic=True,
        strategy="ddp",
        accelerator=device_name,
        check_val_every_n_epoch=config_ns.val_interval,
        precision=config_ns.precision,
    )


    trainer.test(model=model, datamodule=data_module, ckpt_path=config_ns.load)


if __name__ == "__main__":


    params = InferenceParameters()
    forecast_weatehr(params)
