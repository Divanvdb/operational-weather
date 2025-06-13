# Standard library
import json

# Third-party
import torch

from neural_lam.config import NeuralLAMConfig
from neural_lam.datastore import BaseDatastore

from parameters import InferenceParameters

from types import SimpleNamespace

import yaml

# Local
from neural_lam.config import load_config_and_datastore
from neural_lam.models import GraphLAM
from neural_lam.weather_dataset import WeatherDataModule

MODELS = {
    "graph_lam": GraphLAM,
}


def dict_to_namespace(d):
    """Recursively converts a dict to a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def load_params_config(parameters: InferenceParameters):
    with open(parameters.args_path, "r") as f:
        config_dict = yaml.safe_load(f)


    if isinstance(
        config_dict.get("logger", {}).get("var_leads_metrics_watch", {}), str
    ):
        config_dict["logger"]["var_leads_metrics_watch"] = json.loads(
            config_dict["logger"]["var_leads_metrics_watch"]
        )

    config_return = dict_to_namespace(config_dict)

    print("Loaded config:")

    return config_return


def create_data_module(config_ns: SimpleNamespace, datastore: BaseDatastore):
    module =  WeatherDataModule(
        datastore=datastore,
        ar_steps_train=config_ns.ar_steps_train,
        ar_steps_eval=config_ns.ar_steps_eval,
        standardize=True,
        num_past_forcing_steps=config_ns.num_past_forcing_steps,
        num_future_forcing_steps=config_ns.num_future_forcing_steps,
        batch_size=config_ns.batch_size,
        num_workers=config_ns.num_workers,
    )

    module.setup('test')

    return module


def get_device(config_ns: SimpleNamespace):
    if torch.cuda.is_available() and not config_ns.use_cpu:
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )
    else:
        device_name = "cpu"

    return device_name


def create_model(config_ns, config: NeuralLAMConfig, datastore: BaseDatastore):
    ModelClass = MODELS[config_ns.model]
    model = ModelClass(config_ns, config=config, datastore=datastore)

    checkpoint = torch.load(config_ns.load, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])


    return model


def generate_output(model: GraphLAM, data_module: WeatherDataModule):
    batch = next(iter(data_module.test_dataloader()))

    model.eval()

    model.test_step(batch, 1)

    forecast = model.on_test_epoch_end()

    return forecast


def forecast_weather(parameters: InferenceParameters):

    config_ns = load_params_config(parameters)


    config, datastore = load_config_and_datastore(config_path=config_ns.config_path)


    data_module = create_data_module(config_ns, datastore)
    print('Data Module Created')


    device_name = get_device(config_ns)
    print(f"Using device: {device_name}")

    model = create_model(config_ns, config, datastore)
    print("Model created!")


    output = generate_output(model, data_module)


    return output

if __name__ == "__main__":


    params = InferenceParameters()
    forecasted_weather = forecast_weather(params)

    forecasted_weather.to_netcdf("output/forecasted_weather.nc")

    print(forecasted_weather)
