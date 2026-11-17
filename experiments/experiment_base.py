from pathlib import Path
import importlib

from src.utils.config import load_yaml_config
from src.utils.datasets import build_datasets
from src.utils.metrics import resolve_metrics_from_yaml_config

class ExperimentRunner:
    def __init__(
            self,
            experiment_name: str,
            config_path: Path | str,
            ) -> None:
        self.config = load_yaml_config(config_path)

        try:
            self.experiment_config = self.config[experiment_name]
        except KeyError:
            raise KeyError(f"Given experiment name '{experiment_name}' is not in the given config with path {config_path}.")

        
        self.datasets_to_compare = build_datasets(self.config, self.experiment_config["datasets"])
        self.models_to_compare = self.experiment_config["models"]
        self.metrics_to_use = resolve_metrics_from_yaml_config(self.experiment_config["metrics"])

        model_classes = {}
        for model_name in self.models_to_compare:
            if not self._is_model_allowed(model_name): continue
            
            module_name, class_name = self.config["models"][model_name]["import"].split(":", 1)
            model_classes[model_name] = getattr(importlib.import_module(module_name), class_name)

    # In Ablation experiment, override this and put the `if config["models"][name]["kind"] != "nf":` thing
    def _is_model_allowed(model_name: str) -> bool:
        return True


    # TODO: Add custom __str__ definition which includes the config path, selected models, selected datasets, and selected metrics

