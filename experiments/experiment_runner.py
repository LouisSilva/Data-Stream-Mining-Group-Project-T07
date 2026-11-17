import os
import importlib
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from experiments.utils.config import load_yaml_config
from experiments.utils.datasets import build_datasets_from_yaml_config
from experiments.utils.metrics import resolve_metrics_from_yaml_config

class ExperimentRunner(ABC):
    def __init__(
            self,
            experiment_name: str,
            config_path: Path | str,
    ) -> None:
        self.config = load_yaml_config(config_path)
        self.experiment_name = experiment_name
        
        try: self.experiment_config = self.config["experiments"][experiment_name]
        except KeyError: raise ValueError(f"Given experiment name '{experiment_name}' not found in the given config: {config_path}.") from None

        self.models_to_compare = [
            model_name
            for model_name in self.experiment_config["models"]
            if self._is_model_allowed(model_name)
        ]

        self.datasets_to_compare = build_datasets_from_yaml_config(self.config, self.experiment_config["datasets"])
        self.metrics_to_use = resolve_metrics_from_yaml_config(self.experiment_config["metrics"])

        self.drift_markers = {
            name: self.config["datasets"][name]["drift_markers"]
            for name in self.experiment_config["datasets"]
            if "drift_markers" in self.config["datasets"][name]
        }

        self.model_classes = {}
        for model_name in self.models_to_compare:
            module_name, class_name = self.config["models"][model_name]["import"].split(":", 1)
            self.model_classes[model_name] = getattr(importlib.import_module(module_name), class_name)

        if base_outdir := self.experiment_config.get("base_outdir"):
            os.makedirs(base_outdir, exist_ok=True)

    def _create_nf_model(self, model_name: str, n_features: int, n_classes: int):
        """Create a fresh neuro-fuzzy model using its YAML parameters."""
        return self.model_classes[model_name](
            n_features=n_features,
            n_classes=n_classes,
            **self.config["models"][model_name].get("parameters", {}),
        )

    def _make_summary_row(
        self,
        dataset_name: str,
        model_name: str,
        n_samples: int,
        result: dict[str, Any],
        mode: str,
        ablation: str | None = None,
    ) -> dict[str, Any]:
        """Use the same summary schema for comparison and ablation results."""
        row = {
            "mode": mode,
            "dataset": dataset_name,
            "model": model_name,
            "ablation": ablation,
            "N": n_samples,
            "n_rules_final": int(result["n_rules_final"]),
            "time_s": float(result["time_s"]),
            "pred_fallbacks": int(result["pred_fallbacks"]),
            "P_instabilities": int(result["P_instabilities"]),
        }
        
        for metric in self.metrics_to_use:
            values = result["metric_histories"][metric]
            row[f"{metric.suffix}_final"] = float(values[-1])
            row[f"{metric.suffix}_mean"] = float(np.nanmean(values))
        return row

    def _is_model_allowed(self, model_name: str) -> bool:
        return True

    @abstractmethod
    def run(self) -> None:
        pass
        
    def __str__(self) -> str:
        return (
            f"Experiment: {self.experiment_name}\n"
            f"Models: {', '.join(self.models_to_compare)}\n"
            f"Datasets: {', '.join(self.datasets_to_compare)}\n"
            f"Metrics: {', '.join(metric.label for metric in self.metrics_to_use)}"
        )

