"""
Experiment 2 – Ablation study for UnimNeuron-based evolving neuro-fuzzy models
============================================================================

This script runs an *ablation* study focused on the proposed UnimNeuron-based
models

Ablations (antecedent side)
---------------------------
We evaluate how much each component contributes by selectively disabling it:

1) FULL (UnimNeuron): adaptive AND/OR/COMP regime (default unim_aggregate).
2) FIXED_AND: force antecedent aggregation to behave as AND (min).
3) FIXED_OR:  force antecedent aggregation to behave as OR  (max).
4) FIXED_COMP: force antecedent aggregation to behave as COMP (mean).
5) NO_W: disable feature relevance weights by forcing w_j = 1 for all rules
         (keeps FULL UnimNeuron regime unless combined manually).

Outputs
-------
results/ablation/
  <dataset_name>/
     <dataset>_<metric>_ablation.png
     rules_<model>__<ablation>.txt           (optional, best-effort)
  exp_ablation_summary.tex

How to run
----------
python -m experiments.exp_ablation
"""
import importlib
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Callable
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.utils.datasets import load_stream_to_numpy, min_max_scale
from src.utils.metrics import compute_metric_histories
from src.utils.latex import build_ablation_latex_table
from src.utils.plotting import plot_ablation_metric
from experiments.experiment_runner import ExperimentRunner


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "experiment_config.yaml"
#-----------------------------------------------------------------------


@dataclass(frozen=True)
class AblationConfig:
    name: str
    agg_mode: str
    force_no_w: bool


# TODO: Put this into the yaml config
ABLATIONS: List[AblationConfig] = [
    AblationConfig(name="FULL", agg_mode="FULL", force_no_w=False),
    AblationConfig(name="FIXED_AND", agg_mode="FIXED_AND", force_no_w=False),
    AblationConfig(name="FIXED_OR", agg_mode="FIXED_OR", force_no_w=False),
    AblationConfig(name="FIXED_COMP", agg_mode="FIXED_COMP", force_no_w=False),
    AblationConfig(name="NO_W", agg_mode="FULL", force_no_w=True),
]


class AblationExperimentRunner(ExperimentRunner):
    def __init__(self, config_path: Path | str = CONFIG_PATH) -> None:
        super().__init__("ablation", config_path)

    def _is_model_allowed(self, model_name: str) -> bool:
        return self.config["models"][model_name]["kind"] == "nf"

    def __enforce_no_w(self, model: Any) -> None:
        if not hasattr(model, "rules"): 
            return
        try:
            for r in model.rules:
                if hasattr(r, "w"): 
                    r.w = np.ones_like(r.w, dtype=float)
        except Exception:
            return

    def __apply_ablation_patches(self, ablation: AblationConfig, model_cls):
        module = importlib.import_module(model_cls.__module__)
        original = module.unim_aggregate
        module.unim_aggregate = self.__make_fixed_aggregate(ablation.agg_mode)
        return module, original

    def __make_fixed_aggregate(self, mode: str) -> Callable[[np.ndarray, float], float]:
        """Return a unim_aggregate-like function (h, e) -> float."""
        mode = mode.upper()
    
        def full(h: np.ndarray, e: float = 0.5) -> float:
            z = np.asarray(h, dtype=float)
            if z.size == 0:
                return float(e)
            all_le = np.all(z <= e)
            all_ge = np.all(z >= e)
            if all_le and not all_ge:
                return float(np.min(z))
            if all_ge and not all_le:
                return float(np.max(z))
            return float(np.mean(z))
    
        if mode == "FULL":
            return full
        if mode == "FIXED_AND":
            return lambda h, e=0.5: (
                float(np.min(np.asarray(h, dtype=float)))
                if np.asarray(h).size
                else float(e)
            )
        if mode == "FIXED_OR":
            return lambda h, e=0.5: (
                float(np.max(np.asarray(h, dtype=float)))
                if np.asarray(h).size
                else float(e)
            )
        if mode == "FIXED_COMP":
            return lambda h, e=0.5: (
                float(np.mean(np.asarray(h, dtype=float)))
                if np.asarray(h).size
                else float(e)
            )
        raise ValueError(f"Unknown aggregation mode: {mode}")

    def run(self) -> None:
        print(self)
        
        ablations = [ab for ab in ABLATIONS if ab.name in self.experiment_config["ablations"]]
        summary_rows: List[Dict[str, Any]] = []
        
        for dataset_name, dataset in self.datasets_to_compare.items():
            print(f"\n=== DATASET: {dataset_name} ===")

            n_max = self.config["datasets"][dataset_name]["lengths"][self.experiment_config["experiment_mode"]]
            try:
                X_raw, y, _ = load_stream_to_numpy(dataset, n_max=n_max)
            except ValueError as e:
                print(f"  [SKIP] {dataset_name}: non-numeric feature encountered ({e}).")
                continue
            except Exception as e:
                print(f"  [SKIP] {dataset_name}: failed to load ({e}).")
                continue

            X, _, _ = min_max_scale(X_raw)
            n_samples, n_features = X.shape
            n_classes = int(len(np.unique(y)))

            dataset_outdir = os.path.join(self.experiment_config["base_outdir"], dataset_name)
            os.makedirs(dataset_outdir, exist_ok=True)
            results_for_dataset: Dict[str, Dict[str, Any]] = {}

            for model_name in self.models_to_compare:
                model_cls = self.model_classes[model_name]
                for ablation in ablations:
                    label = f"{model_name}-{ablation.name}"
                    print(f"  -> {label}")

                    module, original_aggregate = self.__apply_ablation_patches(ablation, model_cls)

                    try:
                        model = self._create_nf_model(model_name, n_features, n_classes)

                        # Custom prequential loop to allow NO_W enforcement without altering the main evaluator
                        n = len(y)
                        predictions = np.zeros(n, dtype=int)

                        pred_fallbacks = 0
                        t0 = time.perf_counter()
                        for t in tqdm(range(n), desc=f"{label} ({dataset_name})", leave=False):
                            x_t = X[t : t + 1]
                            yt = int(y[t])

                            if t == 0:
                                y_hat = 0
                            else:
                                try:
                                    yhat_raw = np.asarray(model.predict(x_t)).ravel()
                                    if yhat_raw.size == 0 or not np.isfinite(yhat_raw[0]):
                                        raise FloatingPointError
                                    y_hat = int(yhat_raw[0])
                                except Exception:
                                    y_hat = 0
                                    pred_fallbacks += 1

                            predictions[t] = y_hat

                            model.partial_fit(x_t, np.array([yt]))
                            if ablation.force_no_w:
                                self.__enforce_no_w(model)

                        exec_time = time.perf_counter() - t0
                    finally:
                        module.unim_aggregate = original_aggregate

                    result = {
                        "y_pred": predictions,
                        "evaluated_start": 0,
                        "metric_histories": compute_metric_histories(y, predictions, self.metrics_to_use, n_classes),
                        "n_rules_final": len(getattr(model, "rules", [])),
                        "time_s": exec_time,
                        "N": n_samples,
                        "pred_fallbacks": pred_fallbacks,
                        "P_instabilities": int(getattr(model, "n_rls_resets", 0)),
                    }
                    
                    results_for_dataset[label] = result
                    summary_rows.append(
                        self._make_summary_row(
                            dataset_name, model_name, n_samples, result,
                            mode=self.experiment_config["experiment_mode"], ablation=ablation.name,
                        )
                    )

                    report = [
                        f"{metric.label}_final={result['metric_histories'][metric][-1]:.4f}"
                        for metric in self.metrics_to_use
                    ]
                    
                    print(
                        f"     [RESULT] {' | '.join(report)} | "
                        f"Rules_final={result['n_rules_final']} | Time={exec_time:.1f}s"
                    )

            for metric in self.metrics_to_use:
                curves = {
                    label: result["metric_histories"][metric]
                    for label, result in results_for_dataset.items()
                }
                
                if curves:
                    plot_ablation_metric(self.config["plotting"], dataset_name, curves, dataset_outdir, metric, self.drift_markers)

        tex_path = os.path.join(self.experiment_config["base_outdir"], "exp_ablation_summary.tex")
        build_ablation_latex_table(summary_rows, tex_path, self.metrics_to_use)
        print(f"\nLaTeX ablation table saved to: {tex_path}")
        print("Figures saved under:", self.experiment_config["base_outdir"])
                

if __name__ == "__main__":
    AblationExperimentRunner().run()
