"""
Experiment – Stream experiments with evolving fuzzy models
=============================================================================

This script runs *prequential* (predict-then-update) evaluations on a collection
of real and synthetic data streams, comparing:

Baselines (evolvingfuzzysystems.eFS)
-----------------------------------
- ePL, ePL_plus, exTS, Simpl_eTS, eMG

Neuro-fuzzy evolving models (local code)
----------------------------------------
- ENF_UnimNeuron_Safe
- ENF_UnimNeuron_Safe_Local
- ENF_UnimNeuron_PA

Datasets (river)
----------------
Real:
- Elec2_real      -> river.datasets.Elec2
- CreditCard_real -> river.datasets.CreditCard
- Phishing_real   -> river.datasets.Phishing
- Bananas_real    -> river.datasets.Bananas

Synthetic:
- ConceptDrift_Agrawal -> river.datasets.synth.ConceptDriftStream(Agrawal->Agrawal)
- Agrawal_synth        -> river.datasets.synth.Agrawal
- RandomRBFDrift_synth -> river.datasets.synth.RandomRBFDrift
- Sine_synth           -> river.datasets.synth.Sine

Protocol (reproducible, conference-friendly)
--------------------------------------------
1) Stream -> NumPy (X, y_int)
2) Min-max scaling X to [0, 1]
3) Prequential evaluation
4) Save:
   - Console summary lines ([RESULT])
   - Per-dataset plots (configured metric curves and rule evolution)
   - LaTeX tables (configured metric summaries + statistical tests)
   - Optional: exported fuzzy rules (best-effort) for interpretability

Outputs
-------------
results/sota/
  <dataset_name>/
     <dataset>_<metric>_all_models.png
     <dataset>_rules_all_models.png
     rules_<model>.txt          (optional, when export is available)
  exp_stream_summary.tex
  exp_stats_friedman.tex
  exp_stats_posthoc.tex        (if posthoc available)
  
How to run
----------
python -m experiments.exp_sota_comparison
"""
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm.auto import tqdm

from experiments.utils.datasets import load_stream_to_numpy, min_max_scale
from experiments.utils.metrics import compute_metric_histories, Metric
from experiments.utils.latex import build_sota_latex_table, build_stats_tables, export_rules_latex
from experiments.utils.plotting import plot_all_models_metric, plot_all_models_rules
from experiments.utils.evaluation import SafeEFSWrapper, prequential_nf_generic, prequential_efs
from experiments.experiment_runner import ExperimentRunner


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "experiment_config.yaml"
# -------------------------------------------------------------------------


class ModelComparisonExperimentRunner(ExperimentRunner):
    def __init__(self, config_path: Path | str = CONFIG_PATH) -> None:
        super().__init__("sota_comparison", config_path)

    def run(self) -> None:
        print(self)

        summary_rows: List[Dict[str, Any]] = []
        metric_means_for_stats: Dict[Metric, List[List[float]]] = {
            metric: [] for metric in self.metrics_to_use
        }

        for dataset_name, dataset in self.datasets_to_compare.items():
            print(f"\n=== DATASET: {dataset_name} ===")
            
            n_max = self.config["datasets"][dataset_name]["lengths"][self.experiment_config["experiment_mode"]]
            X_raw, y, feature_names = load_stream_to_numpy(dataset, n_max=n_max)
            n_samples, n_features = X_raw.shape
            n_classes = len(np.unique(y))
            X, _, _ = min_max_scale(X_raw)
            
            dataset_outdir = os.path.join(self.experiment_config["base_outdir"], dataset_name)
            os.makedirs(dataset_outdir, exist_ok=True)
            results_for_dataset: Dict[str, Dict[str, Any]] = {}

            for model_name in self.models_to_compare:
                print(f"  -> Model: {model_name}")
                model_cls = self.model_classes[model_name]
                model_config = self.config["models"][model_name]
                
                if model_config["kind"] == "nf":
                    model = self._create_nf_model(model_name, n_features, n_classes)
                    eval_fn = prequential_nf_generic
                    
                elif model_config["kind"] == "efs":
                    ef_model = model_cls(**model_config.get("parameters", {}))
                    model = (SafeEFSWrapper(ef_model, name=model_name)
                             if model_config.get("safe_wrapper", model_name != "ePL") else ef_model)
                    
                    eval_fn = lambda m, X_, y_, desc: prequential_efs(
                        m, X_, y_, warmup=self.experiment_config["warmup"],
                        batch_evolve=self.experiment_config["batch_evolve"], desc=desc,
                    )
                    
                else: raise ValueError(f"Unsupported model kind: {model_config['kind']}")

                t0 = time.perf_counter()
                result = eval_fn(model, X, y, desc=f"{model_name} ({dataset_name}, mode={self.experiment_config["experiment_mode"]})")
                exec_time = time.perf_counter() - t0

                # Evaluate only pre-training test predictions; the first WARMUP
                # examples of eFS models were used for training, not prediction.
                result["metric_histories"] = compute_metric_histories(
                    y_true=y, 
                    y_pred=result["y_pred"], 
                    metrics=self.metrics_to_use,
                    n_classes=n_classes, 
                    evaluated_start=result["evaluated_start"],
                )
                result["time_s"] = exec_time
                result["N"] = n_samples
                
                results_for_dataset[model_name] = result
                summary_rows.append(
                    self._make_summary_row(
                        dataset_name, model_name, n_samples, result, mode=self.experiment_config["experiment_mode"]
                    )
                )
                
                report_metrics = []
                for metric in self.metrics_to_use:
                    values = result["metric_histories"][metric]
                    report_metrics.append(
                        f"{metric.label}_final={values[-1]:.4f} | "
                        f"{metric.label}_mean={np.nanmean(values):.4f}"
                    )

                print(
                    f"[RESULT] mode={self.experiment_config["experiment_mode"]:8s} | dataset={dataset_name:18s} | "
                    f"model={model_name:20s} | N={n_samples:5d} | "
                    + " | ".join(report_metrics)
                    + f" | Rules_final={result['n_rules_final']} | Time={exec_time:.1f}s | "
                    + f"Pred_fallbacks={result['pred_fallbacks']} | "
                    + f"P_instabilities={result['P_instabilities']}"
                )

                if model_name.startswith("ENF_Unim_"):
                    export_rules_latex(
                        model=model,
                        model_name=model_name,
                        dataset_outdir=dataset_outdir,
                        feature_names=feature_names,
                        max_rules_to_write=5,
                        top_features_per_rule=6,
                        latex_env="enumerate",
                    )

            plot_all_models_rules(self.config["plotting"], dataset_name, results_for_dataset, dataset_outdir, self.drift_markers)
            for metric in self.metrics_to_use:
                plot_all_models_metric(self.config["plotting"], dataset_name, results_for_dataset, dataset_outdir, metric, self.drift_markers)

                metric_means_for_stats[metric].append([
                    float(np.nanmean(results_for_dataset[model_name]["metric_histories"][metric]))
                    for model_name in self.models_to_compare
                ])

        tex_path = os.path.join(self.experiment_config["base_outdir"], "exp_stream_summary.tex")
        build_sota_latex_table(summary_rows, tex_path, self.metrics_to_use)
        print(f"\nLaTeX summary table saved to: {tex_path}")

        if len(self.datasets_to_compare) >= 2 and len(self.models_to_compare) >= 3:
            for metric in self.metrics_to_use:
                build_stats_tables(
                    metric_mean_matrix=np.asarray(metric_means_for_stats[metric], dtype=float),
                    model_names=self.models_to_compare,
                    outdir=self.experiment_config["base_outdir"],
                    metric=metric,
                )
                
                print(f"Statistical tables for {metric.label}:")
                print(f"  - {os.path.join(self.experiment_config["base_outdir"], f'exp_stats{metric.suffix}_friedman.tex')}")
                print(f"  - {os.path.join(self.experiment_config["base_outdir"], f'exp_stats{metric.suffix}_posthoc.tex')}")
        else:
            print("Not enough models/datasets for Friedman and Nemenyi tests (need 3 models and 2 datasets).")


if __name__ == "__main__":
    ModelComparisonExperimentRunner().run()
