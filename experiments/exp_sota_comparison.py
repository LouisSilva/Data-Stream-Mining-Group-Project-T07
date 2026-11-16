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

from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive plotting
import matplotlib.pyplot as plt
import numpy as np
import scikit_posthocs as sp
import yaml
from river import datasets
from river.datasets import synth
from scipy import stats
from tqdm.auto import tqdm

from src.utils.config import load_yaml_config
from src.utils.datasets import load_stream_to_numpy, min_max_scale
from src.utils.metrics import compute_metric_histories, Metric, resolve_metrics_from_yaml_config
from src.utils.latex import build_sota_latex_table, build_stats_tables, export_rules_best_effort, export_rules_latex
from src.utils.plotting import plot_all_models_metric, plot_all_models_rules
from src.utils.evaluation import _unim_regime_and_rho, RuleStats, SafeEFSWrapper, prequential_nf_generic, prequential_efs


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parent / "experiment_config.yaml"
# -------------------------------------------------------------------------
# Dataset definitions
# -------------------------------------------------------------------------

REAL_DATASETS = {
    "Elec2_real": datasets.Elec2(),
    "CreditCard_real": datasets.CreditCard(),
    "Phishing_real": datasets.Phishing(),
    "Bananas_real": datasets.Bananas(),  # classic 2D benchmark
}

SYNTH_DATASETS = {
    "ConceptDrift_Agrawal": synth.ConceptDriftStream(
        stream=synth.Agrawal(seed=1),
        drift_stream=synth.Agrawal(seed=2),
        seed=42,
        position=2500,  # known drift point
        width=1000,  # gradual drift window
    ),
    "Agrawal_synth": synth.Agrawal(seed=42),
    "RandomRBFDrift_synth": synth.RandomRBFDrift(),
    "Sine_synth": synth.Sine(seed=42),
}

DATASETS_ALL = {**REAL_DATASETS, **SYNTH_DATASETS}


# -------------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------------
def main():
    config = load_yaml_config(CONFIG_PATH)
    sota_config = config["experiments"]["sota_comparison"]
    os.makedirs(sota_config["base_outdir"], exist_ok=True)

    drift_markers = {
        name: dataset_config["drift_markers"]
        for name, dataset_config in config["datasets"].items()
        if "drift_markers" in dataset_config
    }
    
    models_to_compare = sota_config["models"]
    metrics_to_use = resolve_metrics_from_yaml_config(sota_config["metrics"])

    model_classes = {}
    for name in models_to_compare:
        module_name, class_name = config["models"][name]["import"].split(":", 1)
        model_classes[name] = getattr(importlib.import_module(module_name), class_name)
    
    print(f"Using config: {CONFIG_PATH}")
    print(f"Selected models: {models_to_compare}")
    print(f"Selected metrics: {metrics_to_use}")

    summary: Dict[tuple, Dict[str, Any]] = {}
    metric_means_for_stats: Dict[str, List[List[float]]] = {
        metric: [] for metric in metrics_to_use
    }

    for ds_name in sota_config["datasets"]:
        ds = DATASETS_ALL[ds_name]
        print(f"\n=== DATASET: {ds_name} ===")
        
        n_max = config["datasets"][ds_name]["lengths"][sota_config["experiment_mode"]]
        X_raw, y, feature_names = load_stream_to_numpy(ds, n_max=n_max)
        n_samples, n_features = X_raw.shape
        n_classes = len(np.unique(y))
        X, _, _ = min_max_scale(X_raw)
        outdir_ds = os.path.join(sota_config["base_outdir"], ds_name)
        os.makedirs(outdir_ds, exist_ok=True)
        results_for_ds: Dict[str, Dict[str, Any]] = {}

        for model_name in models_to_compare:
            print(f"  -> Model: {model_name}")
            model_cls = model_classes[model_name]

            spec = config["models"][model_name]
            if spec["kind"] == "nf":
                model = model_cls(n_features=n_features, n_classes=n_classes, **spec.get("parameters", {}))
                eval_fn = prequential_nf_generic
            elif spec["kind"] == "efs":
                ef_model = model_cls(**spec.get("parameters", {}))
                model = (SafeEFSWrapper(ef_model, name=model_name) if spec.get("safe_wrapper", model_name != "ePL") else ef_model)
                eval_fn = lambda m, X_, y_, desc: prequential_efs(m, X_, y_, warmup=sota_config["warmup"], batch_evolve=sota_config["batch_evolve"], desc=desc)
            else:
                raise ValueError(f"Unsupported model kind: {spec['kind']}")

            t0 = time.perf_counter()
            res = eval_fn(model, X, y, desc=f"{model_name} ({ds_name}, mode={sota_config["experiment_mode"]})")
            exec_time = time.perf_counter() - t0

            # Evaluate only pre-training test predictions; the first WARMUP examples of eFS models were used for training, not prediction.
            res["metric_histories"] = compute_metric_histories(
                y_true=y, y_pred=res["y_pred"], metrics=metrics_to_use,
                n_classes=n_classes, evaluated_start=res["evaluated_start"],
            )
            res["time_s"] = exec_time
            res["N"] = n_samples
            results_for_ds[model_name] = res

            row = dict(
                n_rules_final=int(res["n_rules_final"]),
                time_s=float(exec_time),
                pred_fallbacks=int(res["pred_fallbacks"]),
                P_instabilities=int(res["P_instabilities"]),
                N=int(n_samples),
            )
            
            report_metrics = []
            for metric in metrics_to_use:
                values = res["metric_histories"][metric]
                row[f"{metric.suffix}_final"] = float(values[-1])
                row[f"{metric.suffix}_mean"] = float(np.nanmean(values))
                report_metrics.append(
                    f"{metric.label}_final={values[-1]:.4f} | "
                    f"{metric.label}_mean={np.nanmean(values):.4f}"
                )

            summary[(sota_config["experiment_mode"], ds_name, model_name)] = row
            print(
                f"[RESULT] mode={sota_config["experiment_mode"]:8s} | dataset={ds_name:18s} | "
                f"model={model_name:20s} | N={n_samples:5d} | "
                + " | ".join(report_metrics)
                + f" | Rules_final={res['n_rules_final']} | Time={exec_time:.1f}s | "
                + f"Pred_fallbacks={res['pred_fallbacks']} | "
                + f"P_instabilities={res['P_instabilities']}"
            )

            if model_name.startswith("ENF_Unim_"):
                export_rules_latex(
                    model=model,
                    model_name=model_name,
                    dataset_outdir=outdir_ds,
                    feature_names=feature_names,
                    max_rules_to_write=5,
                    top_features_per_rule=6,
                    latex_env="enumerate",
                )

        for metric in metrics_to_use:
            plot_all_models_metric(
                config["plotting"], 
                ds_name, 
                results_for_ds, 
                outdir_ds,
                metric, 
                drift_markers
            )
            
        plot_all_models_rules(config["plotting"], ds_name, results_for_ds, outdir_ds, drift_markers)

        for metric in metrics_to_use:
            metric_means_for_stats[metric].append([
                float(np.nanmean(results_for_ds[model_name]["metric_histories"][metric]))
                for model_name in models_to_compare
            ])

    tex_path = os.path.join(BASE_OUTDIR, "exp_stream_summary.tex")
    build_sota_latex_table(summary, tex_path, metrics_to_use)
    print(f"\nLaTeX summary table saved to: {tex_path}")

    if len(sota_config["datasets"]) >= 2 and len(models_to_compare) >= 3:
        for metric in metrics_to_use:
            build_stats_tables(
                metric_mean_matrix=np.asarray(metric_means_for_stats[metric], dtype=float),
                model_names=models_to_compare,
                outdir=BASE_OUTDIR,
                metric=metric,
            )
            
            print(f"Statistical tables for {metric.label}:")
            print(f"  - {os.path.join(BASE_OUTDIR, f'exp_stats{metric.suffix}_friedman.tex')}")
            print(f"  - {os.path.join(BASE_OUTDIR, f'exp_stats{metric.suffix}_posthoc.tex')}")
    else:
        print("Not enough models/datasets for Friedman and Nemenyi tests (need 3 models and 2 datasets).")


if __name__ == "__main__":
    main()
