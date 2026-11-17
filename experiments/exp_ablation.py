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

from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Callable
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from src.utils.config import load_yaml_config
from src.utils.datasets import load_stream_to_numpy, min_max_scale, build_datasets
from src.utils.metrics import compute_metric_histories, Metric, resolve_metrics_from_yaml_config
from src.utils.latex import build_ablation_latex_table
from src.utils.plotting import plot_ablation_metric


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "experiment_config.yaml"


# ---------------------------------------------------------------------
# Ablation controls
# ---------------------------------------------------------------------
def make_fixed_aggregate(mode: str) -> Callable[[np.ndarray, float], float]:
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


@dataclass(frozen=True)
class AblationConfig:
    name: str
    agg_mode: str
    force_no_w: bool


ABLATIONS: List[AblationConfig] = [
    AblationConfig(name="FULL", agg_mode="FULL", force_no_w=False),
    AblationConfig(name="FIXED_AND", agg_mode="FIXED_AND", force_no_w=False),
    AblationConfig(name="FIXED_OR", agg_mode="FIXED_OR", force_no_w=False),
    AblationConfig(name="FIXED_COMP", agg_mode="FIXED_COMP", force_no_w=False),
    AblationConfig(name="NO_W", agg_mode="FULL", force_no_w=True),
]


def apply_ablation_patches(ab: AblationConfig, model_cls):
    module = importlib.import_module(model_cls.__module__)
    original = module.unim_aggregate
    module.unim_aggregate = make_fixed_aggregate(ab.agg_mode)
    return module, original


def enforce_no_w(model: Any):
    if not hasattr(model, "rules"): return
        
    try:
        for r in model.rules:
            if hasattr(r, "w"): r.w = np.ones_like(r.w, dtype=float)
    except Exception:
        return


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    config = load_yaml_config(CONFIG_PATH)
    ablation_config = config["experiments"]["ablation"]
    os.makedirs(ablation_config["base_outdir"], exist_ok=True)

    drift_markers = {
        name: dataset_config["drift_markers"]
        for name, dataset_config in config["datasets"].items()
        if "drift_markers" in dataset_config
    }

    datasets = build_datasets(config, ablation_config["datasets"])

    models_to_compare = ablation_config["models"]
    metrics_to_use = resolve_metrics_from_yaml_config(ablation_config["metrics"])
    
    model_classes = {}
    for name in models_to_compare:
        if config["models"][name]["kind"] != "nf":
            raise ValueError(f"Ablation requires kind: nf, got {name}")
        
        module_name, class_name = config["models"][name]["import"].split(":", 1)
        model_classes[name] = getattr(importlib.import_module(module_name), class_name)

    print(f"Using config: {CONFIG_PATH}")
    print(f"Selected models: {models_to_compare}")
    print(f"Selected metrics: {metrics_to_use}")

    ablations = [ab for ab in ABLATIONS if ab.name in ablation_config["ablations"]]

    summary_rows: List[Dict[str, Any]] = []
    for ds_name in ablation_config["datasets"]:
        ds = datasets[ds_name]
        print(f"\n=== DATASET: {ds_name} ===")

        n_max = config["datasets"][ds_name]["lengths"][ablation_config["experiment_mode"]]

        try:
            X_raw, y, _ = load_stream_to_numpy(ds, n_max=n_max)
        except ValueError as e:
            print(f"  [SKIP] {ds_name}: non-numeric feature encountered ({e}).")
            continue
        except Exception as e:
            print(f"  [SKIP] {ds_name}: failed to load ({e}).")
            continue

        X, _, _ = min_max_scale(X_raw)
        n_samples, n_features = X.shape
        n_classes = int(len(np.unique(y)))

        outdir_ds = os.path.join(ablation_config["base_outdir"], ds_name)
        os.makedirs(outdir_ds, exist_ok=True)

        curves_for_plot: Dict[Metric, Dict[str, np.ndarray]] = {metric: {} for metric in metrics_to_use}
        
        for model_name in model_classes:
            model_cls = model_classes[model_name]
            
            for ab in ABLATIONS:
                label = f"{model_name}-{ab.name}"
                print(f"  -> {label}")

                module, original_aggregate = apply_ablation_patches(ab, model_cls)

                # Instantiate using model parameters from YAML
                model = model_cls(
                    n_features=n_features, n_classes=n_classes,
                    **config["models"][model_name].get("parameters", {})
                )

                # Custom prequential loop to allow NO_W enforcement without altering the main evaluator
                n = len(y)
                predictions = np.zeros(n, dtype=int)
                correct = np.zeros(n, float)
                acc = np.zeros(n, float)

                t0 = time.perf_counter()
                for t in tqdm(range(n), desc=f"{label} ({ds_name})", leave=False):
                    x_t = X[t : t + 1]
                    yt = int(y[t])

                    if t == 0:
                        y_hat = 0
                    else:
                        try:
                            yhat_raw = np.asarray(model.predict(x_t)).ravel()
                            y_hat = int(yhat_raw[0]) if yhat_raw.size else 0
                        except Exception:
                            y_hat = 0

                    predictions[t] = y_hat

                    model.partial_fit(x_t, np.array([yt]))
                    if ab.force_no_w:
                        enforce_no_w(model)

                exec_time = time.perf_counter() - t0
                module.unim_aggregate = original_aggregate

                n_rules_final = len(getattr(model, "rules", []))
                histories = compute_metric_histories(y, predictions, metrics_to_use, n_classes)
                row = dict(
                    mode=ablation_config["experiment_mode"], 
                    dataset=ds_name,
                    model=model_name, 
                    ablation=ab.name, 
                    N=n_samples,
                    n_rules_final=int(n_rules_final), 
                    time_s=float(exec_time),
                )

                
                report = []
                for metric in metrics_to_use:
                    values = histories[metric]
                    row[f"{metric.suffix}_final"] = float(values[-1])
                    row[f"{metric.suffix}_mean"] = float(np.nanmean(values))
                    curves_for_plot[metric][label] = values
                    report.append(f"{metric.label}_final={values[-1]:.4f}")
                
                summary_rows.append(row)
                print(f"     [RESULT] {' | '.join(report)} | Rules_final={n_rules_final} | Time={exec_time:.1f}s")

        for metric in metrics_to_use:
            if curves_for_plot[metric]:
                plot_ablation_metric(config["plotting"], ds_name, curves_for_plot[metric], outdir_ds, metric, drift_markers)

    tex_path = os.path.join(ablation_config["base_outdir"], "exp_ablation_summary.tex")
    build_ablation_latex_table(summary_rows, tex_path, metrics_to_use)
    print(f"\nLaTeX ablation table saved to: {tex_path}")
    print("Figures saved under:", ablation_config["base_outdir"])


if __name__ == "__main__":
    main()
