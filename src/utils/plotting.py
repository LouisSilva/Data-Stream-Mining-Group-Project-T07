from __future__ import annotations

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.metrics import Metric

PLOT_LINEWIDTH = 1.10
PLOT_GRID_ALPHA = 0.30
PLOT_DPI = 300


def _draw_drift_markers(ax, dataset_name: str, drift_markers: dict, alpha: float = 0.75):
    info = drift_markers.get(dataset_name, None)
    if not info:
        return

    points = info.get("points", [])
    width = int(info.get("width", 0))

    for p in points:
        # drift instant marker
        ax.axvline(p, color="red", linestyle="--", linewidth=1.0, alpha=alpha)

        # optional drift window shading
        if width and width > 0:
            left = max(0, p - width // 2)
            right = p + width // 2
            ax.axvspan(left, right, color="red", alpha=0.08, linewidth=0)


def plot_all_models_metric(dataset_name, model_results, outdir, metric: Metric, drift_markers: dict):
    """Cumulative prequential curve for one selected metric."""
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for model_name, res in model_results.items():
        values = res["metric_histories"][metric]
        steps = np.arange(len(values))
        ax.plot(steps, values, label=model_name, linewidth=PLOT_LINEWIDTH)

    _draw_drift_markers(ax, dataset_name, drift_markers)
    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Cumulative prequential " + metric.label, fontsize=9)
    ax.set_title(dataset_name, fontsize=9)
    ax.set_ylim(-0.025, 1.025)
    ax.grid(alpha=PLOT_GRID_ALPHA)
    ax.legend(
        fontsize=7, loc="lower right", ncol=2,
        frameon=True, framealpha=0.90, borderpad=0.2,
        handlelength=1.6,
    )
    
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, f"{dataset_name}_{metric.suffix}_all_models.png"),
        dpi=PLOT_DPI,
    )
    plt.close(fig)


def plot_all_models_rules(dataset_name, model_results, outdir, drift_markers: dict):
    """One figure per dataset: rule-count evolution curves (when available)."""
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    has_any = False
    for model_name, res in model_results.items():
        rules = res.get("n_rules_time", None)
        if rules is None or np.all(np.isnan(rules)):
            continue
        t = np.arange(len(rules))
        ax.plot(t, rules, label=model_name, linewidth=PLOT_LINEWIDTH)
        has_any = True

    if not has_any:
        plt.close(fig)
        return

    _draw_drift_markers(ax, dataset_name, drift_markers)

    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Number of rules", fontsize=9)
    ax.set_title(dataset_name + " – rule evolution", fontsize=9)
    ax.grid(alpha=PLOT_GRID_ALPHA)

    ax.legend(
        fontsize=7,
        loc="upper left",
        ncol=2,
        frameon=True,
        framealpha=0.90,
        borderpad=0.2,
        handlelength=1.6,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{dataset_name}_rules_all_models.png"), dpi=PLOT_DPI)
    plt.close(fig)


def plot_ablation_accuracy(dataset_name: str, curves: Dict[str, np.ndarray], outdir: str, drift_markers: dict):
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    for label, acc in curves.items():
        t = np.arange(len(acc))
        ax.plot(t, acc, label=label, linewidth=PLOT_LINEWIDTH)

    _draw_drift_markers(ax, dataset_name, drift_markers, alpha=0.70)

    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Prequential accuracy", fontsize=9)
    ax.set_title(dataset_name, fontsize=9)
    ax.grid(alpha=PLOT_GRID_ALPHA)

    leg = ax.legend(
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),  # abaixo do eixo
        ncol=2,
        frameon=True,
        framealpha=0.90,
        borderpad=0.25,
        handlelength=1.6,
        columnspacing=1.2,
    )

    fig.subplots_adjust(bottom=0.32)

    fig.savefig(os.path.join(outdir, f"{dataset_name}_acc_ablation.png"), dpi=PLOT_DPI)
    plt.close(fig)

