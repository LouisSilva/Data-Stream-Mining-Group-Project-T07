from __future__ import annotations

import os
from typing import Dict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from src.utils.metrics import Metric


def _draw_drift_markers(ax, dataset_name: str, drift_markers: dict, alpha: float = 0.75) -> None:
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


def set_plot_style(plotting_config: dict, aspect_ratio=0.618, dpi=None) -> None:
    inches_per_pt = 1 / 72.27
    font_size = plotting_config["overleaf_font_size"]
    fig_width_in = plotting_config["overleaf_text_width_pt"] * inches_per_pt
    fig_height_in = fig_width_in * aspect_ratio

    if dpi == None: dpi = plotting_config["default_dpi"]

    plt.rcdefaults()
    plt.rcParams.update({
        "text.usetex": True,
        "figure.figsize": (fig_width_in, fig_height_in),
        "figure.dpi": dpi,
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
    })


def save_figure(fig: Figure, filepath: str | Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try: fig.savefig(filepath, bbox_inches="tight", pad_inches=0.05)
    except Exception as e: print(f"Caught exception {e} whilst saving figure to {filepath}")
    finally: plt.close(fig)


def plot_all_models_metric(
    plotting_config: dict, 
    dataset_name, 
    model_results, 
    outdir, 
    metric: Metric, 
    drift_markers: dict
) -> None:
    """Cumulative prequential curve for one selected metric."""
    set_plot_style(plotting_config)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for model_name, res in model_results.items():
        values = res["metric_histories"][metric]
        steps = np.arange(len(values))
        ax.plot(steps, values, label=model_name)

    _draw_drift_markers(ax, dataset_name, drift_markers)
    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Cumulative prequential " + metric.label, fontsize=9)
    ax.set_title(dataset_name, fontsize=9)
    ax.set_ylim(-0.025, 1.025)
    ax.grid(alpha=0.3)
    ax.legend(
        fontsize=7, loc="lower right", ncol=2,
        frameon=True, framealpha=0.90, borderpad=0.2,
        handlelength=1.6,
    )
    
    fig.tight_layout()
    save_figure(fig, Path(outdir) / f"{dataset_name}_{metric.suffix}_all_models.png")


def plot_all_models_rules(
    plotting_config: dict, 
    dataset_name, 
    model_results, 
    outdir, 
    drift_markers: dict
) -> None:
    """One figure per dataset: rule-count evolution curves (when available)."""
    set_plot_style(plotting_config)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    has_any = False
    for model_name, res in model_results.items():
        rules = res.get("n_rules_time", None)
        if rules is None or np.all(np.isnan(rules)):
            continue
        t = np.arange(len(rules))
        ax.plot(t, rules, label=model_name)
        has_any = True

    if not has_any:
        plt.close(fig)
        return

    _draw_drift_markers(ax, dataset_name, drift_markers)

    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Number of rules", fontsize=9)
    ax.set_title(dataset_name + " – rule evolution", fontsize=9)
    ax.grid(alpha=0.3)

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
    save_figure(fig, Path(outdir) / f"{dataset_name}_rules_all_models.png")


def plot_ablation_metric(
    plotting_config: dict,
    dataset_name: str, 
    curves: Dict[str, np.ndarray], 
    outdir: str,
    metric: Metric, 
    drift_markers: dict,
) -> None:
    set_plot_style(plotting_config)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    for label, acc in curves.items():
        t = np.arange(len(acc))
        ax.plot(t, acc, label=label)

    _draw_drift_markers(ax, dataset_name, drift_markers, alpha=0.70)

    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Prequential " + metric.label, fontsize=9)
    ax.set_title(dataset_name, fontsize=9)
    ax.grid(alpha=0.3)

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
    save_figure(fig, Path(outdir) / f"{dataset_name}_{metric.suffix}_ablation.png")
