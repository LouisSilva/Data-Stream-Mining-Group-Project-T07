from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


@dataclass(frozen=True)
class Metric:
    label: str
    suffix: str
    calculate: Callable[[np.ndarray], float]


def accuracy(confusion: np.ndarray) -> float:
    return float(np.trace(confusion) / confusion.sum())


def macro_f1(confusion: np.ndarray) -> float:
    tp = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(
        2 * tp, denominator,
        out=np.zeros(len(tp), dtype=float), where=denominator != 0,
    )
    return float(f1.mean())


# Add a new metric here (name -> label, filename suffix, calculation).
METRICS: dict[str, Metric] = {
    "Accuracy": Metric("Accuracy", "acc", accuracy),
    "Macro-F1": Metric("Macro-F1", "macro_f1", macro_f1),
}


def resolve_metrics_from_yaml_config(names: List[str]) -> List[Metric]:
    """Convert YAML metric names into Metric objects."""
    unknown = set(names) - METRICS.keys()
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Supported: {list(METRICS)}")
    return [METRICS[name] for name in names]


def compute_metric_histories(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[Metric],
    n_classes: int,
    evaluated_start: int = 0,
) -> Dict[str, np.ndarray]:
    """Cumulative prequential scores; unscored warm-up entries are NaN."""
    n = len(y_true)
    if len(y_pred) != n or not (0 <= evaluated_start < n):
        raise ValueError("Invalid prediction lengths or evaluated_start")

    histories = {metric: np.full(n, np.nan, dtype=float) for metric in metrics}
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t in range(evaluated_start, n):
        yt, yh = int(y_true[t]), int(y_pred[t])
        if not (0 <= yt < n_classes and 0 <= yh < n_classes):
            raise ValueError(f"Invalid predicted/actual class at t={t}: {yh}, {yt}")
            
        confusion[yt, yh] += 1
        for metric in metrics:
            histories[metric][t] = metric.calculate(confusion)

    return histories
