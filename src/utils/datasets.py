from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from river import datasets


def load_stream_to_numpy(ds, n_max: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert a river dataset/stream into NumPy arrays (X, y_int).

    Returns
    -------
    X : (N, d) float
    y_int : (N,) int labels encoded as {0,...,C-1}
    feature_names : list[str] feature key order used for vectorization

    Notes
    -----
    Assumes numeric features. Text datasets require vectorization.
    """
    X_list = []
    y_raw_list = []

    it = ds.__iter__() if hasattr(ds, "__iter__") else iter(ds)

    try:
        x0, y0 = next(it)
    except StopIteration:
        raise RuntimeError("Empty stream/dataset")

    feature_names = list(x0.keys())
    x0_vec = np.array([x0[f] for f in feature_names], dtype=float)
    X_list.append(x0_vec)
    y_raw_list.append(y0)

    for i, (x, y) in enumerate(it, start=1):
        if i >= n_max:
            break
        x_vec = np.array([x[f] for f in feature_names], dtype=float)
        X_list.append(x_vec)
        y_raw_list.append(y)

    X = np.vstack(X_list).astype(float)

    # encode labels to integers
    unique_labels = list(dict.fromkeys(y_raw_list))  # preserves order
    label_to_int = {lab: idx for idx, lab in enumerate(unique_labels)}
    y_int = np.array([label_to_int[lab] for lab in y_raw_list], dtype=int)

    return X, y_int, feature_names


def min_max_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max scaling to [0, 1]."""
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    denom = x_max - x_min
    denom[denom == 0.0] = 1.0
    X_scaled = (X - x_min) / denom
    return X_scaled, x_min, x_max


def safe_instantiate_real_datasets(candidates: List[str]) -> Dict[str, Any]:
    """Instantiate candidate River real datasets if available in this River version."""
    out: Dict[str, Any] = {}
    for cls_name in candidates:
        cls = getattr(datasets, cls_name, None)
        if cls is None:
            continue
        try:
            ds = cls()
        except Exception:
            continue
        key = f"{cls_name}_real"
        out[key] = ds
    return out

