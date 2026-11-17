from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm


def _unim_regime_and_rho(h: np.ndarray, e: float) -> Tuple[str, float]:
    """
    Decide regime AND/OR/COMP based on h_j relative to e
    and compute rho in [-1,1] using the same indicator definition you used.

    rho = (1/n) sum( 1{h>=e} - 1{h<=e} ).
    Note: if h==e, both indicators are 1 -> cancels to 0 (neutral).
    """
    h = np.asarray(h, dtype=float).ravel()
    if h.size == 0:
        return "COMP", 0.0

    all_le = np.all(h <= e)
    all_ge = np.all(h >= e)

    if all_le and not all_ge:
        reg = "AND"
    elif all_ge and not all_le:
        reg = "OR"
    else:
        reg = "COMP"

    ge = (h >= e).astype(float)
    le = (h <= e).astype(float)
    rho = float(np.mean(ge - le))
    return reg, rho


@dataclass
class RuleStats:
    act_sum: float = 0.0
    rho_sum: float = 0.0
    n_and: float = 0.0
    n_or: float = 0.0
    n_comp: float = 0.0

    def add(self, act: float, reg: str, rho: float) -> None:
        # Use activation-weighted accounting (more faithful for evolving rules)
        a = float(max(act, 0.0))
        self.act_sum += a
        self.rho_sum += a * float(rho)
        if reg == "AND":
            self.n_and += a
        elif reg == "OR":
            self.n_or += a
        else:
            self.n_comp += a

    def summary(self) -> Dict[str, float]:
        denom = self.act_sum if self.act_sum > 0 else 1.0
        return {
            "rho_mean": self.rho_sum / denom,
            "AND_pct": 100.0 * self.n_and / denom,
            "OR_pct": 100.0 * self.n_or / denom,
            "COMP_pct": 100.0 * self.n_comp / denom,
            "act_sum": self.act_sum,
        }


class SafeEFSWrapper:
    """
    Wraps an evolvingfuzzysystems model so that:
      - internal index/value errors do NOT crash the experiment
      - non-finite predictions fall back to class 0
      - keeps counters:
            stabilizations: internal errors caught
            fallback_preds: NaN/inf predictions
    """

    def __init__(self, model, name="eFS"):
        self.model = model
        self.name = name
        self.stabilizations = 0
        self.fallback_preds = 0

    def _safe_call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (IndexError, ValueError, FloatingPointError):
            self.stabilizations += 1
            return None

    def fit(self, X, y):
        return self._safe_call(self.model.fit, X, y)

    def evolve(self, X, y):
        if hasattr(self.model, "evolve"):
            return self._safe_call(self.model.evolve, X, y)
        return self.fit(X, y)

    def predict(self, X):
        try:
            y = self.model.predict(X)
            y = np.asarray(y).ravel()
            if not np.isfinite(y).all():
                raise FloatingPointError
            return y
        except Exception:
            self.fallback_preds += 1
            return np.zeros(len(X), dtype=int)

    def n_rules(self):
        try:
            return self.model.n_rules()
        except Exception:
            return -1


def prequential_nf_generic(model, X, y, desc="NF") -> Dict[str, Any]:
    """
    Prequential evaluation (predict-then-update) for NF-type models.

    Additionally logs per-rule UnimNeuron regime (AND/OR/COMP) and rho when possible.
    It stores stats into: model._rule_stats (list[RuleStats]) for later export.

    Expected API:
      - predict(X) -> label(s)
      - partial_fit(X, y)
      - optional: n_rls_resets (count of repaired numerical instabilities)
      - optional: rules list with (center, sigma, w), and scalar neutral element 'e'
    """
    n = len(y)
    y_pred = np.zeros(n, int)
    correct = np.zeros(n, float)
    acc = np.zeros(n, float)
    rules_time = []

    pred_fallbacks = 0

    # Initialize rule stats holder (best-effort)
    # We refresh size dynamically because rules can be added/removed.
    if not hasattr(model, "_rule_stats"):
        model._rule_stats = []  # type: ignore

    for t in tqdm(range(n), desc=desc, leave=False):
        x_t = X[t : t + 1]
        yt = int(y[t])

        # ---- predict
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

        y_pred[t] = y_hat
        correct[t] = 1.0 if y_hat == yt else 0.0
        acc[t] = correct[: t + 1].mean()

        # ---- update model
        model.partial_fit(x_t, np.array([yt]))

        # ---- track rule count over time
        if hasattr(model, "history_n_rules") and getattr(model, "history_n_rules"):
            rules_time.append(model.history_n_rules[-1])
        elif hasattr(model, "n_rules_"):
            rules_time.append(model.n_rules_)
        elif hasattr(model, "rules"):
            rules_time.append(len(model.rules))
        else:
            rules_time.append(np.nan)

        # ---- collect UnimNeuron semantics per rule (best-effort)
        # Works for your ENF_Unim_* models because they have:
        #   model.rules : list of rules with center, sigma, w
        #   model.e : neutral element
        if hasattr(model, "rules") and hasattr(model, "e"):
            try:
                e = float(getattr(model, "e"))
                rules_list = list(getattr(model, "rules"))
                R = len(rules_list)

                # ensure stats list matches current rule count
                stats_list: List[RuleStats] = list(getattr(model, "_rule_stats"))
                if len(stats_list) != R:
                    # Resize conservatively: keep prefix; new rules get fresh stats
                    new_stats = [RuleStats() for _ in range(R)]
                    m = min(len(stats_list), R)
                    for i in range(m):
                        new_stats[i] = stats_list[i]
                    setattr(model, "_rule_stats", new_stats)
                    stats_list = new_stats

                # compute per-rule activation + regime on current sample
                x_vec = np.asarray(x_t, dtype=float).ravel()
                for i, r in enumerate(rules_list):
                    # Rule must have center/sigma/w
                    c = np.asarray(getattr(r, "center"), dtype=float).ravel()
                    sig = float(getattr(r, "sigma"))
                    w = np.asarray(getattr(r, "w"), dtype=float).ravel()

                    # Gaussian membership per feature
                    sig_eff = max(sig, 1e-6)
                    a = np.exp(-((x_vec - c) ** 2) / (2.0 * sig_eff**2))

                    # feature contributions h_j (same as in your models)
                    h = w * a + (1.0 - w) * e

                    # UnimNeuron regime + rho
                    reg, rho = _unim_regime_and_rho(h, e=e)

                    # Activation value of the rule: use the same unim aggregation
                    # (matches your implementation)
                    # AND: min, OR: max, COMP: mean
                    if reg == "AND":
                        act = float(np.min(h))
                    elif reg == "OR":
                        act = float(np.max(h))
                    else:
                        act = float(np.mean(h))

                    stats_list[i].add(act=act, reg=reg, rho=rho)

                setattr(model, "_rule_stats", stats_list)

            except Exception:
                # do not break experiments
                pass

    rules_arr = np.array(rules_time, float)
    n_rules_final = int(rules_arr[-1]) if not np.isnan(rules_arr[-1]) else -1
    P_instabilities = int(getattr(model, "n_rls_resets", 0))

    return dict(
        accuracy=acc,
        y_pred=y_pred,
        evaluated_start=0,
        n_rules_time=rules_arr,
        n_rules_final=n_rules_final,
        pred_fallbacks=int(pred_fallbacks),
        P_instabilities=P_instabilities,
    )


def prequential_efs(model, X, y, warmup=200, batch_evolve=20, desc="eFS"):
    """
    Prequential evaluation for evolvingfuzzysystems (Kaike eFS models).
    """
    n = len(y)
    y_pred = np.zeros(n, int)
    correct = np.zeros(n, float)
    acc = np.zeros(n, float)

    # These eFS models output continuous regression values, not class IDs.
    # The configured benchmark uses integer-encoded binary class labels (0/1).
    # A 0.5 decision boundary converts the continuous output to that label set.
    # Multiclass classification requires a separate, explicitly defined adapter.
    n_classes = len(np.unique(y))
    if n_classes != 2:
        raise ValueError(
            f"eFS regression-to-class conversion supports binary streams only; "
            f"found {n_classes} classes in {desc}."
        )

    if warmup >= n:
        warmup = max(1, n // 10)

    model.fit(X[:warmup], y[:warmup])

    bufX = []
    bufY = []

    for t in tqdm(range(warmup, n), desc=desc, leave=False):
        x_t = X[t : t + 1]
        yt = int(y[t])

        yhat_raw = model.predict(x_t)
        yhat_raw = np.asarray(yhat_raw).ravel()

        # Convert eFS continuous regression output to a binary class.
        # Do not cast to int: e.g., int(2.3) == 2 is an invalid class label.
        if yhat_raw.size != 1 or not np.isfinite(yhat_raw[0]):
            raise ValueError(
                f"Invalid eFS prediction at t={t}: {yhat_raw!r} ({desc})"
            )
        yh = int(float(yhat_raw[0]) >= 0.5)

        y_pred[t] = yh
        correct[t] = 1.0 if yh == yt else 0.0
        acc[t] = correct[: t + 1].mean()

        bufX.append(x_t.ravel())
        bufY.append(yt)

        if len(bufY) >= batch_evolve:
            X_block = np.vstack(bufX)
            y_block = np.array(bufY, int)
            model.fit(X_block, y_block)
            bufX.clear()
            bufY.clear()

    if n > warmup:
        acc[:warmup] = acc[warmup]

    try:
        n_rules_final = int(model.n_rules())
    except Exception:
        n_rules_final = -1

    rules_time = np.full_like(acc, n_rules_final, float)

    pred_fallbacks = int(getattr(model, "fallback_preds", 0))
    P_instabilities = int(getattr(model, "stabilizations", 0))

    return dict(
        accuracy=acc,
        y_pred=y_pred,
        evaluated_start=warmup,
        n_rules_time=rules_time,
        n_rules_final=n_rules_final,
        pred_fallbacks=pred_fallbacks,
        P_instabilities=int(P_instabilities),
    )

