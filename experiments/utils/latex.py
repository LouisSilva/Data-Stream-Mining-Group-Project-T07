from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

from experiments.utils.metrics import Metric
from experiments.utils.evaluation import RuleStats


def _latex_escape(s: str) -> str:
    return (
        s.replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _linguistic_term(u: float) -> str:
    """Map scalar in [0,1] to a simple linguistic label (IEEE-friendly, deterministic)."""
    if u < 1.0 / 3.0:
        return "Low"
    elif u < 2.0 / 3.0:
        return "Medium"
    else:
        return "High"


def _topk_indices_by_weight(w: np.ndarray, k: int = 5) -> List[int]:
    w = np.asarray(w, dtype=float).ravel()
    k = int(max(1, min(k, w.size)))
    return list(np.argsort(-w)[:k])


def build_sota_latex_table(rows: List[Dict[str, Any]], outpath: str, metrics: List[Metric]):
    """Create columns only for the configured metrics."""
    metric_columns = []
    for metric in metrics:
        metric_columns.extend(
            [(f"{metric.label}$_{{final}}$", f"{metric.suffix}_final"),
             (f"{metric.label}$_{{mean}}$", f"{metric.suffix}_mean")]
        )

    header_labels = ["Mode", "Dataset", "Model", "$N$"]
    header_labels += [name for name, _ in metric_columns]
    header_labels += ["$N_{\\mathrm{rules}}^{final}$", "Time [s]", "Fallbacks", "Instabilities"]
    colspec = "l" * 3 + "c" * (len(header_labels) - 3)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Prequential metrics and rule growth on real and synthetic data streams.}",
        rf"\begin{{tabular}}{{{colspec}}} \toprule",
        " & ".join(header_labels) + r" \\ \midrule",
    ]

    for row in sorted(rows, key=lambda row: (row["mode"], row["dataset"], row["model"])):
        entries = [
            _latex_escape(row["mode"]), _latex_escape(row["dataset"]),
            _latex_escape(row["model"]), str(int(row["N"])),
        ]
        
        entries += [f"{row[key]:.3f}" for _, key in metric_columns]
        entries += [
            str(int(row["n_rules_final"])), f"{row['time_s']:.1f}",
            str(int(row["pred_fallbacks"])), str(int(row["P_instabilities"])),
        ]
        
        lines.append(" & ".join(entries) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(outpath, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def build_ablation_latex_table(rows: List[Dict[str, Any]], outpath: str, metrics: List[Metric]):
    metric_columns = [
        (f"{metric.label}$_{{{part}}}$", f"{metric.suffix}_{part}")
        for metric in metrics for part in ("final", "mean")
    ]
    
    headers = ["Mode", "Dataset", "Model", "Ablation", "$N$"]
    headers += [label for label, _ in metric_columns]
    headers += ["$N_{\\mathrm{rules}}$", "Time [s]"]
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Ablation study on UnimNeuron components (prequential evaluation).}",
        rf"\begin{{tabular}}{{llll{'c' * (len(headers)-4)}}} \toprule",
        " & ".join(headers) + r" \\ \midrule",
    ]
    
    for row in rows:
        entries = [
            _latex_escape(row["mode"]), _latex_escape(row["dataset"]),
            _latex_escape(row["model"]), _latex_escape(row["ablation"]),
            str(int(row["N"])),
        ]
        
        entries += [f"{row[key]:.3f}" for _, key in metric_columns]
        entries += [str(int(row["n_rules_final"])), f"{row['time_s']:.1f}"]
        lines.append(" & ".join(entries) + r" \\ ")
        
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_stats_tables(
    metric_mean_matrix: np.ndarray,
    model_names: List[str],
    outdir: str,
    metric: Metric,
):
    os.makedirs(outdir, exist_ok=True)

    friedman_path = os.path.join(outdir, f"exp_stats_{metric.suffix}_friedman.tex")
    posthoc_path = os.path.join(outdir, f"exp_stats_{metric.suffix}_posthoc.tex")

    cols = [metric_mean_matrix[:, j] for j in range(metric_mean_matrix.shape[1])]
    stat, p = stats.friedmanchisquare(*cols)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{Global statistical comparison via Friedman test ({metric.label} mean).}}")
    lines.append(r"\begin{tabular}{lc} \toprule")
    lines.append(r"Statistic & $p$-value \\ \midrule")
    lines.append(f"$\\chi^2_F = {stat:.4f}$ & ${p:.4g}$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(friedman_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    import scikit_posthocs as sp
    nemenyi = sp.posthoc_nemenyi_friedman(metric_mean_matrix)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{Post-hoc Nemenyi p-values for {metric.label} (pairwise).}}")
    colspec = "l" + "c" * len(model_names)
    lines.append(rf"\begin{{tabular}}{{{colspec}}} \toprule")
    header = (
        "Model & "
        + " & ".join([_latex_escape(m) for m in model_names])
        + r" \\ \midrule"
    )
    lines.append(header)

    for i, mi in enumerate(model_names):
        row = [_latex_escape(mi)]
        for j in range(len(model_names)):
            row.append(f"{nemenyi.iloc[i, j]:.4f}")
        lines.append(" & ".join(row) + r" \\\\")
        
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(posthoc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -------------------------------------------------------------------------
# Interpretability: export fuzzy rules (best-effort, non-breaking)
# -------------------------------------------------------------------------
def export_rules_best_effort(
    model: Any,
    model_name: str,
    dataset_outdir: str,
    feature_names: List[str],
    max_rules_to_write: int = 5,
    top_features_per_rule: int = 6,
) -> None:
    """
    Export human-readable fuzzy rules for UnimNeuron-based models.

    Produces:
      - Linguistic antecedents: feature is {Low,Medium,High} based on rule center in [0,1]
      - Feature relevance weights w_j
      - UnimNeuron semantics per rule: AND/OR/COMP percentages + rho_mean
      - Consequent:
          * Safe: argmax class_probs
          * PA: argmax row W[1+idx,:] (per-rule consequent weights)

    Notes
    -----
    - Requires that during evaluation `prequential_nf_generic` stored `model._rule_stats`.
    - This function is best-effort: never crashes experiments.
    """
    os.makedirs(dataset_outdir, exist_ok=True)
    outpath = os.path.join(dataset_outdir, f"rules_{model_name}.txt")

    try:
        if not hasattr(model, "rules") or len(getattr(model, "rules")) == 0:
            return

        rules_list = list(getattr(model, "rules"))
        k = min(int(max_rules_to_write), len(rules_list))

        # Stats collected online
        stats_list: List[RuleStats] = []
        if hasattr(model, "_rule_stats"):
            stats_list = list(getattr(model, "_rule_stats"))

        # Prepare consequent accessors
        has_safe_probs = False
        has_pa_W = False
        W = None
        if hasattr(model, "W") and getattr(model, "W") is not None:
            W = np.asarray(getattr(model, "W"), dtype=float)
            has_pa_W = True

        # Safe model rules have class_probs method
        if hasattr(rules_list[0], "class_probs"):
            has_safe_probs = True

        lines: List[str] = []
        lines.append(f"Model: {model_name}")
        lines.append(
            "Interpretable rule export (UnimNeuron semantics + linguistic antecedents)\n"
        )
        lines.append("Legend:")
        lines.append("  - center[j] in [0,1] -> {Low, Medium, High}")
        lines.append("  - w_j is feature relevance in [0,1]")
        lines.append("  - rho_mean in [-1,1]: +1 OR-like, -1 AND-like, 0 COMP-like\n")

        # Sort rules by support (descending) to show the most relevant first
        supports = np.array(
            [int(getattr(r, "support", 0)) for r in rules_list], dtype=int
        )
        order = list(np.argsort(-supports))

        shown = 0
        for ridx in order:
            if shown >= k:
                break
            r = rules_list[ridx]

            center = np.asarray(getattr(r, "center"), dtype=float).ravel()
            sigma = float(getattr(r, "sigma"))
            support = int(getattr(r, "support", 0))
            w = np.asarray(getattr(r, "w"), dtype=float).ravel()

            # Rule semantics summary if available
            sem = None
            if ridx < len(stats_list):
                sem = stats_list[ridx].summary()

            # Consequent
            consequent_str = "THEN class = ?"
            if has_safe_probs:
                probs = np.asarray(r.class_probs(), dtype=float).ravel()
                c = int(np.argmax(probs))
                consequent_str = f"THEN class = {c} (p={probs[c]:.3f})"
            elif has_pa_W and W is not None:
                # Row per rule is W[1+ridx,:] (bias at row 0)
                if W.shape[0] > 1 + ridx:
                    row = W[1 + ridx, :].ravel()
                    c = int(np.argmax(row))
                    # show top-2 classes as extra info
                    top2 = list(np.argsort(-row)[:2])
                    consequent_str = (
                        f"THEN class = {c}  "
                        f"[top: {top2[0]}({row[top2[0]]:.3f}), {top2[1]}({row[top2[1]]:.3f})]"
                    )

            # Antecedent: show only top features by relevance weight to keep it readable
            top_idx = _topk_indices_by_weight(w, k=min(top_features_per_rule, w.size))
            antecedents = []
            for j in top_idx:
                fname = feature_names[j] if j < len(feature_names) else f"x{j+1}"
                term = _linguistic_term(float(center[j]))
                antecedents.append(f"{fname} is {term} (w={w[j]:.2f})")

            # Build rule text
            lines.append(f"Rule {shown+1}  [internal_id={ridx}]")
            lines.append(f"  support = {support}, sigma = {sigma:.4f}")
            if sem is not None:
                lines.append(
                    f"  semantics: rho_mean={sem['rho_mean']:.3f} | "
                    f"AND={sem['AND_pct']:.1f}% OR={sem['OR_pct']:.1f}% COMP={sem['COMP_pct']:.1f}%"
                )
            else:
                lines.append(
                    "  semantics: (not available)  -- run via prequential_nf_generic to log regimes"
                )

            lines.append("  IF " + " AND ".join(antecedents))
            lines.append("  " + consequent_str)
            lines.append("")

            shown += 1

        with open(outpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    except Exception:
        return


def export_rules_latex(
    model: Any,
    model_name: str,
    dataset_outdir: str,
    feature_names: List[str],
    max_rules_to_write: int = 5,
    top_features_per_rule: int = 6,
    latex_env: str = "enumerate",  # "enumerate" or "align"
) -> None:
    """
    Export interpretable fuzzy rules to a LaTeX .tex file.

    Output:
      <dataset_outdir>/rules_<model_name>.tex

    Contents (per rule):
      - support, sigma
      - semantics: rho_mean and AND/OR/COMP percentages (if model._rule_stats exists)
      - IF antecedent using linguistic terms derived from center[j]
      - THEN consequent (Safe: class_probs; PA: W row per rule)

    Notes:
      - Best-effort: never breaks experiment if some info is missing.
      - For anonymous submission, this contains no author info.
    """
    os.makedirs(dataset_outdir, exist_ok=True)
    outpath = os.path.join(dataset_outdir, f"rules_{model_name}.tex")

    try:
        if not hasattr(model, "rules") or len(getattr(model, "rules")) == 0:
            return

        rules_list = list(getattr(model, "rules"))
        n_rules = len(rules_list)

        # stats collected online (optional)
        stats_list = []
        if hasattr(model, "_rule_stats"):
            try:
                stats_list = list(getattr(model, "_rule_stats"))
            except Exception:
                stats_list = []

        # consequent: Safe vs PA
        has_safe_probs = hasattr(rules_list[0], "class_probs")
        W = None
        has_pa_W = False
        if hasattr(model, "W") and getattr(model, "W") is not None:
            try:
                W = np.asarray(getattr(model, "W"), dtype=float)
                has_pa_W = True
            except Exception:
                W = None
                has_pa_W = False

        # sort by support descending (more publishable)
        supports = np.array([int(getattr(r, "support", 0)) for r in rules_list], dtype=int)
        order = list(np.argsort(-supports))

        k = min(int(max_rules_to_write), n_rules)

        lines: List[str] = []
        lines.append(r"% Auto-generated interpretable rules (UnimNeuron models)")
        lines.append(r"% Copy/paste this block into the paper where rules are discussed.")
        lines.append("")

        if latex_env == "align":
            lines.append(r"\begin{align}")
        else:
            lines.append(r"\begin{enumerate}")

        shown = 0
        for ridx in order:
            if shown >= k:
                break

            r = rules_list[ridx]
            center = np.asarray(getattr(r, "center"), dtype=float).ravel()
            sigma = float(getattr(r, "sigma", np.nan))
            support = int(getattr(r, "support", 0))
            w = np.asarray(getattr(r, "w", np.ones_like(center)), dtype=float).ravel()

            # semantics summary (optional)
            sem_str = r"\textit{semantics: not available}"
            if ridx < len(stats_list):
                try:
                    sem = stats_list[ridx].summary()
                    sem_str = (
                        rf"\textit{{semantics: }} $\bar{{\rho}}={sem['rho_mean']:.3f}$, "
                        rf"AND={sem['AND_pct']:.1f}\%, OR={sem['OR_pct']:.1f}\%, COMP={sem['COMP_pct']:.1f}\%"
                    )
                except Exception:
                    pass

            # consequent string
            then_str = r"\text{THEN class } = ?"
            if has_safe_probs:
                try:
                    probs = np.asarray(r.class_probs(), dtype=float).ravel()
                    c = int(np.argmax(probs))
                    then_str = rf"\text{{THEN class}} = {c}\, (p={probs[c]:.3f})"
                except Exception:
                    pass
            elif has_pa_W and W is not None:
                try:
                    if W.shape[0] > 1 + ridx:
                        row = W[1 + ridx, :].ravel()
                        c = int(np.argmax(row))
                        top2 = list(np.argsort(-row)[:2])
                        then_str = (
                            rf"\text{{THEN class}} = {c}"
                            rf"\, [\text{{top: }} {top2[0]}({row[top2[0]]:.3f}), {top2[1]}({row[top2[1]]:.3f})]"
                        )
                except Exception:
                    pass

            # antecedent (top-K by weight)
            top_idx = _topk_indices_by_weight(w, k=min(top_features_per_rule, w.size))
            ants = []
            for j in top_idx:
                fname = feature_names[j] if j < len(feature_names) else f"x{j+1}"
                fname = _latex_escape(fname)
                term = _linguistic_term(float(center[j]))
                ants.append(rf"{fname}\ \text{{is}}\ {term}\ (w={w[j]:.2f})")

            # build rule latex
            if_part = r"\ \wedge\ ".join(ants) if len(ants) > 0 else r"\text{(no antecedent available)}"
            meta = rf"\textit{{support}}={support},\ \sigma={sigma:.3f}"

            if latex_env == "align":
                lines.append(
                    rf"\text{{Rule {shown+1}:}}\ & \text{{IF }} {if_part}\ \text{{ THEN }} {then_str} \\"
                )
                lines.append(rf"& {meta},\ {sem_str} \\")
                lines.append(r"")
            else:
                lines.append(r"\item")
                lines.append(rf"\textbf{{Rule {shown+1}.}} {meta}. {sem_str}.")
                lines.append(r"\\")
                lines.append(rf"$\textbf{{IF}}\ {if_part}\ \ \textbf{{THEN}}\ {then_str}.$")
                lines.append("")

            shown += 1

        if latex_env == "align":
            lines.append(r"\end{align}")
        else:
            lines.append(r"\end{enumerate}")

        with open(outpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    except Exception:
        return
