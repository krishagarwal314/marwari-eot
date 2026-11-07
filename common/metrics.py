"""EOT metrics. Convention everywhere: label 1 = COMPLETE (turn ended), 0 = INCOMPLETE.

false_completion_rate: incomplete clips called complete (the agent interrupts the user).
false_wait_rate:       complete clips called incomplete (the agent waits too long).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np


def binary_metrics(y_true, p_complete, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_complete, dtype=np.float64)
    pred = (p > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n_pos, n_neg = tp + fn, tn + fp
    tpr = tp / n_pos if n_pos else float("nan")
    tnr = tn / n_neg if n_neg else float("nan")
    out = {
        "n": int(len(y)),
        "n_complete": n_pos,
        "n_incomplete": n_neg,
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "balanced_accuracy": (tpr + tnr) / 2,
        "false_completion_rate": fp / n_neg if n_neg else float("nan"),
        "false_wait_rate": fn / n_pos if n_pos else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tpr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    out["roc_auc"] = roc_auc(y, p)
    return out


def roc_auc(y_true, scores) -> float:
    """Mann-Whitney AUC with tie handling; NaN if only one class is present."""
    from scipy.stats import rankdata

    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_ci(y_true, p_complete, threshold: float = 0.5, n_boot: int = 1000,
                 seed: int = 0, keys=("accuracy", "balanced_accuracy", "roc_auc",
                                       "false_completion_rate", "false_wait_rate"),
                 groups=None) -> dict:
    """95% percentile bootstrap CIs. If ``groups`` (e.g. speaker ids) is given, resample whole
    groups (cluster bootstrap) so clips from one speaker are not treated as independent."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_complete, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        index_of = {g: np.flatnonzero(groups == g) for g in uniq}
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            idx = np.concatenate([index_of[g] for g in rng.choice(uniq, len(uniq))])
        m = binary_metrics(y[idx], p[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for k, v in samples.items()}


def report(y_true, p_complete, threshold=0.5, n_boot=1000, groups=None, seed=0) -> dict:
    m = binary_metrics(y_true, p_complete, threshold)
    for k, (lo, hi) in bootstrap_ci(y_true, p_complete, threshold, n_boot, seed, groups=groups).items():
        m[f"{k}_ci_lo"], m[f"{k}_ci_hi"] = lo, hi
    return m


def per_group(y_true, p_complete, group_values, threshold=0.5) -> dict:
    y = np.asarray(y_true)
    p = np.asarray(p_complete)
    g = np.asarray(group_values)
    return {str(v): binary_metrics(y[g == v], p[g == v], threshold) for v in np.unique(g)}


def cohens_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    po = float((a == b).mean())
    pe = float(sum((a == l).mean() * (b == l).mean() for l in labels))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def seed_summary(values) -> tuple[float, float]:
    """mean and sample std over seeds."""
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def latency_ms(fn: Callable[[], object], n: int = 200, warmup: int = 20) -> dict:
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    return {"p50_ms": float(np.percentile(t, 50)), "p90_ms": float(np.percentile(t, 90))}


def onnx_cpu_session(path: str, threads: int = 1):
    """Single-threaded CPU session configured like Smart Turn's inference.py."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
