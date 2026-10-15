"""
tutorial_utils.py -- small helpers shared by the model tutorials (notebooks/model_tutorials/*.ipynb).

Every tutorial imports this first:  ``from tutorial_utils import *``.  It puts ``src/`` on the path, fixes the plot
style, and gives three helpers used in every notebook:

    toy_scan(n_packets, seed, **kw)  ->  df, curves, Y, groups    a synthetic scan of the toy CCD world
    report(name, Y, mean, std)       ->  prints the error (in %) and the error-bar coverage of sigma_e, lambda, gain
    grouped_cv(make_model, X, Y, groups)  -> out-of-fold (mean, std), packets kept together

The toy CCD world (src/ml/synthetic.py) mimics the packets: settings, trace features, curves, and image labels with
realistic noise.  Its physics is INVENTED: the numbers compare methods, they say nothing about the real CCD.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
np.set_printoptions(precision=4, suppress=True)

from ml import data, evaluate, models, synthetic  # noqa: E402

TARGETS = ("sigma_e", "lambda", "gain")


def toy_scan(n_packets: int, seed: int = 0, **kw):
    """A synthetic scan: df (one row per amplifier per packet), curves, Y = (log sigma_e, log lambda, log gain), groups (packet ids)."""
    df, curves = synthetic.make_dataset(n_packets, seed=seed, **kw)
    Y, _ = data.make_targets(df)
    return df, curves, Y, data.groups_of(df)


def grouped_cv(make_model, X, Y, groups, n_splits: int = 5, seed: int = 0):
    """Out-of-fold predictions: every row is predicted by a model that never saw its packet."""
    return evaluate.grouped_cv(make_model, X, Y, groups, n_splits, seed)


def report(name: str, Y, mean, std=None, targets=TARGETS) -> dict:
    """Print and return the RMS error of log(predicted/true) (= relative error) and the +-1 sigma coverage per target."""
    Y, mean = np.atleast_2d(Y), np.atleast_2d(mean)
    out = {}
    parts = []
    for t, tn in enumerate(targets[: Y.shape[1]]):
        ok = np.isfinite(Y[:, t]) & np.isfinite(mean[:, t])
        err = mean[ok, t] - Y[ok, t]
        cov = float(np.mean(np.abs(err) <= std[ok, t])) if std is not None else np.nan
        out[tn] = {"error": float(np.sqrt(np.mean(err ** 2))), "coverage68": cov}
        parts.append(f"{tn}: {100 * out[tn]['error']:5.1f} %" + (f" (cover {cov:.2f})" if std is not None else ""))
    print(f"{name:40s} " + "   ".join(parts))
    return out


__all__ = ["np", "pd", "plt", "matplotlib", "ROOT", "data", "evaluate", "models", "synthetic", "toy_scan", "grouped_cv", "report", "TARGETS"]
