"""
data.py -- turn dataset.csv (process_packet.py) or a synthetic scan into model inputs and targets.

A row of the dataset is one (packet, board, channel): one amplifier at one operating point.

Inputs (choose any combination; ``make_inputs``)
-------------------------------------------------
``theta``   the operating parameters (what we set)
``delta``   their change from the reference
``trace``   the hand-made trace features at theta (mean over the repeated trace sets)
``ref``     the same features at the reference, taken in the same packet minutes earlier: they carry
            the amplifier's identity and the CCD's current state
``diff``    trace - ref: the change the operating parameters caused, with the amplifier's own
            signature removed
``curves``  the trace curves (mean skip waveform, transfer shapes, noise spectra), for networks
            that learn their own features (``curve_tensor``)

Targets (``make_targets``): log sigma_e, log lambda, log gain.  Logs because all three are positive,
span factors, and an error in log is a relative error (0.05 = 5 %).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = {"sigma_e": "label.sigma_e", "lambda": "label.lambda_e_per_bin", "gain": "label.gain_adu_per_e"}
INPUT_PREFIXES = ("theta", "delta", "trace", "ref", "diff")


def load_dataset(path: str | Path, only_ok: bool = True) -> pd.DataFrame:
    """Read dataset.csv; keep rows whose labels have no flags (``label.ok``) if ``only_ok``."""
    df = pd.read_csv(path)
    if only_ok and "label.ok" in df:
        df = df[df["label.ok"].astype(str).str.lower() == "true"].reset_index(drop=True)
    return df


def columns(df: pd.DataFrame, prefix: str) -> list[str]:
    """The columns of one input group (e.g. 'trace' -> 'trace.<feature>'), without the bookkeeping ones."""
    skip = ("n_usable", "n_rejected_values")
    return [c for c in df.columns if c.startswith(prefix + ".") and c.split(".", 1)[1] not in skip]


def amplifier_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """One column per amplifier (board, channel): 1 for the rows of that amplifier, else 0."""
    key = df["board"].astype(str) + "_ch" + df["channel"].astype(str)
    return pd.get_dummies(key, prefix="amp", prefix_sep=".").astype(float)


def make_inputs(df: pd.DataFrame, groups: tuple[str, ...] = ("theta", "trace"),
                drop_constant: bool = True) -> tuple[np.ndarray, list[str]]:
    """
    The input matrix X (rows of df, columns of the chosen groups) and the column names.

    Group ``amp`` is the amplifier identity (one-hot): a model that sees only theta cannot
    otherwise know that it predicts the noisy amplifier or the quiet one.  (The ``ref`` features
    carry the identity implicitly, and also work for an amplifier never seen in training.)
    Columns with any missing value are dropped (a feature that cannot exist for some trace), and
    columns that never change (e.g. a parameter not scanned) if ``drop_constant``.
    """
    frames = [amplifier_onehot(df) if g == "amp" else df[columns(df, g)] for g in groups]
    X = pd.concat(frames, axis=1).apply(pd.to_numeric, errors="coerce")
    names = list(X.columns)
    keep = X.notna().all()
    if drop_constant:
        keep &= X.std() > 0
    names = [n for n in names if keep[n]]
    return X[names].to_numpy(float), names


def make_targets(df: pd.DataFrame, targets: tuple[str, ...] = ("sigma_e", "lambda", "gain")) -> tuple[np.ndarray, list[str]]:
    """log of the labels, one column per target."""
    Y = np.column_stack([np.log(pd.to_numeric(df[TARGETS[t]], errors="coerce").to_numpy(float)) for t in targets])
    return Y, list(targets)


def groups_of(df: pd.DataFrame, by: str = "packet_id") -> np.ndarray:
    """Group labels for cross-validation: rows of one packet must never be split between train and test."""
    return df[by].astype(str).to_numpy()


def resample(curve: np.ndarray, length: int) -> np.ndarray:
    """A curve brought to a fixed length by linear interpolation (curves differ in length with the geometry)."""
    curve = np.asarray(curve, float)
    if curve.size == length:
        return curve
    return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, curve.size), curve)


def log_frequency(curve: np.ndarray) -> np.ndarray:
    """
    A spectrum sampled on a linear frequency axis (first point f = 0), resampled on a LOGARITHMIC axis with
    the same number of points (f = 0 kept first).  On a linear axis the lowest frequencies, where noise that
    does not average down shows up, are 2 points out of hundreds; on a log axis they get as many points as
    every other decade.  (Self-supervised pre-training on the synthetic world only learned the noise floor
    from log-frequency spectra: probe R^2 0.70 against 0.08.)
    """
    curve = np.atleast_2d(np.asarray(curve, float))
    m = curve.shape[1]
    f = np.linspace(0, 1, m)
    g = np.r_[0, np.geomspace(f[1], 1, m)[:-1]]
    return np.array([np.interp(g, f, row) for row in curve])


def curve_tensor(curves: dict[str, np.ndarray], names: list[str] | None = None, length: int = 256,
                 log_names: tuple[str, ...] = ("noise: Allan deviation", "noise: CDS spectrum"),
                 log_frequency_spectra: bool = False) -> np.ndarray:
    """
    Stack curves as channels of one array (n_rows, n_curves, length) for a 1D CNN.

    Every curve is resampled to ``length`` points; spectra and Allan deviations are taken in log
    (they span decades).  ``log_frequency_spectra``: curves whose name contains "spectrum" are first put
    on a logarithmic frequency axis (``log_frequency``).  Normalization (per position, from the training rows only) is done by
    the model, not here, so that no information leaks from the test rows.
    """
    names = names or sorted(curves)
    stacked = []
    for name in names:
        c = np.asarray(curves[name], float)
        if log_frequency_spectra and "spectrum" in name:
            c = log_frequency(c)
        if name in log_names:
            c = np.log(np.clip(c, 1e-30, None))
        stacked.append(np.array([resample(row, length) for row in c]))
    return np.stack(stacked, axis=1)


def curves_from_processed(out_dir: str | Path, df: pd.DataFrame, kind: str = "theta",
                          names: list[str] | None = None) -> dict[str, np.ndarray]:
    """
    The trace curves of real packets, aligned with the rows of ``df`` (from process_packet.py output).

    For every row (packet, board, channel), the curves of all usable trace sets of ``kind``
    ('theta' or 'ref') are read from their .npz files and their median is taken.  Curves of
    different lengths (other geometries) are resampled to the length of the first one found.
    Rows without curves get NaN.
    """
    out_dir = Path(out_dir)
    found: dict[tuple, dict[str, list]] = {}
    for result_path in sorted(out_dir.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        for t in result.get("traces", []):
            if t.get("kind") != kind or not t.get("usable") or not t.get("npz"):
                continue
            with np.load(t["npz"]) as arrays:
                entry = found.setdefault((result["packet_id"], t["board"], t["channel"]), {})
                for key in arrays.files:
                    if key.startswith("curve: "):
                        entry.setdefault(key[7:], []).append(arrays[key])
    all_names = names or sorted({n for e in found.values() for n in e})
    lengths = {n: next((len(e[n][0]) for e in found.values() if n in e), 1) for n in all_names}
    out = {n: np.full((len(df), lengths[n]), np.nan) for n in all_names}
    for i, row in enumerate(df.itertuples(index=False)):
        entry = found.get((getattr(row, "packet_id"), int(getattr(row, "board")), int(getattr(row, "channel"))), {})
        for n in all_names:
            if n in entry:
                out[n][i] = np.nanmedian([resample(c, lengths[n]) for c in entry[n]], axis=0)
    return out
