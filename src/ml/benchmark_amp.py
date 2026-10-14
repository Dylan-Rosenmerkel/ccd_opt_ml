"""
benchmark_amp.py -- does a model need to be told WHICH amplifier it predicts?

Until now most models were given the amplifier identity (a one-hot column, or the amplifier level of the
hierarchical GP), because the amplifiers differ a lot (one is ~4x noisier, gains differ by up to 2x).  That makes
a model amplifier-SPECIFIC: it cannot say anything about an amplifier it has never seen.  This experiment compares

* explicit identity       the amplifier one-hot / the hierarchical GP's amplifier level;
* implicit identity       no identity, but the amplifier's OWN trace and reference features (its signature);
* no amplifier information at all                (settings, the packet's state, and for A the trace physics);

on amplifiers seen in training and on a module never seen (0 or 2 of its packets in training), with three scores:

* abs    the usual error (log, so relative);
* shape  the error after removing each amplifier's own mean error: does the model get the DEPENDENCE on the
         settings right, even if it has the amplifier's overall level wrong?
* rank   the Spearman correlation, within each amplifier, between predicted and true values over the test
         packets: does it ORDER the settings correctly?  This is what an optimizer needs.

    python -m ml.benchmark_amp --out ../docs/ml_results [--seeds 1 2 3]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import data, models, structured as st, synthetic
from .benchmark import ALL_CURVES, inputs_for

TARGETS = ["sigma_e", "lambda", "gain"]


def two_scans(n_train, n_test, seed):
    """One table with a training scan and a test scan (2 modules each), packet ids kept apart."""
    tr, ctr = synthetic.make_dataset(n_train, seed=seed, n_boards=2)
    te, cte = synthetic.make_dataset(n_test, seed=99, n_boards=2)
    tr = tr.assign(split="train", packet=tr["packet_id"].str[4:].astype(int), packet_id="tr_" + tr["packet_id"])
    te = te.assign(split="test", packet=te["packet_id"].str[4:].astype(int), packet_id="te_" + te["packet_id"])
    df = pd.concat([tr, te], ignore_index=True)
    curves = {k: np.concatenate([ctr[k], cte[k]]) for k in ctr}
    return df, curves


def metrics(df, mean):
    """abs, shape and within-amplifier rank scores per target (see the module docstring)."""
    truth = np.log(df[["truth.sigma_e", "truth.lambda", "truth.gain"]].to_numpy(float))
    amp = st.amplifier_key(df)
    out = {}
    for t, name in enumerate(TARGETS):
        r = mean[:, t] - truth[:, t]
        shape = r.copy()
        for a in np.unique(amp):
            shape[amp == a] -= r[amp == a].mean()
        ranks = [spearmanr(mean[amp == a, t], truth[amp == a, t])[0] for a in np.unique(amp) if (amp == a).sum() > 3]
        out[name] = {"abs": float(np.sqrt(np.mean(r ** 2))), "shape": float(np.sqrt(np.mean(shape ** 2))),
                     "rank": float(np.nanmean(ranks))}
    return out


def variants():
    """label -> how to build the model and its inputs."""
    feat = lambda groups: ("first", groups)
    return {
        "GP: settings + amplifier ID": feat(("theta", "amp")),
        "GP: settings + amplifier ID + features": feat(("theta", "amp", "trace", "ref")),
        "GP: settings + features (no ID)": feat(("theta", "trace", "ref")),
        "GP: settings only (no amplifier information)": feat(("theta",)),
        "CNN: settings + curves (no ID)": ("cnn", ("theta",)),
        "A+B: physics + hierarchical GP (amplifier levels)": ("structured", ("hier", ())),
        "A: physics, no amplifier information": ("structured", ("none", ())),
        "A: physics + own features (no ID)": ("structured", ("none", ("trace", "ref"))),
    }


def run_variant(spec, df, curves, train, test):
    kind, arg = spec
    Y = data.make_targets(df)[0]
    if kind == "first":
        X, _ = data.make_inputs(df, arg, drop_constant=False)
        model = models.GP().fit(X[train], Y[train])
        return model.predict(X[test])[0]
    if kind == "cnn":
        X, _ = inputs_for(df, curves, arg, True, ALL_CURVES, log_frequency_spectra=True)
        model = models.CurveCNN().fit(models.take(X, train), Y[train])
        return model.predict(models.take(X, test))[0]
    gp_kind, own = arg
    prep = st.StructuredPrep(own_features=own).fit(df.iloc[train].reset_index(drop=True))
    X = prep.transform(df, curves)
    model = st.PhysicsStructured(gp_kind, use_components=False).fit(models.take(X, train), Y[train])
    return model.predict(models.take(X, test))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="../docs/ml_results")
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--n-train", type=int, default=240)
    parser.add_argument("--n-test", type=int, default=150)
    parser.add_argument("--ks", nargs="*", type=int, default=[0, 2])
    args = parser.parse_args()
    out = Path(args.out)
    results = {}
    for seed in args.seeds:
        df, curves = two_scans(args.n_train, args.n_test, seed)
        is_train, board, packet = (df["split"] == "train").to_numpy(), df["board"].to_numpy(), df["packet"].to_numpy()
        test = np.flatnonzero(df["split"] == "test")
        groups = {"known amplifiers": board[test] == 100, "new module": board[test] == 101}
        for k in args.ks:
            train = np.flatnonzero(is_train & ((board == 100) | (packet < k)))
            for label, spec in variants().items():
                t0 = time.time()
                mean = run_variant(spec, df, curves, train, test)
                for gname, mask in groups.items():
                    if gname == "known amplifiers" and k != args.ks[0]:
                        continue                                  # the known amplifiers do not depend on k
                    key = "known amplifiers" if gname == "known amplifiers" else f"new module, k={k}"
                    m = metrics(df.iloc[test[mask]].reset_index(drop=True), mean[mask])
                    results.setdefault(key, {}).setdefault(label, []).append(m)
                    print(f"[amp seed={seed}] {key:18s} {label:52s} " + "  ".join(
                        f"{t}: abs {m[t]['abs'] * 100:5.1f}% shape {m[t]['shape'] * 100:4.1f}% rank {m[t]['rank']:.2f}" for t in TARGETS)
                          + f"  {time.time() - t0:4.0f}s", flush=True)
        (out / "amp_information.json").write_text(json.dumps(results, indent=1))
    print("\n=== mean over training scans (abs / shape in %, rank) ===")
    summary = {}
    for key, by_label in results.items():
        for label, runs in by_label.items():
            e = {t: {s: float(np.mean([r[t][s] for r in runs])) for s in ("abs", "shape", "rank")} for t in TARGETS}
            summary.setdefault(key, {})[label] = e
            print(f"{key:18s} {label:52s} " + "  ".join(
                f"{t}: {e[t]['abs'] * 100:5.1f} / {e[t]['shape'] * 100:4.1f} / {e[t]['rank']:.2f}" for t in TARGETS))
    (out / "amp_information_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
