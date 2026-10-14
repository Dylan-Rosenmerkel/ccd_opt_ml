"""
benchmark.py -- the model comparisons of docs/ml_models.tex, on the SYNTHETIC world (synthetic.py).

    python -m ml.benchmark --out ../docs/ml_results            # all experiments (~1 h on a laptop)
    python -m ml.benchmark --only inputs bo --quick            # a subset, smaller

Experiments
-----------
inputs    which INPUTS help, for each MODEL type: theta only, trace features only, both, curves
          (CNN), curves + features.  Trained on one scan of 240 packets (the size of a 10-parameter
          x 20-value scan), scored on an independent scan of 200 packets.
learning  error against the number of training packets (30 ... 960), for the main candidates;
          also on the packets where the HIDDEN noise floor is large, which only the curves show.
holdout   a second module (4 amplifiers) that the models have seen in 0, 5 or 20 of its packets:
          how much data from a new module is needed.
representation  hand-made features, the curves, or both: every candidate sees the traces at theta
          and at the reference, only the representation differs.
bo        Bayesian optimization against random search: the best objective found against the
          number of packets (images) taken.
relevance the Gaussian process's length scales: which inputs it found to matter.

All numbers come from the toy physics of synthetic.py and only compare methods; they say nothing
about the real CCD.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data, evaluate, models, optimize, synthetic

TARGET_NAMES = ["sigma_e", "lambda", "gain"]
INPUT_SETS = {                                    # label -> (groups, uses curves)
    "theta + amplifier": (("theta", "amp"), False),
    "trace features only": (("trace", "ref"), False),
    "theta + amp + features": (("theta", "amp", "trace", "ref"), False),
    "curves + theta + amp": (("theta", "amp"), True),
    "curves + theta + amp + features": (("theta", "amp", "trace", "ref"), True),
}


def scan(n_packets, seed):
    """One synthetic scan: (df, X builder, Y)."""
    df, curves = synthetic.make_dataset(n_packets, seed=seed)
    Y, _ = data.make_targets(df)
    return df, curves, Y


THETA_CURVES = sorted(synthetic.CURVES)                            # curves of the traces at theta
ALL_CURVES = THETA_CURVES + ["ref: " + n for n in THETA_CURVES]    # ... and of the reference traces


def inputs_for(df, curves, groups, use_curves, curve_names=None, log_frequency_spectra=False):
    """Model input for one input set (the same columns, in the same order, for every synthetic scan)."""
    if groups:
        X, cols = data.make_inputs(df, groups, drop_constant=False)
    else:
        X, cols = None, []
    if use_curves:
        names = curve_names or THETA_CURVES
        log_names = tuple(n for n in names if "Allan" in n or "spectrum" in n or "variation" in n)
        return {"curves": data.curve_tensor(curves, names, log_names=log_names, log_frequency_spectra=log_frequency_spectra),
                "scalars": X}, cols
    return X, cols


def fit_score(make_model, train, test, groups, use_curves, mask=None, curve_names=None):
    df_tr, cv_tr, Y_tr = train
    df_te, cv_te, Y_te = test
    X_tr, _ = inputs_for(df_tr, cv_tr, groups, use_curves, curve_names)
    X_te, _ = inputs_for(df_te, cv_te, groups, use_curves, curve_names)
    t0 = time.time()
    model = make_model().fit(X_tr, Y_tr)
    mean, std = model.predict(X_te)
    out = {"all": evaluate.scores(Y_te, mean, std, TARGET_NAMES), "seconds": time.time() - t0}
    if mask is not None and mask.any():
        out["subset"] = evaluate.scores(Y_te[mask], mean[mask], std[mask], TARGET_NAMES)
    return out, model


# --------------------------------------------------------------------------- #
# experiments                                                                  #
# --------------------------------------------------------------------------- #

def exp_inputs(out, figs, n_train, n_test):
    train, test = scan(n_train, 1), scan(n_test, 99)
    results = {}
    plan = [(label, key) for label, (g, c) in INPUT_SETS.items()
            for key in (["cnn"] if c else ["ridge", "gbt", "gp", "mlp"])]
    for label, key in plan:
        groups, use_curves = INPUT_SETS[label]
        r, _ = fit_score(models.MODELS[key], train, test, groups, use_curves)
        results[f"{label} | {key}"] = r
        print(f"[inputs] {label:34s} {key:5s} {r['seconds']:6.1f}s  "
              + "  ".join(f"{t} {r['all'][t]['rel_error'] * 100:5.1f}% (cov {r['all'][t]['coverage']:.2f})" for t in TARGET_NAMES), flush=True)
    (out / "inputs.json").write_text(json.dumps(results, indent=1))
    # figure: relative error per target, grouped by input set
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=False)
    labels = list(results)
    colors = {"ridge": "#8c8c8c", "gbt": "#2a9d8f", "gp": "#e76f51", "mlp": "#264653", "cnn": "#e9c46a"}
    for ax, t in zip(axes, TARGET_NAMES):
        vals = [results[l]["all"][t]["rel_error"] * 100 for l in labels]
        y = np.arange(len(labels))[::-1]
        ax.barh(y, vals, color=[colors[l.split(" | ")[1]] for l in labels])
        for yi, v in zip(y, vals):
            ax.text(v * 1.02, yi, f"{v:.1f}", va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels([l.replace(" | ", "  |  ") for l in labels] if t == "sigma_e" else [], fontsize=7.5)
        ax.set_xscale("log")
        ax.set_title({"sigma_e": "read noise $\\sigma_e$", "lambda": "dark charge $\\lambda$", "gain": "gain"}[t])
        ax.set_xlabel("relative error on new packets (%)")
    fig.suptitle(f"Synthetic world: {n_train} training packets, {n_test} test packets (4 amplifiers each)", y=1.0)
    plt.tight_layout()
    fig.savefig(figs / "ml_inputs_models.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return results


def exp_learning(out, figs, sizes, n_test):
    test = scan(n_test, 99)
    floor_mask = test[0]["truth.floor"].to_numpy() > 0.1          # packets where the hidden pickup floor matters
    candidates = [("GP, theta + amp", "gp", ("theta", "amp"), False),
                  ("GP, theta + amp + features", "gp", ("theta", "amp", "trace", "ref"), False),
                  ("boosted trees, theta + amp + features", "gbt", ("theta", "amp", "trace", "ref"), False),
                  ("NN ensemble, theta + amp + features", "mlp", ("theta", "amp", "trace", "ref"), False),
                  ("CNN, curves + theta + amp", "cnn", ("theta", "amp"), True),
                  ("CNN, curves + theta + amp + features", "cnn", ("theta", "amp", "trace", "ref"), True)]
    results = {c[0]: {} for c in candidates}
    for n in sizes:
        train = scan(n, 1)
        for label, key, groups, use_curves in candidates:
            r, _ = fit_score(models.MODELS[key], train, test, groups, use_curves, mask=floor_mask)
            results[label][n] = r
            print(f"[learning] n={n:4d} {label:40s} sigma_e {r['all']['sigma_e']['rel_error'] * 100:5.1f}%  "
                  f"(floor packets {r['subset']['sigma_e']['rel_error'] * 100:5.1f}%)  lambda {r['all']['lambda']['rel_error'] * 100:5.1f}%"
                  f"  {r['seconds']:6.1f}s", flush=True)
        (out / "learning.json").write_text(json.dumps(results, indent=1))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for label in results:
        ns = sorted(results[label])
        style = "--" if "CNN" in label else "-"
        axes[0].plot(ns, [results[label][n]["all"]["sigma_e"]["rel_error"] * 100 for n in ns], style, marker="o", label=label)
        axes[1].plot(ns, [results[label][n]["subset"]["sigma_e"]["rel_error"] * 100 for n in ns], style, marker="o")
        axes[2].plot(ns, [results[label][n]["all"]["lambda"]["rel_error"] * 100 for n in ns], style, marker="o")
    titles = ["$\\sigma_e$, all test packets", "$\\sigma_e$, packets with the hidden noise floor", "$\\lambda$, all test packets"]
    for ax, title in zip(axes, titles):
        ax.set(xscale="log", yscale="log", xlabel="training packets", ylabel="relative error (%)", title=title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7.5)
    plt.tight_layout()
    fig.savefig(figs / "ml_learning_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return results


def exp_representation(out, figs, sizes, n_test):
    """
    Hand-made features, the trace curves, or both?  Every candidate sees the traces at theta AND
    the reference traces (as numbers or as curves), so they differ only in the representation.
    """
    test = scan(n_test, 99)
    floor_mask = test[0]["truth.floor"].to_numpy() > 0.1
    candidates = [
        ("features only (NN ensemble)", "mlp", ("trace", "ref"), False),
        ("curves only (CNN)", "cnn", (), True),
        ("features + curves (CNN)", "cnn", ("trace", "ref"), True),
        ("theta + amp + features (NN ensemble)", "mlp", ("theta", "amp", "trace", "ref"), False),
        ("theta + amp + curves (CNN)", "cnn", ("theta", "amp"), True),
        ("theta + amp + features + curves (CNN)", "cnn", ("theta", "amp", "trace", "ref"), True),
    ]
    results = {c[0]: {} for c in candidates}
    for n in sizes:
        train = scan(n, 1)
        for label, key, groups, use_curves in candidates:
            r, _ = fit_score(models.MODELS[key], train, test, groups, use_curves, mask=floor_mask, curve_names=ALL_CURVES)
            results[label][n] = r
            print(f"[representation] n={n:4d} {label:40s} " + "  ".join(
                f"{t} {r['all'][t]['rel_error'] * 100:5.1f}%" for t in TARGET_NAMES)
                + f"  (sigma_e on pickup packets {r['subset']['sigma_e']['rel_error'] * 100:5.1f}%)  {r['seconds']:5.1f}s", flush=True)
        (out / "representation.json").write_text(json.dumps(results, indent=1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    styles = {"features only (NN ensemble)": ("#2a9d8f", ":"), "curves only (CNN)": ("#e9c46a", ":"),
              "features + curves (CNN)": ("#e76f51", ":"), "theta + amp + features (NN ensemble)": ("#2a9d8f", "-"),
              "theta + amp + curves (CNN)": ("#e9c46a", "-"), "theta + amp + features + curves (CNN)": ("#e76f51", "-")}
    for label in results:
        ns = sorted(results[label])
        color, ls = styles[label]
        for ax, t in zip(axes, TARGET_NAMES):
            ax.plot(ns, [results[label][k]["all"][t]["rel_error"] * 100 for k in ns], ls, color=color, marker="o", ms=3, label=label)
    for ax, t in zip(axes, TARGET_NAMES):
        ax.set(xscale="log", yscale="log", xlabel="training packets", ylabel="relative error on new packets (%)",
               title={"sigma_e": "read noise $\\sigma_e$", "lambda": "dark charge $\\lambda$", "gain": "gain"}[t])
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Hand-made features, curves, or both (each sees the traces at theta and at the reference); dotted: without theta")
    plt.tight_layout()
    fig.savefig(figs / "ml_representation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return results


def exp_holdout(out, figs, n_train):
    """
    A second module (board 101, 4 amplifiers) that the models have not seen, or have seen only in its
    first k packets: how many packets of a new module are needed before the predictions are good?
    """
    df, curves = synthetic.make_dataset(n_train, seed=1, n_boards=2)
    Y, _ = data.make_targets(df)
    packet = df["packet_id"].str[4:].astype(int).to_numpy()
    new = (df["board"] == 101).to_numpy()
    results = {}
    for k in (0, 5, 20):
        train = ~new | (packet < k)                                  # board 100 always, board 101 only its first k packets
        test = new & (packet >= max(k, 20))                          # the same test packets for every k
        for label, groups in [("theta + amp", ("theta", "amp")), ("theta + amp + trace + ref", ("theta", "amp", "trace", "ref"))]:
            X, _ = data.make_inputs(df, groups, drop_constant=False)
            for key in ("gbt", "gp"):
                model = models.MODELS[key]().fit(X[train], Y[train])
                mean, std = model.predict(X[test])
                name = f"k={k} | {label} | {key}"
                results[name] = evaluate.scores(Y[test], mean, std, TARGET_NAMES)
                print(f"[holdout] new module, {k:2d} of its packets in training | {label:26s} {key:4s} " + "  ".join(
                    f"{t} {results[name][t]['rel_error'] * 100:5.1f}%" for t in TARGET_NAMES), flush=True)
    (out / "holdout.json").write_text(json.dumps(results, indent=1))
    return results


def exp_bo(out, figs, n_seeds, n_total, n_initial):
    low, high = synthetic.LOW, synthetic.HIGH
    rng_noise = np.random.default_rng(7)

    def measure(theta):                                              # one packet: noisy objective (label noise ~2 %)
        return synthetic.true_objective(theta) + rng_noise.normal(0, 0.02)

    allowed = synthetic.allowed                                      # a relation the optimizer must respect

    curves = {"Bayesian optimization": [], "random search": []}
    for seed in range(n_seeds):
        for label, strategy in (("Bayesian optimization", "bo"), ("random search", "random")):
            r = optimize.run_loop(measure, low, high, n_initial, n_total, seed=seed, strategy=strategy, allowed=allowed)
            best_true = [synthetic.true_objective(r["thetas"][int(np.argmin(r["J"][:k + 1]))]) for k in range(len(r["J"]))]
            curves[label].append(best_true)
            print(f"[bo] seed {seed} {label:22s} best true objective after {n_total}: {best_true[-1]:.4f}", flush=True)
    nominal = synthetic.true_objective(synthetic.NOMINAL)
    grid = optimize_reference(low, high, allowed)
    (out / "bo.json").write_text(json.dumps({"curves": curves, "nominal": nominal, "reference_optimum": grid}, indent=1))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for label, color in (("Bayesian optimization", "#e76f51"), ("random search", "#8c8c8c")):
        arr = np.array(curves[label])
        k = np.arange(1, arr.shape[1] + 1)
        ax.plot(k, arr.mean(0), color=color, label=label)
        ax.fill_between(k, arr.min(0), arr.max(0), color=color, alpha=0.2)
    ax.axhline(nominal, color="k", ls=":", label="nominal settings")
    ax.axhline(grid, color="k", ls="--", lw=0.8, label="best of 50,000 random points (reference)")
    ax.axvline(n_initial, color="0.6", lw=0.8)
    ax.set(xlabel="packets taken (each with an image)", ylabel="true objective of the best point so far\n"
           "log $\\sigma_e$ + 0.5 log $\\lambda$ (lower = better)",
           title=f"Finding better settings in 10 parameters ({n_seeds} repeats; band = min-max)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(figs / "ml_bo.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return curves


def optimize_reference(low, high, allowed, n=50000, seed=3):
    """The best true objective among many random allowed points (to put the BO result in context)."""
    rng = np.random.default_rng(seed)
    c = low + (high - low) * rng.random((n, len(low)))
    c = c[allowed(c)]
    amps = synthetic.make_amplifiers(np.random.default_rng(1234))
    vals = np.mean([synthetic.objective(*np.array([[synthetic.physics(t, amps[ch], 0.0)[k] for k in ("sigma_e", "lambda")]
                                                    for t in c]).T) for ch in (0, 1, 3)], axis=0)
    return float(vals.min())


def exp_relevance(out, figs, n_train):
    df, curves, Y = scan(n_train, 1)
    X, names = data.make_inputs(df, ("theta", "amp", "trace", "ref"), drop_constant=False)
    gp = models.GP(max_points=400).fit(X, Y)
    fig, axes = plt.subplots(1, 3, figsize=(17, 9), sharey=True)
    for t, ax in enumerate(axes):
        rel = 1 / gp.length_scales(t)
        order = np.argsort(rel)
        ax.barh(np.arange(len(names)), rel[order], color=["#e76f51" if names[i].startswith("theta") else
                                                          "#264653" if names[i].startswith("amp") else
                                                          "#2a9d8f" if names[i].startswith("trace") else "#e9c46a" for i in order])
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels([names[i] for i in order], fontsize=6.5)
        ax.set_xscale("log")
        ax.set_title(["log $\\sigma_e$", "log $\\lambda$", "log gain"][t])
        ax.set_xlabel("relevance = 1 / length scale (log)")
    fig.suptitle("What the Gaussian process uses (red: theta, dark: amplifier, green: trace features, yellow: reference features)")
    plt.tight_layout()
    fig.savefig(figs / "ml_gp_relevance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    rel = {TARGET_NAMES[t]: dict(sorted(zip(names, (1 / gp.length_scales(t)).tolist()), key=lambda kv: -kv[1])[:8])
           for t in range(3)}
    (out / "relevance.json").write_text(json.dumps(rel, indent=1))
    print("[relevance]", json.dumps(rel, indent=1), flush=True)
    return rel


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="../docs/ml_results")
    parser.add_argument("--figs", default="../docs/overleaf_ml_models/figures")
    parser.add_argument("--only", nargs="*", default=["inputs", "relevance", "holdout", "learning", "representation", "bo"])
    parser.add_argument("--quick", action="store_true", help="smaller sizes, for testing the script")
    args = parser.parse_args()
    out, figs = Path(args.out), Path(args.figs)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    q = args.quick
    if "inputs" in args.only:
        exp_inputs(out, figs, 60 if q else 240, 50 if q else 200)
    if "relevance" in args.only:
        exp_relevance(out, figs, 60 if q else 240)
    if "holdout" in args.only:
        exp_holdout(out, figs, 60 if q else 240)
    if "learning" in args.only:
        exp_learning(out, figs, [30, 60] if q else [30, 60, 120, 240, 480, 960], 50 if q else 200)
    if "representation" in args.only:
        exp_representation(out, figs, [60, 120] if q else [60, 120, 240, 480, 960], 50 if q else 200)
    if "bo" in args.only:
        exp_bo(out, figs, 2 if q else 5, 25 if q else 60, 10)


if __name__ == "__main__":
    main()
