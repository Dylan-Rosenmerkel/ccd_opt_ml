"""
benchmark_structured.py -- the second-round models (structured.py) against the first round, on the SYNTHETIC world.

    python -m ml.benchmark_structured --out ../docs/ml_results          # all experiments (~30 min)
    python -m ml.benchmark_structured --only newmodule --quick

Experiments
-----------
learning   error against the number of training packets (15 ... 480) in two worlds:
           v2  the world of the first round (benchmark.py): the trace physics is complete;
           v3  the image sees a noise term the trace cannot (low-frequency noise over the 4.8-min image),
               and the image gives lambda row by row (dark current and CIC separately).
           Models: the first round's best (GP on theta + amp + features; CNN on curves), B (hierarchical GP,
           direct), A (physics-structured with an ordinary GP), A + B, A + B without the noise-floor
           measurement, and in v3 A + B with lambda = dark + CIC.  Scored against the labels (which have
           1.5 % / 3 % / 0.8 % noise) and against the noise-free truth.
newmodule  a second module seen in 0, 1, 2, 5, 10 or 20 of its packets (as benchmark.py's holdout).
design     the scan design: one-at-a-time (the planned 10 parameters x 20 values), random points, or half
           of each, with the same number of packets; tested on random points (where an optimizer looks).
bo         F: Bayesian optimization with a DECOMPOSED surrogate (one hierarchical GP for log sigma_e and one for
           log lambda, per amplifier, combined into J) against the first round's GP on J itself.
conformal  H: the coverage of the 68 % and 95 % error bars before and after conformal calibration
           (calibrate.py), for the boosted trees, the network ensemble, the CNN and A + B.

All numbers come from toy physics and only compare methods.
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

from . import calibrate, data, evaluate, models, structured as st, synthetic
from .benchmark import THETA_CURVES, inputs_for

TARGETS = ["sigma_e", "lambda", "gain"]
WORLDS = {"v2": {}, "v3": {"image_only_noise": True, "per_row_lambda": True}}
FEATURE_GROUPS = ("theta", "amp", "trace", "ref")


def truth_targets(df) -> np.ndarray:
    return np.log(df[["truth.sigma_e", "truth.lambda", "truth.gain"]].to_numpy(float))


def score_both(df_te, mean, std, seconds):
    Y = data.make_targets(df_te)[0]
    return {"labels": evaluate.scores(Y, mean, std, TARGETS),
            "truth": evaluate.scores(truth_targets(df_te), mean, std, TARGETS), "seconds": seconds}


def structured_candidates(world):
    out = {"B: hierarchical GP": lambda: st.DirectGP("hier"),
           "A: physics + ordinary GP": lambda: st.PhysicsStructured("plain", use_components=False),
           "A+B: physics + hierarchical GP": lambda: st.PhysicsStructured("hier", use_components=False),
           "A+B, noise floor not measured": lambda: st.PhysicsStructured("hier", use_components=False, use_floor=False)}
    if world == "v3":
        out["A+B, lambda = dark + CIC"] = lambda: st.PhysicsStructured("hier", use_components=True)
    return out


def run_structured(make, train, test, full=None):
    """A second-round model.  ``full`` = (df, curves, train rows, test rows): the inputs are built on whole
    packets (the packet state vector needs every amplifier of the packet), then split."""
    (df_tr, cv_tr), (df_te, cv_te) = train, test
    prep = st.StructuredPrep().fit(df_tr)
    if full is None:
        X_tr, X_te = prep.transform(df_tr, cv_tr), prep.transform(df_te, cv_te)
    else:
        X = prep.transform(full[0], full[1])
        X_tr, X_te = models.take(X, full[2]), models.take(X, full[3])
    t0 = time.time()
    model = make().fit(X_tr, data.make_targets(df_tr)[0], st.component_targets(df_tr))
    mean, std = model.predict(X_te)
    return score_both(df_te, mean, std, time.time() - t0), model


def run_first_round(key, groups, use_curves, train, test, full=None):
    """A first-round model.  ``full`` = (df, curves, train rows, test rows): build the inputs on the whole
    dataset and then split, so that the amplifier one-hot has the same columns on both sides."""
    (df_tr, cv_tr), (df_te, cv_te) = train, test
    if full is None:
        X_tr, _ = inputs_for(df_tr, cv_tr, groups, use_curves, THETA_CURVES)
        X_te, _ = inputs_for(df_te, cv_te, groups, use_curves, THETA_CURVES)
    else:
        X, _ = inputs_for(full[0], full[1], groups, use_curves, THETA_CURVES)
        X_tr, X_te = models.take(X, full[2]), models.take(X, full[3])
    t0 = time.time()
    model = models.MODELS[key]().fit(X_tr, data.make_targets(df_tr)[0])
    mean, std = model.predict(X_te)
    return score_both(df_te, mean, std, time.time() - t0)


def _line(tag, label, r):
    return (f"[{tag}] {label:34s} " + "  ".join(
        f"{t} {r['labels'][t]['rel_error'] * 100:5.1f}% (truth {r['truth'][t]['rel_error'] * 100:4.1f}, cov {r['labels'][t]['coverage']:.2f})"
        for t in TARGETS) + f"  {r['seconds']:5.1f}s")


def exp_learning(out, figs, sizes, n_test, cnn_sizes):
    results = {}
    for world, kw in WORLDS.items():
        test = synthetic.make_dataset(n_test, seed=99, **kw)
        results[world] = {}
        for n in sizes:
            train = synthetic.make_dataset(n, seed=1, **kw)
            plan = dict(structured_candidates(world))
            plan["first round: GP, theta + amp + features"] = ("gp", FEATURE_GROUPS, False)
            if n in cnn_sizes:
                plan["first round: CNN, curves + theta + amp"] = ("cnn", ("theta", "amp"), True)
            for label, make in plan.items():
                if isinstance(make, tuple):
                    r = run_first_round(*make, train, test)
                else:
                    r, _ = run_structured(make, train, test)
                results[world].setdefault(label, {})[n] = r
                print(_line(f"learning {world} n={n}", label, r), flush=True)
            (out / "structured_learning.json").write_text(json.dumps(results, indent=1))
    plot_learning(results, figs)
    return results


STYLE = {"first round: GP, theta + amp + features": ("#8c8c8c", "-", "o"),
         "first round: CNN, curves + theta + amp": ("#e9c46a", "-", "s"),
         "B: hierarchical GP": ("#2a9d8f", "-", "o"),
         "A: physics + ordinary GP": ("#264653", "--", "^"),
         "A+B: physics + hierarchical GP": ("#e76f51", "-", "D"),
         "A+B, noise floor not measured": ("#e76f51", ":", "x"),
         "A+B, lambda = dark + CIC": ("#9b2226", "-.", "v")}


def plot_learning(results, figs):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4))
    for row, world in enumerate(results):
        for col, t in enumerate(TARGETS):
            ax = axes[row, col]
            for label, by_n in results[world].items():
                ns = sorted(by_n, key=int)
                color, ls, marker = STYLE[label]
                ax.plot([int(n) for n in ns], [by_n[n]["truth"][t]["rel_error"] * 100 for n in ns], ls, color=color,
                        marker=marker, ms=4, label=label)
            ax.set(xscale="log", yscale="log", xlabel="training packets", ylabel="error against the truth (%)")
            ax.set_title(f"world {world}: " + {"sigma_e": "read noise $\\sigma_e$", "lambda": "dark charge $\\lambda$",
                                               "gain": "gain"}[t], fontsize=10)
            ax.grid(True, which="both", alpha=0.3)
    axes[1, 0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Second-round models: v2 = trace physics complete; v3 = image-only noise + lambda row by row", y=1.0)
    plt.tight_layout()
    fig.savefig(figs / "ml_structured_learning.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def exp_newmodule(out, figs, n_train, ks):
    df, curves = synthetic.make_dataset(n_train, seed=1, n_boards=2)
    packet = df["packet_id"].str[4:].astype(int).to_numpy()
    new = (df["board"] == 101).to_numpy()
    sub = lambda m: (df[m].reset_index(drop=True), {k: v[m] for k, v in curves.items()})
    results = {}
    plan = {"first round: GP, theta + amp + features": ("gp", FEATURE_GROUPS, False),
            "first round: boosted trees, theta + amp + features": ("gbt", FEATURE_GROUPS, False),
            "B: hierarchical GP": structured_candidates("v2")["B: hierarchical GP"],
            "A: physics + ordinary GP": structured_candidates("v2")["A: physics + ordinary GP"],
            "A+B: physics + hierarchical GP": structured_candidates("v2")["A+B: physics + hierarchical GP"]}
    test_rows = new & (packet >= max(ks))
    test = sub(test_rows)
    for k in ks:
        train_rows = ~new | (packet < k)
        train = sub(train_rows)
        for label, make in plan.items():
            if isinstance(make, tuple):
                r = run_first_round(*make, train, test, full=(df, curves, np.flatnonzero(train_rows), np.flatnonzero(test_rows)))
            else:
                r, _ = run_structured(make, train, test, full=(df, curves, np.flatnonzero(train_rows), np.flatnonzero(test_rows)))
            results.setdefault(label, {})[k] = r
            print(_line(f"new module k={k:2d}", label, r), flush=True)
        (out / "structured_newmodule.json").write_text(json.dumps(results, indent=1))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    colors = {"first round: GP, theta + amp + features": "#8c8c8c", "first round: boosted trees, theta + amp + features": "#2a9d8f",
              "B: hierarchical GP": "#264653", "A: physics + ordinary GP": "#e9c46a", "A+B: physics + hierarchical GP": "#e76f51"}
    for ax, t in zip(axes, TARGETS):
        for label, by_k in results.items():
            kk = sorted(by_k, key=int)
            ax.plot([int(k) + 0.5 for k in kk], [by_k[k]["truth"][t]["rel_error"] * 100 for k in kk], "-o", ms=4,
                    color=colors[label], label=label)
        ax.set(xscale="log", yscale="log", xlabel="packets of the new module in training (+0.5)",
               ylabel="error on the new module (%)",
               title={"sigma_e": "read noise $\\sigma_e$", "lambda": "dark charge $\\lambda$", "gain": "gain"}[t])
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7.5)
    fig.suptitle(f"A module never seen: trained on {n_train} packets of module 100 plus k packets of module 101", y=1.0)
    plt.tight_layout()
    fig.savefig(figs / "ml_structured_newmodule.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return results


def exp_design(out, figs, n_test):
    test = synthetic.make_dataset(n_test, seed=99)
    oat = synthetic.oat_thetas(20)
    n = len(oat)
    rng = np.random.default_rng(5)
    oat_half = synthetic.oat_thetas(10)
    designs = {"one-at-a-time": oat, "random": synthetic.sample_theta(rng, n),
               "half one-at-a-time, half random": np.vstack([oat_half, synthetic.sample_theta(rng, n - len(oat_half))])}
    plan = {"first round: GP, theta + amp + features": ("gp", FEATURE_GROUPS, False),
            "first round: boosted trees, theta + amp + features": ("gbt", FEATURE_GROUPS, False),
            "B: hierarchical GP": structured_candidates("v2")["B: hierarchical GP"],
            "A+B: physics + hierarchical GP": structured_candidates("v2")["A+B: physics + hierarchical GP"]}
    results = {}
    for dname, thetas in designs.items():
        train = synthetic.make_dataset(len(thetas), seed=1, thetas=np.random.default_rng(2).permutation(thetas))
        for label, make in plan.items():
            r = run_first_round(*make, train, test) if isinstance(make, tuple) else run_structured(make, train, test)[0]
            results.setdefault(dname, {})[label] = r
            print(_line(f"design {dname} ({len(thetas)} packets)", label, r), flush=True)
        (out / "structured_design.json").write_text(json.dumps(results, indent=1))
    return results


GOOD_CHANNELS = (0, 1, 3)


def measure_packet(theta, amps, rng):
    """One packet at state 0 without pickup: per good amplifier the labels log sigma_e, log lambda (with label noise)."""
    rows = []
    for ch in GOOD_CHANNELS:
        ph = synthetic.physics(np.asarray(theta, float), amps[ch], 0.0)
        rows.append((np.log(ph["sigma_e"] * (1 + 0.015 * rng.normal())), np.log(ph["lambda"] * (1 + 0.03 * rng.normal()))))
    return np.array(rows)                                            # (3, 2)


def decomposed_propose(thetas, labels, low, high, rng, allowed, n_candidates=4000):
    """EI on J = mean over amplifiers of (log sigma_e + 1/2 log lambda), from two hierarchical GPs on theta."""
    n = len(thetas)
    X = {"x": np.repeat(thetas, len(GOOD_CHANNELS), axis=0), "amp": np.tile(np.arange(len(GOOD_CHANNELS)), n)}
    Y = labels.reshape(-1, 2)
    gp = st.HierGP(levels=("amp",), iters=60).fit(X, Y)
    cand = low + (high - low) * rng.random((n_candidates, len(low)))
    cand = cand[allowed(cand)]
    m, sd = gp.predict({"x": np.repeat(cand, len(GOOD_CHANNELS), axis=0), "amp": np.tile(np.arange(len(GOOD_CHANNELS)), len(cand))})
    m = m.reshape(len(cand), len(GOOD_CHANNELS), 2)
    sd = sd.reshape(len(cand), len(GOOD_CHANNELS), 2)
    mu = (m[..., 0] + 0.5 * m[..., 1]).mean(1)
    s = np.sqrt(sd[..., 0] ** 2 + 0.25 * sd[..., 1] ** 2).mean(1)    # amplifiers fully correlated: conservative
    J = (labels[..., 0] + 0.5 * labels[..., 1]).mean(1)
    from .optimize import expected_improvement
    ei = expected_improvement(mu, s, float(J.min()))
    return cand[int(np.argmax(ei))]


def exp_bo(out, figs, n_seeds, n_total, n_initial):
    from .optimize import propose
    low, high, allowed = synthetic.LOW, synthetic.HIGH, synthetic.allowed
    amps = synthetic.make_amplifiers(np.random.default_rng(1234))
    curves = {"GP on the objective (first round)": [], "decomposed: hierarchical GPs on sigma_e and lambda": []}
    for seed in range(n_seeds):
        for label in curves:
            rng, rng_m = np.random.default_rng(seed), np.random.default_rng(1000 + seed)
            thetas = synthetic.sample_theta(rng, n_initial)
            labels = np.array([measure_packet(t, amps, rng_m) for t in thetas])
            while len(thetas) < n_total:
                J = (labels[..., 0] + 0.5 * labels[..., 1]).mean(1)
                if label.startswith("GP on"):
                    nxt, _ = propose(thetas, J, low, high, rng, allowed=allowed)
                else:
                    nxt = decomposed_propose(thetas, labels, low, high, rng, allowed)
                thetas = np.vstack([thetas, nxt])
                labels = np.concatenate([labels, measure_packet(nxt, amps, rng_m)[None]])
            J = (labels[..., 0] + 0.5 * labels[..., 1]).mean(1)
            best_true = [synthetic.true_objective(thetas[int(np.argmin(J[:k + 1]))]) for k in range(len(J))]
            curves[label].append(best_true)
            print(f"[bo] seed {seed} {label:52s} best true objective after {n_total}: {best_true[-1]:.4f} (after 30: {best_true[min(29, len(best_true) - 1)]:.4f})", flush=True)
            (out / "structured_bo.json").write_text(json.dumps({"curves": curves, "nominal": synthetic.true_objective(synthetic.NOMINAL)}, indent=1))
    ref = json.loads((out / "bo.json").read_text())["reference_optimum"] if (out / "bo.json").exists() else None
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for label, color in zip(curves, ("#8c8c8c", "#e76f51")):
        arr = np.array(curves[label])
        k = np.arange(1, arr.shape[1] + 1)
        ax.plot(k, arr.mean(0), color=color, label=label)
        ax.fill_between(k, arr.min(0), arr.max(0), color=color, alpha=0.2)
    ax.axhline(synthetic.true_objective(synthetic.NOMINAL), color="k", ls=":", label="nominal settings")
    if ref is not None:
        ax.axhline(ref, color="k", ls="--", lw=0.8, label="best of 50,000 random points")
    ax.axvline(n_initial, color="0.6", lw=0.8)
    ax.set(xlabel="packets taken (each with an image)", ylabel="true objective of the best point so far",
           title=f"Decomposed surrogate vs GP on the objective ({n_seeds} repeats; band = min-max)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(figs / "ml_structured_bo.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return curves


def exp_conformal(out, figs, sizes, n_test):
    test = synthetic.make_dataset(n_test, seed=99)
    Y_te = data.make_targets(test[0])[0]
    results = {}
    for n in sizes:
        train = synthetic.make_dataset(n, seed=1)
        Y_tr = data.make_targets(train[0])[0]
        groups = train[0]["packet_id"].to_numpy()
        prep = st.StructuredPrep().fit(train[0])
        plan = {"boosted trees, theta + amp + features": (models.GBT, *[inputs_for(*d, FEATURE_GROUPS, False)[0] for d in (train, test)], None),
                "NN ensemble, theta + amp + features": (models.MLPEnsemble, *[inputs_for(*d, FEATURE_GROUPS, False)[0] for d in (train, test)], None),
                "CNN, curves + theta + amp": (models.CurveCNN, *[inputs_for(*d, ("theta", "amp"), True, THETA_CURVES)[0] for d in (train, test)], None),
                "A+B: physics + hierarchical GP": (lambda: st.PhysicsStructured("hier", use_components=False),
                                                   prep.transform(*train), prep.transform(*test), None)}
        for label, (make, X_tr, X_te, extra) in plan.items():
            t0 = time.time()
            c = calibrate.Conformal(make).fit(X_tr, Y_tr, groups, extra)
            m, s = c.model.predict(X_te)
            r = {"coverage": calibrate.coverage_report(Y_te, m, s, c.factors),
                 "factors": {str(k): v.tolist() for k, v in c.factors.items()},
                 "rel_error": [float(np.sqrt(np.mean((m[:, t] - Y_te[:, t]) ** 2))) for t in range(3)],
                 "seconds": time.time() - t0}
            results.setdefault(label, {})[n] = r
            cov = r["coverage"]
            print(f"[conformal n={n}] {label:38s} " + "  ".join(
                f"{t}: 68% {cov['raw 0.68'][i]:.2f}->{cov['conformal 0.68'][i]:.2f}, 95% {cov['raw 0.95'][i]:.2f}->{cov['conformal 0.95'][i]:.2f}"
                for i, t in enumerate(TARGETS)) + f"  {r['seconds']:5.0f}s", flush=True)
            (out / "structured_conformal.json").write_text(json.dumps(results, indent=1))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="../docs/ml_results")
    parser.add_argument("--figs", default="../docs/overleaf_ml_models/figures")
    parser.add_argument("--only", nargs="*", default=["learning", "newmodule", "conformal", "design", "bo"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out, figs = Path(args.out), Path(args.figs)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    q = args.quick
    if "newmodule" in args.only:
        exp_newmodule(out, figs, 60 if q else 240, [0, 2, 20] if q else [0, 1, 2, 5, 10, 20])
    if "learning" in args.only:
        exp_learning(out, figs, [15, 30] if q else [15, 30, 60, 120, 240, 480], 50 if q else 200,
                     [30] if q else [60, 240, 480])
    if "conformal" in args.only:
        exp_conformal(out, figs, [30] if q else [60, 240], 50 if q else 200)
    if "design" in args.only:
        exp_design(out, figs, 50 if q else 200)
    if "bo" in args.only:
        exp_bo(out, figs, 2 if q else 5, 20 if q else 60, 10)


if __name__ == "__main__":
    main()
