"""
benchmark_advanced.py -- the remaining models (C, D, E, F, G of docs/overleaf_ml_models/ml_models.tex) on the SYNTHETIC world.

    python -m ml.benchmark_advanced --out ../docs/ml_results            # all experiments (~40 min)
    python -m ml.benchmark_advanced --only sets --quick

Experiments
-----------
pretrain   C and D: self-supervised pre-training on 3000 unlabeled packets (12,000 rows of curves), then
           30 ... 240 labeled packets: a CNN from scratch (linear- and log-frequency spectra), the
           pre-trained encoder frozen (D, linear probe) or fine-tuned (D), the deep kernel GP with the
           pre-trained encoder frozen or fine-tuned (C).  Also: what the embedding knows (probe R^2 of the
           hidden floor and state).
sets       E: world v4 (pickup on during only some trace sets; 10 % damaged sets).  A+B with the sets
           combined by fixed rules or by the learned set network; two training scans per size.
modules    G: a hierarchical GP trained on ONE module or on FOUR, then a fifth module seen in 0-5 packets.
           With several modules the module-to-module spread is learned (meta-learning in GP form).
bo         F: trust region, learned / known / no constraint, multi-fidelity (trace-only packets at 0.25 or
           0.5 of a full packet), ParEGO (Pareto front, hypervolume), against the global decomposed surrogate
           and random search.  5 repeats, a budget of 40 full packets.
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

from . import bo_advanced, data, deep, evaluate, models, setmodels, structured as st, synthetic
from .benchmark import ALL_CURVES, inputs_for
from .benchmark_structured import TARGETS, _plain_log_axis, run_structured, truth_targets


def _scores(df_te, mean, std, seconds):
    return {"truth": evaluate.scores(truth_targets(df_te), mean, std, TARGETS),
            "labels": evaluate.scores(data.make_targets(df_te)[0], mean, std, TARGETS), "seconds": seconds}


def _line(tag, label, r):
    return f"[{tag}] {label:46s} " + "  ".join(
        f"{t} {r['truth'][t]['rel_error'] * 100:5.1f}% (cov {r['labels'][t]['coverage']:.2f})" for t in TARGETS) + f"  {r['seconds']:5.1f}s"


# --------------------------------------------------------------------------- #
# C, D                                                                         #
# --------------------------------------------------------------------------- #

def exp_pretrain(out, figs, sizes, n_unlabeled, n_test, epochs):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    un, cu = synthetic.make_dataset(n_unlabeled, seed=7)
    te, cte = synthetic.make_dataset(n_test, seed=99)
    results = {"probe": {}, "learning": {}}
    pre = {}
    for logf in (False, True):
        Xu, _ = inputs_for(un, cu, ("theta",), True, ALL_CURVES, log_frequency_spectra=logf)
        Xp, _ = inputs_for(te, cte, ("theta",), True, ALL_CURVES, log_frequency_spectra=logf)
        t0 = time.time()
        p = deep.Pretrainer(epochs=epochs).fit(Xu["curves"], Xu["scalars"])
        pre[logf] = p
        zu, zt = p.encode(Xu["curves"]), p.encode(Xp["curves"])
        key = "log-frequency spectra" if logf else "linear-frequency spectra"
        with threadpool_limits(1):
            results["probe"][key] = {q: float(HistGradientBoostingRegressor(max_iter=300).fit(zu, un[q]).score(zt, te[q]))
                                     for q in ("truth.floor", "truth.state", "truth.sigma1")}
        results["probe"][key]["losses"] = p.losses
        print(f"[pretrain] {key}: {time.time() - t0:.0f}s, probe R2 " + ", ".join(
            f"{q} {v:.2f}" for q, v in results["probe"][key].items() if q != "losses"), flush=True)
    p = pre[True]
    for n in sizes:
        tr, ctr = synthetic.make_dataset(n, seed=1)
        Ytr = data.make_targets(tr)[0]
        X = {lf: [inputs_for(d, c, ("theta", "amp"), True, ALL_CURVES, log_frequency_spectra=lf)[0] for d, c in ((tr, ctr), (te, cte))]
             for lf in (False, True)}
        prep = st.StructuredPrep().fit(tr)
        P = [prep.transform(tr, ctr), prep.transform(te, cte)]
        dk = [{"curves": X[True][i]["curves"], "x": P[i]["x"], "amp": P[i]["amp"]} for i in range(2)]
        plan = {"CNN from scratch (first round)": (lambda: models.CurveCNN(), X[False]),
                "CNN from scratch, log-frequency spectra": (lambda: models.CurveCNN(), X[True]),
                "D: pre-trained encoder frozen + NN ensemble": (lambda: deep.PretrainedHead(p), X[True]),
                "D: pre-trained encoder fine-tuned": (lambda: deep.PretrainedCNN(p), X[True]),
                "C: deep kernel GP, pre-trained encoder frozen": (lambda: deep.DeepKernelGP(p, finetune=False), dk),
                "C: deep kernel GP, pre-trained encoder fine-tuned": (lambda: deep.DeepKernelGP(p, finetune=True), dk)}
        for label, (make, (Xtr, Xte)) in plan.items():
            t0 = time.time()
            m, s = make().fit(Xtr, Ytr).predict(Xte)
            r = _scores(te, m, s, time.time() - t0)
            results["learning"].setdefault(label, {})[n] = r
            print(_line(f"pretrain n={n}", label, r), flush=True)
        (out / "advanced_pretrain.json").write_text(json.dumps(results, indent=1))
    plot_learning(results["learning"], figs / "ml_advanced_pretrain.png",
                  "C and D: pre-training on 3000 unlabeled packets, then n labeled packets (world v2)", out)
    return results


def plot_learning(res, path, title, out, reference=True):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ref = None
    if reference and (out / "structured_learning.json").exists():
        ref = json.loads((out / "structured_learning.json").read_text())["v2"].get("A+B: physics + hierarchical GP")
    for ax, t in zip(axes, TARGETS):
        for label, by_n in res.items():
            ns = sorted(by_n, key=int)
            ax.plot([int(n) for n in ns], [by_n[n]["truth"][t]["rel_error"] * 100 for n in ns], "-o", ms=3.5, label=label)
        if ref is not None:
            ns = sorted(ref, key=int)
            ax.plot([int(n) for n in ns], [ref[n]["truth"][t]["rel_error"] * 100 for n in ns], "k--", lw=1, label="A+B (reference)")
        ax.set(xscale="log", yscale="log", xlabel="labeled training packets", ylabel="error against the truth (%)",
               title={"sigma_e": "read noise $\\sigma_e$", "lambda": "dark charge $\\lambda$", "gain": "gain"}[t])
        _plain_log_axis(ax, sorted({int(n) for b in res.values() for n in b}))
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(title, y=1.0)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# E                                                                            #
# --------------------------------------------------------------------------- #

V4 = {"within_packet_pickup": True, "damaged_fraction": 0.1}


def exp_sets(out, figs, sizes, n_test, seeds):
    te, cte = synthetic.make_dataset(n_test, seed=99, **V4)
    results = {}
    for n in sizes:
        for seed in seeds:
            tr, ctr = synthetic.make_dataset(n, seed=seed, **V4)
            prep = st.StructuredPrep().fit(tr)
            Xtr = dict(prep.transform(tr, ctr), sets=ctr["sets: noise: Allan deviation"])
            Xte = dict(prep.transform(te, cte), sets=cte["sets: noise: Allan deviation"])
            for comb in setmodels.COMBINERS:
                t0 = time.time()
                m, s = setmodels.SetStructured(comb).fit(Xtr, data.make_targets(tr)[0]).predict(Xte)
                r = _scores(te, m, s, time.time() - t0)
                results.setdefault(comb, {}).setdefault(n, []).append(r)
                print(_line(f"sets n={n} seed={seed}", comb, r), flush=True)
            (out / "advanced_sets.json").write_text(json.dumps(results, indent=1))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for comb, by_n in results.items():
        ns = sorted(by_n, key=int)
        vals = np.array([[r["truth"]["sigma_e"]["rel_error"] * 100 for r in by_n[n]] for n in ns])
        ax.errorbar([int(n) for n in ns], vals.mean(1), yerr=vals.std(1), marker="o", ms=4, capsize=2,
                    lw=2 if comb == "learned" else 1, label=comb)
    ax.set(xscale="log", yscale="log", xlabel="training packets", ylabel="$\\sigma_e$ error against the truth (%)",
           title="E: combining the 10 trace sets (pickup in some sets, 10 % damaged)")
    _plain_log_axis(ax, sorted({int(n) for b in results.values() for n in b}))
    ax.legend(fontsize=7.5, title="A+B with the sets combined by", title_fontsize=7.5)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(figs / "ml_advanced_sets.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return results


# --------------------------------------------------------------------------- #
# G                                                                            #
# --------------------------------------------------------------------------- #

def exp_modules(out, figs, n_train, ks):
    df, curves = synthetic.make_dataset(n_train, seed=1, n_boards=5)
    packet = df["packet_id"].str[4:].astype(int).to_numpy()
    board = df["board"].to_numpy()
    new = board == 104
    test_rows = new & (packet >= max(ks))
    sub = lambda m: (df[m].reset_index(drop=True), {k: v[m] for k, v in curves.items()})
    results = {}
    plan = {"B: hierarchical GP": lambda: st.DirectGP("hier"),
            "A+B: physics + hierarchical GP": lambda: st.PhysicsStructured("hier", use_components=False)}
    for n_modules in (1, 4):
        known = np.isin(board, np.arange(100, 100 + n_modules))
        for k in ks:
            train_rows = known | (new & (packet < k))
            for label, make in plan.items():
                r, model = run_structured(make, sub(train_rows), sub(test_rows),
                                          full=(df, curves, np.flatnonzero(train_rows), np.flatnonzero(test_rows)))
                name = f"{label}, trained on {n_modules} module{'s' if n_modules > 1 else ''}"
                results.setdefault(name, {})[k] = r
                print(_line(f"modules k={k}", name, r), flush=True)
            (out / "advanced_modules.json").write_text(json.dumps(results, indent=1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, by_k in results.items():
        kk = sorted(by_k, key=int)
        style = "-" if "4 modules" in name else "--"
        for ax, (what, key) in zip(axes, (("rel_error", "truth"), ("coverage", "labels"))):
            vals = [by_k[k][key]["sigma_e"][what] * (100 if what == "rel_error" else 1) for k in kk]
            ax.plot([int(k) for k in kk], vals, style, marker="o", ms=4, label=name)
    axes[0].set(yscale="log", xlabel="packets of the new module in training", ylabel="$\\sigma_e$ error on the new module (%)")
    axes[1].set(xlabel="packets of the new module in training", ylabel="68 % coverage of $\\sigma_e$ (target 0.68)")
    axes[1].axhline(0.68, color="k", lw=0.8, ls=":")
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("G: learning the module-to-module spread from several modules", y=1.0)
    plt.tight_layout()
    fig.savefig(figs / "ml_advanced_modules.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return results


# --------------------------------------------------------------------------- #
# F                                                                            #
# --------------------------------------------------------------------------- #

def exp_bo(out, figs, seeds, budget):
    strategies = {"global (decomposed surrogate)": dict(strategy="global"), "trust region": dict(strategy="trust region"),
                  "learned constraint": dict(strategy="learned constraint"), "no constraint": dict(strategy="no constraint"),
                  "multi-fidelity, trace = 0.25 packet": dict(strategy="multi-fidelity", trace_cost=0.25),
                  "multi-fidelity, trace = 0.5 packet": dict(strategy="multi-fidelity", trace_cost=0.5),
                  "ParEGO (multi-objective)": dict(strategy="parego"), "random": dict(strategy="random")}
    results = {}
    sim = bo_advanced.Simulator(0)
    # hypervolume reference: 20 % worse than nominal in both (the nominal sigma_e is already near its best,
    # so no point beats nominal in BOTH quantities; the front is about what lambda costs in sigma_e)
    ref = sim.true_parts(synthetic.NOMINAL) + np.log(1.2)
    for label, kw in strategies.items():
        for seed in seeds:
            t0 = time.time()
            r = bo_advanced.run(seed=seed, n_total=int(budget), budget=budget, **kw)
            if "parts" in r:
                r["hypervolume"] = bo_advanced.hypervolume_2d(np.array(r["parts"]), ref)
            results.setdefault(label, []).append(r)
            print(f"[bo] {label:38s} seed {seed}: best true J {r['best_true'][-1]:.3f}, after 20 cost "
                  f"{_best_at(r, 20):.3f}, failures {r['failures']}, traces {r['n_trace']}"
                  + (f", hypervolume {r['hypervolume']:.4f}" if "hypervolume" in r else "") + f"  {time.time() - t0:.0f}s", flush=True)
        (out / "advanced_bo.json").write_text(json.dumps({"results": results, "nominal_parts": ref.tolist()}, indent=1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    grid = np.linspace(1, budget, 80)
    for label, runs in results.items():
        curves = np.array([[_best_at(r, c) for c in grid] for r in runs])
        axes[0].plot(grid, curves.mean(0), label=label, lw=2 if "global" in label else 1)
    axes[0].axhline(synthetic.true_objective(synthetic.NOMINAL), color="k", ls=":", lw=0.8, label="nominal")
    axes[0].set(xlabel="cost (full-packet equivalents)", ylabel="true J of the best point (mean of repeats)",
                title="F: strategies at equal cost")
    axes[0].set_ylim(-2.85, -2.45)
    axes[0].legend(fontsize=7)
    for label in ("ParEGO (multi-objective)", "global (decomposed surrogate)", "random"):
        p = np.array(results[label][0]["parts"])
        axes[1].plot(p[:, 0], p[:, 1], "o", ms=3, alpha=0.6, label=label)
    axes[1].plot(*(ref - np.log(1.2)), "k*", ms=10, label="nominal")
    axes[1].set(xlabel="mean log $\\sigma_e$ (true)", ylabel="mean log $\\lambda$ (true)", title="points tried (first repeat)")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(figs / "ml_advanced_bo.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return results


def _best_at(r, cost):
    """Best true J among the full packets taken by the given cost (nan before the first)."""
    k = int(np.searchsorted(r["costs"], cost + 1e-9))
    return r["best_true"][k - 1] if k > 0 else np.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="../docs/ml_results")
    parser.add_argument("--figs", default="../docs/overleaf_ml_models/figures")
    parser.add_argument("--only", nargs="*", default=["sets", "modules", "bo", "pretrain"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out, figs = Path(args.out), Path(args.figs)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    q = args.quick
    if "sets" in args.only:
        exp_sets(out, figs, [30, 60] if q else [30, 60, 120, 240], 50 if q else 200, [1] if q else [1, 2])
    if "modules" in args.only:
        exp_modules(out, figs, 40 if q else 120, [0, 2] if q else [0, 1, 2, 5])
    if "bo" in args.only:
        exp_bo(out, figs, [0] if q else [0, 1, 2, 3, 4], 15.0 if q else 40.0)
    if "pretrain" in args.only:
        exp_pretrain(out, figs, [30] if q else [30, 60, 120, 240], 300 if q else 3000, 50 if q else 200, 3 if q else 40)


if __name__ == "__main__":
    main()
