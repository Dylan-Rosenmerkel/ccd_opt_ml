"""
benchmark_seeds.py -- are the benchmark conclusions robust?  The key comparisons, repeated over 5 training scans.

Most experiments of benchmark.py, benchmark_structured.py and benchmark_advanced.py used ONE training scan
(seed 1).  With two scans, the set-model conclusion (E) reversed, so differences smaller than the scatter
between scans are not established.  Here every comparison is repeated with training scans of seeds 1-5 (the
test scan, seed 99, and the toy CCDs stay the same) and reported as mean +- standard deviation.

    python -m ml.benchmark_seeds --out ../docs/ml_results            # all (~1 h)
    python -m ml.benchmark_seeds --only design --seeds 1 2

Experiments: learning (worlds v2 and v3), design, newmodule, pretrain (C, D), sets (E).
Output: seeds_<experiment>.json with every run, and seeds_summary.json / the printed table with mean, sd.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import data, deep, models, setmodels, structured as st, synthetic
from .benchmark import ALL_CURVES, THETA_CURVES, inputs_for
from .benchmark_advanced import V4, _scores
from .benchmark_structured import FEATURE_GROUPS, WORLDS, run_first_round, run_structured

TARGETS = ["sigma_e", "lambda", "gain"]


def _cnn(train, test, logf, groups=("theta", "amp"), names=THETA_CURVES, make=None):
    (df_tr, cv_tr), (df_te, cv_te) = train, test
    X_tr, _ = inputs_for(df_tr, cv_tr, groups, True, names, log_frequency_spectra=logf)
    X_te, _ = inputs_for(df_te, cv_te, groups, True, names, log_frequency_spectra=logf)
    t0 = time.time()
    m, s = (make or models.CurveCNN)().fit(X_tr, data.make_targets(df_tr)[0]).predict(X_te)
    return _scores(df_te, m, s, time.time() - t0)


def _record(results, key, label, r, tag):
    results.setdefault(key, {}).setdefault(label, []).append(r)
    print(f"[{tag}] {key} | {label:44s} " + "  ".join(f"{t} {r['truth'][t]['rel_error'] * 100:5.1f}%" for t in TARGETS)
          + f"  cov(sigma_e) {r['labels']['sigma_e']['coverage']:.2f}", flush=True)


def exp_learning(out, seeds, sizes, n_test):
    results = {}
    for world, kw in WORLDS.items():
        test = synthetic.make_dataset(n_test, seed=99, **kw)
        for seed in seeds:
            for n in sizes:
                train = synthetic.make_dataset(n, seed=seed, **kw)
                key = f"{world} n={n}"
                _record(results, key, "first round: GP, theta + amp + features",
                        run_first_round("gp", FEATURE_GROUPS, False, train, test), "learning")
                _record(results, key, "B: hierarchical GP", run_structured(lambda: st.DirectGP("hier"), train, test)[0], "learning")
                _record(results, key, "A+B", run_structured(lambda: st.PhysicsStructured("hier", use_components=False), train, test)[0],
                        "learning")
                if n >= 60:
                    _record(results, key, "CNN, curves + theta + amp", _cnn(train, test, False), "learning")
                    _record(results, key, "CNN, log-frequency spectra", _cnn(train, test, True), "learning")
            (out / "seeds_learning.json").write_text(json.dumps(results, indent=1))
    return results


def exp_design(out, seeds, n_test):
    test = synthetic.make_dataset(n_test, seed=99)
    results = {}
    for seed in seeds:
        rng = np.random.default_rng(seed + 100)
        oat, oat_half = synthetic.oat_thetas(20), synthetic.oat_thetas(10)
        n = len(oat)
        designs = {"one-at-a-time": oat, "random": synthetic.sample_theta(rng, n),
                   "half and half": np.vstack([oat_half, synthetic.sample_theta(rng, n - len(oat_half))])}
        for dname, thetas in designs.items():
            train = synthetic.make_dataset(len(thetas), seed=seed, thetas=np.random.default_rng(seed).permutation(thetas))
            _record(results, dname, "first round: GP, theta + amp + features", run_first_round("gp", FEATURE_GROUPS, False, train, test), "design")
            _record(results, dname, "A+B", run_structured(lambda: st.PhysicsStructured("hier", use_components=False), train, test)[0], "design")
        (out / "seeds_design.json").write_text(json.dumps(results, indent=1))
    return results


def exp_newmodule(out, seeds, n_train, ks):
    results = {}
    for seed in seeds:
        df, curves = synthetic.make_dataset(n_train, seed=seed, n_boards=2)
        packet = df["packet_id"].str[4:].astype(int).to_numpy()
        new = (df["board"] == 101).to_numpy()
        sub = lambda m: (df[m].reset_index(drop=True), {k: v[m] for k, v in curves.items()})
        test_rows = new & (packet >= max(ks))
        for k in ks:
            train_rows = ~new | (packet < k)
            full = (df, curves, np.flatnonzero(train_rows), np.flatnonzero(test_rows))
            key = f"k={k}"
            _record(results, key, "first round: GP, theta + amp + features",
                    run_first_round("gp", FEATURE_GROUPS, False, sub(train_rows), sub(test_rows), full=full), "newmodule")
            _record(results, key, "B: hierarchical GP", run_structured(lambda: st.DirectGP("hier"), sub(train_rows), sub(test_rows), full=full)[0],
                    "newmodule")
            _record(results, key, "A+B", run_structured(lambda: st.PhysicsStructured("hier", use_components=False), sub(train_rows),
                                                        sub(test_rows), full=full)[0], "newmodule")
        (out / "seeds_newmodule.json").write_text(json.dumps(results, indent=1))
    return results


def exp_pretrain(out, seeds, sizes, n_unlabeled, n_test, epochs):
    te, cte = synthetic.make_dataset(n_test, seed=99)
    results = {}
    for seed in seeds:
        un, cu = synthetic.make_dataset(n_unlabeled, seed=1000 + seed)
        Xu, _ = inputs_for(un, cu, ("theta",), True, ALL_CURVES, log_frequency_spectra=True)
        pre = deep.Pretrainer(epochs=epochs, seed=seed).fit(Xu["curves"], Xu["scalars"])
        for n in sizes:
            tr, ctr = synthetic.make_dataset(n, seed=seed)
            key = f"n={n}"
            _record(results, key, "CNN from scratch", _cnn((tr, ctr), (te, cte), False, names=ALL_CURVES), "pretrain")
            _record(results, key, "CNN from scratch, log-frequency", _cnn((tr, ctr), (te, cte), True, names=ALL_CURVES), "pretrain")
            _record(results, key, "D: pre-trained, fine-tuned",
                    _cnn((tr, ctr), (te, cte), True, names=ALL_CURVES, make=lambda: deep.PretrainedCNN(pre)), "pretrain")
            prep = st.StructuredPrep().fit(tr)
            Xc = [inputs_for(d, c, ("theta",), True, ALL_CURVES, log_frequency_spectra=True)[0]["curves"] for d, c in ((tr, ctr), (te, cte))]
            P = [prep.transform(tr, ctr), prep.transform(te, cte)]
            t0 = time.time()
            m, s = deep.DeepKernelGP(pre, finetune=False).fit({"curves": Xc[0], "x": P[0]["x"], "amp": P[0]["amp"]},
                                                              data.make_targets(tr)[0]).predict({"curves": Xc[1], "x": P[1]["x"], "amp": P[1]["amp"]})
            _record(results, key, "C: deep kernel, encoder frozen", _scores(te, m, s, time.time() - t0), "pretrain")
        (out / "seeds_pretrain.json").write_text(json.dumps(results, indent=1))
    return results


def exp_sets(out, seeds, sizes, n_test):
    te, cte = synthetic.make_dataset(n_test, seed=99, **V4)
    results = {}
    for seed in seeds:
        for n in sizes:
            tr, ctr = synthetic.make_dataset(n, seed=seed, **V4)
            prep = st.StructuredPrep().fit(tr)
            Xtr = dict(prep.transform(tr, ctr), sets=ctr["sets: noise: Allan deviation"])
            Xte = dict(prep.transform(te, cte), sets=cte["sets: noise: Allan deviation"])
            for comb in ("median", "robust mean", "robust mean of variances", "learned"):
                t0 = time.time()
                m, s = setmodels.SetStructured(comb).fit(Xtr, data.make_targets(tr)[0]).predict(Xte)
                _record(results, f"n={n}", comb, _scores(te, m, s, time.time() - t0), "sets")
        (out / "seeds_sets.json").write_text(json.dumps(results, indent=1))
    return results


def summarize(results: dict) -> dict:
    """mean and sd over the training scans of the error against the truth (%), per target, and of the sigma_e coverage."""
    out = {}
    for key, by_label in results.items():
        for label, runs in by_label.items():
            entry = {t: [100 * np.mean([r["truth"][t]["rel_error"] for r in runs]), 100 * np.std([r["truth"][t]["rel_error"] for r in runs])]
                     for t in TARGETS}
            entry["coverage sigma_e"] = [float(np.mean([r["labels"]["sigma_e"]["coverage"] for r in runs])),
                                         float(np.std([r["labels"]["sigma_e"]["coverage"] for r in runs]))]
            entry["n_scans"] = len(runs)
            out.setdefault(key, {})[label] = {k: ([float(x) for x in v] if isinstance(v, list) else v) for k, v in entry.items()}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="../docs/ml_results")
    parser.add_argument("--only", nargs="*", default=["design", "newmodule", "sets", "learning", "pretrain"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    q, seeds = args.quick, args.seeds
    summary_path = out / "seeds_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    runs = {"design": lambda: exp_design(out, seeds, 50 if q else 200),
            "newmodule": lambda: exp_newmodule(out, seeds, 60 if q else 240, [0, 2] if q else [0, 1, 2, 5]),
            "sets": lambda: exp_sets(out, seeds, [30] if q else [30, 60, 120], 50 if q else 200),
            "learning": lambda: exp_learning(out, seeds, [15, 60] if q else [15, 60, 240], 50 if q else 200),
            "pretrain": lambda: exp_pretrain(out, seeds, [60] if q else [60, 240], 300 if q else 3000, 50 if q else 200, 3 if q else 40)}
    for name in args.only:
        summary[name] = summarize(runs[name]())
        summary_path.write_text(json.dumps(summary, indent=1))
        print(f"\n=== {name}: error against the truth, mean +- sd over {len(seeds)} training scans ===")
        for key, by_label in summary[name].items():
            for label, e in by_label.items():
                print(f"{key:16s} {label:44s} " + "  ".join(f"{t} {e[t][0]:5.1f} +- {e[t][1]:4.1f}%" for t in TARGETS)
                      + f"  cov {e['coverage sigma_e'][0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
