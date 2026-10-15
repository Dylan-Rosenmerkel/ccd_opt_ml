"""
train_lvl1.py -- LEVEL 1: can the image's noise parameters be predicted from the traces?  One command.

Every model is trained and tested on the processed packets, WITHOUT amplifier labels: a model sees an amplifier's
own traces (its trace and reference features or curves) and the settings, never which amplifier it is.  One
training example = one amplifier's traces in one packet, with that amplifier's labels from the image (a packet
read by 4 amplifiers gives 4 examples).

Testing is by packet-grouped cross-validation: the packets are split into K folds; each fold's packets are
predicted by models trained on the other folds only.  The 4 examples of a packet always stay in the same fold
(they share the CCD's state at that moment; splitting them would make the scores look better than they are).

Reported per model and per target (read noise sigma_e, dark charge lambda, gain):

* error     RMS of log(predicted / true), i.e. the typical relative error (0.05 = 5 %);
* bias      mean of log(predicted / true);
* shape     the error after removing each amplifier's own mean error (is the dependence right, even if one
            amplifier's level is off?);
* rank      Spearman correlation between predicted and true values within each amplifier (is the ORDER right?);
* coverage  fraction of truths within the predicted +-1 sigma (should be 0.68) and +-1.96 sigma (0.95);
* cal. cov. the same after conformal calibration, with each fold's factor taken from the OTHER folds (honest);
* factor    by how much the error bars must be scaled for 68 % coverage (1 = calibrated);
* label     the median error of the labels themselves (WADERs): no model can be judged below this.

The amplifier and board are used ONLY to group rows for these scores, never as model inputs.

    python lvl1/train_lvl1.py --inputs <processed>/model_inputs [--models physics gp cnn ...] [--board 101]
                              [--folds 5] [--out DIR] [--quick]

Models (family file in brackets; see lvl1/README.md):
    trace_physics   [physics]   no learning: noise from the trace's Allan curve, the training mean gain
    physics         [physics]   A: trace physics + learned gain and correction (the default model)
    physics_joint   [physics]   A with the joint dark-current / CIC fit of lambda (needs per-row lambda)
    sets            [sets]      E: A with the trace sets combined by a learned set network
    ridge           [features]  Bayesian ridge on settings + own trace + own reference features
    trees           [features]  gradient-boosted trees, same inputs
    gp              [features]  Gaussian process, same inputs
    nn              [features]  neural-network ensemble, same inputs
    cnn             [curves]    1D CNN on the trace and reference curves + settings
    pretrained      [curves + pretrain]  D: CNN whose encoder was pre-trained on every unlabeled trace, fine-tuned
    deep_kernel     [curves + pretrain]  C: GP on the pre-trained encoder's embedding (frozen), no amplifier term
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scipy.stats import spearmanr  # noqa: E402

from ml import models, real_inputs, structured as st  # noqa: E402
from ml.evaluate import grouped_folds  # noqa: E402

TARGETS = ("sigma_e", "lambda", "gain")
ALL_MODELS = ("trace_physics", "physics", "physics_joint", "sets", "ridge", "trees", "gp", "nn", "cnn", "pretrained", "deep_kernel")
FAMILY = {"trace_physics": "physics", "physics": "physics", "physics_joint": "physics", "sets": "sets",
          "ridge": "features", "trees": "features", "gp": "features", "nn": "features",
          "cnn": "curves", "pretrained": "curves", "deep_kernel": "curves"}
MAX_NAN_FRACTION = 0.2


# --------------------------------------------------------------------------- #
# inputs: everything fitted here uses the TRAINING rows of the fold only        #
# --------------------------------------------------------------------------- #


def prep_columns(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep columns with at most 20 % missing and some variation in the training rows; fill gaps with the training median."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nan_frac = np.isnan(train).mean(0)
        sd = np.nanstd(train, 0)
        keep = (nan_frac <= MAX_NAN_FRACTION) & np.isfinite(sd) & (sd > 0)
        med = np.nanmedian(train[:, keep], 0)
    fill = lambda a: np.where(np.isnan(a[:, keep]), med, a[:, keep])
    return fill(train), fill(test), keep


def packet_state(packet_id: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    """Per row: the mean of the reference features over its packet's rows (the CCD's state; no amplifier identity)."""
    out = np.full_like(X_ref, np.nan)
    for p in np.unique(packet_id):
        rows = packet_id == p
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[rows] = np.nanmean(X_ref[rows], 0)
    return out


def scalars(d: dict, with_features: bool, with_state: bool) -> np.ndarray:
    blocks = [d["theta"]]
    if with_features:
        blocks += [d["X_trace"], d["X_ref"]]
    if with_state:
        blocks.append(packet_state(d["packet_id"], d["X_ref"]))
    return np.hstack(blocks)


def fill_sets(sets: np.ndarray, n_sets: np.ndarray, rng) -> np.ndarray:
    """Rows with fewer trace sets than the maximum: the missing slots are filled with copies of existing sets."""
    out = sets.copy()
    for i in range(len(out)):
        ok = np.flatnonzero(np.isfinite(out[i]).all(1))
        missing = np.flatnonzero(~np.isfinite(out[i]).all(1))
        if len(ok) and len(missing):
            out[i, missing] = out[i, rng.choice(ok, len(missing))]
    return out


# --------------------------------------------------------------------------- #
# models: fit on the training rows, return (mean, std) for the test rows        #
# --------------------------------------------------------------------------- #


class Runner:
    """Holds the loaded files and the shared pre-trained encoder; ``run(name, train, test)``."""

    def __init__(self, files: dict, rows: np.ndarray, quick: bool):
        self.files, self.rows, self.quick = files, rows, quick
        self.pretrainer = None
        d = files[next(iter(files))]
        self.Y, self.Y_err = d["Y"][rows], d["Y_err"][rows]

    #: arrays that are not one-per-example (names, block lengths)
    NOT_PER_ROW = ("allan_ns", "theta_names", "feature_names", "curve_names", "targets")

    def f(self, family):
        """The family's arrays restricted to the selected examples."""
        d = self.files[family]
        n = len(d["packet_id"])
        return {k: (v[self.rows] if k not in self.NOT_PER_ROW and isinstance(v, np.ndarray) and v.ndim and len(v) == n else v)
                for k, v in d.items()}

    # -- the physics family -------------------------------------------------------
    def _physics_X(self, d, train, test):
        xs = scalars(d, True, True)
        xtr, xte, _ = prep_columns(xs[train], xs[test])
        a, b = st.allan_fit(np.nan_to_num(d["allan"], nan=np.nanmedian(d["allan"])), d["allan_ns"])
        def part(idx, x):
            X = {"x": x, "a": a[idx], "b": b[idx], "n_skips": d["n_skips"][idx], "y_var": self.Y_err[idx] ** 2}
            if "lambda_parts" in d:
                X["lambda_parts"], X["lambda_cov"] = d["lambda_parts"][idx], d["lambda_cov"][idx]
            return X
        return part(train, xtr), part(test, xte)

    def trace_physics(self, train, test):
        d = self.f("physics")
        Xtr, Xte = self._physics_X(d, train, test)
        A = st.PhysicsStructured()
        g, lam = self.Y[train, 2].mean(), self.Y[train, 1].mean()
        m = np.column_stack([A.physics_log_sigma_adu(Xte) - g, np.full(len(test), lam), np.full(len(test), g)])
        s = np.column_stack([np.full(len(test), self.Y[train, 2].std()), np.full(len(test), self.Y[train, 1].std()),
                             np.full(len(test), self.Y[train, 2].std())])
        return m, s

    def physics(self, train, test, joint=False):
        d = self.f("physics")
        Xtr, Xte = self._physics_X(d, train, test)
        if joint:
            # (a part fitted at or below zero is fine: the joint fit clips it at 5 % of lambda, in log space)
            if "lambda_parts" not in Xtr or not (np.isfinite(Xtr["lambda_parts"]).all() and (Xtr["lambda_parts"].sum(1) > 0).all()):
                raise SkipModel("no per-row lambda (dark current / CIC) for every training row")
            model = st.PhysicsStructured(lambda_model="joint")
        else:
            model = st.PhysicsStructured()
        return model.fit(Xtr, self.Y[train]).predict(Xte)

    def physics_joint(self, train, test):
        return self.physics(train, test, joint=True)

    def sets(self, train, test):
        from ml import setmodels
        d, ds = self.f("physics"), self.f("sets")
        Xtr, Xte = self._physics_X(d, train, test)
        S = fill_sets(ds["sets"], ds["n_sets"], np.random.default_rng(0))
        if not np.isfinite(S).all():
            raise SkipModel("some rows have no usable trace set")
        Xtr["sets"], Xte["sets"] = S[train], S[test]
        model = setmodels.SetStructured(combiner="learned", kind="none", allan_ns=ds["allan_ns"])
        return model.fit(Xtr, self.Y[train]).predict(Xte)

    # -- the features family --------------------------------------------------------
    def _features(self, train, test):
        d = self.f("features")
        xs = scalars(d, True, False)
        xtr, xte, _ = prep_columns(xs[train], xs[test])
        return xtr, xte

    def ridge(self, train, test):
        xtr, xte = self._features(train, test)
        return models.BayesRidge().fit(xtr, self.Y[train]).predict(xte)

    def trees(self, train, test):
        xtr, xte = self._features(train, test)
        return models.GBT(max_iter=100 if self.quick else 300).fit(xtr, self.Y[train]).predict(xte)

    def gp(self, train, test):
        xtr, xte = self._features(train, test)
        return models.GP(max_iter=60 if self.quick else 150).fit(xtr, self.Y[train]).predict(xte)

    def nn(self, train, test):
        xtr, xte = self._features(train, test)
        return models.MLPEnsemble(n_models=2 if self.quick else 5, epochs=100 if self.quick else 600).fit(xtr, self.Y[train]).predict(xte)

    # -- the curves family -------------------------------------------------------------
    def _curves(self, train, test, both=True):
        d = self.f("curves")
        c = np.concatenate([d["curves_theta"], d["curves_ref"]], 1) if both else d["curves_theta"]
        c = np.nan_to_num(c)
        sc = np.hstack([d["theta"], self.f("features")["X_ref"]]) if "features" in self.files else d["theta"]
        str_, ste, _ = prep_columns(sc[train], sc[test])
        return {"curves": c[train], "scalars": str_}, {"curves": c[test], "scalars": ste}

    def cnn(self, train, test):
        Xtr, Xte = self._curves(train, test)
        return models.CurveCNN(n_models=2 if self.quick else 3, epochs=60 if self.quick else 400).fit(Xtr, self.Y[train]).predict(Xte)

    def _pretrained(self):
        if self.pretrainer is None:
            from ml import deep
            if "pretrain" not in self.files:
                raise SkipModel("no pretrain.npz")
            p = self.files["pretrain"]
            keep = np.isin(p["board"], np.unique(self.files["curves"]["board"][self.rows]))
            c = np.nan_to_num(p["curves"][keep])
            th = np.nan_to_num(p["theta"][keep]) if p["theta"].size else None
            # unlabeled traces only: no label of any packet is used here
            self.pretrainer = deep.Pretrainer(epochs=10 if self.quick else 40).fit(c, th)
        return self.pretrainer

    def pretrained(self, train, test):
        from ml import deep
        pre = self._pretrained()
        Xtr, Xte = self._curves(train, test, both=False)
        return deep.PretrainedCNN(pre, n_models=2 if self.quick else 3, epochs=60 if self.quick else 400).fit(Xtr, self.Y[train]).predict(Xte)

    def deep_kernel(self, train, test):
        from ml import deep
        pre = self._pretrained()
        Xtr, Xte = self._curves(train, test, both=False)
        model = deep.DeepKernelGP(pre, finetune=False, amp_term=False, steps=60 if self.quick else 250)
        return model.fit({"curves": Xtr["curves"], "x": Xtr["scalars"]}, self.Y[train]).predict({"curves": Xte["curves"], "x": Xte["scalars"]})


class SkipModel(Exception):
    pass


# --------------------------------------------------------------------------- #
# scores                                                                        #
# --------------------------------------------------------------------------- #


def conformal_q(z: np.ndarray, c: float) -> float:
    z = np.sort(z[np.isfinite(z)])
    if not len(z):
        return np.nan
    k = int(np.ceil((len(z) + 1) * c)) - 1
    return float(z[min(max(k, 0), len(z) - 1)])


def scores(Y, Y_err, mean, std, amp, folds) -> dict:
    out = {}
    for t, name in enumerate(TARGETS):
        ok = np.isfinite(Y[:, t]) & np.isfinite(mean[:, t])
        r = mean[:, t] - Y[:, t]
        shape = r.copy()
        for a in np.unique(amp):
            sel = (amp == a) & ok
            if sel.any():
                shape[sel] -= r[sel].mean()
        ranks = []
        for a in np.unique(amp):
            sel = (amp == a) & ok
            if sel.sum() >= 4 and np.std(Y[sel, t]) > 0 and np.std(mean[sel, t]) > 0:
                ranks.append(spearmanr(mean[sel, t], Y[sel, t])[0])
        z = np.abs(r) / np.maximum(std[:, t], 1e-12)
        cal68, cal95 = np.zeros(len(r), bool), np.zeros(len(r), bool)
        for test in folds:                                   # cross-conformal: each fold's factor from the others
            other = np.setdiff1d(np.flatnonzero(ok), test)
            cal68[test] = z[test] <= conformal_q(z[other], 0.68)
            cal95[test] = z[test] <= conformal_q(z[other], 0.95)
        out[name] = {"error": float(np.sqrt(np.mean(r[ok] ** 2))), "bias": float(np.mean(r[ok])),
                     "shape": float(np.sqrt(np.mean(shape[ok] ** 2))),
                     "rank": float(np.mean(ranks)) if ranks else None,
                     "coverage68": float(np.mean(z[ok] <= 1.0)), "coverage95": float(np.mean(z[ok] <= 1.96)),
                     "calibrated_coverage68": float(np.mean(cal68[ok])), "calibrated_coverage95": float(np.mean(cal95[ok])),
                     "factor68": conformal_q(z[ok], 0.68),
                     "label_error": float(np.nanmedian(Y_err[ok, t])) if np.isfinite(Y_err[ok, t]).any() else None,
                     "n": int(ok.sum())}
    return out


def fmt(v, pct=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{100 * v:.1f} %" if pct else f"{v:.2f}"


def report_md(results: dict, info: dict) -> str:
    lines = [f"# Level 1 results ({info['date']})", "",
             f"Inputs: `{info['inputs']}`; boards {info['boards']}; {info['n_rows']} examples from {info['n_packets']} packets; "
             f"{info['n_folds']}-fold cross-validation grouped by packet.  No amplifier labels are model inputs.", ""]
    for name in TARGETS:
        lines += [f"## {name}", "", "| model | error | bias | shape | rank | cover 68 / 95 | calibrated 68 / 95 | factor | label error |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for model, res in results.items():
            if "skipped" in res:
                continue
            s = res["scores"][name]
            lines.append(f"| {model} | {fmt(s['error'])} | {fmt(s['bias'])} | {fmt(s['shape'])} | {fmt(s['rank'], False)} | "
                         f"{fmt(s['coverage68'], False)} / {fmt(s['coverage95'], False)} | {fmt(s['calibrated_coverage68'], False)} / "
                         f"{fmt(s['calibrated_coverage95'], False)} | {fmt(s['factor68'], False)} | {fmt(s['label_error'])} |")
        lines.append("")
    skipped = {m: r["skipped"] for m, r in results.items() if "skipped" in r}
    if skipped:
        lines += ["## Skipped", ""] + [f"* {m}: {why}" for m, why in skipped.items()] + [""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main                                                                          #
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", required=True, help="the model_inputs directory written by process_packet.py")
    parser.add_argument("--models", nargs="*", default=list(ALL_MODELS), choices=ALL_MODELS)
    parser.add_argument("--board", type=int, nargs="*", default=[101], help="boards whose amplifiers are used (default 101)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--include-flagged", action="store_true", help="also use labels that WADERs flagged")
    parser.add_argument("--allow-fake", action="store_true", help="also use FAKE (rehearsal) packets")
    parser.add_argument("--quick", action="store_true", help="short training (to test the pipeline)")
    parser.add_argument("--out", help="results directory (default: <processed>/lvl1_results/<date>)")
    args = parser.parse_args()

    inputs = Path(args.inputs)
    files = {p.stem: real_inputs.load(p) for p in sorted(inputs.glob("*.npz"))}
    base = files.get("physics") or files.get("features")
    if base is None:
        raise SystemExit(f"no physics.npz or features.npz in {inputs}")
    rows = np.isin(base["board"], args.board) & np.isfinite(base["Y"]).all(1)
    if not args.include_flagged:
        rows &= base["label_ok"]
    if not args.allow_fake:
        rows &= ~base["fake"]
    rows = np.flatnonzero(rows)
    if len(rows) < 6:
        raise SystemExit(f"only {len(rows)} usable examples (boards {args.board}, labels ok, not fake): take more packets")
    packets = base["packet_id"][rows]
    amp = np.array([f"{b}_ch{c}" for b, c in zip(base["board"][rows], base["channel"][rows])])
    n_folds = min(args.folds, len(np.unique(packets)))
    folds = grouped_folds(packets, n_folds, seed=0)
    runner = Runner(files, rows, args.quick)
    out = Path(args.out) if args.out else inputs.parent / "lvl1_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    info = {"date": datetime.now().isoformat(timespec="seconds"), "inputs": str(inputs), "boards": args.board,
            "n_rows": int(len(rows)), "n_packets": int(len(np.unique(packets))), "n_folds": n_folds, "quick": args.quick,
            "models": args.models}
    print(f"{len(rows)} examples from {info['n_packets']} packets, boards {args.board}, {n_folds} folds (grouped by packet)")
    results, preds = {}, {}
    for name in args.models:
        if FAMILY[name] not in files:
            results[name] = {"skipped": f"{FAMILY[name]}.npz not found"}
            print(f"{name:14s} skipped: {results[name]['skipped']}")
            continue
        t0 = time.time()
        mean, std = np.full(runner.Y.shape, np.nan), np.full(runner.Y.shape, np.nan)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for test in folds:
                    train = np.setdiff1d(np.arange(len(rows)), test)
                    m, s = getattr(runner, name)(train, test)
                    mean[test], std[test] = m, s
        except SkipModel as exc:
            results[name] = {"skipped": str(exc)}
            print(f"{name:14s} skipped: {exc}")
            continue
        sc = scores(runner.Y, runner.Y_err, mean, std, amp, folds)
        results[name] = {"scores": sc, "seconds": time.time() - t0}
        preds[name] = (mean, std)
        print(f"{name:14s} " + "  ".join(f"{t}: {100 * sc[t]['error']:5.1f} % (rank {fmt(sc[t]['rank'], False)}, cover {sc[t]['coverage68']:.2f})"
                                          for t in TARGETS) + f"   {time.time() - t0:5.0f} s", flush=True)
    (out / "report.json").write_text(json.dumps({"info": info, "results": results}, indent=1))
    (out / "report.md").write_text(report_md(results, info))
    import pandas as pd
    table = {"packet_id": packets, "board": base["board"][rows], "channel": base["channel"][rows]}
    for t, tn in enumerate(TARGETS):
        table[f"true log {tn}"] = runner.Y[:, t]
        table[f"label error {tn}"] = runner.Y_err[:, t]
        for name, (m, s) in preds.items():
            table[f"{name}: log {tn}"], table[f"{name}: sd {tn}"] = m[:, t], s[:, t]
    pd.DataFrame(table).to_csv(out / "predictions.csv", index=False)
    print(f"\nwrote {out / 'report.md'}, report.json, predictions.csv")


if __name__ == "__main__":
    main()
