"""
explain_models.py -- what every level-1 model has learned: its fitted parameters and the inputs its predictions rely
on (for the first-data report, docs/overleaf_first_data_2026-11-03, "The models one by one").

Each model is fitted ONCE on the A+B packets, exactly as the session C test of ``train_lvl1.py --holdout-session C``
fits it, at the trace-feature set the A+B cross-validation chose for each target (the ladder run; physics_hier, which
is not in it: the holdout run's own cross-validation; the baselines: no trace features).  The fitted model predicts
every row (A+B and C), and then:

* native parameters: what the fit itself says -- GP inverse length scales on standardized inputs (ARD relevance),
  SAAS relevances, ridge coefficients, the trees' split gains, the deep-kernel GP's relevances, the physics model's
  three terms and the hierarchical GP's amplitudes;
* permutation sensitivity: how far the predictions move (RMS of the change of the log prediction, in %) when one
  input, or one group of inputs, is shuffled across all rows (A+B and C).  It measures what the model USES, right or
  wrong;
* permutation loss on C: how much the session C error grows (percentage points) when the group is shuffled among the
  C rows only.  It measures what HELPS on new packets.

No refit is needed: the fitted model is captured at its final ``predict`` call and re-run on shuffled inputs.

    .venv/bin/python lvl1/explain_models.py [--out processed/lvl1_results/2026-11-03_explain] [--models gp trees ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lvl1"))
sys.path.insert(0, str(ROOT / "src"))

import train_lvl1 as tl  # noqa: E402
from ml import bogp, deep, real_inputs, setmodels  # noqa: E402
from ml import models as mm  # noqa: E402
from ml import structured as st  # noqa: E402

TIER_FILE = ROOT / "lvl1" / "feature_study" / "real_AB" / "evidence_tiers.json"
LADDER = ROOT / "processed_AB" / "lvl1_results" / "2026-11-03_ladder_raw" / "report.json"
HOLDOUT = ROOT / "processed" / "lvl1_results" / "2026-11-03_holdoutC"
PLAN = ROOT / "lvl1" / "packet_plan" / "plan.json"
TIER_SETS = ("none", "P", "P+A1", "P+A", "P+A+B", "all")
BASELINES = ("mean", "settings", "settings_amp", "settings_amp_trees")
TRACE_MODELS = ("ridge", "trees", "gp", "gp_dim", "nn", "cnn", "pretrained", "deep_kernel",
                "physics", "physics_white", "physics_joint", "sets", "physics_hier")
TARGETS = tl.TARGETS

# --------------------------------------------------------------------------- #
# capture the fitted model at its final predict call                           #
# --------------------------------------------------------------------------- #

_CAP = {"depth": 0, "last": None}


def _capture(cls):
    orig = cls.__dict__["predict"]

    def predict(self, X, *a, **k):
        _CAP["depth"] += 1
        try:
            out = orig(self, X, *a, **k)
        finally:
            _CAP["depth"] -= 1
        if _CAP["depth"] == 0:                  # an outermost call (a model's own GPs inside it are not recorded)
            _CAP["last"] = (self, X)
        return out
    cls.predict = predict


for _cls in (mm.BayesRidge, mm.GBT, mm.GP, mm.MLPEnsemble, mm.CurveCNN, bogp.BoTorchGP, st.HierGP,
             st.PhysicsStructured, setmodels.SetStructured, deep.DeepKernelGP):
    if "predict" in _cls.__dict__:
        _capture(_cls)


# --------------------------------------------------------------------------- #
# inputs: names and groups                                                     #
# --------------------------------------------------------------------------- #

def setting_group(name: str) -> str:
    """The knob family of a setting (theta) name."""
    n = name.replace("theta.", "")
    if n == "delayInt":
        return "settings: integration time"
    if re.match(r"delay_H", n):
        return "settings: H-clock timing"
    if re.match(r"delay_V", n) or n == "NVWAIT":
        return "settings: V-clock timing"
    if n.startswith("delay"):
        return "settings: RG/SW/OG/DG timing"
    if n.startswith("RG_"):
        return "settings: RG voltages"
    if re.match(r"(SW|OG|DG)_", n):
        return "settings: SW/OG/DG voltages"
    if re.match(r"(V\d|TG)_", n):
        return "settings: V/TG clock voltages"
    if re.match(r"H\d_", n):
        return "settings: H clock voltages"
    if n == "VSUB":
        return "settings: VSUB"
    if re.match(r"(VDD|VDRAIN|VR)", n):
        return "settings: VDD/VDRAIN/VR"
    return "settings: offsets, NPSD"


def slot(name: str, addr: tuple) -> dict:
    """One input of a model: its display name, its group, its coarse group and where it sits in X."""
    if name.startswith("theta."):
        return {"name": "setting " + name[6:], "group": setting_group(name), "coarse": "settings", "addr": addr}
    if name.startswith("amp "):
        return {"name": name, "group": "amplifier identity", "coarse": "amplifier identity", "addr": addr}
    for prefix, coarse in (("own: ", "own trace"), ("ref: ", "reference trace"), ("state: ", "packet state")):
        if name.startswith(prefix):
            fam = name[len(prefix):].split(":")[0]
            group = "packet state" if prefix == "state: " else f"{coarse}: {fam}"
            return {"name": name, "group": group, "coarse": coarse, "addr": addr}
    raise ValueError(name)


def scalar_block(runner, model: str, train, test) -> tuple[list[str], str | None]:
    """The names of the scalar inputs the model sees, in the order of its prepared input matrix, and the key of that
    matrix in X (None: X is the matrix itself)."""
    fam = tl.FAMILY[model]
    th_names = [str(n) for n in runner.f("features")["theta_names"]]
    if model == "settings":
        names, xs, key = th_names, runner.f("features")["theta"], None
    elif model.startswith("settings_amp"):
        n_amp = runner.amp_id.max() + 1
        keys = np.unique(np.array([f"{b}_ch{c}" for b, c in zip(runner.files["features"]["board"][runner.rows],
                                                                 runner.files["features"]["channel"][runner.rows])]))
        names = th_names + [f"amp {k.split('_')[1]}" for k in keys[:n_amp]]
        xs, key = np.hstack([runner.f("features")["theta"], np.eye(n_amp)[runner.amp_id]]), None
    elif fam == "features":
        d = runner.f("features")
        fn = [str(n) for n in d["feature_names"]]
        names, xs, key = th_names + ["own: " + n for n in fn] + ["ref: " + n for n in fn], tl.scalars(d, True, False), None
    elif fam in ("physics", "sets"):
        d = runner.f("physics")
        fn = [str(n) for n in d["feature_names"]]
        names = th_names + ["own: " + n for n in fn] + ["ref: " + n for n in fn] + ["state: " + n for n in fn]
        xs, key = tl.scalars(d, True, True), "x"
    elif fam == "curves":
        d, df = runner.f("curves"), runner.f("features")
        fn = [str(n) for n in df["feature_names"]]
        names, xs = th_names + ["ref: " + n for n in fn], np.hstack([d["theta"], df["X_ref"]])
        key = "x" if model == "deep_kernel" else "scalars"
    else:
        raise ValueError(model)
    _, _, keep = tl.prep_columns(xs[train], xs[test])
    return [n for n, k in zip(names, keep) if k], key


def layout(runner, model: str, X, train, test) -> list[dict]:
    """Every input slot of the model's captured X."""
    if model == "mean":
        return []
    names, key = scalar_block(runner, model, train, test)
    mat = X if key is None else X[key]
    if mat.shape[1] != len(names):
        raise RuntimeError(f"{model}: {mat.shape[1]} input columns but {len(names)} names")
    slots = [slot(n, (key, j)) for j, n in enumerate(names)]
    fam = tl.FAMILY[model]
    if fam == "curves":
        cn = [str(n) for n in runner.f("curves")["curve_names"]]
        own = [("own curve: " + n, "own curves") for n in cn]
        ref = [("ref curve: " + n, "reference curves") for n in cn] if model == "cnn" else []
        for c, (n, coarse) in enumerate(own + ref):
            slots.append({"name": n, "group": coarse, "coarse": coarse, "addr": ("curves", c)})
    if fam in ("physics", "sets"):
        slots.append({"name": "trace noise term (white noise / Allan fit)", "group": "trace noise term",
                      "coarse": "trace noise term", "addr": ("phys",)})
    if fam == "sets":
        slots.append({"name": "trace sets (every set's Allan curve)", "group": "trace sets", "coarse": "trace sets",
                      "addr": ("sets",)})
    if model == "physics_hier":
        slots.append({"name": "amplifier and module identity", "group": "amplifier identity",
                      "coarse": "amplifier identity", "addr": ("ids",)})
    return slots


def permuted(X, addrs: list[tuple], rows: np.ndarray, perm: np.ndarray):
    """A copy of X in which, on ``rows``, the inputs at ``addrs`` are replaced by those of ``rows[perm]``."""
    src = rows[perm]
    if not isinstance(X, dict):
        out = X.copy()
        for _, j in addrs:
            out[rows, j] = X[src, j]
        return out
    out = dict(X)
    touched = set()

    def arr(key):
        if key not in touched:
            out[key] = np.array(X[key], copy=True)
            touched.add(key)
        return out[key]
    for a in addrs:
        if a[0] == "phys":
            for k in ("white", "a", "b"):
                if k in X and X[k] is not None:
                    arr(k)[rows] = np.asarray(X[k])[src]
        elif a[0] == "ids":
            for k in ("amp", "module"):
                arr(k)[rows] = np.asarray(X[k])[src]
        elif a[0] == "sets":
            arr("sets")[rows] = X["sets"][src]
        elif a[0] == "curves":
            arr("curves")[rows, a[1]] = X["curves"][src, a[1]]
        else:
            arr(a[0])[rows, a[1]] = X[a[0]][src, a[1]]
    return out


# --------------------------------------------------------------------------- #
# native parameters                                                            #
# --------------------------------------------------------------------------- #

def _ranked(names, values) -> list[list]:
    v = np.asarray(values, float).reshape(-1)
    order = np.argsort(-np.abs(v))
    return [[names[i], float(v[i])] for i in order]


def _hier_params(gp, t: int = 0) -> dict:
    q = gp.fits[t]["p"]
    return {"relevance": np.exp(-q["ls"].numpy()), "amplitude": float(q["s"].exp()), "noise": float(q["noise"].exp()),
            **({k: float(v.exp()) for k, v in q.items() if k.startswith(("c_", "s_"))})}


def _gp_part(gp, xnames, t: int = 0) -> dict:
    """The fitted parameters of one GP inside a model, by kind."""
    if isinstance(gp, st.HierGP):
        p = _hier_params(gp, t)
        rel = p.pop("relevance")
        return {"kind": "ARD Matern-5/2 GP (MAP, weak log-normal prior on log length scales)", "relevance": _ranked(xnames, rel),
                **p}
    if isinstance(gp, bogp.BoTorchGP):
        return {"kind": f"BoTorch {gp.kind} GP", "relevance": _ranked(xnames, gp.relevance(t))}
    if isinstance(gp, st.JointLambdaGP):
        out = {"kind": "joint dark-current + CIC GP"}
        for name, q in zip(("dark current", "CIC"), gp.q):
            out[f"relevance ({name})"] = _ranked(xnames, np.exp(-q["ls"].numpy()))
        return out
    return {"kind": type(gp).__name__}


def native(model: str, obj, X, xnames: list[str], Y_train: np.ndarray) -> dict:
    """What the fit itself says, per target."""
    out: dict = {}
    if model == "mean":
        return {t: {"value": float(np.exp(Y_train[:, i].mean())), "sd_log": float(Y_train[:, i].std(ddof=1))}
                for i, t in enumerate(TARGETS)}
    if isinstance(obj, mm.GP):
        for i, t in enumerate(TARGETS):
            k = obj.models[i].kernel_
            out[t] = {"kind": "sklearn GP, Matern-5/2 ARD, maximum likelihood", "relevance": _ranked(xnames, 1 / obj.length_scales(i)),
                      "amplitude": float(k.k1.k1.constant_value), "noise": float(k.k2.noise_level)}
    elif isinstance(obj, mm.BayesRidge):
        for i, t in enumerate(TARGETS):
            m = obj.models[i]
            out[t] = {"kind": "Bayesian ridge", "coefficient": _ranked(xnames, m.coef_), "intercept": float(m.intercept_),
                      "noise_precision": float(m.alpha_), "weight_precision": float(m.lambda_)}
    elif isinstance(obj, mm.GBT):
        for i, t in enumerate(TARGETS):
            gain = np.zeros(len(xnames))
            for it in obj.models[i][0]._predictors:
                nodes = it[0].nodes
                inner = ~nodes["is_leaf"].astype(bool)
                np.add.at(gain, nodes["feature_idx"][inner], nodes["gain"][inner])
            out[t] = {"kind": "boosted trees", "split_gain_share": _ranked(xnames, gain / max(gain.sum(), 1e-30)),
                      "n_inputs_used": int((gain > 0).sum())}
    elif isinstance(obj, bogp.BoTorchGP):
        for i, t in enumerate(TARGETS):
            m = obj.models[i]
            out[t] = {"kind": f"BoTorch {obj.kind} GP", "relevance": _ranked(xnames, obj.relevance(i)),
                      "noise": float(m.likelihood.noise.mean())}
    elif isinstance(obj, deep.DeepKernelGP):
        names = [f"embedding {k + 1}" for k in range(obj.emb_dim)] + xnames
        for i, t in enumerate(TARGETS):
            q = obj.q[i]
            out[t] = {"kind": "GP on [embedding, scalars]", "relevance": _ranked(names, np.exp(-q["ls"].detach().numpy())),
                      "amplitude": float(q["s"].exp()), "noise": float(q["noise"].exp())}
    elif isinstance(obj, st.PhysicsStructured):
        T = obj.physics_log_sigma_adu(X)
        G = obj.gain.predict(X)[0][:, 0]
        R = obj.resid.predict(X)[0][:, 0]
        out["terms"] = {"trace": T.tolist(), "minus_log_gain": (-G).tolist(), "correction": R.tolist()}
        out["sigma_e"] = {"gain GP": _gp_part(obj.gain, xnames), "correction GP": _gp_part(obj.resid, xnames),
                          "n_outliers": int(np.sum(obj.outliers))}
        out["gain"] = {"gain GP": _gp_part(obj.gain, xnames)}
        lam = obj.lam if hasattr(obj, "lam") else None
        out["lambda"] = {"lambda GP": _gp_part(lam, xnames) if lam is not None else {"kind": "split"}}
    return out


# --------------------------------------------------------------------------- #
# the run                                                                      #
# --------------------------------------------------------------------------- #

def chosen_sets() -> dict[str, dict[str, str]]:
    """Per model and target, the feature set with the lowest A+B cross-validation error."""
    ladder = json.loads(LADDER.read_text())["results"]
    out = {m: {t: "none" for t in TARGETS} for m in BASELINES}
    for m in TRACE_MODELS:
        src = ladder
        if not any(k.startswith(m + " [") for k in ladder):                  # physics_hier: the holdout run's own CV
            src = {f"{m} [{s}]": json.loads((HOLDOUT / s / "report.json").read_text())["results"].get(m, {})
                   for s in TIER_SETS if (HOLDOUT / s / "report.json").exists()}
        out[m] = {}
        for t in TARGETS:
            cv = {s: src[f"{m} [{s}]"]["scores"][t]["error"] for s in TIER_SETS
                  if "scores" in src.get(f"{m} [{s}]", {})}
            out[m][t] = min(cv, key=cv.get)
    return out


def rms(a, axis=0):
    return np.sqrt(np.nanmean(np.asarray(a) ** 2, axis=axis))


def explain_one(runner, model: str, spec: str, train, test_all, ho, rng, repeats: int, repeats_single: int) -> dict:
    runner.feature_spec = spec
    _CAP["last"] = None
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, _ = getattr(runner, model)(train, test_all)
    fit_s = time.time() - t0
    Y = runner.Y
    err_c = rms(mean[ho] - Y[ho]) * 100
    res = {"model": model, "set": spec, "fit_seconds": fit_s, "error_C": dict(zip(TARGETS, err_c.tolist())),
           "error_AB_in_sample": dict(zip(TARGETS, (rms(mean[train] - Y[train]) * 100).tolist()))}
    if model == "mean":
        res["native"] = native(model, None, None, [], Y[train])
        return res
    obj, X = _CAP["last"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = obj.predict(X)[0]
        if not np.allclose(base, mean, atol=1e-6, equal_nan=True):
            print(f"  note: {model} re-prediction differs from the run by {np.nanmax(np.abs(base - mean)):.2g}")
        slots = layout(runner, model, X, train, test_all)
        xnames = [s["name"] for s in slots if s["addr"][0] in (None, "x", "scalars")]
        res["native"] = native(model, obj, X, xnames, Y[train])
        res["n_inputs"] = {"scalars": len(xnames), "slots": len(slots)}
        all_rows = np.arange(len(test_all))

        def sens(addrs, reps):
            s = []
            for _ in range(reps):
                p = obj.predict(permuted(X, addrs, all_rows, rng.permutation(len(all_rows))))[0]
                s.append(rms(p - base) * 100)
            return np.mean(s, 0)

        def loss_c(addrs, reps):
            d = []
            for _ in range(reps):
                p = obj.predict(permuted(X, addrs, ho, rng.permutation(len(ho))))[0]
                d.append(rms(p[ho] - Y[ho]) * 100 - err_c)
            return np.mean(d, 0)

        def within_amp(reps, addrs, rows):
            """Shuffles that keep every row's values within its own amplifier: what the inputs say beyond WHICH
            amplifier a row is (the CCD's state, the setting's effect), not the amplifier's fingerprint."""
            amp = runner.amp_id[rows]
            s, d = [], []
            for _ in range(reps):
                perm = np.arange(len(rows))
                for a in np.unique(amp):
                    idx = np.flatnonzero(amp == a)
                    perm[idx] = rng.permutation(idx)
                p = obj.predict(permuted(X, addrs, rows, perm))[0]
                s.append(rms(p - base) * 100)
                d.append(rms(p[ho] - Y[ho]) * 100 - err_c)
            return np.mean(s, 0), np.mean(d, 0)

        groups: dict[str, list] = {}
        coarse: dict[str, list] = {}
        for s in slots:
            groups.setdefault(s["group"], []).append(s["addr"])
            coarse.setdefault(s["coarse"], []).append(s["addr"])
        res["groups"] = {g: {"n": len(a), "sensitivity": dict(zip(TARGETS, sens(a, repeats).tolist())),
                             "loss_C": dict(zip(TARGETS, loss_c(a, repeats).tolist()))} for g, a in groups.items()}
        res["coarse"] = {}
        for g, a in coarse.items():
            ws, _ = within_amp(repeats, a, all_rows)
            _, wl = within_amp(repeats, a, ho)
            res["coarse"][g] = {"n": len(a), "sensitivity": dict(zip(TARGETS, sens(a, repeats).tolist())),
                                "loss_C": dict(zip(TARGETS, loss_c(a, repeats).tolist())),
                                "sensitivity_within_amp": dict(zip(TARGETS, ws.tolist())),
                                "loss_C_within_amp": dict(zip(TARGETS, wl.tolist()))}
        single = [(s["name"], s["group"], sens([s["addr"]], repeats_single)) for s in slots]
        res["single"] = {t: sorted(([n, g, float(v[i])] for n, g, v in single), key=lambda r: -r[2])[:25]
                         for i, t in enumerate(TARGETS)}
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", default=str(ROOT / "processed" / "model_inputs"))
    parser.add_argument("--out", default=str(ROOT / "processed" / "lvl1_results" / "2026-11-03_explain"))
    parser.add_argument("--models", nargs="*", default=list(BASELINES + TRACE_MODELS))
    parser.add_argument("--repeats", type=int, default=20, help="shuffles per input group (default 20)")
    parser.add_argument("--repeats-single", type=int, default=3, help="shuffles per single input (default 3)")
    args = parser.parse_args()

    files = {p.stem: real_inputs.load(p) for p in sorted(Path(args.inputs).glob("*.npz"))}
    base = files["physics"]
    rows = (np.isin(base["board"], [101]) & np.isfinite(base["Y"]).all(1) & base["label_ok"] & ~base["fake"]
            & base["trace_ok"].astype(bool))
    rows = np.flatnonzero(rows)
    packets = base["packet_id"][rows].astype(str)
    held = np.array([tl.matches_packet(p, tl.session_packets(PLAN, "C")) for p in packets])
    train, ho = np.flatnonzero(~held), np.flatnonzero(held)
    test_all = np.arange(len(rows))
    runner = tl.Runner(files, rows, quick=False, tiers=tl.load_tiers(TIER_FILE))
    print(f"{len(train)} A+B rows (fit), {len(ho)} session C rows")

    sets = chosen_sets()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "explain.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results["_chosen_sets"] = sets
    results["_rows"] = {"packet_id": packets.tolist(), "held_out": held.tolist(),
                        "amplifier": [f"ch{c}" for c in base["channel"][rows]], "Y": runner.Y.tolist()}
    rng = np.random.default_rng(0)
    for model in args.models:
        for spec in dict.fromkeys(sets[model][t] for t in TARGETS):
            key = f"{model} [{spec}]"
            print(f"{key} ...", flush=True)
            t0 = time.time()
            try:
                results[key] = explain_one(runner, model, spec, train, test_all, ho, rng, args.repeats, args.repeats_single)
            except tl.SkipModel as exc:
                results[key] = {"skipped": str(exc)}
            print(f"  {time.time() - t0:.0f} s; C error " + ", ".join(f"{t} {v:.1f}" for t, v in results[key].get("error_C", {}).items()),
                  flush=True)
            path.write_text(json.dumps(results, indent=1))
    print(f"written {path}")


if __name__ == "__main__":
    main()
