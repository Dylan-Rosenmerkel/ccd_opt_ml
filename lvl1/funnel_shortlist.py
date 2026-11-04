"""
funnel_shortlist.py -- session F: pick the settings worth an image from the trace-only screens.

1. Train on the labelled packets (a dataset.csv; board 101 without ch2 by default):
   * "change": boosted trees on the change of the trace features (log for positive ones) between two traces of one
     amplifier, trained on within-packet pairs (reference -> own trace; the reference's label is the nominal label
     interpolated in time, or, for a reference at another packet's settings, that packet's label) and on neighbouring
     packets (n -> n+1, n+2), in both orders;
   * "absolute": boosted trees on one trace's features (never the settings), applied to a screen's new trace AND its
     nominal reference trace, so its change is f(new) - f(reference).
   The prediction is the mean of the chosen models' changes (--models).
2. Read the screens (processed packet folders: features.json, the combined trace features at the new settings and at
   the reference), predict each screen's dlog sigma_e and dlog lambda against nominal per amplifier, and average over
   the board's amplifiers: a setting applies to all of them.
3. Order the screens by the joint objective J(w) = w dlog sigma_e + (1-w) dlog lambda (--weight w; lvl1/shortlist.py),
   with the rule "weighted" (default) or "pareto" (the predicted Pareto front first).  On the first night the Pareto
   rule lost at the extreme weights (lvl1/scorecard.py, ranking).  Several weights (--weight 1 0.5 0) share the
   shortlist in turn: slot 1 = the best under the first weight, slot 2 = the best under the second, ... then the
   second best under each, never the same screen twice, so the images of session F test the weighting too.
4. With --plan: write the picks' changes into the plan's SHORTLIST packets (in slot order), skipping screens already
   imaged as random CONTROLs (their image is taken anyway), and shortlist.json next to the plan.  run_plan.sh then
   continues with session F.

--evaluate: the screens are labelled packets (a test on old data): print the true J of the shortlist against all.

    python lvl1/funnel_shortlist.py --dataset processed_v2/dataset.csv --screens-dir processed_v2 \
        --plan lvl1/packet_plan_v2/plan.json --weight 0.5
    python lvl1/funnel_shortlist.py --screens L1_33 ... L1_43 --exclude-train L1_33 ... L1_43 --evaluate   # first night
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

import change_test as ct  # noqa: E402
import shortlist as sl  # noqa: E402
import train_lvl1 as tl  # noqa: E402
from ml import models  # noqa: E402

T = list(ct.LABELS)                 # sigma_e, lambda, gain
MODELS = ("change", "absolute")


# --------------------------------------------------------------------------- data
def features() -> list[str]:
    tiers = tl.load_tiers(ct.TIERS)
    return [f for f, ts in tiers.items() if ts & {"P", "A1", "A2"}]


def logf(X: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """log of the features that are positive in training (a non-positive value there becomes NaN)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pos, np.log(np.where(X > 0, X, np.nan)), X)


def load_labelled(path: Path, board: int, skip_channels, exclude) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d = d[(d.board == board) & ~d.channel.isin(skip_channels) & ~d.packet_id.isin(exclude)]
    d = d[d["label.ok"].astype(str).str.lower().eq("true") & (d["trace.n_usable"] > 0)].copy()
    ts = pd.to_datetime(d["date_start"])
    d["hours"] = (ts - ts.min()).dt.total_seconds() / 3600
    return d.reset_index(drop=True)


def reference_labels(d: pd.DataFrame) -> np.ndarray:
    """Per row, the label (log) of its reference traces: nominal -> the nominal packets' label interpolated in time;
    another packet's settings -> the latest earlier packet of the amplifier at those settings (NaN if none)."""
    th = [c for c in d.columns if c.startswith("theta.")]
    theta = d[th].to_numpy(float)
    ref_theta = theta - d[["delta." + c[6:] for c in th]].to_numpy(float)
    is_nominal = np.nan_to_num(np.abs(d[["delta." + c[6:] for c in th]].to_numpy(float))).sum(1) == 0
    nominal_theta = theta[is_nominal][0] if is_nominal.any() else None
    Y = np.log(d[[ct.LABELS[t] for t in T]].to_numpy(float))
    out = np.full(Y.shape, np.nan)
    for ch, g in d.groupby("channel"):
        r = g.index.to_numpy()
        nom = r[is_nominal[r]][np.argsort(d.hours.to_numpy()[r[is_nominal[r]]])]
        for i in r:
            if nominal_theta is not None and np.allclose(ref_theta[i], nominal_theta, atol=1e-3, equal_nan=True):
                if len(nom) and not is_nominal[i]:
                    out[i] = [np.interp(d.hours[i], d.hours[nom], Y[nom, k]) for k in range(3)]
            else:
                same = [j for j in r if d.hours[j] < d.hours[i] and np.allclose(theta[j], ref_theta[i], atol=1e-3, equal_nan=True)]
                if same:
                    out[i] = Y[max(same, key=lambda j: d.hours[j])]
    return out


def load_screens(dirs: list[Path], board: int, skip_channels, names: list[str]) -> pd.DataFrame:
    """The screens' combined trace features (process_packet.py features.json): one row per (packet, channel) with
    trace.<f> (new settings) and ref.<f> (reference)."""
    rows = []
    for p in dirs:
        f = json.loads((p / "features.json").read_text())
        comb = {(c["kind"], c["board"], c["channel"]): c for c in f["combined"]}
        for ch in sorted({c for k, b, c in comb if b == board and c not in skip_channels}):
            new, ref = comb.get(("theta", board, ch), {}), comb.get(("ref", board, ch), {})
            if not new.get("n_usable") or not ref.get("n_usable"):
                continue
            row = {"packet_id": p.name, "channel": ch}
            row.update({f"trace.{n}": new["mean"].get(n, np.nan) for n in names})
            row.update({f"ref.{n}": ref["mean"].get(n, np.nan) for n in names})
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- models
def fit_trees(X, Y):
    xtr, _, keep = tl.prep_columns(X, X[:1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = models.GBT().fit(xtr, Y)
    return lambda Z: m.predict(tl.prep_columns(X, Z)[1])[0]


def train(d: pd.DataFrame, names: list[str]):
    """The change and absolute models (predicting log sigma_e, log lambda, log gain or their changes)."""
    own, ref = d[[f"trace.{n}" for n in names]].to_numpy(float), d[[f"ref.{n}" for n in names]].to_numpy(float)
    pos = (np.nanmin(own, 0) > 0) & (np.nanmin(ref, 0) > 0)
    own, ref = logf(own, pos), logf(ref, pos)
    Y = np.log(d[[ct.LABELS[t] for t in T]].to_numpy(float))
    Yref = reference_labels(d)
    XA, XB, D = [], [], []
    ok = np.isfinite(Yref).all(1)                                     # within-packet pairs
    XA.append(ref[ok]), XB.append(own[ok]), D.append(Y[ok] - Yref[ok])
    for ch, g in d.groupby("channel"):                                # neighbouring packets
        r = g.index.to_numpy()[np.argsort(g.hours.to_numpy())]
        for step in (1, 2):
            XA.append(own[r[:-step]]), XB.append(own[r[step:]]), D.append(Y[r[step:]] - Y[r[:-step]])
    XA, XB, D = np.vstack(XA), np.vstack(XB), np.vstack(D)
    X = np.vstack([XB - XA, XA - XB])
    change = fit_trees(X, np.vstack([D, -D]))
    absolute = fit_trees(own, Y)
    return pos, {"change": lambda a, b: change(b - a), "absolute": lambda a, b: absolute(b) - absolute(a)}


def predict_screens(s: pd.DataFrame, names, pos, fitted, use) -> pd.DataFrame:
    """Per screen (packet): the mean over its amplifiers of the predicted dlog sigma_e, dlog lambda, dlog gain."""
    A = logf(s[[f"ref.{n}" for n in names]].to_numpy(float), pos)
    B = logf(s[[f"trace.{n}" for n in names]].to_numpy(float), pos)
    pred = np.mean([fitted[m](A, B) for m in use], 0)
    out = s[["packet_id", "channel"]].copy()
    for k, t in enumerate(T):
        out[f"d_{t}"] = pred[:, k]
    return out.groupby("packet_id")[[f"d_{t}" for t in T]].mean()


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(ROOT / "processed" / "dataset.csv"), help="labelled training packets")
    parser.add_argument("--screens-dir", default=str(ROOT / "processed"), help="processed packet folders")
    parser.add_argument("--screens", nargs="*", help="screen packet ids (default: the plan's screens)")
    parser.add_argument("--plan", help="plan.json: its screens, controls and SHORTLIST packets (changes files written)")
    parser.add_argument("--weight", type=float, nargs="+", default=None,
                        help="sigma_e weight(s) w of J = w dlog sigma_e + (1-w) dlog lambda, taking the shortlist slots in turn "
                             "(default: the plan's funnel weights, else 0.5)")
    parser.add_argument("--rule", choices=sl.RULES, default="weighted")
    parser.add_argument("--k", type=int, help="shortlist length (default: the plan's SHORTLIST packets, else 5)")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--board", type=int, default=101)
    parser.add_argument("--skip-channels", type=int, nargs="*", default=[2])
    parser.add_argument("--exclude-train", nargs="*", default=[], help="packets never used for training (e.g. the screens)")
    parser.add_argument("--evaluate", action="store_true", help="the screens are labelled: compare the shortlist's true J")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text()) if args.plan else None
    screen_ids = args.screens or [p["packet_id"] for p in (plan or {}).get("packets", []) if p.get("screen")]
    if not screen_ids:
        raise SystemExit("no screens: give --screens or a --plan with screen packets")
    names = features()
    d = load_labelled(Path(args.dataset), args.board, args.skip_channels, set(args.exclude_train) | set(screen_ids))
    names = [n for n in names if f"trace.{n}" in d and f"ref.{n}" in d]
    pos, fitted = train(d, names)
    found = [Path(args.screens_dir) / p for p in screen_ids if (Path(args.screens_dir) / p / "features.json").exists()]
    missing = sorted(set(screen_ids) - {p.name for p in found})
    s = load_screens(found, args.board, args.skip_channels, names)
    pred = predict_screens(s, names, pos, fitted, args.models)
    weights = args.weight or (plan or {}).get("funnel", {}).get("weights") or [0.5]
    pred["pareto"] = sl.pareto_front(pred.d_sigma_e, pred.d_lambda)
    ranked = {}
    for w in weights:
        pred[f"J {sl.weight_label(w)}"] = sl.objective(pred.d_sigma_e, pred.d_lambda, w)
        ranked[w] = list(pred.index[sl.order(pred.d_sigma_e.to_numpy(), pred.d_lambda.to_numpy(), w, args.rule)])
    print(f"trained on {len(d)} rows ({d.packet_id.nunique()} packets); {len(pred)} screens"
          + (f" ({len(missing)} without features: {missing[:5]})" if missing else "")
          + f"; weights {', '.join(sl.weight_label(w) for w in weights)} (sigma_e:lambda), rule {args.rule}, "
          f"models {'+'.join(args.models)}")
    print(pred.loc[ranked[weights[0]]].head(10).round(3).to_string())

    def picks(n, skip=()):
        """n screens: the weights in turn, each its best not yet picked (and not in skip)."""
        out, pos_ = [], {w: 0 for w in weights}
        while len(out) < min(n, len(pred) - len(skip)):
            for w in weights:
                while pos_[w] < len(ranked[w]) and (ranked[w][pos_[w]] in out or ranked[w][pos_[w]] in skip):
                    pos_[w] += 1
                if pos_[w] < len(ranked[w]) and len(out) < n:
                    out.append((ranked[w][pos_[w]], w))
        return out

    if args.evaluate:
        lab = load_labelled(Path(args.dataset), args.board, args.skip_channels, set())
        lab["ref_label"] = list(reference_labels(lab))
        k = args.k or 5
        rng = np.random.default_rng(0)
        for w in weights:                       # each weight's own top k, judged by its own J
            true = {}
            for p, g in lab[lab.packet_id.isin(pred.index)].groupby("packet_id"):
                ds = [np.log(r[ct.LABELS["sigma_e"]]) - r.ref_label[0] for _, r in g.iterrows()]
                dl = [np.log(r[ct.LABELS["lambda"]]) - r.ref_label[1] for _, r in g.iterrows()]
                if np.all(np.isfinite(ds + dl)):
                    true[p] = sl.objective(np.mean(ds), np.mean(dl), w)
            tj = pd.Series(true)
            top = [p for p in ranked[w] if p in tj.index][:k]
            rand = [tj.loc[rng.choice(tj.index, min(k, len(tj)), replace=False)].min() for _ in range(2000)]
            print(f"\nweight {sl.weight_label(w)}: true J (x100) of the {len(tj)} labelled screens: best {100 * tj.min():.1f} "
                  f"({tj.idxmin()}), median {100 * tj.median():.1f}")
            print(f"  its top {k} {top}: best {100 * tj.loc[top].min():.1f} (regret {100 * (tj.loc[top].min() - tj.min()):.1f}), "
                  f"mean {100 * tj.loc[top].mean():.1f};  {k} random screens: best {100 * np.mean(rand):.1f} on average")

    if plan:
        controls = {p["changes"] and json.dumps(p["changes"], sort_keys=True) for p in plan["packets"] if "control_of_screen" in p}
        changes = {p["packet_id"]: p["changes"] for p in plan["packets"]}
        slots = sorted((p for p in plan["packets"] if "shortlist" in p), key=lambda p: p["shortlist"])
        skip = {p for p in pred.index if json.dumps(changes.get(p, {}), sort_keys=True) in controls}
        chosen = picks(len(slots), skip)
        out_dir = Path(args.plan).parent
        for slot, (pick, w) in zip(slots, chosen):
            lines = [f"# {slot['packet_id']} (session F): funnel SHORTLIST {slot['shortlist']} = screen {pick}",
                     f"# picked by lvl1/funnel_shortlist.py: weight {sl.weight_label(w)} (sigma_e:lambda), rule "
                     f"{args.rule}, models {'+'.join(args.models)}; predicted dlog sigma_e {pred.d_sigma_e[pick]:+.3f}, "
                     f"dlog lambda {pred.d_lambda[pick]:+.3f}"]
            lines += [f"acm101:{k}={v}" if isinstance(v, int) else f"acm101:{k}={v:g}" for k, v in changes[pick].items()]
            (out_dir / f"{slot['packet_id']}.txt").write_text("\n".join(lines) + "\n")
        (out_dir / "shortlist.json").write_text(json.dumps(
            {"weights": weights, "rule": args.rule, "models": args.models,
             "picks": {s_["packet_id"]: {"screen": p, "weight": w} for s_, (p, w) in zip(slots, chosen)},
             "skipped_as_controls": sorted(skip), "screens": pred.reset_index().to_dict(orient="records"),
             "missing": missing}, indent=1, default=float))
        print(f"\nwrote {len(chosen)} shortlist changes files and shortlist.json to {out_dir}: "
              + ", ".join(f"{s_['packet_id']} = {p} ({sl.weight_label(w)})" for s_, (p, w) in zip(slots, chosen)))


if __name__ == "__main__":
    main()
