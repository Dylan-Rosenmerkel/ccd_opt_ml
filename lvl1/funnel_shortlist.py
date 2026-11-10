"""
funnel_shortlist.py -- session F: from the trace-only screens (new traces only, no reference), pick the settings
worth an image: can a shortlist of the best settings be predicted from ~100 traces?

1. Train on the labelled packets (a dataset.csv; board 101 without ch2 by default):
   * the change model, which RANKS: boosted trees on [features(A), features(B), features(B) - features(A)] of two
     traces of one amplifier (log for positive features), one per target (sigma_e, lambda, gain); by default
     CLASSIFIERS of which trace is better (--target direction: P(B worse than A), changes of the size of the label
     noise down-weighted; Bradley-Terry counts the probabilities as soft wins), or regressions of the size of the change
     (--target size); trained in both orders on the within-packet pairs (reference -> own trace) and on
     every two packets of an amplifier with drift-corrected labels (--pairs random, the default: each packet's
     setting effect = log y - log y(nominal interpolated in time), the pair's target the difference), or on
     neighbouring packets (--pairs neighbours: n -> n+1, n+2, measured labels);
   * the noise model, which only ATTACHES a predicted sigma_e (electrons) to each screen after the ranking: boosted
     trees on one trace's features and its amplifier -> log sigma_e (the best sigma_e model on known amplifiers in the
     scorecard, ~10 %); its typical error is estimated by 5-fold cross-validation on the training packets.
2. Read the screens (processed packet folders: features.json, the new traces).
3. Rank: per amplifier, the change model compares every two screens (both orders, averaged); a sorting method (--sort,
   lvl1/shortlist.py: mean change, wins, Bradley-Terry; default Bradley-Terry, best on the first night,
   lvl1/rank_methods.py) turns the comparisons of J(w) = w dlog sigma_e + (1-w) dlog lambda into one order; the
   scores are averaged over the board's amplifiers (a setting is for all of them).  First round: w = 0, the screens
   are ranked on lambda (dark charge) alone, and the predicted noise is attached to the picks.
4. The shortlist = the best screens in order (with several weights, they take turns), skipping screens already imaged
   as random CONTROLs.  With --plan the picks are written into the plan's SHORTLIST changes files with shortlist.json
   (the full ranking, scores and predicted noise); run_plan.sh then continues with session F.

--evaluate: the screens are labelled packets (a test on old data): the shortlist against the truth and random picks.

    python lvl1/funnel_shortlist.py --dataset processed_v2/dataset.csv --screens-dir processed_v2 \
        --plan lvl1/packet_plan_v2/plan.json [--weight 0] [--sort bradley-terry] [--pairs random]
    python lvl1/funnel_shortlist.py --screens L1_26 ... L1_42 --evaluate        # first night (the screens are held out)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

import change_test as ct  # noqa: E402
import shortlist as sl  # noqa: E402
import train_lvl1 as tl  # noqa: E402

T = list(ct.LABELS)                 # sigma_e, lambda, gain
METHODS = ("without reference",)    # the screens are compared with each other; no nominal reference (user, 2026-11-04)


# --------------------------------------------------------------------------- data
def features() -> list[str]:
    tiers = tl.load_tiers(ct.TIERS)
    return [f for f, ts in tiers.items() if ts & {"P", "A1", "A2"}]


def logf(X: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """log of the features that are positive in training (a non-positive value there becomes NaN)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pos, np.log(np.where(X > 0, X, np.nan)), X)


def rounded(changes: dict, step: int) -> dict:
    """delayInt to the nearest multiple of step (half up); the rest unchanged."""
    if not step or "delayInt" not in changes:
        return dict(changes)
    return {**changes, "delayInt": int(np.floor(changes["delayInt"] / step + 0.5) * step)}


def load_labelled(path: Path, board: int, skip_channels, exclude, keep_fractional: bool = False) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d = d[(d.board == board) & ~d.channel.isin(skip_channels) & ~d.packet_id.isin(exclude)]
    if not keep_fractional and "label.fractional_window" in d:
        # delayInt not a whole number of ADC samples: N / N+1 sample windows bias the image labels (2026-11)
        d = d[~d["label.fractional_window"].astype(str).str.lower().eq("true")]
    d = d[d["label.ok"].astype(str).str.lower().eq("true") & (d["trace.n_usable"] > 0)].copy()
    ts = pd.to_datetime(d["date_start"])
    d["hours"] = (ts - ts.min()).dt.total_seconds() / 3600
    return d.reset_index(drop=True)


def _nominal(d: pd.DataFrame):
    th = [c for c in d.columns if c.startswith("theta.")]
    theta = d[th].to_numpy(float)
    delta = d[["delta." + c[6:] for c in th]].to_numpy(float)
    is_nominal = np.nan_to_num(np.abs(delta)).sum(1) == 0
    return theta, theta - delta, is_nominal


def nominal_interp(d: pd.DataFrame) -> np.ndarray:
    """Per row, the amplifier's nominal label (log) interpolated in time between its nominal packets."""
    _, _, is_nominal = _nominal(d)
    Y = np.log(d[[ct.LABELS[t] for t in T]].to_numpy(float))
    out = np.full(Y.shape, np.nan)
    for ch, g in d.groupby("channel"):
        r = g.index.to_numpy()
        nom = r[is_nominal[r]][np.argsort(d.hours.to_numpy()[r[is_nominal[r]]])]
        if len(nom):
            for k in range(3):
                out[r, k] = np.interp(d.hours.to_numpy()[r], d.hours.to_numpy()[nom], Y[nom, k])
    return out


def reference_labels(d: pd.DataFrame) -> np.ndarray:
    """Per row, the label (log) of its reference traces: nominal -> the nominal packets' label interpolated in time;
    another packet's settings -> the latest earlier packet of the amplifier at those settings (NaN if none)."""
    theta, ref_theta, is_nominal = _nominal(d)
    nominal_theta = theta[is_nominal][0] if is_nominal.any() else None
    Y = np.log(d[[ct.LABELS[t] for t in T]].to_numpy(float))
    nom = nominal_interp(d)
    out = np.full(Y.shape, np.nan)
    for ch, g in d.groupby("channel"):
        r = g.index.to_numpy()
        for i in r:
            if nominal_theta is not None and np.allclose(ref_theta[i], nominal_theta, atol=1e-3, equal_nan=True):
                if not is_nominal[i]:
                    out[i] = nom[i]
            else:
                same = [j for j in r if d.hours[j] < d.hours[i] and np.allclose(theta[j], ref_theta[i], atol=1e-3, equal_nan=True)]
                if same:
                    out[i] = Y[max(same, key=lambda j: d.hours[j])]
    return out


def load_screens(dirs: list[Path], board: int, skip_channels, names: list[str]) -> pd.DataFrame:
    """The screens' combined trace features (process_packet.py features.json): one row per (packet, channel) with
    trace.<f> (new settings) and ref.<f> (reference; NaN when the screen has none)."""
    rows = []
    for p in dirs:
        f = json.loads((p / "features.json").read_text())
        comb = {(c["kind"], c["board"], c["channel"]): c for c in f["combined"]}
        for ch in sorted({c for k, b, c in comb if b == board and c not in skip_channels}):
            new, ref = comb.get(("theta", board, ch), {}), comb.get(("ref", board, ch), {})
            if not new.get("n_usable"):
                continue
            row = {"packet_id": p.name, "channel": ch}
            row.update({f"trace.{n}": new["mean"].get(n, np.nan) for n in names})
            row.update({f"ref.{n}": (ref["mean"].get(n, np.nan) if ref.get("n_usable") else np.nan) for n in names})
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- models
def gbt(X, Y):
    """One boosted-tree model per target; returns predict(Z)."""
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    xtr, _, _ = tl.prep_columns(X, X[:1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = [H(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, random_state=0).fit(xtr, Y[:, k])
              for k in range(Y.shape[1])]
    return lambda Z: np.column_stack([m.predict(tl.prep_columns(X, Z)[1]) for m in ms])


def direction_model(X, Y, noise=(0.019, 0.12, 0.008)):
    """Boosted-tree CLASSIFIERS, one per target: is the change of the pair positive (B worse than A)?  A change of the
    size of the label noise is down-weighted (w = 1 - exp(-d^2 / 2 s^2), s = the label noise of a change).  Returns
    predict(Z) = P(B worse) - 0.5 per target (positive: B worse), the margins Bradley-Terry counts as soft wins."""
    from sklearn.ensemble import HistGradientBoostingClassifier as C
    xtr, _, _ = tl.prep_columns(X, X[:1])
    ms = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(Y.shape[1]):
            w = 1 - np.exp(-Y[:, k] ** 2 / (2 * noise[k] ** 2))
            ms.append(C(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, random_state=0)
                      .fit(xtr, (Y[:, k] > 0).astype(int), sample_weight=w))
    return lambda Z: np.column_stack([m.predict_proba(tl.prep_columns(X, Z)[1])[:, 1] - 0.5 for m in ms])


def train(d: pd.DataFrame, names: list[str], pairs: str = "random", target: str = "direction"):
    """The change model (A, B -> log y(B) - log y(A); ranks the screens) and the sigma_e model (one trace + which
    amplifier -> log sigma_e; only attaches a predicted noise to each screen, never ranks).  Returns also the sigma_e
    model's typical error (5-fold cross-validation by packet on the training rows, RMS of the log error)."""
    own, ref = d[[f"trace.{n}" for n in names]].to_numpy(float), d[[f"ref.{n}" for n in names]].to_numpy(float)
    pos = (np.nanmin(own, 0) > 0) & (np.nanmin(ref, 0) > 0)
    own, ref = logf(own, pos), logf(ref, pos)
    Y = np.log(d[[ct.LABELS[t] for t in T]].to_numpy(float))
    eff = Y - nominal_interp(d)                                       # setting effect (drift removed)
    Yref = reference_labels(d)
    XA, XB, D = [], [], []
    ok = np.isfinite(Yref).all(1)                                     # within-packet pairs
    XA.append(ref[ok]), XB.append(own[ok]), D.append(Y[ok] - Yref[ok])
    for ch in d.channel.unique():
        r = np.flatnonzero(d.channel.to_numpy() == ch)
        if pairs == "random":                                         # every two packets, drift-corrected labels
            pr = np.array([(i, j) for i, j in combinations(r, 2)])
            dy = eff[pr[:, 1]] - eff[pr[:, 0]]
        else:                                                         # neighbours n -> n+1, n+2, measured labels
            r = r[np.argsort(d.hours.to_numpy()[r])]
            pr = np.array([(i, j) for step in (1, 2) for i, j in zip(r[:-step], r[step:])])
            dy = Y[pr[:, 1]] - Y[pr[:, 0]]
        keep = np.isfinite(dy).all(1)
        XA.append(own[pr[keep, 0]]), XB.append(own[pr[keep, 1]]), D.append(dy[keep])
    XA, XB, D = np.vstack(XA), np.vstack(XB), np.vstack(D)
    XX, YY = np.vstack([np.hstack([XA, XB, XB - XA]), np.hstack([XB, XA, XA - XB])]), np.vstack([D, -D])
    f_change = direction_model(XX, YY) if target == "direction" else gbt(XX, YY)
    # the sigma_e predictor: trees on the trace features and the amplifier (the best sigma_e model on known amplifiers
    # in the scorecard, ~10 %); the funnel's amplifiers are the trained ones
    chans = sorted(d.channel.unique())
    onehot = lambda ch: np.eye(len(chans))[[chans.index(c) for c in ch]]
    Xs = np.hstack([own, onehot(d.channel)])
    f_sig = gbt(Xs, Y[:, :1])
    pk = d.packet_id.to_numpy()
    folds = np.searchsorted(np.unique(pk), pk) % 5                    # packets dealt into 5 folds
    cv = np.full(len(d), np.nan)
    for k in range(5):
        cv[folds == k] = gbt(Xs[folds != k], Y[folds != k, :1])(Xs[folds == k])[:, 0]
    err = float(np.sqrt(np.nanmean((cv - Y[:, 0]) ** 2)))
    return pos, {"change": lambda A, B: f_change(np.hstack([A, B, B - A])), "target": target,
                 "sigma_e": lambda B, ch: f_sig(np.hstack([B, onehot(ch)]))[:, 0], "sigma_e_error": err}, len(D)


def rank_screens(s: pd.DataFrame, names, pos, fitted, weights, sort: str) -> dict:
    """{(method, w): screen ids in order (best first)} and the per-screen board scores."""
    B = logf(s[[f"trace.{n}" for n in names]].to_numpy(float), pos)
    # every two screens of an amplifier compared with the change model (both orders)
    pairwise = {}
    for ch in s.channel.unique():
        r = np.flatnonzero(s.channel.to_numpy() == ch)
        D = np.full((len(r), len(r), 3), np.nan)
        ii, jj = np.array(list(combinations(range(len(r)), 2))).T
        D[ii, jj] = fitted["change"](B[r[ii]], B[r[jj]])
        D[jj, ii] = fitted["change"](B[r[jj]], B[r[ii]])
        pairwise[ch] = (s.packet_id.to_numpy()[r], D)
    screens = sorted(set(s.packet_id))
    out, scores = {}, {}
    for w in weights:
        per_amp = []
        for ch, (ids, D) in pairwise.items():
            per_amp.append(pd.Series(sl.sort_scores(sl.objective(D[:, :, 0], D[:, :, 1], w), sort,
                                                    soft=fitted["target"] == "direction"), index=ids))
        scores[("without reference", w)] = pd.DataFrame({k: v for k, v in enumerate(per_amp)}).mean(axis=1).reindex(screens)
        for m in METHODS:
            sc_ = scores[(m, w)].dropna()
            out[(m, w)] = list(sc_.sort_values(kind="stable").index)
    # the predicted noise attached to every screen (never used to rank): sigma_e in electrons per amplifier, and the
    # mean over the amplifiers
    sig = pd.DataFrame({"packet_id": s.packet_id, "channel": s.channel,
                        "sigma_e": np.exp(fitted["sigma_e"](B, s.channel.to_numpy()))})
    noise = sig.pivot_table(index="packet_id", columns="channel", values="sigma_e")
    noise.columns = [f"sigma_e ch{c}" for c in noise.columns]
    noise["sigma_e mean"] = noise.mean(axis=1)
    return out, scores, noise.reindex(screens)


def pick_slots(order: dict, weights, n: int, skip=()) -> list[tuple[str, float, str]]:
    """n picks (screen, weight, method): per round, each weight's best not yet picked from each method in turn."""
    out, used = [], set(skip)
    pos = {k: 0 for k in order}
    while len(out) < n and any(pos[k] < len(order[k]) for k in order):
        for w in weights:
            for m in METHODS:
                k = (m, w)
                while pos[k] < len(order[k]) and order[k][pos[k]] in used:
                    pos[k] += 1
                if pos[k] < len(order[k]) and len(out) < n:
                    out.append((order[k][pos[k]], w, m))
                    used.add(order[k][pos[k]])
    return out


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(ROOT / "processed" / "dataset.csv"), help="labelled training packets")
    parser.add_argument("--screens-dir", default=str(ROOT / "processed"), help="processed packet folders")
    parser.add_argument("--screens", nargs="*", help="screen packet ids (default: the plan's screens)")
    parser.add_argument("--plan", help="plan.json: its screens, controls and SHORTLIST packets (changes files written)")
    parser.add_argument("--weight", type=float, nargs="+", default=None,
                        help="sigma_e weight(s) w of J = w dlog sigma_e + (1-w) dlog lambda (default: the plan's funnel "
                             "weights, else 0 = rank on lambda (dark charge) only, the first round)")
    parser.add_argument("--sort", choices=sl.SORTS, default=None,
                        help="sorting of the screen-vs-screen comparisons (default: the plan's, else bradley-terry: best on the first night, lvl1/rank_methods.py)")
    parser.add_argument("--target", choices=("direction", "size"), default="direction",
                        help="the change model: classify which trace is better (default; best lambda ranking on the first "
                             "night, lvl1/rank_methods.py --target direction) or regress the size of the change")
    parser.add_argument("--pairs", choices=("neighbours", "random"), default="random",
                        help="the change model's training pairs besides the within-packet ones: every two packets with "
                             "drift-corrected labels (default: ranks the screens best on the first night, "
                             "lvl1/rank_methods.py) or neighbouring packets (n -> n+1, n+2, measured labels)")
    parser.add_argument("--k", type=int, default=5, help="top k per ranking for --evaluate")
    parser.add_argument("--board", type=int, default=101)
    parser.add_argument("--skip-channels", type=int, nargs="*", default=[2])
    parser.add_argument("--exclude-train", nargs="*", default=[], help="packets never used for training (the screens always are)")
    parser.add_argument("--evaluate", action="store_true", help="the screens are labelled: compare each ranking's top k")
    parser.add_argument("--keep-fractional", action="store_true", help="keep the training rows with fractional CDS windows")
    parser.add_argument("--round-delayint", type=int, default=20,
                        help="imaged packets (shortlist and controls): delayInt rounded to a multiple of this (whole ADC "
                             "samples: no fractional-window labels; 0 = as screened)")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text()) if args.plan else None
    funnel = (plan or {}).get("funnel", {})
    weights = args.weight or funnel.get("weights") or [0.0]
    sort = args.sort or funnel.get("sort") or "bradley-terry"
    screen_ids = args.screens or [p["packet_id"] for p in (plan or {}).get("packets", []) if p.get("screen")]
    if not screen_ids:
        raise SystemExit("no screens: give --screens or a --plan with screen packets")
    names = features()
    d = load_labelled(Path(args.dataset), args.board, args.skip_channels, set(args.exclude_train) | set(screen_ids),
                      args.keep_fractional)
    names = [n for n in names if f"trace.{n}" in d and f"ref.{n}" in d]
    pos, fitted, n_pairs = train(d, names, args.pairs, args.target)
    found = [Path(args.screens_dir) / p for p in screen_ids if (Path(args.screens_dir) / p / "features.json").exists()]
    missing = sorted(set(screen_ids) - {p.name for p in found})
    s = load_screens(found, args.board, args.skip_channels, names)
    order, scores, noise = rank_screens(s, names, pos, fitted, weights, sort)
    err = fitted["sigma_e_error"]
    print(f"trained on {len(d)} rows ({d.packet_id.nunique()} packets; {n_pairs} change pairs, {args.pairs}, "
          f"{args.target} model); {s.packet_id.nunique()} screens"
          + (f" ({len(missing)} without features: {missing[:5]})" if missing else "")
          + f"; weights {', '.join(sl.weight_label(w) for w in weights)} (sigma_e:lambda); screen-vs-screen sort: {sort}")
    print(f"predicted noise attached to each screen: sigma_e (e-) from trees on the trace + amplifier, typical error "
          f"{100 * err:.0f} % (cross-validated)")
    for w in weights:
        for m in METHODS:
            top = order[(m, w)][:8]
            print(f"\n  {m}, weight {sl.weight_label(w)}" + (f" (sort {sort})" if m == "without reference" else "")
                  + ": best first")
            print(noise.loc[top].round(3).assign(score=scores[(m, w)].loc[top].round(3)).to_string())

    if args.evaluate:
        lab = load_labelled(Path(args.dataset), args.board, args.skip_channels, set())
        Y = np.log(lab[[ct.LABELS[t] for t in T]].to_numpy(float))
        eff = pd.DataFrame(Y - nominal_interp(lab), columns=T)
        eff["packet_id"] = lab.packet_id
        truth = eff[eff.packet_id.isin(screen_ids)].groupby("packet_id")[T].mean().dropna()
        rng = np.random.default_rng(0)
        k = args.k
        for w in weights:
            tj = pd.Series(sl.objective(truth.sigma_e, truth["lambda"], w), index=truth.index)
            rand = np.mean([tj.loc[rng.choice(tj.index, min(k, len(tj)), replace=False)].min() for _ in range(2000)])
            line = [f"weight {sl.weight_label(w)}: {len(tj)} labelled screens, true best {100 * tj.min():.1f} ({tj.idxmin()})"]
            for m in METHODS:
                top = [p for p in order[(m, w)] if p in tj.index][:k]
                line.append(f"{m}: best of top {k} {100 * tj.loc[top].min():.1f} (regret {100 * (tj.loc[top].min() - tj.min()):.1f})")
            line.append(f"random {k}: best {100 * rand:.1f} on average")
            print("\n  ".join(line))
        tsig = lab[lab.packet_id.isin(screen_ids)].groupby("packet_id")[ct.LABELS["sigma_e"]].mean()
        both = noise["sigma_e mean"].dropna().index.intersection(tsig.index)
        print(f"attached sigma_e vs the image (board mean, {len(both)} screens): typical error "
              f"{100 * np.sqrt(np.mean(np.log(noise.loc[both, 'sigma_e mean'] / tsig.loc[both]) ** 2)):.1f} %")

    if plan:
        controls = {json.dumps(p["changes"], sort_keys=True) for p in plan["packets"] if "control_of_screen" in p}
        changes = {p["packet_id"]: p["changes"] for p in plan["packets"]}
        slots = sorted((p for p in plan["packets"] if "shortlist" in p), key=lambda p: p["shortlist"])
        skip = {p for p in scores[(METHODS[0], weights[0])].index if json.dumps(changes.get(p, {}), sort_keys=True) in controls}
        chosen = pick_slots(order, weights, len(slots), skip)
        out_dir = Path(args.plan).parent
        for slot, (pick, w, m) in zip(slots, chosen):
            lines = [f"# {slot['packet_id']} (session F): funnel SHORTLIST {slot['shortlist']} = screen {pick}",
                     f"# picked by lvl1/funnel_shortlist.py: ranking {m}, weight {sl.weight_label(w)} (sigma_e:lambda)"
                     + (f", sort {sort}" if m == "without reference" else ""),
                     "# predicted noise (not used to pick): " + ", ".join(
                         f"{c} {v:.3f}" for c, v in noise.loc[pick].items() if np.isfinite(v)) + f" e- (+-{100 * err:.0f} %)"]
            ch = rounded(changes[pick], args.round_delayint)
            if ch != changes[pick]:
                lines.append(f"# delayInt rounded to a multiple of {args.round_delayint} for the image "
                             f"(screened at {changes[pick]['delayInt']})")
            lines += [f"acm101:{k}={v}" if isinstance(v, int) else f"acm101:{k}={v:g}" for k, v in ch.items()]
            (out_dir / f"{slot['packet_id']}.txt").write_text("\n".join(lines) + "\n")
        rounded_controls = {}
        for p in plan["packets"]:
            if "control_of_screen" in p:
                ch = rounded(p["changes"], args.round_delayint)
                if ch != p["changes"]:
                    f = out_dir / f"{p['packet_id']}.txt"
                    head = [x for x in f.read_text().splitlines() if x.startswith("#") and "rounded" not in x]
                    f.write_text("\n".join(head + [f"# delayInt rounded to a multiple of {args.round_delayint} for the "
                                                    f"image (screened at {p['changes']['delayInt']})"]
                                           + [f"acm101:{k}={v}" if isinstance(v, int) else f"acm101:{k}={v:g}"
                                              for k, v in ch.items()]) + "\n")
                    rounded_controls[p["packet_id"]] = [p["changes"]["delayInt"], ch["delayInt"]]
        (out_dir / "shortlist.json").write_text(json.dumps(
            {"weights": weights, "sort": sort, "methods": list(METHODS), "sigma_e_typical_error": err,
             "picks": {s_["packet_id"]: {"screen": p, "weight": w, "ranking": m,
                                         "predicted_noise_e": noise.loc[p].dropna().to_dict()}
                       for s_, (p, w, m) in zip(slots, chosen)},
             "predicted_noise_e": noise.dropna(how="all").to_dict(orient="index"),
             "skipped_as_controls": sorted(skip),
             "rankings": {f"{m} | {sl.weight_label(w)}": order[(m, w)] for m in METHODS for w in weights},
             "scores": {f"{m} | {sl.weight_label(w)}": scores[(m, w)].dropna().to_dict() for m in METHODS for w in weights},
             "missing": missing, "round_delayint": args.round_delayint, "rounded_controls": rounded_controls,
             "keep_fractional_training_rows": args.keep_fractional}, indent=1, default=float))
        print(f"\nwrote {len(chosen)} shortlist changes files and shortlist.json to {out_dir}: "
              + ", ".join(f"{s_['packet_id']} = {p} ({m}, {sl.weight_label(w)})" for s_, (p, w, m) in zip(slots, chosen)))


if __name__ == "__main__":
    main()
