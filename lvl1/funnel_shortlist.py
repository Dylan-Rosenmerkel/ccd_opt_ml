"""
funnel_shortlist.py -- session F: pick the settings worth an image from the trace-only screens, with TWO rankings side
by side on the same screens.

1. Train on the labelled packets (a dataset.csv; board 101 without ch2 by default):
   * the change model: boosted trees on [features(A), features(B), features(B) - features(A)] of two traces of one
     amplifier (log for positive features), trained on the within-packet pairs (reference -> own trace; the reference's
     label is the nominal label interpolated in time, or, for a reference at another packet's settings, that packet's
     label) and on every pair of packets of an amplifier with drift-corrected labels ("random pairs": each packet's
     setting effect = log y - log y(nominal interpolated in time), the pair's target the difference), in both orders;
   * the one-trace model: boosted trees, one trace's features (never the settings) -> setting effect.
2. Read the screens (processed packet folders: features.json; new traces, and the reference trace if there is one).
3. Rank them twice, per amplifier, then average each screen's score over the board's amplifiers (a setting is for all):
   * WITH REFERENCE: each screen's change against its own nominal reference trace = the mean of the change model
     (reference -> new) and the one-trace model (f(new) - f(reference));
   * WITHOUT REFERENCE: the references are ignored.  The change model compares every two screens (both orders,
     averaged), and a sorting method (--sort, lvl1/shortlist.py: mean change, wins, Bradley-Terry) turns the
     comparisons into one order.
   The joint objective J(w) = w dlog sigma_e + (1-w) dlog lambda (lvl1/shortlist.py) is applied to every prediction
   (for screen vs screen: to every comparison, before sorting).
4. The shortlist slots go round: for each weight (--weight, default the plan's funnel weights), the best screen of the
   with-reference ranking, then the best of the without-reference ranking, then the next weight; then the second best
   of each, never the same screen twice, skipping screens already imaged as random CONTROLs.  So session F's images
   test both rankings and the weighting.  With --plan the picks are written into the plan's SHORTLIST changes files
   (in slot order) with shortlist.json (both full rankings); run_plan.sh then continues with session F.

--evaluate: the screens are labelled packets (a test on old data): each ranking's top k against the truth and random.

    python lvl1/funnel_shortlist.py --dataset processed_v2/dataset.csv --screens-dir processed_v2 \
        --plan lvl1/packet_plan_v2/plan.json [--weight 1 0.5 0] [--sort bradley-terry]
    python lvl1/funnel_shortlist.py --screens L1_33 ... L1_42 --evaluate        # first night (the screens are held out)
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
METHODS = ("with reference", "without reference")


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


def train(d: pd.DataFrame, names: list[str]):
    """The change model (A, B -> log y(B) - log y(A)) and the one-trace model (trace -> setting effect)."""
    own, ref = d[[f"trace.{n}" for n in names]].to_numpy(float), d[[f"ref.{n}" for n in names]].to_numpy(float)
    pos = (np.nanmin(own, 0) > 0) & (np.nanmin(ref, 0) > 0)
    own, ref = logf(own, pos), logf(ref, pos)
    Y = np.log(d[[ct.LABELS[t] for t in T]].to_numpy(float))
    eff = Y - nominal_interp(d)                                       # setting effect (drift removed)
    Yref = reference_labels(d)
    XA, XB, D = [], [], []
    ok = np.isfinite(Yref).all(1)                                     # within-packet pairs
    XA.append(ref[ok]), XB.append(own[ok]), D.append(Y[ok] - Yref[ok])
    for ch in d.channel.unique():                                     # random pairs, drift-corrected
        r = np.flatnonzero(d.channel.to_numpy() == ch)
        pr = np.array([(i, j) for i, j in combinations(r, 2)])
        dy = eff[pr[:, 1]] - eff[pr[:, 0]]
        keep = np.isfinite(dy).all(1)
        XA.append(own[pr[keep, 0]]), XB.append(own[pr[keep, 1]]), D.append(dy[keep])
    XA, XB, D = np.vstack(XA), np.vstack(XB), np.vstack(D)
    f_change = gbt(np.vstack([np.hstack([XA, XB, XB - XA]), np.hstack([XB, XA, XA - XB])]), np.vstack([D, -D]))
    ok1 = np.isfinite(eff).all(1)
    f_one = gbt(own[ok1], eff[ok1])
    return pos, {"change": lambda A, B: f_change(np.hstack([A, B, B - A])), "one trace": f_one}, len(D)


def rank_screens(s: pd.DataFrame, names, pos, fitted, weights, sort: str) -> dict:
    """{(method, w): screen ids in order (best first)} and the per-screen board scores."""
    A = logf(s[[f"ref.{n}" for n in names]].to_numpy(float), pos)
    B = logf(s[[f"trace.{n}" for n in names]].to_numpy(float), pos)
    has_ref = np.isfinite(A).any(1)
    # with reference: per (screen, amplifier), the mean of the change model and the one-trace difference
    wr = np.full((len(s), 3), np.nan)
    if has_ref.any():
        wr[has_ref] = (fitted["change"](A[has_ref], B[has_ref])
                       + fitted["one trace"](B[has_ref]) - fitted["one trace"](A[has_ref])) / 2
    # without reference: every two screens of an amplifier compared with the change model
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
        j = pd.Series(sl.objective(wr[:, 0], wr[:, 1], w), index=s.index).groupby(s.packet_id).mean()
        scores[("with reference", w)] = j.reindex(screens)
        per_amp = []
        for ch, (ids, D) in pairwise.items():
            per_amp.append(pd.Series(sl.sort_scores(sl.objective(D[:, :, 0], D[:, :, 1], w), sort), index=ids))
        scores[("without reference", w)] = pd.DataFrame({k: v for k, v in enumerate(per_amp)}).mean(1).reindex(screens)
        for m in METHODS:
            sc_ = scores[(m, w)].dropna()
            out[(m, w)] = list(sc_.sort_values(kind="stable").index)
    return out, scores


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
                             "weights, else 1 0.5 0)")
    parser.add_argument("--sort", choices=sl.SORTS, default=None,
                        help="sorting of the screen-vs-screen comparisons (default: the plan's, else bradley-terry: best on the first night, lvl1/rank_methods.py)")
    parser.add_argument("--k", type=int, default=5, help="top k per ranking for --evaluate")
    parser.add_argument("--board", type=int, default=101)
    parser.add_argument("--skip-channels", type=int, nargs="*", default=[2])
    parser.add_argument("--exclude-train", nargs="*", default=[], help="packets never used for training (the screens always are)")
    parser.add_argument("--evaluate", action="store_true", help="the screens are labelled: compare each ranking's top k")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text()) if args.plan else None
    funnel = (plan or {}).get("funnel", {})
    weights = args.weight or funnel.get("weights") or [1.0, 0.5, 0.0]
    sort = args.sort or funnel.get("sort") or "bradley-terry"
    screen_ids = args.screens or [p["packet_id"] for p in (plan or {}).get("packets", []) if p.get("screen")]
    if not screen_ids:
        raise SystemExit("no screens: give --screens or a --plan with screen packets")
    names = features()
    d = load_labelled(Path(args.dataset), args.board, args.skip_channels, set(args.exclude_train) | set(screen_ids))
    names = [n for n in names if f"trace.{n}" in d and f"ref.{n}" in d]
    pos, fitted, n_pairs = train(d, names)
    found = [Path(args.screens_dir) / p for p in screen_ids if (Path(args.screens_dir) / p / "features.json").exists()]
    missing = sorted(set(screen_ids) - {p.name for p in found})
    s = load_screens(found, args.board, args.skip_channels, names)
    order, scores = rank_screens(s, names, pos, fitted, weights, sort)
    print(f"trained on {len(d)} rows ({d.packet_id.nunique()} packets; {n_pairs} change pairs); {s.packet_id.nunique()} screens"
          + (f" ({len(missing)} without features: {missing[:5]})" if missing else "")
          + f"; weights {', '.join(sl.weight_label(w) for w in weights)} (sigma_e:lambda); screen-vs-screen sort: {sort}")
    for w in weights:
        print(f"  {sl.weight_label(w)}: with reference {order[('with reference', w)][:5]}   "
              f"without reference {order[('without reference', w)][:5]}")

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
                     + (f", sort {sort}" if m == "without reference" else "")]
            lines += [f"acm101:{k}={v}" if isinstance(v, int) else f"acm101:{k}={v:g}" for k, v in changes[pick].items()]
            (out_dir / f"{slot['packet_id']}.txt").write_text("\n".join(lines) + "\n")
        (out_dir / "shortlist.json").write_text(json.dumps(
            {"weights": weights, "sort": sort, "methods": list(METHODS),
             "picks": {s_["packet_id"]: {"screen": p, "weight": w, "ranking": m} for s_, (p, w, m) in zip(slots, chosen)},
             "skipped_as_controls": sorted(skip),
             "rankings": {f"{m} | {sl.weight_label(w)}": order[(m, w)] for m in METHODS for w in weights},
             "scores": {f"{m} | {sl.weight_label(w)}": scores[(m, w)].dropna().to_dict() for m in METHODS for w in weights},
             "missing": missing}, indent=1, default=float))
        print(f"\nwrote {len(chosen)} shortlist changes files and shortlist.json to {out_dir}: "
              + ", ".join(f"{s_['packet_id']} = {p} ({m}, {sl.weight_label(w)})" for s_, (p, w, m) in zip(slots, chosen)))


if __name__ == "__main__":
    main()
