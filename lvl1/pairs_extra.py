"""
pairs_extra.py -- two questions for the v2 plan, on the first night (board 101 ch0/1/3, the scorecard's time blocks):

1. RANDOM PAIRS.  Train the change models on every pair of packets of an amplifier (not only neighbours), with the
   labels drift-corrected: each packet's setting effect e = log y - log y(nominal, interpolated in time between the
   nominal packets), and the pair's target e(B) - e(A).  Against the current training set (within-packet pairs and
   neighbours n -> n+1, n+2 with raw labels).  Both in both orders.  Scored on test neighbours (raw labels, as in the
   scorecard) and on every test pair (drift-corrected).
2. DOES EACH TRACE NEED ITS OWN REFERENCE?  For the within-packet changes (nominal -> new setting) the reference
   trace is replaced by the reference traces of the packet d places earlier (same amplifier; d = 0 is its own, taken
   minutes before; d = 3 / 6 ~ 45 / 90 minutes earlier), as if one nominal trace set were shared by several screens.
   Scored by direction, size and the ranking regret of the setting effect (board mean, w = 0.5).

    .venv/bin/python lvl1/pairs_extra.py
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

import change_test as ct  # noqa: E402
import scorecard as sc  # noqa: E402
import shortlist as sl  # noqa: E402

T = sc.T


def setting_effect(d, Y):
    """log y - log y(nominal interpolated in time), per amplifier (drift removed)."""
    eff = np.full(Y.shape, np.nan)
    for ch, g in d.groupby("channel"):
        r = g.index.to_numpy()
        m = r[g.packet_id.isin(ct.NOMINAL).to_numpy()]
        m = m[np.argsort(d.hours[m].to_numpy())]
        for k in range(3):
            eff[r, k] = Y[r, k] - np.interp(d.hours[r], d.hours[m], Y[m, k])
    return eff


def fit(kind, FA, FB, Y):
    """Change trees on B - A, or combined trees on [A, B, B - A]; both orders."""
    if kind == "B - A":
        X, Xs = FB - FA, FA - FB
    else:
        X, Xs = np.hstack([FA, FB, FB - FA]), np.hstack([FB, FA, FA - FB])
    Xtr, Ytr = np.vstack([X, Xs]), np.vstack([Y, -Y])
    def predict(GA, GB):
        Z = GB - GA if kind == "B - A" else np.hstack([GA, GB, GB - GA])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sc.fit_predict("trees", Xtr, Ytr, Z)
    return predict


def score(y, p):
    out = []
    for i, t in enumerate(T):
        dr = sc.direction(y[:, i], p[:, i], sc.CLEAR[i])
        out.append(f"{t} {100 * dr:3.0f} % / {100 * sc.rms(p[:, i] - y[:, i]):5.1f}")
    return "   ".join(out)


def main():
    files, rows, d = sc.load((0, 1, 3))
    Y = files["physics"]["Y"][rows]
    TR, P = sc.trace_table(files, rows, d), sc.make_pairs(d, Y)
    n = len(d)
    F = TR["F"]
    eff = setting_effect(d, Y)
    amp, blk, hours = d.channel.to_numpy(), d.block.to_numpy(), d.hours.to_numpy()
    # every pair of packets of one amplifier (own traces), drift-corrected target
    allp = np.array([(i, j) for ch in np.unique(amp) for i, j in combinations(np.flatnonzero(amp == ch), 2)])
    allY = eff[allp[:, 1]] - eff[allp[:, 0]]
    ok_all = np.isfinite(allY).all(1)
    allp, allY = allp[ok_all], allY[ok_all]
    print(f"{len(P['A'])} current pairs; {len(allp)} drift-corrected pairs of any two packets")

    # within-packet pairs with the reference of the packet d places earlier (same amplifier, time order)
    within = np.flatnonzero(P["type"] == 0)
    order = {ch: np.flatnonzero(amp == ch)[np.argsort(hours[amp == ch])] for ch in np.unique(amp)}
    pos = {i: (ch, k) for ch, idx in order.items() for k, i in enumerate(idx)}

    def shifted_ref(rb, dist):
        ch, k = pos[rb]
        return order[ch][max(0, k - dist)]

    for cell in sc.CELLS:
        print(f"\n=== {sc.CELL_NAME[cell]} ===")
        res = {}
        for kind in ("B - A", "A, B, B - A"):
            for tset in ("current (within + neighbours, raw)", "within + ALL pairs, drift-corrected"):
                pn = np.full((len(P["A"]), 3), np.nan)         # predictions for the scorecard pairs
                pa = np.full((len(allp), 3), np.nan)            # predictions for every pair
                pw = {dist: np.full((len(within), 3), np.nan) for dist in (0, 3, 6)}
                for k in range(sc.N_BLOCKS):
                    for a in (np.unique(amp) if cell == "new" else [None]):
                        keep = (lambda r: amp[r] != a) if a is not None else (lambda r: np.ones(len(r), bool))
                        trc = np.flatnonzero((blk[P["ra"]] != k) & (blk[P["rb"]] != k) & keep(P["ra"]))
                        if tset.startswith("current"):
                            FA, FB, YY = F[P["A"][trc]], F[P["B"][trc]], P["Y"][trc]
                        else:
                            wtr = trc[P["type"][trc] == 0]
                            tra = np.flatnonzero((blk[allp[:, 0]] != k) & (blk[allp[:, 1]] != k) & keep(allp[:, 0]))
                            FA = np.vstack([F[P["A"][wtr]], F[allp[tra, 0]]])
                            FB = np.vstack([F[P["B"][wtr]], F[allp[tra, 1]]])
                            YY = np.vstack([P["Y"][wtr], allY[tra]])
                        pred = fit(kind, FA, FB, YY)
                        sel_a = (lambda r: amp[r] == a) if a is not None else (lambda r: np.ones(len(r), bool))
                        te = np.flatnonzero((blk[P["ra"]] == k) & (blk[P["rb"]] == k) & sel_a(P["ra"]))
                        if len(te):
                            pn[te] = pred(F[P["A"][te]], F[P["B"][te]])
                        ta = np.flatnonzero((blk[allp[:, 0]] == k) & (blk[allp[:, 1]] == k) & sel_a(allp[:, 0]))
                        if len(ta):
                            pa[ta] = pred(F[allp[ta, 0]], F[allp[ta, 1]])
                        tw = np.flatnonzero((blk[P["rb"][within]] == k) & sel_a(P["rb"][within]))
                        for dist in pw:
                            refs = np.array([n + shifted_ref(P["rb"][within[i]], dist) for i in tw], int)
                            if len(tw):
                                pw[dist][tw] = pred(F[refs], F[P["B"][within[tw]]])
                nb = P["type"] == 1
                okn, oka = np.isfinite(pn[nb, 0]), np.isfinite(pa[:, 0])
                print(f"{kind:12s} | {tset:38s} | n->n+1 (raw): {score(P['Y'][nb][okn], pn[nb][okn])}")
                print(f"{'':12s} | {'':38s} | any test pair (corrected, {oka.sum()}): {score(allY[oka], pa[oka])}")
                res[(kind, tset)] = pw
        # 2. shared references (the combined model trained on everything)
        print(f"\n  within-packet changes with the reference of the packet d earlier ({sc.CELL_NAME[cell]}):")
        yw = P["Y"][within]
        pk = d.packet_id.to_numpy()[P["rb"][within]]
        common = sorted(set.intersection(*[set(pk[amp[P["rb"][within]] == a]) for a in np.unique(amp)]))
        for (kind, tset), pw in res.items():
            for dist, p in pw.items():
                okw = np.isfinite(p[:, 0])
                rho = [spearmanr(yw[okw, i], p[okw, i])[0] for i in range(2)]
                # ranking: board mean of the predicted setting effect, J(0.5), regret of the best of 5
                idx = [np.array([np.flatnonzero((pk == q) & (amp[P["rb"][within]] == a))[0] for q in common]) for a in np.unique(amp)]
                tj = np.mean([sl.objective(yw[i, 0], yw[i, 1], 0.5) for i in idx], 0)
                ps, pl = np.mean([p[i, 0] for i in idx], 0), np.mean([p[i, 1] for i in idx], 0)
                top = sl.order(ps, pl, 0.5)[:5]
                print(f"  {kind:12s} {tset[:7]:8s} d={dist}: {score(yw[okw], p[okw])}   Spearman sigma_e {rho[0]:.2f}, "
                      f"lambda {rho[1]:.2f}; regret of the best of 5 (w=0.5) {100 * (tj[top].min() - tj.min()):.1f} %")


if __name__ == "__main__":
    main()
