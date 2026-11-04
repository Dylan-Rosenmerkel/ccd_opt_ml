"""
rank_methods.py -- the funnel's two rankings side by side, on the first night (board 101 ch0/1/3; the non-nominal
packets play the screens; truth = each setting's effect, log y minus the nominal label interpolated in time, averaged
over the 3 amplifiers because a setting is for the whole board):

WITH REFERENCE (each screen has its own nominal trace minutes earlier):
  * change model: predicted change reference -> new trace;
  * one-trace model: f(new) - f(reference).
WITHOUT REFERENCE (the screens' new traces only):
  * one-trace model: f(new);
  * screen vs screen: the change model compares every two screens of an amplifier (pair (i, j) by a model trained
    without both their time blocks), then a sorting method turns the comparisons into one order (lvl1/shortlist.py:
    mean change, wins, Bradley-Terry).
Change model: boosted trees on [A, B, B - A], trained on the within-packet pairs and on every pair of packets of an
amplifier with drift-corrected labels ("random pairs"), both orders.  One-trace model: boosted trees, trace features ->
setting effect.  Trained-amplifier and new-amplifier cells.  Score: regret of the best of the 5 shortlisted settings
under J(w) (w = 1, 0.5, 0) and the Spearman correlation with the truth.

    .venv/bin/python lvl1/rank_methods.py [--pairs neighbours]      (default: random drift-corrected pairs)
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

import change_test as ct  # noqa: E402
import pairs_extra as pe  # noqa: E402
import scorecard as sc  # noqa: E402
import shortlist as sl  # noqa: E402
import train_lvl1 as tl  # noqa: E402

WEIGHTS = (1.0, 0.5, 0.0)


def gbt(X, Y):
    """One boosted-tree model per target (no quantile models: faster than models.GBT); returns predict(Z)."""
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    xtr, _, _ = tl.prep_columns(X, X[:1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = [H(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, random_state=0).fit(xtr, Y[:, k])
              for k in range(Y.shape[1])]
    return lambda Z: np.column_stack([m.predict(tl.prep_columns(X, Z)[1]) for m in ms])


def change_model(FA, FB, Y):
    X, Xs = np.hstack([FA, FB, FB - FA]), np.hstack([FB, FA, FA - FB])
    f = gbt(np.vstack([X, Xs]), np.vstack([Y, -Y]))
    return lambda GA, GB: f(np.hstack([GA, GB, GB - GA]))


def main():
    files, rows, d = sc.load((0, 1, 3))
    Y = files["physics"]["Y"][rows]
    TR, P = sc.trace_table(files, rows, d), sc.make_pairs(d, Y)
    n = len(d)
    F = TR["F"]                                              # own: 0..n-1, reference: n..2n-1
    eff = pe.setting_effect(d, Y)[:, :2]
    amp, blk = d.channel.to_numpy(), d.block.to_numpy()
    amps = np.unique(amp)
    nominal = d.packet_id.isin(ct.NOMINAL).to_numpy()
    allp = np.array([(i, j) for ch in amps for i, j in combinations(np.flatnonzero(amp == ch), 2)])
    allY = eff[allp[:, 1]] - eff[allp[:, 0]]
    ok = np.isfinite(allY).all(1)
    allp, allY = allp[ok], allY[ok]
    within = np.flatnonzero(P["type"] == 0)                  # reference (n + r) -> own (r)
    wr = P["rb"][within]
    screens = ~nominal & np.isfinite(eff).all(1)
    common = sorted(set.intersection(*[set(d.packet_id[screens & (amp == a)]) for a in amps]))
    key = {(p, c): i for i, (p, c) in enumerate(zip(d.packet_id, d.channel))}
    idx = [np.array([key[(q, a)] for q in common]) for a in amps]
    truth = np.mean([eff[i] for i in idx], 0)

    nbr = np.array([(i, j) for ch in amps for step in (1, 2)
                    for i, j in zip(*(lambda r: (r[:-step], r[step:]))(np.flatnonzero(amp == ch)[np.argsort(d.hours.to_numpy()[amp == ch])]))])
    nbrY = Y[nbr[:, 1], :2] - Y[nbr[:, 0], :2]
    PAIRS, PY = (nbr, nbrY) if "--pairs" in sys.argv and sys.argv[sys.argv.index("--pairs") + 1] == "neighbours" else (allp, allY)
    print(f"training pairs: {'neighbours' if PAIRS is nbr else 'random (drift-corrected)'}", flush=True)

    def training(excl_blocks, excl_amp):
        """Within-packet pairs and the other pairs (random or neighbours) with no packet in excl_blocks (and not excl_amp)."""
        keep_w = ~np.isin(blk[wr], excl_blocks) & ((amp[wr] != excl_amp) if excl_amp is not None else True)
        keep_a = (~np.isin(blk[PAIRS[:, 0]], excl_blocks) & ~np.isin(blk[PAIRS[:, 1]], excl_blocks)
                  & ((amp[PAIRS[:, 0]] != excl_amp) if excl_amp is not None else True))
        w = within[keep_w]
        FA = np.vstack([F[P["A"][w]], F[PAIRS[keep_a, 0]]])
        FB = np.vstack([F[P["B"][w]], F[PAIRS[keep_a, 1]]])
        return FA, FB, np.vstack([P["Y"][w, :2], PY[keep_a]])

    for cell in sc.CELLS:
        groups = [None] if cell == "trained" else list(amps)
        S = {k: np.full((n, 2), np.nan) for k in ("with reference: change model", "with reference: one trace",
                                                  "without reference: one trace")}
        D = {a: np.full((n, n, 2), np.nan) for a in amps}     # pairwise predicted change i -> j, per amplifier
        for a in groups:
            on = (amp == a) if a is not None else np.ones(n, bool)
            for k in range(sc.N_BLOCKS):
                te = np.flatnonzero((blk == k) & on & screens)
                if not len(te):
                    continue
                cm = change_model(*training([k], a))
                S["with reference: change model"][te] = cm(F[n + te], F[te])
                tr = np.flatnonzero((blk != k) & ((amp != a) if a is not None else True) & np.isfinite(eff).all(1))
                f1 = gbt(F[tr], eff[tr])
                S["with reference: one trace"][te] = f1(F[te]) - f1(F[n + te])
                S["without reference: one trace"][te] = f1(F[te])
            for k1, k2 in combinations_with_replacement(range(sc.N_BLOCKS), 2):
                cm = change_model(*training([k1, k2], a))
                for ch in ([a] if a is not None else amps):
                    s1 = np.flatnonzero((blk == k1) & (amp == ch) & screens)
                    s2 = np.flatnonzero((blk == k2) & (amp == ch) & screens)
                    pr = [(i, j) for i in s1 for j in s2 if i != j]
                    if not pr:
                        continue
                    pi, pj = np.array(pr).T
                    D[ch][pi, pj] = cm(F[pi], F[pj])
                    D[ch][pj, pi] = cm(F[pj], F[pi])
            print(f"  {cell}: {a if a is not None else 'all amplifiers'} done", flush=True)

        print(f"\n=== {sc.CELL_NAME[cell]}: {len(common)} settings. Regret of the best of the 5 shortlisted (%) / "
              f"Spearman with the truth, per weight sigma_e:lambda")
        print(f"{'':52s}" + "".join(f"{sl.weight_label(w):>16s}" for w in WEIGHTS))
        rng = np.random.default_rng(0)
        rand = []
        for w in WEIGHTS:
            tj = sl.objective(truth[:, 0], truth[:, 1], w)
            rand.append(f"{100 * np.mean([tj[rng.choice(len(tj), 5, replace=False)].min() - tj.min() for _ in range(4000)]):8.1f} /  -  ")
        print(f"{'random':52s}" + "".join(f"{r:>16s}" for r in rand))

        def line(name, board_scores):
            out = []
            for w in WEIGHTS:
                tj = sl.objective(truth[:, 0], truth[:, 1], w)
                pj = board_scores(w)
                top = np.argsort(pj, kind="stable")[:5]
                out.append(f"{100 * (tj[top].min() - tj.min()):8.1f} / {spearmanr(tj, pj)[0]:4.2f}")
            print(f"{name:52s}" + "".join(f"{o:>16s}" for o in out))

        for name, s in S.items():
            line(name, lambda w, s=s: np.mean([sl.objective(s[i, 0], s[i, 1], w) for i in idx], 0))
        for sort in sl.SORTS:
            def board(w, sort=sort):
                per_amp = []
                for a, i in zip(amps, idx):
                    Dj = sl.objective(D[a][:, :, 0], D[a][:, :, 1], w)[np.ix_(i, i)]
                    per_amp.append(sl.sort_scores(Dj, sort))
                return np.mean(per_amp, 0)
            line(f"without reference: screen vs screen, {sort}", board)


if __name__ == "__main__":
    main()
