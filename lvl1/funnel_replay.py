"""
funnel_replay.py -- the session-F funnel replayed on the first night, keeping everything for the figures of the
walkthrough deck (docs/overleaf_walkthrough_2026-11-05): the screen-vs-screen comparisons, the Bradley-Terry scores
(with the MM iterations), the true setting effects, and the sigma_e predictor's cross-validated predictions.

Same set-up as lvl1/rank_methods.py --target direction (random drift-corrected pairs): the 35 non-nominal settings of
board 101 ch0/1/3 play the screens; pair (i, j) is compared by a change model trained without both their time blocks
(and, for the new-amplifier cell, without the amplifier).

    .venv/bin/python lvl1/funnel_replay.py      -> processed/lvl1_results/2026-11-05_funnel_replay/replay.npz
"""

from __future__ import annotations

import sys
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))
sys.argv += ["--target", "direction"]                     # rank_methods reads its change-model target at import

import change_test as ct  # noqa: E402
import pairs_extra as pe  # noqa: E402
import rank_methods as rm  # noqa: E402
import scorecard as sc  # noqa: E402
import shortlist as sl  # noqa: E402

OUT = ROOT / "processed" / "lvl1_results" / "2026-11-05_funnel_replay"


def bt_history(D, n_iter=60):
    """Bradley-Terry scores (soft wins) after each MM iteration, as shortlist.sort_scores computes them."""
    M = sl.antisymmetric(D)
    beats = np.nan_to_num(np.clip(0.5 + M, 0, 1))
    played = np.isfinite(M).astype(float)
    W = beats.sum(1) + 0.5
    p, hist = np.ones(len(M)), []
    for _ in range(n_iter):
        p = W / ((played / (p[:, None] + p[None, :])).sum(1) + 1.0 / (p + 1.0))
        p /= np.exp(np.mean(np.log(p)))
        hist.append(-np.log(p))
    return np.array(hist)


def main():
    files, rows, d = sc.load((0, 1, 3))
    Y = files["physics"]["Y"][rows]
    TR, P = sc.trace_table(files, rows, d), sc.make_pairs(d, Y)
    n, F = len(d), TR["F"]
    eff = pe.setting_effect(d, Y)[:, :2]
    amp, blk = d.channel.to_numpy(), d.block.to_numpy()
    amps = np.unique(amp)
    nominal = d.packet_id.isin(ct.NOMINAL).to_numpy()
    allp = np.array([(i, j) for ch in amps for i, j in combinations(np.flatnonzero(amp == ch), 2)])
    allY = eff[allp[:, 1]] - eff[allp[:, 0]]
    ok = np.isfinite(allY).all(1)
    allp, allY = allp[ok], allY[ok]
    within = np.flatnonzero(P["type"] == 0)
    wr = P["rb"][within]
    screens = ~nominal & np.isfinite(eff).all(1)
    common = sorted(set.intersection(*[set(d.packet_id[screens & (amp == a)]) for a in amps]))
    key = {(p, c): i for i, (p, c) in enumerate(zip(d.packet_id, d.channel))}
    idx = [np.array([key[(q, a)] for q in common]) for a in amps]

    def training(excl_blocks, excl_amp):
        keep_w = ~np.isin(blk[wr], excl_blocks) & ((amp[wr] != excl_amp) if excl_amp is not None else True)
        keep_a = (~np.isin(blk[allp[:, 0]], excl_blocks) & ~np.isin(blk[allp[:, 1]], excl_blocks)
                  & ((amp[allp[:, 0]] != excl_amp) if excl_amp is not None else True))
        w = within[keep_w]
        FA = np.vstack([F[P["A"][w]], F[allp[keep_a, 0]]])
        FB = np.vstack([F[P["B"][w]], F[allp[keep_a, 1]]])
        return FA, FB, np.vstack([P["Y"][w, :2], allY[keep_a]])

    out = {"packets": np.array(common), "amps": amps, "truth_amp": np.stack([eff[i] for i in idx]),
           "knob": np.array([str(d.knob[i]) if "knob" in d else "" for i in idx[0]])}
    for cell in sc.CELLS:
        D = {a: np.full((n, n, 2), np.nan) for a in amps}
        for a in ([None] if cell == "trained" else list(amps)):
            for k1, k2 in combinations_with_replacement(range(sc.N_BLOCKS), 2):
                cm = rm.change_model(*training([k1, k2], a))
                for ch in ([a] if a is not None else amps):
                    s1 = np.flatnonzero((blk == k1) & (amp == ch) & screens)
                    s2 = np.flatnonzero((blk == k2) & (amp == ch) & screens)
                    pr = [(i, j) for i in s1 for j in s2 if i != j]
                    if not pr:
                        continue
                    pi, pj = np.array(pr).T
                    D[ch][pi, pj] = cm(F[pi], F[pj])
                    D[ch][pj, pi] = cm(F[pj], F[pi])
            print(f"{cell}: {a if a is not None else 'all amplifiers'} done", flush=True)
        Dm = np.stack([D[a][np.ix_(i, i)] for a, i in zip(amps, idx)])      # amp x screen x screen x target
        out[f"D_{cell}"] = Dm
        for w in (1.0, 0.0):
            Dj = sl.objective(Dm[..., 0], Dm[..., 1], w)
            out[f"bt_{cell}_w{w:g}"] = np.mean([sl.sort_scores(x, "bradley-terry", soft=True) for x in Dj], 0)
            out[f"bthist_{cell}_w{w:g}"] = np.mean([bt_history(x) for x in Dj], 0)

    # the sigma_e predictor attached to each pick: trees on the own trace + amplifier one-hot, 5 packet folds
    chans = list(amps)
    Xs = np.hstack([F[:n], np.eye(len(chans))[[chans.index(c) for c in amp]]])
    pk = d.packet_id.to_numpy()
    folds = np.searchsorted(np.unique(pk), pk) % 5
    cv = np.full(n, np.nan)
    for k in range(5):
        cv[folds == k] = rm.gbt(Xs[folds != k], Y[folds != k, :1])(Xs[folds == k])[:, 0]
    out.update(sigma_true=Y[:, 0], sigma_cv=cv, sigma_amp=amp, sigma_nominal=nominal)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "replay.npz", **out)
    print("saved", OUT / "replay.npz")


if __name__ == "__main__":
    main()
