"""
relative_extra.py -- two follow-ups to the scorecard's relative models (board 101 ch0/1/3, same blocks and splits as
lvl1/scorecard.py):

1. COMBINED change models: boosted trees on [features(A), (features(B),) features(B) - features(A), (hand-picked
   difference-curve bands, dlog white noise)], trained on both orders (the swapped row exchanges A and B and negates the
   differences, the bands and the target).  Variants: A + (B - A); A + B + (B - A); each with and without the bands;
   and the scorecard's change trees (differences only), refitted here on the same splits.
2. Does it work for every kind of change?  Direction and size per kind of pair: within a packet = nominal vs one kind
   of knob (timing, output-stage voltages, clock rails, VSUB, all at once); between packets = which kinds are compared
   (e.g. "delayInt vs VR": both away from nominal, different knobs).  Also the settings-only reference.

    .venv/bin/python lvl1/relative_extra.py [--base processed/lvl1_results/2026-11-04_scorecard]
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

import change_handpicked as hp  # noqa: E402
import scorecard as sc  # noqa: E402

T = sc.T
CLEAR = dict(zip(T, sc.CLEAR))
GROUP = [("timing", ("delay",)), ("output stage (VDD, VR, RG, SW)", ("VDD", "VR", "RG", "SW", "OG", "DG")),
         ("clock rails (H, V, TG)", ("H1", "H2", "H3", "V1", "V2", "V3", "TG")), ("VSUB", ("VSUB",))]


def kind_of(changes: dict) -> str:
    kinds = {g for g, prefixes in GROUP for k in changes if k.startswith(prefixes)}
    if not kinds:
        return "nominal"
    return "all at once" if len(kinds) > 1 else kinds.pop()


def combined_X(TR, P, bands_fit, tr, te, with_bands: bool, with_b: bool = False):
    A, B = TR["F"][P["A"]], TR["F"][P["B"]]
    D = B - A
    parts_tr, parts_trs, parts_te = [A[tr], D[tr]], [B[tr], -D[tr]], [A[te], D[te]]
    if with_b:                                   # B's own features too (swapped: A's)
        parts_tr.insert(1, B[tr]), parts_trs.insert(1, A[tr]), parts_te.insert(1, B[te])
    if with_bands:
        Dc = TR["C"][P["B"]] - TR["C"][P["A"]]
        w = TR["white"][P["B"]] - TR["white"][P["A"]]
        for i in range(3):
            h = hp.Handpicked(6).fit(Dc[tr], P["Y"][tr, i])
            btr, bte = h.transform(Dc[tr]), h.transform(Dc[te])
            parts_tr.append(btr), parts_trs.append(-btr), parts_te.append(bte)
        parts_tr.append(w[tr, None]), parts_trs.append(-w[tr, None]), parts_te.append(w[te, None])
    Xtr = np.vstack([np.hstack(parts_tr), np.hstack(parts_trs)])
    return Xtr, np.vstack([P["Y"][tr], -P["Y"][tr]]), np.hstack(parts_te)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default=str(ROOT / "processed/lvl1_results/2026-11-04_scorecard"))
    args = parser.parse_args()
    files, rows, d = sc.load((0, 1, 3))
    Y = files["physics"]["Y"][rows]
    TR, P = sc.trace_table(files, rows, d), sc.make_pairs(d, Y)
    amp, blk = d.channel.to_numpy(), d.block.to_numpy()
    ba, bb, pamp = blk[P["ra"]], blk[P["rb"]], amp[P["ra"]]
    preds = {}
    for cell in sc.CELLS:
        splits = []
        for k in range(sc.N_BLOCKS):
            for a in (np.unique(amp) if cell == "new" else [None]):
                te = np.flatnonzero((ba == k) & (bb == k) & ((pamp == a) if a is not None else True))
                tr = np.flatnonzero((ba != k) & (bb != k) & ((pamp != a) if a is not None else True))
                if len(te) and len(tr):
                    splits.append((tr, te))
        for name in ("change trees (B - A)", "combined: A, B - A", "combined: A, B - A, hand-picked bands",
                     "combined: A, B, B - A", "combined: A, B, B - A, hand-picked bands"):
            pr = np.full(P["Y"].shape, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for tr, te in splits:
                    if name.startswith("change"):
                        X = TR["F"][P["B"]] - TR["F"][P["A"]]
                        pr[te] = sc.fit_predict("trees", np.vstack([X[tr], -X[tr]]), np.vstack([P["Y"][tr], -P["Y"][tr]]), X[te])
                    else:
                        Xtr, Ytr, Xte = combined_X(TR, P, None, tr, te, "bands" in name, "A, B," in name)
                        pr[te] = sc.fit_predict("trees", Xtr, Ytr, Xte)
            preds[(cell, name)] = pr
            print(f"{cell:8s} {name} done", flush=True)
    # the settings-only reference and the scorecard's best, from the saved scorecard run (same pairs, matched by key)
    base = pd.read_csv(Path(args.base) / "relative_predictions.csv")
    key = pd.DataFrame({"type": [sc.PAIR_TYPES[k] for k in P["type"]], "packet A": d.packet_id.to_numpy()[P["ra"]],
                        "packet B": d.packet_id.to_numpy()[P["rb"]], "channel": d.channel.to_numpy()[P["ra"]]})
    m = key.merge(base, on=["type", "packet A", "packet B", "channel"], how="left")
    for cell in sc.CELLS:
        ref = sc.REL_REF[cell]
        for name in (ref, "absolute trace: pretrained cnn, differenced", "absolute trace: trees, differenced"):
            if f"{cell} | {name} | sigma_e" in m:
                preds[(cell, name if name != ref else f"settings only ({ref})")] = m[[f"{cell} | {name} | {t}" for t in T]].to_numpy(float)

    plan = {p["packet_id"]: p for p in json.loads((ROOT / "lvl1/packet_plan/plan.json").read_text())["packets"]}
    kind = {p: kind_of(plan[p]["changes"]) for p in d.packet_id.unique()}
    ka, kb = np.array([kind[p] for p in d.packet_id.to_numpy()[P["ra"]]]), np.array([kind[p] for p in d.packet_id.to_numpy()[P["rb"]]])
    pair_kind = np.where(P["type"] == 0, "nominal vs " + kb,
                         np.where(ka == kb, "same kind: " + ka,
                                  np.where((ka == "nominal") | (kb == "nominal"), "nominal vs another packet's change",
                                           "two different kinds of change")))
    L = ["# Relative models: a combined model, and every kind of change", "",
         "Direction right on clear changes (%, number of clear changes) and size error (RMS %) per pair kind; board 101 "
         "ch0/1/3, same blocks and splits as the scorecard. Within-packet pairs are nominal -> one kind of change (their "
         "sigma_e changes almost all go up: the base rate is given); between-packet pairs (n -> n+1 and n -> n+2) compare "
         "two settings directly.", ""]
    groups = [("all within-packet pairs", P["type"] == 0), ("all between-packet pairs (n+1, n+2)", P["type"] > 0)]
    groups += [(g, pair_kind == g) for g in sorted(set(pair_kind)) if (pair_kind == g).sum() >= 6]
    for cell in sc.CELLS:
        L += [f"## {sc.CELL_NAME[cell]}", ""]
        names = [n for c, n in preds if c == cell]
        for gname, sel in groups:
            y = P["Y"][sel]
            ncl = {t: int((np.abs(y[:, i]) > CLEAR[t]).sum()) for i, t in enumerate(T)}
            up = {t: np.mean(y[np.abs(y[:, i]) > CLEAR[t], i] > 0) if ncl[t] else np.nan for i, t in enumerate(T)}
            L += [f"### {gname} ({int(sel.sum())} pairs; clear changes " + ", ".join(f"{t} {ncl[t]} ({100 * up[t]:.0f} % up)" for t in T) + ")", "",
                  "| model | " + " | ".join(f"{t} direction | {t} size" for t in T) + " |", "|---|" + "---|---|" * len(T)]
            for n in names:
                p = preds[(cell, n)][sel]
                ok = np.isfinite(p[:, 0])
                if ok.sum() < 3:
                    continue
                cells = []
                for i, t in enumerate(T):
                    dr = sc.direction(y[ok, i], p[ok, i], CLEAR[t])
                    cells.append(f"{100 * dr:.0f} % | {100 * sc.rms(p[ok, i] - y[ok, i]):.1f}" if np.isfinite(dr) else f"- | {100 * sc.rms(p[ok, i] - y[ok, i]):.1f}")
                L.append(f"| {n} | " + " | ".join(cells) + " |")
            L.append("")
    out = Path(args.base) / "relative_extra.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
