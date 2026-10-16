"""
check_real_images.py -- how well do WADERs labels work on real images of the planned geometry?

For the averaged images (avg_*.fits[.bz2]) of one board in a run directory, independent of any packet:

1. per channel, WITHOUT WADERs: the read noise of the overscan (row by row), the row-to-row pedestal jumps, the width
   of the zero-electron peak in the active region, the baseline of the active region relative to the overscan along the
   row (8 blocks of columns), lambda row by row (image_rows.py) and its dark-current / CIC line;
2. run_WADERs on the images as they are, and on copies with the along-row baseline trend removed (per column, the
   median of the zero-electron pixels over the rows, smoothed over 25 columns; overscan untouched), both with
   --no-clusters --keep-dead;
3. for repeated images at the same settings: the scatter of sigma_e, gain and lambda between repeats against their
   quoted errors, and chi2/ndf -- the label noise a model would have to live with.

    python lvl1/check_real_images.py <run dir> [--board 101] [--out <dir>] [--run-waders <path to run_WADERs>]

First used on data/2026-10-16-PA03_PQ04_16rows (3 repeats at nominal, board 101): docs/context/2026-10-16_session_context.md.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import io
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import image_rows as ir  # noqa: E402

RUN_WADERS = Path.home() / "Documents" / "Software" / "comm_scripts" / "bin" / "run_WADERs"
rob = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))


def open_fits(path: Path):
    from astropy.io import fits
    data = path.read_bytes()
    return fits.open(io.BytesIO(bz2.decompress(data) if path.suffix == ".bz2" else data))


def n_active(header) -> int:
    return int(header.get("PHYSCOL", 6144) + header.get("PRESCAN", 8) - header.get("NCOLPRE", 0))


def column_trend(d: np.ndarray, na: int) -> np.ndarray:
    """The along-row baseline of the active region relative to each row's overscan pedestal (0 in the overscan)."""
    from scipy.ndimage import median_filter
    ped = np.median(d[:, na + 8:], 1)
    act = d - ped[:, None]
    z = np.where(np.abs(act) < 0.6, act, np.nan)
    trend = np.nan_to_num(np.nanmedian(z, 0))
    trend = median_filter(trend, 25, mode="nearest")
    trend[na:] = 0
    return trend


def run_waders(listfile: Path, out: Path, exe: Path) -> Path:
    subprocess.run([str(exe), str(listfile), "-o", str(out), "--no-clusters", "-j", "4", "--autoconfig-args=--keep-dead"],
                   cwd=out.parent, capture_output=True, text=True)
    return next(out.glob("**/fit_quality.csv"))


def repeat_table(csv_path: Path, board: int) -> list[str]:
    rows = [r for r in csv.DictReader(open(csv_path)) if f"_{board}_" in r["image"]]
    lines = ["| channel | images | σe [e-] | σe scatter (quoted) | gain scatter | λ [e-/pix] | λ scatter (quoted) | χ²/ndf |",
             "|---|---|---|---|---|---|---|---|"]
    for ext in ("EXT1", "EXT2", "EXT3", "EXT4"):
        rr = [r for r in rows if r["ext"] == ext]
        if not rr:
            continue
        f = lambda k: np.array([float(r[k]) for r in rr])
        n, g, lam, c = f("noise"), f("gain"), f("dc"), f("chi2ndf")
        sd = lambda v: f"{100 * np.std(v, ddof=1) / np.mean(v):.1f} %" if len(v) > 1 else "–"
        lines.append(f"| ch{int(ext[3]) - 1} | {len(rr)} | {np.mean(n):.4f} | {sd(n)} ({100 * np.mean(f('noise_err_s') / n):.1f} %) | {sd(g)} | "
                     f"{np.mean(lam):.4f} | {sd(lam)} ({100 * np.mean(f('dc_err_s') / lam):.1f} %) | {', '.join(f'{x:.1f}' for x in c)} |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--board", type=int, default=101)
    parser.add_argument("--out", default=None)
    parser.add_argument("--run-waders", default=str(RUN_WADERS))
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    out = Path(args.out or f"check_real_images_{run.name}").resolve()
    (out / "trend_removed").mkdir(parents=True, exist_ok=True)
    images = sorted(run.glob(f"avg_*_{args.board}_*.fits*"))
    report = [f"# Real-image check: {run.name}, board {args.board} ({len(images)} images)", "",
              "## Without WADERs", "",
              "| image | ch | overscan noise [ADU] | pedestal jumps between rows [ADU] | 0e- width, as is → trend removed [ADU] | "
              "active − overscan baseline along the row, first → last eighth [ADU] | λ first 4 / last 4 rows | DC, CIC [e-/pix] |",
              "|---|---|---|---|---|---|---|---|"]
    for p in images:
        h = open_fits(p)
        na = n_active(h[0].header)
        meta_path = run / (p.name[4:].split(".fits")[0] + ".meta")
        for ch in range(4):
            d = h[ch + 1].data.astype(float)
            ped = np.median(d[:, na + 8:], 1)
            act = d[:, :na] - ped[:, None]
            tr = column_trend(d, na)
            act2 = act - tr[None, :na]
            blocks = [np.median(b[np.abs(b) < 0.6]) for b in np.array_split(act, 8, axis=1)]
            g = 1.3
            s_adu = rob(act2[act2 < 0.65])
            lam = np.array([ir.row_lambda(r, g, s_adu)[0] for r in act2])
            err = np.array([ir.row_lambda(r, g, s_adu)[1] for r in act2])
            line = ir.line_fit(lam, err)
            report.append(f"| {p.name[-22:-9]} | {ch} | {np.median([rob(r) for r in d[:, na + 8:]]):.3f} | {np.std(ped):.3f} | "
                          f"{rob(act[act < 0.65]):.3f} → {s_adu:.3f} | {blocks[0]:+.2f} → {blocks[-1]:+.2f} | "
                          f"{lam[:4].mean():.3f} / {lam[-4:].mean():.3f} | {line.get('lambda_dark', np.nan):.3f} ± {line.get('lambda_dark_err', np.nan):.3f}, "
                          f"{line.get('lambda_cic', np.nan):.3f} ± {line.get('lambda_cic_err', np.nan):.3f} |")
            h[ch + 1].data = (d - tr[None, :]).astype(np.float32)
        h.writeto(out / "trend_removed" / (p.name.split(".fits")[0] + ".fits"), overwrite=True)
    report += ["", "(λ row by row here with a nominal gain of 1.3 ADU/e- and the zero-peak width after the trend is removed; "
               "the WADERs gain changes it by a few %.)", ""]
    lists = {"as is": out / "as_is.txt", "trend removed": out / "trend_removed.txt"}
    lists["as is"].write_text("".join(f"{p}\n" for p in images))
    lists["trend removed"].write_text("".join(f"{p}\n" for p in sorted((out / "trend_removed").glob("*.fits"))))
    for label, lst in lists.items():
        csv_path = run_waders(lst, out / f"waders_{label.replace(' ', '_')}", Path(args.run_waders))
        report += [f"## run_WADERs, {label} (repeats: scatter between images vs the quoted error)", ""] + repeat_table(csv_path, args.board) + [""]
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
