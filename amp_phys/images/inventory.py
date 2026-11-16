"""Inventory of skipper-CCD image files (avg_img_*.fits.bz2) in the PA_PQ_testing runs.

Reads only the primary header of each file (streaming bz2, no full decompression) and writes a CSV with
the acquisition settings (shape, skips, exposure, routine, pocket-pump settings, clock delays, voltages).

Usage:
    python inventory.py [--root /pool/storage/DAQ1_Data/run/PA_PQ_testing] [--out results/images/inventory.csv]
"""
import argparse, bz2, csv, glob, os, re, sys
from concurrent.futures import ProcessPoolExecutor
from astropy.io import fits

KEYS = ["ACMID", "ROUTINE", "SEQNAME", "NROW", "NCOL", "NDCM", "NPBIN", "NSBIN", "EXPOSURE", "NCOLPRE", "NROWPRE",
        "PP_NCYCLES", "PP_SHIFT", "DELAY_VA", "DELAY_VB", "DELAY_HA", "DELAY_HB", "DELAY_HC", "VERTICAL",
        "HORIZONTAL", "VSUB", "VR1", "VDD1", "VDRAIN1", "V1_A_H", "V1_A_L", "H1_A_H", "H1_A_L", "TG_A_H",
        "DATEINI", "DATEEND", "CH0", "CH1", "CH2", "CH3"]


def read_header(path, nbytes=40 * 2880):
    with bz2.open(path) as f:
        raw = f.read(nbytes)
    return fits.Header.fromstring(raw[: raw.find(b"END" + b" " * 77) + 80].decode("ascii", "replace"))


def one(path):
    try:
        h = read_header(path)
    except Exception as e:  # noqa
        return {"file": path, "error": str(e)}
    row = {"run": os.path.basename(os.path.dirname(path)), "file": os.path.basename(path)}
    for k in KEYS:
        row[k] = h.get(k, "")
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/pool/storage/DAQ1_Data/run/PA_PQ_testing")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/images/inventory.csv"))
    ap.add_argument("-j", type=int, default=48)
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.root, "*", "avg_img_*.fits.bz2")))
    with ProcessPoolExecutor(a.j) as ex:
        rows = list(ex.map(one, files, chunksize=4))
    keys = ["run", "file"] + KEYS + ["error"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} files -> {a.out}")


if __name__ == "__main__":
    main()
