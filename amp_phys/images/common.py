"""Shared helpers for amp_phys phase-3 image analysis.

- load(path): read a .fits.bz2 skipper image (decompressed in memory), return (header, [4 x 2D float32 arrays]);
  cached as .npz in a cache dir (default: $AMP_IMG_CACHE or the scratch directory) to avoid repeated bz2 decoding.
- ROOT: raw-data root (read-only).
- AMPS: the 5 working amplifiers (board, ext index) + dead controls.
"""
import bz2, io, os, glob
import numpy as np
from astropy.io import fits

ROOT = "/pool/storage/DAQ1_Data/run/PA_PQ_testing"
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "images"))
CACHE = os.environ.get("AMP_IMG_CACHE",
                       "/tmp/claude-313758494/-pool-homes-drosenm1/b851f3f2-6fdb-4a21-a197-260d9e3378ad/scratchpad/cache")
WORKING = [(101, 0), (101, 1), (101, 2), (101, 3), (102, 3)]
DEAD = [(102, 0), (102, 1), (102, 2)]
ALL = WORKING + DEAD
# from the header: ch0=CCD_D, ch1=CCD_C, ch2=CCD_A, ch3=CCD_B (same on both boards)
CCD_LABEL = {0: "D", 1: "C", 2: "A", 3: "B"}


# What the IMAGES show (2026-10 runs): on board 102 only ch2 (EXT3) carries CCD charge (tracks, single-e peaks,
# overscan step); ch0, ch1 and ch3 show no charge at all. This differs from the phase-1 trace label (102 ch3 working).
IMG_CHARGE = [(101, 0), (101, 1), (101, 2), (101, 3), (102, 2)]


def amp_name(board, ch):
    if (board, ch) in IMG_CHARGE:
        return f"{board} ch{ch}"
    return f"{board} ch{ch} (no charge)"


def find(run, pattern, board):
    fs = sorted(glob.glob(os.path.join(ROOT, run, f"avg_img_{pattern}_L2_{board}_*.fits.bz2")))
    return fs


def partner(path, board):
    """File of the other board taken at the same time (same timestamp)."""
    b = os.path.basename(path)
    parts = b.split("_L2_")
    stamp = "_".join(parts[1].split("_")[1:3])
    fs = glob.glob(os.path.join(os.path.dirname(path), parts[0] + f"_L2_{board}_{stamp}_*.fits.bz2"))
    return fs[0] if fs else None


def load(path, use_cache=True):
    key = os.path.join(CACHE, os.path.basename(path).replace(".fits.bz2", ".npz"))
    if use_cache and os.path.exists(key):
        z = np.load(key, allow_pickle=True)
        return fits.Header.fromstring(str(z["hdr"])), [z[f"e{i}"] for i in range(4)]
    with bz2.open(path) as f:
        h = fits.open(io.BytesIO(f.read()))
    hdr = h[0].header
    data = [np.asarray(h[i].data, dtype=np.float32) for i in range(1, 5)]
    if use_cache:
        os.makedirs(CACHE, exist_ok=True)
        np.savez(key, hdr=hdr.tostring(), **{f"e{i}": d for i, d in enumerate(data)})
    return hdr, data


def mad_std(x, axis=None):
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)
