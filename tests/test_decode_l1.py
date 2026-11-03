"""Tests of src/decode_l1.py: a written level-1 file reads back, a cut file reads to its last complete pixel, a damaged
pixel is caught, and (if the data are present) the real 2026-10-06 1x1 level-1 file passes every layout check."""

from __future__ import annotations

import glob
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decode_l1 import EPIX, read_l1, write_l1  # noqa: E402


class TestDecodeL1(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        self.truth = rng.normal(0, 5.0, (3 * 20, 30)) + rng.poisson(0.3, (60, 1)) * 1.33
        self.path = Path(self.tmp.name) / "s_3x20x30_1x1_L1_101_20261104_000000_0.RJ45ADCDATA0"
        write_l1(self.path, self.truth, ncol=20)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        im = read_l1(self.path)                      # geometry from the file name
        self.assertEqual((im.n_pixels, im.complete_rows, im.ncol, im.nskip), (60, 3, 20, 30))
        self.assertEqual(im.problems, [])
        self.assertLess(np.abs(im.values - self.truth).max(), 1 / 150 + 1e-9)
        np.testing.assert_array_equal(im.rows, np.repeat([0, 1, 2], 20))

    def test_cut_file(self):
        w = np.fromfile(self.path, dtype=">u4")
        pix = 4 * 30 + 2 + 13
        cut = Path(self.tmp.name) / "cut_3x20x30_1x1_L1_101_0.RJ45ADCDATA0"
        w[:4 + 1 + 20 * pix + 1 + 7 * pix + 50].tofile(cut)      # row 0, then 7 pixels and a bit of row 1
        im = read_l1(cut)
        self.assertEqual((im.complete_rows, im.n_pixels), (1, 27))

    def test_damaged_pixel(self):
        w = np.fromfile(self.path, dtype=">u4").astype(np.int64)
        pix = 4 * 30 + 2 + 13
        w[4 + 5 * pix + 3] += 1000                    # a signal sum of pixel 5: no longer equal to the trailer's sum
        bad = Path(self.tmp.name) / "bad_3x20x30_1x1_L1_101_0.RJ45ADCDATA0"
        w.astype(">u4").tofile(bad)
        im = read_l1(bad)
        self.assertEqual(im.n_pixels, 59)
        self.assertIn("row 0: 1 of 20 pixels fail", im.problems[0])

    def test_real_partial_file(self):
        files = sorted(glob.glob(str(ROOT / "data" / "2026-10-06-*" / "img_100x1000x3000_1x1_L1_101_*.RJ45ADCDATA0")))
        if not files:
            self.skipTest("2026-10-06 level-1 data not present")
        im = read_l1(files[0])
        self.assertEqual((im.n_pixels, im.n_samples, im.problems), (80, 150, []))
        v = im.values[:, 20:].mean(1)
        self.assertAlmostEqual(1.4826 * np.median(np.abs(v - np.median(v))), 0.102, places=2)


if __name__ == "__main__":
    unittest.main()
