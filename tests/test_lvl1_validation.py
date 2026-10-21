"""
Tests of the level-1 validation scores (src/ml/evaluate.py, lvl1/train_lvl1.py): CRPS, packet bootstrap, paired
differences, setting groups, the held-out session.

    .venv/bin/python -m unittest tests/test_lvl1_validation.py        (or: .venv/bin/python -m pytest tests)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

from ml.evaluate import bootstrap_weights, crps_gaussian, grouped_folds, percentile_interval, setting_groups  # noqa: E402
import train_lvl1 as tl  # noqa: E402


def fake_predictions(n_packets: int, err: float = 0.1, seed: int = 0, n_amp: int = 4):
    """Y, mean, std, amp, packet ids for n_packets packets of n_amp amplifiers, with relative errors ~ err."""
    rng = np.random.default_rng(seed)
    n = n_packets * n_amp
    packets = np.repeat([f"p{i:03d}" for i in range(n_packets)], n_amp)
    amp = np.tile([f"101_ch{a}" for a in range(n_amp)], n_packets)
    Y = rng.normal(0, 1, (n, 3))
    mean = Y + rng.normal(0, err, (n, 3))
    std = np.full((n, 3), err)
    return Y, mean, std, amp, packets


class TestCRPS(unittest.TestCase):
    def test_against_numerical_integral(self):
        # CRPS(F, y) = integral (F(x) - 1{x >= y})^2 dx
        for mu, sigma, y in [(0.0, 1.0, 0.0), (0.3, 0.2, 0.1), (-1.0, 2.0, 3.0), (0.0, 0.05, -0.02)]:
            F = lambda x: norm.cdf(x, mu, sigma)
            lo, hi = mu - 12 * sigma - abs(y), mu + 12 * sigma + abs(y)
            num = integrate.quad(lambda x: F(x) ** 2, lo, y, limit=200)[0] + integrate.quad(lambda x: (1 - F(x)) ** 2, y, hi, limit=200)[0]
            self.assertAlmostEqual(float(crps_gaussian(y, mu, sigma)), num, places=6)

    def test_closed_form_at_the_mean(self):
        # z = 0: CRPS = sigma (sqrt(2) - 1) / sqrt(pi)
        for sigma in (0.01, 0.5, 3.0):
            self.assertAlmostEqual(float(crps_gaussian(1.0, 1.0, sigma)), sigma * (np.sqrt(2) - 1) / np.sqrt(np.pi), places=12)

    def test_sharp_limit_and_propriety(self):
        self.assertAlmostEqual(float(crps_gaussian(0.3, 0.0, 1e-9)), 0.3, places=6)      # -> |y - mu|
        rng = np.random.default_rng(1)
        y = rng.normal(0, 0.1, 20000)                                                  # truth ~ N(0, 0.1^2)
        honest, over, under = (crps_gaussian(y, 0, s).mean() for s in (0.1, 0.3, 0.03))
        self.assertLess(honest, over)
        self.assertLess(honest, under)


class TestBootstrap(unittest.TestCase):
    def test_weights(self):
        W = bootstrap_weights(np.repeat(["a", "b", "c"], 4), 50, seed=0)
        self.assertEqual(W.shape, (50, 12))
        np.testing.assert_allclose(W.sum(1), 12)                       # 3 packets drawn, 4 rows each
        for p in range(3):                                             # rows of one packet share a weight
            self.assertTrue((W[:, 4 * p:4 * p + 4] == W[:, [4 * p]]).all())

    def test_unit_weights_reproduce_the_scores(self):
        Y, m, s, amp, packets = fake_predictions(20)
        folds = grouped_folds(packets, 5)
        full = tl.scores(Y, np.full_like(Y, 0.01), m, s, amp, folds)
        b = tl.bootstrap_scores(Y, m, s, amp, folds, np.ones((1, len(Y))))
        for t, name in enumerate(tl.TARGETS):
            for key in ("error", "shape", "rank", "calibrated_coverage68", "crps", "worst_amp_calibrated_coverage68"):
                self.assertAlmostEqual(b[t][key][0], full[name][key], places=10, msg=key)

    def test_weighted_spearman_equals_spearman_on_the_resample(self):
        rng = np.random.default_rng(3)
        x, y = rng.normal(size=12), rng.normal(size=12)
        W = rng.integers(0, 3, (20, 12)).astype(float)
        rho = tl.weighted_spearman(x, y, W)
        for b in range(20):
            idx = np.repeat(np.arange(12), W[b].astype(int))
            if len(np.unique(idx)) >= 4:
                self.assertAlmostEqual(rho[b], spearmanr(x[idx], y[idx])[0], places=10)

    def test_intervals_shrink_with_more_packets(self):
        widths = {}
        for n_packets in (10, 40, 160):
            w = []
            for seed in range(5):
                Y, m, s, amp, packets = fake_predictions(n_packets, seed=seed)
                b = tl.bootstrap_scores(Y, m, s, amp, grouped_folds(packets, 5), bootstrap_weights(packets, 500, seed))
                iv = percentile_interval(b[0]["error"])["95"]
                w.append(iv[1] - iv[0])
            widths[n_packets] = np.mean(w)
        self.assertGreater(widths[10], widths[40])
        self.assertGreater(widths[40], widths[160])
        self.assertAlmostEqual(widths[10] / widths[160], 4, delta=1.5)          # ~ 1 / sqrt(n)

    def test_paired_difference(self):
        Y, m, s, amp, packets = fake_predictions(40, err=0.1)
        worse = Y + np.random.default_rng(9).normal(0, 0.2, Y.shape)
        W = bootstrap_weights(packets, 500)
        p = tl.paired_vs({"ref": (m, s), "worse": (worse, s), "same": (m.copy(), s)}, Y, "ref", W)
        d = p["worse"]["sigma_e"]
        self.assertGreater(d["delta_error"], 0.05)
        self.assertGreater(d["delta_error_interval"]["95"][0], 0)               # clearly worse
        self.assertEqual(d["p_better"], 0.0)
        self.assertAlmostEqual(p["same"]["sigma_e"]["delta_error"], 0.0)
        self.assertTrue(p["same"]["sigma_e"]["identical"])
        self.assertIsNone(p["same"]["sigma_e"]["p_better"])
        self.assertFalse(d["identical"])
        self.assertNotIn("ref", p)


class TestSettingGroups(unittest.TestCase):
    def test_identical_settings_share_a_group(self):
        nominal = [10.0, -2.0, 1000.0]
        theta = np.array([nominal, [10.0, -2.0, 300.0], nominal, [10.5, -2.0, 1000.0], [10.0, -2.0, 300.0],
                          [10.0, -2.0, 1000.0 + 1e-7], [11.0, -1.5, 1500.0]])
        pid = np.array(["L1_01", "L1_04", "L1_07", "L1_15", "L1_13", "L1_14", "L1_33"])
        rows_theta, rows_pid = np.repeat(theta, 4, 0), np.repeat(pid, 4)            # 4 amplifiers per packet
        g = setting_groups(rows_theta, rows_pid)
        group = dict(zip(rows_pid, g))
        self.assertEqual(group["L1_01"], "L1_01+L1_07+L1_14")
        self.assertEqual(group["L1_07"], group["L1_01"])
        self.assertEqual(group["L1_13"], "L1_04+L1_13")
        self.assertEqual(group["L1_15"], "L1_15")
        self.assertEqual(len(np.unique(g)), 4)
        # no fold splits a group: a test packet's twin is never in training
        for seed in range(5):
            for test in grouped_folds(g, 3, seed):
                train = np.setdiff1d(np.arange(len(g)), test)
                self.assertFalse(set(g[test]) & set(g[train]))

    def test_tolerance_and_nan(self):
        theta = np.array([[0.0, np.nan], [0.0, np.nan], [0.0, 1.0], [0.004, np.nan], [1.0, np.nan]])
        g = setting_groups(theta, np.array(["a", "b", "c", "d", "e"]), tol=0.01)
        self.assertEqual(list(g), ["a+b+d", "a+b+d", "c", "a+b+d", "e"])          # NaN = NaN; NaN != 1; 0.004 < 1 % of 1
        g = setting_groups(theta, np.array(["a", "b", "c", "d", "e"]), tol=0.001)
        self.assertEqual(g[3], "d")


class TestHoldout(unittest.TestCase):
    def test_session_c_of_the_plan(self):
        ids = tl.session_packets(ROOT / "lvl1" / "packet_plan" / "plan.json", "C")
        self.assertEqual(ids, [f"L1_{i}" for i in range(33, 44)])

    def test_matching(self):
        ids = ["L1_33", "syn_0001"]
        self.assertTrue(tl.matches_packet("L1_33", ids))
        self.assertTrue(tl.matches_packet("L1_33b", ids))
        self.assertTrue(tl.matches_packet("L1_33_retake", ids))
        self.assertFalse(tl.matches_packet("L1_330", ids))
        self.assertFalse(tl.matches_packet("syn_00010", ids))
        self.assertFalse(tl.matches_packet("L1_32", ids))


if __name__ == "__main__":
    unittest.main()
