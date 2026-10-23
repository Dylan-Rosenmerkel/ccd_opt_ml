"""
Tests of the tier-1/2 trace features (src/trace_features_extra.py) on synthetic signals with known answers, and an
end-to-end check on one real trace of 2026-10-12 against the pilot numbers of docs/literature/signal_analysis_features.md.

    .venv/bin/python -m unittest tests/test_trace_features_new.py
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import trace_features as tf  # noqa: E402
import trace_features_extra as tx  # noqa: E402

RNG = np.random.default_rng(1)
REAL = ROOT / "data" / "2026-10-12-PA03_PQ04_Packet_Tests" / "trace_2x5x500_20x20_ch0_101_20261012_185604_32.RJ45ADCDATA0"


class Registry(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(len(tf.names_by_tier((0,))), 85)
        self.assertGreaterEqual(len(tf.names_by_tier((1,))), 20)
        self.assertGreaterEqual(len(tf.names_by_tier((2,))), 20)
        for name, *_ , tier in tx.EXTRA:
            self.assertIn(name, tf.FEATURES)
            self.assertEqual(tf.FEATURES[name].tier, tier)


class JumpRobust(unittest.TestCase):
    def test_gaussian_has_no_jumps(self):
        e = RNG.normal(0, 3.0, 5000)
        self.assertAlmostEqual(math.sqrt(tx.med_rv(e) / len(e)), 3.0, delta=0.15)
        self.assertLess(max(0.0, 1 - tx.med_rv(e) / np.sum(e ** 2)), 0.05)

    def test_jumps_raise_the_share(self):
        e = RNG.normal(0, 1.0, 5000)
        e[RNG.choice(5000, 50, replace=False)] += RNG.choice([-15, 15], 50)
        share = 1 - tx.med_rv(e) / np.sum(e ** 2)
        self.assertGreater(share, 0.5)                      # jumps carry most of the variance (50 x 225 vs 5000)
        self.assertAlmostEqual(math.sqrt(tx.med_rv(e) / len(e)), 1.0, delta=0.15)
        self.assertGreater(tx.lee_mykland_count(e), 20)
        self.assertLess(tx.lee_mykland_count(RNG.normal(0, 1, 5000)), 15)


class Steps(unittest.TestCase):
    def test_window_steps(self):
        W = RNG.normal(0, 5.0, (400, 150))
        rate0, _ = tx.window_steps(W)
        self.assertLess(rate0, 1.0)
        W[:40, 70:] += 50.0                                  # 10 % of the windows get a 50-ADU telegraph step
        rate, amp = tx.window_steps(W)
        self.assertAlmostEqual(rate, 10.0, delta=1.5)
        self.assertAlmostEqual(amp, 50.0, delta=5.0)


class AveragingDeviations(unittest.TestCase):
    def test_white_noise_hadamard_and_allan(self):
        x = RNG.normal(0, 2.0, 20000)
        ns = np.array([1, 10, 50])
        had = tx.hadamard_deviation(x, ns)
        self.assertTrue(np.allclose(had, 2.0 / np.sqrt(ns), rtol=0.1))     # white noise: sigma / sqrt(n)
        mdev = tx.modified_allan(x, ns)
        # modified Allan of white noise falls faster than 1/sqrt(n) (~ n^-1.5 for phase-white; >= 1/sqrt(n) slope here)
        self.assertLess(mdev[2] / mdev[0], 1 / np.sqrt(50))

    def test_hadamard_ignores_linear_drift(self):
        x = RNG.normal(0, 1.0, 5000) + np.linspace(0, 50, 5000)
        self.assertAlmostEqual(tx.hadamard_deviation(x, [50])[0], 1 / np.sqrt(50), delta=0.05)


class Spectra(unittest.TestCase):
    def test_one_over_f_corner(self):
        f = np.geomspace(10, 15000, 40)
        W, A, g = 1.0, 1000.0, 1.0
        fc, gamma = tx.fit_one_over_f(f, W + A * f ** -g)
        self.assertAlmostEqual(fc, (A / W) ** (1 / g), delta=50)
        self.assertAlmostEqual(gamma, 1.0, delta=0.05)

    def test_lorentzian_corner(self):
        f = np.geomspace(10, 15000, 40)
        self.assertAlmostEqual(tx.fit_lorentzian(f, 1 + 50 / (1 + (f / 800.0) ** 2)), 800.0, delta=40)

    def test_multitaper_finds_a_sine(self):
        rate = 30690.0
        t = np.arange(490) / rate
        series = [RNG.normal(0, 5, 490) + 4.0 * np.cos(2 * np.pi * 2250 * t + 0.3) for _ in range(8)]
        freqs, F, mu, Fcrit = tx.multitaper_lines(series, rate)
        i = int(np.argmax(np.median(F, 0)))
        self.assertAlmostEqual(freqs[i], 2250, delta=rate / 490)
        self.assertAlmostEqual(np.median(2 * np.abs(mu[:, i])), 4.0, delta=1.0)
        self.assertTrue((F[:, i] > Fcrit).mean() >= 0.5)


class Transient(unittest.TestCase):
    def test_matrix_pencil_damped_oscillation(self):
        k = np.arange(30)
        y = 100 * 0.9 ** k * np.cos(2 * np.pi * 0.2 * k) + 40 * 0.6 ** k
        z, c = tx.matrix_pencil(y)
        osc = [q for q in z if q.imag > 1e-6]
        self.assertTrue(any(abs(np.angle(q) / (2 * np.pi) - 0.2) < 0.01 and abs(abs(q) - 0.9) < 0.01 for q in osc))
        self.assertTrue(any(abs(q.imag) < 1e-6 and abs(q.real - 0.6) < 0.01 for q in z))

    def test_slopes(self):
        W = np.outer(np.ones(3), np.arange(10.0)) * np.array([[1.0], [2.0], [-0.5]])
        self.assertTrue(np.allclose(tx.slopes(W), [1.0, 2.0, -0.5]))


class Complexity(unittest.TestCase):
    def test_entropies_and_dfa(self):
        white = RNG.normal(0, 1, 2000)
        walk = np.cumsum(white)
        self.assertGreater(tx.permutation_entropy(white), 0.98)
        self.assertLess(tx.permutation_entropy(walk), tx.permutation_entropy(white))
        self.assertAlmostEqual(tx.dfa_alpha(white), 0.5, delta=0.1)
        self.assertAlmostEqual(tx.dfa_alpha(walk), 1.5, delta=0.15)
        self.assertGreater(tx.sample_entropy(white[:500]), tx.sample_entropy(np.sin(np.arange(500) / 5)))
        self.assertAlmostEqual(tx.red_noise_beta(white), 1.0, delta=0.35)


@unittest.skipUnless(REAL.exists(), "real 2026-10-12 trace not available")
class RealTrace(unittest.TestCase):
    """The pilot numbers of the signal-analysis report on a good amplifier (ch0, reference)."""
    @classmethod
    def setUpClass(cls):
        cls.r = tf.trace_features(REAL)

    def test_no_errors(self):
        self.assertEqual({k: v for k, v in self.r.info["errors"].items() if k.startswith("extra")}, {})

    def test_start_transient(self):
        s = self.r.scalars
        self.assertTrue(-230 < s["start: CDS-slope lever arm (samples)"] < -195)
        self.assertGreater(s["start: slope-explained fraction"], 0.9)
        self.assertTrue(9 < s["start: transient time constant (us)"] < 16)
        self.assertTrue(400 < s["start: transient amplitude (ADU)"] < 900)
        self.assertTrue(0.8 < s["start: row-start/inner first-skip ratio"] < 1.05)     # NCOLPRE 0

    def test_good_amplifier_has_no_switching(self):
        s = self.r.scalars
        self.assertLess(s["discrete: in-window step rate (%)"], 0.5)
        self.assertTrue(abs(s["discrete: null-CDS kurtosis"]) < 1)
        self.assertTrue(6 < s["noise: null-CDS noise (ADU)"] < 9)

    def test_clock_regions(self):
        s = self.r.scalars                              # pilot (lvl1/feature_study/clock_pilot): -160, 40951, 3.08
        self.assertTrue(-250 < s["clock: V-block level slope (ADU/transfer)"] < -80)
        self.assertTrue(38000 < s["clock: V-transfer peak-to-peak (ADU)"] < 48000)
        self.assertTrue(2.5 < s["clock: H/V peak-to-peak ratio"] < 3.5)
        # since 2026-10-23 against the same block's settled second half (the reference trace's row-start block has 20
        # transfers: first five 126.6 kADU vs 126.8 kADU settled)
        self.assertTrue(-600 < s["clock: first H block of a row, excess (ADU)"] < 200)
        self.assertGreater(s["clock: leading V-block level slope (ADU/transfer)"], 0)   # the first V block climbs

    def test_prepulse_dip(self):
        s = self.r.scalars                              # pilot (lvl1/feature_study/prepulse_pilot): ch0 -886 ADU, 0.019
        self.assertTrue(-1000 < s["edge: pre-pulse dip depth (ADU)"] < -780)
        self.assertTrue(-0.025 < s["edge: pre-pulse dip / pulse height"] < -0.014)

    def test_curves(self):
        self.assertEqual(len(self.r.curves["start: mean transient"]), 30)
        self.assertEqual(len(self.r.curves["noise: Hadamard deviation"]), len(tf.ALLAN_N))


class MatchedTraceGeometry(unittest.TestCase):
    """The clock features at the binning of the matched traces (1x1, prescan 100; docs/packet_design.md section 6)."""
    def test_clock_features_at_1x1(self):
        import json, tempfile
        from simulate_trace import simulate_trace
        from decode_trace import decode_trace
        from trace_regions import split_regions
        with tempfile.TemporaryDirectory() as tmp:
            f = simulate_trace(Path(tmp) / "m", charge_per_pixel=0.0, NSBIN=1, NPBIN=1, NCOLPRE=100, NROW=2, NCOL=5, NDCM=100)
            t = decode_trace(f)
            m = json.loads(Path(str(f).rsplit(".", 1)[0] + ".meta").read_text())
            s = tf.trace_features(t, meta=m, regions=split_regions(t, m)).scalars
        for name in ("clock: V-transfer peak-to-peak (ADU)", "clock: H/V peak-to-peak ratio", "clock: first H block of a row, excess (ADU)"):
            self.assertTrue(math.isfinite(s[name]), name)
        self.assertTrue(math.isnan(s["clock: V-block level slope (ADU/transfer)"]))      # one V transfer per row


if __name__ == "__main__":
    unittest.main()
