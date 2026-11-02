"""
Every packet must set every voltage and timing constant, so nothing a previous packet changed
survives into the next packet's reference traces, and every apply step must read back what it set.

Fake boards (src/fake_daq.py) keep every value until it is set again, like acmpy.
Run: python -m unittest tests/test_packet_reset.py
"""

from __future__ import annotations

import copy
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccd_limits import Limits  # noqa: E402
import scan_packets as sp  # noqa: E402
import take_packet as tp  # noqa: E402


def _setup(work: str) -> tp.DaqSetup:
    setup = tp.DaqSetup.load(ROOT / "configs" / "daq_setup.toml")
    setup.work_dir = work
    setup.fits_wait_s = 0.0           # the fake boards write no FITS files
    setup.poll_interval_s = 0.01
    return setup


class TestPacketReset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.setup = _setup(self.tmp.name)
        self.limits = Limits.load(ROOT / "configs" / "limits.toml")
        with open(ROOT / "configs" / "scan_recommended.toml", "rb") as handle:
            self.config = tomllib.load(handle)
        self.reference = tp.board_thetas(self.config["reference"], self.config.get("reference_board", {}),
                                         self.setup.theta_boards)
        # delayInt and RG_1_H are NOT in the scan's reference table
        self.assertNotIn("delayInt", self.reference[101])
        self.assertNotIn("RG_1_H", self.reference[101])
        self.change = {"delayInt": 1500, "RG_1_H": 8.5}

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, plan, controller=None):
        controller = controller or tp.make_controller("fake", self.setup, fake_repeat_flush=False)
        sp.run_scan(controller, plan, self.limits, self.config, self.setup, allow_unreviewed=True)
        return controller

    def _manifest(self, packet_id):
        import json
        return json.loads((Path(self.tmp.name) / "packets" / f"{packet_id}.json").read_text())

    def test_complete_theta(self):
        request = sp.make_request("t_0", sp.apply_change(self.reference, self.change, self.setup),
                                  self.config, self.setup)
        tp.plan_packet(request, self.setup, self.limits, None)
        for board in self.setup.theta_boards:
            for theta in (request.theta[board], request.theta_ref[board]):
                self.assertEqual(tp._primitive_keys(theta), set(tp.COMPLETE_KEYS))
            self.assertEqual(request.theta[board]["delayInt"], 1500)
            self.assertEqual(request.theta_ref[board]["delayInt"], tp.NOMINAL_TIMING["delayInt"])
            self.assertEqual(request.theta_ref[board]["RG_1_H"], self.limits.nominal["RG_1_H"])

    def test_next_reference_is_restored(self):
        changed = sp.apply_change(self.reference, self.change, self.setup)
        controller = self._run([("t_00001", changed), ("t_00002", copy.deepcopy(self.reference))])
        second = self._manifest("t_00002")
        self.assertEqual(second["status"], "ok")
        apply_ref = [s for s in second["steps"] if s["kind"] == "apply"][0]
        self.assertTrue(apply_ref["step"].endswith("apply_ref_acm101"))
        readback = apply_ref["receipts"]["101"]["readback"]
        self.assertEqual(readback["delayInt"], 1000)
        self.assertAlmostEqual(readback["RG_1_H"], 9.0)
        for board, fake in controller.fakes.items():
            self.assertEqual(fake.state["delayInt"], 1000)
            self.assertAlmostEqual(fake.state["RG_1_H"], 9.0)
            self.assertIn(("load_sequencer", None), fake.calls)

    def test_readback_mismatch_stops_the_packet(self):
        controller = tp.make_controller("fake", self.setup, fake_repeat_flush=False)
        controller.fakes[101].ignore_sets = {"delayInt"}
        controller.fakes[101].state["delayInt"] = 1000      # the board keeps its old value
        changed = sp.apply_change(self.reference, self.change, self.setup)
        with self.assertRaises(tp.PacketError) as caught:
            self._run([("t_00001", changed)], controller)
        self.assertIn("delayInt reads 1000, set 1500", str(caught.exception))
        self.assertEqual(self._manifest("t_00001")["status"], "failed")

    def test_readback_problems(self):
        expected = {"delayInt": 1000, "VSUB": 45.0, "H1_A_L": 6.7}
        self.assertEqual(tp.readback_problems(expected, {"delayInt": 1000, "VSUB": 45.095, "H1_A_L": [6.7, 6.7]}), [])
        problems = tp.readback_problems(expected, {"delayInt": 1500, "VSUB": 45.5, "H1_A_L": [6.7, 6.2]})
        self.assertEqual(len(problems), 3)
        self.assertEqual(tp.readback_problems(expected, None), ["the receipt has no readback"])


if __name__ == "__main__":
    unittest.main()
