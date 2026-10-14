"""
check_limits.py -- regression check of the safety limits (ccd_limits.py + configs/limits.toml).

The nominal configuration must pass; every rule of docs/voltage_rules.txt must catch a change that breaks
it; hardware ranges from the ACM manual must hold; timing constants must be accepted in wide windows.

    cd src && ../.venv/bin/python check_limits.py          # prints OK / FAIL per case, exits 1 on any failure
"""

from __future__ import annotations

import sys
from pathlib import Path

from ccd_limits import Limits, bias_hw_range

LIMITS = Limits.load(Path(__file__).resolve().parent.parent / "configs" / "limits.toml")
NOMINAL = dict(LIMITS.nominal)


def case(label: str, changes: dict, expect_ok: bool, expect_text: str = "") -> bool:
    theta = {**NOMINAL, **changes}
    problems = LIMITS.violations(theta)
    ok = (not problems) if expect_ok else any(expect_text in p for p in problems)
    print(f"{'OK  ' if ok else 'FAIL'} {label:62s} " + ("accepted" if not problems else f"refused: {problems[0][:90]}"))
    return ok


def main() -> None:
    results = [
        case("nominal configuration", {}, True),
        case("partial change only (VR) checked against nominal DG/OG", {"VR": -3.5}, True),
        case("HIGH >= LOW (TG_B_H below TG_B_L)", {"TG_B_H": 7.0}, False, "below TG_B_L"),
        case("VDD > VDRAIN", {"VDRAIN1": -19.5, "VDD1": -20.0}, False, "must be above VDRAIN1"),
        case("SW_H > OG_H", {"OG_1_H": 1.5}, False, "SW_1_H"),
        case("DG_H > VR", {"DG_2_H": -1.5, "VR2": -2.5, "OG_2_H": -1.6}, True),
        case("DG_H > VR (broken)", {"VR": -2.0, "DG_1_H": -2.0}, False, "must be above VR1"),
        case("DG_L < VR", {"DG_1_L": -3.0, "VR1": -3.5}, False, "must be below VR1"),
        case("OG_L < VR", {"OG_2_L": -2.5, "VR2": -3.0}, False, "OG_2_L"),
        case("OG_L >= SW_L", {"OG_1_L": -7.0}, False, "at or above SW_1_L"),
        case("V clock lows above TG", {"TG_A_H": 8.0, "TG_A_L": 8.0}, False, "must be above every TG level"),
        case("H clock lows below TG", {"H1_A_L": 7.5}, False, "must be below every TG level"),
        case("H3 lows above SW_H", {"SW_1_H": 2.5, "SW_2_H": 2.5}, True),
        case("VDRAIN < DG_L", {"VDRAIN2": -6.5, "DG_2_L": -7.0}, False, "must be below DG_2_L"),
        case("VDD below VR", {"VDD": -6.5, "VR": -7.0, "VDRAIN": -8.0}, False, "must be below VR1"),
        case("VSUB is unrestricted by the rules", {"VSUB": 55.0}, True),
        case("V clock high above the manual's +10 V", {"V1_A_H": 10.5}, False, "hardware range"),
        case("timing: delayInt 2000, delay_Ha 2400, delayRGH 1500", {"delayInt": 2000, "delay_Ha": 2400, "delayRGH": 1500}, True),
        case("timing: delayInt 0 (below 50)", {"delayInt": 0}, False, "below allowed minimum"),
        case("unknown parameter (OFS_CLK) is refused", {"OFS_CLK": -5}, False, "unknown"),
    ]
    warnings = LIMITS.warnings(NOMINAL)
    print(f"{'OK  ' if not warnings else 'FAIL'} {'no rule warnings at nominal':62s} {warnings}")
    results.append(not warnings)
    vr = bias_hw_range("VR1")
    results.append(abs(vr[1] - (-2.0)) < 1e-9)
    print(f"{'OK  ' if results[-1] else 'FAIL'} {'VR hardware range capped by the manual (max -2 V)':62s} {vr}")
    print("\nALL OK" if all(results) else "\nSOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
