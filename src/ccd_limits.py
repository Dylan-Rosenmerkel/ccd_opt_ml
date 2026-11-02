"""
ccd_limits.py -- safety limits and consistency checks for CCD operating parameters.

Every set of operating parameters (``theta``) that the acquisition scripts want to
send to the hardware must pass through :meth:`Limits.check` first.  Nothing in
this project talks to the ACM without that check, so a mistake in a scan
definition is caught on the computer instead of on the CCD.

What is checked
---------------
1. **Hardware range**  - the DAC / bias-potentiometer ranges taken from the ``acmpy``
   source (``control/daccontrol.py`` and ``control/biascontrol.py``).
2. **Allowed window**  - a per-parameter ``min`` / ``max`` read from a TOML file (see
   ``configs/limits.toml``).  A parameter with *no* matching entry is rejected: unknown
   knobs are never touched.  The optional ``max_step`` is not a rejection rule: voltage
   changes larger than ``max_step`` are applied as a ramp of smaller steps
   (:meth:`Limits.ramp_plan`).
3. **Relationships**   - rules that tie several parameters together (:func:`relational_constraints`),
   from the group's safe-operation rules (``docs/voltage_rules.txt``, 2026-10-14): clock HIGH >= LOW,
   VDD > VDRAIN, SW_H > OG_H, DG_H > VR, DG_L < VR, OG_L < VR, OG_L >= SW_L, V clock lows above TG,
   H clock lows below TG, H3 lows above SW_H, VDRAIN < DG_L, VDD below VR.  Rules that involve
   pins a parameter set does not contain are checked with the NOMINAL values of those pins (``[nominal]``
   in limits.toml, the module's .bcf), so a partial change cannot break a rule unnoticed.  (The rules file's
   line "DGL<=SWL" is a typo and is disregarded, as the group decided on 2026-10-14.)

The module only uses the Python standard library so that it runs on the DAQ
computer without extra installs.

Naming
------
Voltage keys follow the ``acmpy`` conventions, e.g. ``V1_A_H`` (single DAC pin) or
``V_H`` (group of pins), ``VSUB``, ``VDD`` (group: VDD1 + VDD2).  Sequencer keys
(``NDCM``, ``NPBIN``, ``delay_Ha`` ...) are integers written into the sequencer file.
"""

from __future__ import annotations

import fnmatch
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# --------------------------------------------------------------------------- #
# 1. Static knowledge about the ACM hardware (copied from acmpy)               #
# --------------------------------------------------------------------------- #

#: Sequencer constants that are plain integers (clock cycles, counts, seconds).
SEQUENCER_KEYS = frozenset(
    {
        "EXPOSURE", "NDCM", "NROW", "NCOL", "NPBIN", "NSBIN",
        "NDCMPRE", "NROWPRE", "NROWPOST", "NCOLPRE", "NCOLPOST",
        "BOOLCLR", "NCOLCLR", "NSBINCLR", "DUMPLESS",
        "delay_Va", "delay_Vb", "delay_Ha", "delay_Hb", "delay_Hc",
        "delayInt", "delayRGL", "delayRGH", "delaySWH", "delaySWL",
        "delayOGL", "delayOGH", "delayDGL", "delayDGH",
        "PP_SHIFT", "PP_NCYCLES", "NVWAIT", "NPSD",
    }
)

#: Names of the 40 DAC pins (``DACControl._pin_names``).
DAC_PINS = (
    "SW_1_H", "SW_1_L", "OG_1_H", "OG_1_L", "RG_1_H", "RG_1_L", "DG_1_H", "DG_1_L",
    "SW_2_H", "SW_2_L", "OG_2_H", "OG_2_L", "RG_2_H", "RG_2_L", "DG_2_H", "DG_2_L",
    "V1_B_H", "V1_B_L", "V2_C_H", "V2_C_L", "V3_B_H", "V3_B_L", "TG_B_H", "TG_B_L",
    "V1_A_H", "V1_A_L", "V3_A_H", "V3_A_L", "TG_A_H", "TG_A_L", "H1_B_H", "H1_B_L",
    "H2_C_H", "H2_C_L", "H3_B_H", "H3_B_L", "H1_A_H", "H1_A_L", "H3_A_H", "H3_A_L",
)

#: Bias circuits: name -> (v_ref, v_offset, R1 [ohm], Rb [ohm]); ``BiasControl`` in acmpy.
_BIAS_CIRCUIT = {
    "VSUB": (1.173, 0.0, 98.8e3, 1.33e3),
    "VDRAIN1": (-1.175, 0.0, 98.8e3, 4.32e3),
    "VDRAIN2": (-1.175, 0.0, 98.8e3, 4.32e3),
    "VDD1": (-1.175, 0.0, 98.8e3, 4.32e3),
    "VDD2": (-1.175, 0.0, 98.8e3, 4.32e3),
    "VR1": (-1.175, 2.0, 50.0e3, 4.00e3),
    "VR2": (-1.175, 2.0, 50.0e3, 4.00e3),
}
#: Names of the 7 bias outputs.
BIAS_PINS = tuple(_BIAS_CIRCUIT)
_POT_RAB = 20e3          # digital potentiometer end-to-end resistance
_POT_CODES = 1024        # 10-bit potentiometer
BIAS_GROUPS = {"VDD": ("VDD1", "VDD2"), "VDRAIN": ("VDRAIN1", "VDRAIN2"), "VR": ("VR1", "VR2")}

#: Clock rail offsets are assumed to stay at the reference value (-10 V).  The ACM manual (docs/Sequencer
#: Notes.pdf, A.3.4) gives the rails as offset .. offset + 20 V, i.e. [-10, +10] V for every rail; acmpy's
#: DAC code suggested gain 2 for the vertical pins ([-20, +20] V).  The narrower (manual) range is used.
_RAIL_HW_RANGE = {"normal": (-10.0, 10.0), "vertical": (-10.0, 10.0)}

#: Bias ranges of the ACM manual (Table A.1).  The potentiometer formula below gives nearly the same ends;
#: the intersection of both is used.
_MANUAL_BIAS_RANGE = {"VR": (-13.8, -2.0), "VDD": (-27.9, -6.0), "VDRAIN": (-27.9, -6.0), "VSUB": (6.7, 88.3)}


def bias_hw_range(name: str) -> tuple[float, float]:
    """Return the (min, max) voltage a bias circuit can physically produce."""
    v_ref, v_off, r1, rb = _BIAS_CIRCUIT[name]

    def volts(code: int) -> float:
        r_wb = code / _POT_CODES * _POT_RAB
        return v_ref * (r1 / (r_wb + rb) + 1.0) + v_off

    ends = (volts(0), volts(_POT_CODES - 1))
    low, high = min(ends), max(ends)
    manual = _MANUAL_BIAS_RANGE.get(name.rstrip("12"))
    if manual is not None:
        low, high = max(low, manual[0]), min(high, manual[1])
    return low, high


def rail_hw_range(pin: str) -> tuple[float, float]:
    """Return the (min, max) voltage a DAC clock pin can produce (gain 2 if 'V' in name)."""
    return _RAIL_HW_RANGE["vertical" if "V" in pin else "normal"]


def expand_key(key: str) -> list[str]:
    """
    Expand a voltage key into the primitive pins/biases it controls.

    Mirrors ``acmpy.control.Control._iter_targets``:

    * ``"V1_A_H"``           -> ``["V1_A_H"]``
    * ``"VDD"``              -> ``["VDD1", "VDD2"]``
    * ``"H1_H"``             -> ``["H1_A_H", "H1_B_H"]``  (group of one phase)
    * ``"V_L"`` / ``"H_H"``  -> all vertical / horizontal pins of that level

    Raises ``KeyError`` for names that are neither pins nor known groups.
    """
    key = key.upper()
    if key in DAC_PINS or key in _BIAS_CIRCUIT:
        return [key]
    if key in BIAS_GROUPS:
        return list(BIAS_GROUPS[key])
    if key in ("V_L", "V_H", "H_L", "H_H"):
        return [key.replace("_", "1_A_"), key.replace("_", "1_B_"),
                key.replace("_", "2_C_"), key.replace("_", "3_A_"),
                key.replace("_", "3_B_")]
    if len(key) == 4 and key[2] == "_" and key[3] in "HL" and key[:2] in (
            "SW", "OG", "RG", "DG", "V1", "V2", "V3", "H1", "H2", "H3", "TG"):
        prefix, level = key[:2], key[3]
        if prefix in ("H1", "H3", "V1", "V3", "TG"):
            return [f"{prefix}_A_{level}", f"{prefix}_B_{level}"]
        if prefix in ("H2", "V2"):
            return [f"{prefix}_C_{level}"]
        return [f"{prefix}_1_{level}", f"{prefix}_2_{level}"]
    raise KeyError(key)


def split_theta(theta: Mapping[str, float]) -> tuple[dict[str, int], dict[str, float]]:
    """Split ``theta`` into (sequencer constants, expanded voltage pins/biases).

    Voltage groups are expanded in dictionary order, so a later key overrides an
    earlier one -- the same order in which the keys are later sent to the ACM.
    Unknown names raise ``KeyError`` (they would be rejected by ``daq[...]`` anyway).
    """
    sequencer: dict[str, int] = {}
    voltages: dict[str, float] = {}
    for key, value in theta.items():
        if key in SEQUENCER_KEYS:
            sequencer[key] = value
        else:
            for pin in expand_key(key):
                voltages[pin] = float(value)
    return sequencer, voltages


# --------------------------------------------------------------------------- #
# 2. Relationships between parameters                                          #
# --------------------------------------------------------------------------- #

def relational_constraints(pins: Mapping[str, float], sequencer: Mapping[str, int],
                           nominal: Mapping[str, float] | None = None) -> list[str]:
    """
    Check the relationships *between* parameters; return a list of violation messages.

    The rules are the group's safe-operation rules (``docs/voltage_rules.txt``).  ``pins`` are the
    voltages of the parameter set; pins it does not contain take their ``nominal`` value (the module's
    .bcf), so every rule is checked on the complete configuration that will be on the board.

    Readout stage i (SW_i, OG_i, RG_i, DG_i) is paired with the biases VRi, VDDi, VDRAINi (i = 1, 2).
    """
    v = {**(nominal or {}), **pins}
    violations: list[str] = []

    def need(*names):
        return all(n in v for n in names)

    # every clock: HIGH >= LOW ("above"; equality allowed: the TGs run with HIGH = LOW nominally)
    for pin, value in v.items():
        if pin.endswith("_H"):
            low_pin = pin[:-2] + "_L"
            if low_pin in v and value < v[low_pin]:
                violations.append(f"{pin}={value} is below {low_pin}={v[low_pin]}")
    for i in ("1", "2"):
        sw_h, sw_l, og_h, og_l = f"SW_{i}_H", f"SW_{i}_L", f"OG_{i}_H", f"OG_{i}_L"
        dg_h, dg_l, vr, vdd, vdrain = f"DG_{i}_H", f"DG_{i}_L", f"VR{i}", f"VDD{i}", f"VDRAIN{i}"
        if need(vdd, vdrain) and not v[vdd] > v[vdrain]:
            violations.append(f"{vdd}={v[vdd]} must be above {vdrain}={v[vdrain]}")
        if need(sw_h, og_h) and not v[sw_h] > v[og_h]:
            violations.append(f"{sw_h}={v[sw_h]} must be above {og_h}={v[og_h]}")
        if need(dg_h, vr) and not v[dg_h] > v[vr]:
            violations.append(f"{dg_h}={v[dg_h]} must be above {vr}={v[vr]}")
        if need(dg_l, vr) and not v[dg_l] < v[vr]:
            violations.append(f"{dg_l}={v[dg_l]} must be below {vr}={v[vr]}")
        if need(og_l, vr) and not v[og_l] < v[vr]:
            violations.append(f"{og_l}={v[og_l]} must be below {vr}={v[vr]}")
        if need(og_l, sw_l) and not v[og_l] >= v[sw_l]:
            violations.append(f"{og_l}={v[og_l]} must be at or above {sw_l}={v[sw_l]}")
        if need(vdrain, dg_l) and not v[vdrain] < v[dg_l]:
            violations.append(f"{vdrain}={v[vdrain]} must be below {dg_l}={v[dg_l]}")
        if need(vdd, vr) and not v[vdd] < v[vr]:
            violations.append(f"{vdd}={v[vdd]} must be below {vr}={v[vr]} (VDD is generally much lower than VR)")
    tg = [p for p in v if p.startswith("TG_")]
    v_lows = [p for p in v if p[0] == "V" and p[1] in "123" and p.endswith("_L")]
    h_lows = [p for p in v if p[0] == "H" and p[1] in "123" and p.endswith("_L")]
    if tg:
        tg_max, tg_min = max(tg, key=v.get), min(tg, key=v.get)
        for p in v_lows:
            if not v[p] > v[tg_max]:
                violations.append(f"{p}={v[p]} must be above every TG level ({tg_max}={v[tg_max]})")
        for p in h_lows:
            if not v[p] < v[tg_min]:
                violations.append(f"{p}={v[p]} must be below every TG level ({tg_min}={v[tg_min]})")
    sw_highs = [p for p in ("SW_1_H", "SW_2_H") if p in v]
    for p in ("H3_A_L", "H3_B_L"):
        for q in sw_highs:
            if p in v and not v[p] > v[q]:
                violations.append(f"{p}={v[p]} must be above {q}={v[q]}")
    return violations


def relational_warnings(pins: Mapping[str, float], nominal: Mapping[str, float] | None = None) -> list[str]:
    """Written rules that are reported but not enforced.  None at present (kept for rules still to be confirmed)."""
    return []


# --------------------------------------------------------------------------- #
# 3. The user-editable allowed windows                                         #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Window:
    """Allowed range of one parameter (``None`` means "not restricted")."""
    pattern: str
    minimum: float | None = None
    maximum: float | None = None
    max_step: float | None = None   # largest voltage change applied in one go (ramped beyond)


class LimitsError(ValueError):
    """Raised when a parameter set violates the limits (message lists every violation)."""


class Limits:
    """Allowed windows loaded from a TOML file, plus the hardware/relationship checks."""

    def __init__(self, windows: list[Window], reviewed: bool, source: str = "<memory>",
                 nominal: Mapping[str, float] | None = None):
        self.windows = windows
        self.reviewed = reviewed
        self.source = source
        self.nominal = dict(nominal or {})          # pin -> nominal voltage, for the relationship rules

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "Limits":
        """Read a limits TOML file.  Table order matters: the first matching pattern wins."""
        path = Path(path)
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
        windows = [
            Window(pattern, entry.get("min"), entry.get("max"), entry.get("max_step"))
            for pattern, entry in data.get("limits", {}).items()
        ]
        nominal = {k: float(v) for k, v in data.get("nominal", {}).items()}
        return cls(windows, bool(data.get("meta", {}).get("reviewed", False)), str(path), nominal)

    def window_for(self, name: str) -> Window | None:
        """Return the first window whose glob pattern matches ``name``."""
        for window in self.windows:
            if fnmatch.fnmatchcase(name, window.pattern):
                return window
        return None

    # -- checking -----------------------------------------------------------
    def violations(self, theta: Mapping[str, float]) -> list[str]:
        """Return a list of human-readable violations (empty list = acceptable)."""
        try:
            sequencer, pins = split_theta(theta)
        except KeyError as err:
            return [f"unknown parameter name: {err}"]

        problems: list[str] = []

        # (a) hardware range of every voltage
        for name, value in pins.items():
            low, high = (bias_hw_range(name) if name in _BIAS_CIRCUIT else rail_hw_range(name))
            if not low <= value <= high:
                problems.append(f"{name}={value} V outside hardware range [{low:.2f}, {high:.2f}] V")

        # (b) integer-ness of sequencer constants
        for name, value in sequencer.items():
            if not float(value).is_integer() or value < 0:
                problems.append(f"{name}={value} must be a non-negative integer")

        # (c) allowed windows, for sequencer constants and voltages
        for name, value in {**sequencer, **pins}.items():
            window = self.window_for(name)
            if window is None:
                problems.append(f"{name}: no entry in {self.source} (unknown parameters are refused)")
                continue
            if window.minimum is not None and value < window.minimum:
                problems.append(f"{name}={value} below allowed minimum {window.minimum}")
            if window.maximum is not None and value > window.maximum:
                problems.append(f"{name}={value} above allowed maximum {window.maximum}")

        # (d) relationships between parameters (pins not in theta: their nominal values)
        problems += relational_constraints(pins, sequencer, self.nominal)
        return problems

    def warnings(self, theta: Mapping[str, float]) -> list[str]:
        """Written rules that are not enforced (see :func:`relational_warnings`); empty when all is fine."""
        try:
            _, pins = split_theta(theta)
        except KeyError:
            return []
        return relational_warnings(pins, self.nominal)

    def check(self, theta: Mapping[str, float]) -> None:
        """Raise :class:`LimitsError` if ``theta`` is not acceptable."""
        problems = self.violations(theta)
        if problems:
            raise LimitsError("parameter set rejected:\n  - " + "\n  - ".join(problems))

    # -- ramping -------------------------------------------------------------
    def ramp_plan(self, theta: Mapping[str, float],
                  theta_prev: Mapping[str, float] | None) -> list[dict[str, float]]:
        """
        Return the voltage settings to apply, in order, to go from ``theta_prev`` to ``theta``.

        Each entry maps primitive pins/biases to volts.  The change of every voltage is
        split into equal steps no larger than its ``max_step`` (all voltages move together,
        so the number of steps is set by the one that needs the most).  The last entry is
        always the target.  Without a previous state the target is applied directly.
        Intermediate points are checked against the limits by the caller.
        """
        _, target = split_theta(theta)
        if theta_prev is None:
            return [target]
        _, start = split_theta(theta_prev)

        n_steps = 1
        for name, value in target.items():
            window = self.window_for(name)
            if window and window.max_step and name in start:
                n_steps = max(n_steps, math.ceil(abs(value - start[name]) / window.max_step - 1e-9))

        plan = []
        for k in range(1, n_steps + 1):
            fraction = k / n_steps
            plan.append({name: (start[name] + fraction * (value - start[name]) if name in start else value)
                         for name, value in target.items()})
        return plan
