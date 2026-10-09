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
3. **Relationships**   - rules that tie several parameters together, e.g. a clock
   high level must not be below its low level.  The physics relationships between
   biases (to be provided by the group) go into :func:`relational_constraints`.

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
        "PP_SHIFT", "PP_NCYCLES",
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
_POT_RAB = 20e3          # digital potentiometer end-to-end resistance
_POT_CODES = 1024        # 10-bit potentiometer
BIAS_GROUPS = {"VDD": ("VDD1", "VDD2"), "VDRAIN": ("VDRAIN1", "VDRAIN2"), "VR": ("VR1", "VR2")}

#: Clock rail offsets are assumed to stay at the reference value (-10 V).  With this
#: offset a DAC pin can reach [-10, +10] V, vertical pins (gain 2) [-20, +20] V.
_RAIL_HW_RANGE = {"normal": (-10.0, 10.0), "vertical": (-20.0, 20.0)}


def bias_hw_range(name: str) -> tuple[float, float]:
    """Return the (min, max) voltage a bias circuit can physically produce."""
    v_ref, v_off, r1, rb = _BIAS_CIRCUIT[name]

    def volts(code: int) -> float:
        r_wb = code / _POT_CODES * _POT_RAB
        return v_ref * (r1 / (r_wb + rb) + 1.0) + v_off

    ends = (volts(0), volts(_POT_CODES - 1))
    return min(ends), max(ends)


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

def relational_constraints(pins: Mapping[str, float], sequencer: Mapping[str, int]) -> list[str]:
    """
    Check relationships *between* parameters; return a list of violation messages.

    Built-in rules
    --------------
    * every clock rail pair must satisfy ``HIGH >= LOW``.

    PLACEHOLDER -- to be extended
    -----------------------------
    The CCD only works if certain biases and clock levels keep fixed relationships
    (e.g. between VSUB, the output-stage biases and the clock levels).  Those rules
    are *not* encoded yet: add them here once they have been written down from the
    manuals, one ``violations.append(...)`` per rule, so every acquisition is
    checked against them automatically.
    """
    violations: list[str] = []

    for pin, value in pins.items():
        if pin.endswith("_H"):
            low_pin = pin[:-2] + "_L"
            if low_pin in pins and value < pins[low_pin]:
                violations.append(f"{pin}={value} is below {low_pin}={pins[low_pin]}")

    # TODO(bias/clock relationships): add rules here when provided.

    return violations


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

    def __init__(self, windows: list[Window], reviewed: bool, source: str = "<memory>"):
        self.windows = windows
        self.reviewed = reviewed
        self.source = source

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
        return cls(windows, bool(data.get("meta", {}).get("reviewed", False)), str(path))

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

        # (d) relationships between parameters
        problems += relational_constraints(pins, sequencer)
        return problems

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
