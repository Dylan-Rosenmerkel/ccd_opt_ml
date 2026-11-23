"""
channel_scorecard.py -- one row per channel, the same trace-derived physics numbers for every setup: PA03 (board 101,
102 ch3 and the dead 102 ch0 as electronics reference, Nov v3 traces, 130 K) and the LSM DM07/09/10 (ACM 106/109/110,
180 K).  Medians over the traces of each channel.  Sources: trace_anatomy.py and trap_from_trace.py results.

    .venv/bin/python amp_phys/channel_scorecard.py --out amp_phys/results/per_channel
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "results"
COLS = {
    "CDS white (nV/rtHz)": "CDS noise (nV/rtHz)",
    "skip corr": "skip-to-skip corr",
    "settling tau after reset (us)": "reset settling (us)",
    "signal-window slope (ADU/us)": "signal-window slope (ADU/us)",
    "excess kurtosis of the CDS": "kurtosis",
    "burst fraction (1 ms blocks > 1.5x median rms)": "burst share",
    "ASD 200-1000 Hz (nV/rtHz)": "slow noise 0.2-1 kHz (uV/rtHz)",
    "ASD 1000-10000 Hz (nV/rtHz)": "noise 1-10 kHz (uV/rtHz)",
}
TRAP = {"white ASD (nV/rtHz)": "white part (nV/rtHz)", "1/f corner (kHz)": "1/f corner (kHz)",
        "1/f share at 10 us": "1/f share", "N_t A=0.85 C_ox (eV^-1 cm^-3)": "N_t (1e16 /eV/cm3)",
        "fit rms rel. residual": "trap-fit residual"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    an = pd.concat([pd.read_csv(f) for f in glob.glob(str(R / "per_channel/anat_*/trace_anatomy.csv"))])
    dead = pd.read_csv(R / "anatomy/trace_anatomy.csv")
    dead = dead[dead.label == "Nov v3 dead 102ch0"].assign(label="PA03 102 ch0 (dead)")
    lsm = pd.read_csv(R / "lsm_hightemp/anatomy/trace_anatomy.csv")
    ours = pd.concat([an, dead]).assign(setup="PA03 130 K", **{"skip corr": lambda d: d["CDS lag-1 autocorr"]})
    lsm = lsm.assign(setup="LSM 180 K", **{"skip corr": lambda d: d["CDS lag-1 autocorr"] + 1 / 6})  # 7 skips/pixel bias
    A = pd.concat([ours, lsm]).groupby(["setup", "label"], sort=False)[list(COLS)].median().rename(columns=COLS)
    for c in ("slow noise 0.2-1 kHz (uV/rtHz)", "noise 1-10 kHz (uV/rtHz)"):
        A[c] /= 1e3
    T = pd.concat([pd.read_csv(R / "trap_from_trace_v3/trap_from_trace.csv"),
                   pd.read_csv(R / "lsm_hightemp/traps/trap_from_trace.csv")]).groupby("label")[list(TRAP)].median()
    T = T.rename(columns=TRAP)
    T["N_t (1e16 /eV/cm3)"] /= 1e16
    A = A.join(T, on="label")
    st = pd.read_csv(R / "lsm_hightemp/static_psd/static_psd.csv")
    st["label"] = "ACM" + st.board.astype(str) + " ch" + st.channel.astype(str)
    A = A.join(st.set_index("label")[["ASD 200-1000 Hz (nV/rtHz)"]].rename(
        columns={"ASD 200-1000 Hz (nV/rtHz)": "static 0.2-1 kHz (uV/rtHz)"}) / 1e3, on="label")
    sa = R / "per_channel/skip_averaging/skip_averaging.csv"            # episodic behaviour over all traces
    if sa.exists():
        S = pd.read_csv(sa)
        g = S.groupby("label")
        A = A.join(pd.DataFrame({"traces": g.size(), "kurtosis p90 (all traces)": g["kurtosis"].quantile(0.9),
                                 "kurtosis max (all traces)": g["kurtosis"].max(),
                                 "CDS noise p90/median (all traces)": g["sigma_1 (ADU)"].quantile(0.9) / g["sigma_1 (ADU)"].median()}),
                   on="label")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A.to_csv(out / "channel_scorecard.csv")
    (out / "channel_scorecard.md").write_text(A.round(3).to_markdown() + "\n")
    pd.set_option("display.width", 300)
    print(A.to_string(float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
