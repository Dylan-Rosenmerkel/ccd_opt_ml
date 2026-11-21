# Board 101 ch2 (CCD_A): excess noise over time

Run with the repo root as data root (`.venv/bin/python -I <script> <repo root> <out dir>`; `timeline.py` takes `<out dir> <repo root>`).

- `october.py`: decodes the raw October board-101 traces (data/2026-10-06, -08, -12) -> `october_traces.csv`
  (single-skip noise sigma1, per-pixel noise spread, kurtosis per trace and channel).
- `analyze.py`: the decoded Nov 2-3 packets (processed/L1_*) -> `ch2_trace_features.csv`, ch2 vs ch0/1/3 per packet and trace.
- `oct_cds.py`: CDS along the skips and its spectrum for October ch2 vs ch0 -> `oct_ch2_cds.png`.
- `timeline.py`: ch2 sigma1 / median of the other channels in the same acquisition -> `ch2_timeline.csv/.png`.

Result: ch2/others = 18x (Oct 6, installation check), 19-21x (Oct 8), 2.5-7.6x (Oct 12, varies with settings and time),
1.03-1.04x median (Nov 2-3, 533 trace sets, range 0.95-1.15). The other three channels are unchanged throughout
(sigma1 5.3 / 5.5 / 6.3 ADU in October). October excess is white along the skips (lag-1 correlation -0.06), with heavy
tails on Oct 12 (kurtosis 1-14).
