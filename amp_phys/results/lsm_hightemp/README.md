# LSM production modules DM07/DM09/DM10 at 180 K (2025-08-28), ACM 106/109/110

Data: /pool/homes/drosenm1/data/2025-08-28-DM07_DM09_DM10_High_Temp (copied from CC-IN2P3).
Traces are ccd-cdaq CSVs: columns RD,RU,TG,SW,OG,DG,H2,V2 = our flag bits 0..7; 15 MS/s, 10 skips/pixel.
psd_*.csv are static records (all clocks frozen).

    .venv/bin/python amp_phys/trace_anatomy.py --traces <trace_chC_B_*.csv> --label "ACMB chC" --drop 3 --out .../anatomy --plot
    .venv/bin/python amp_phys/static_psd.py --files <psd_*.csv> --out .../static_psd

Notes: first skip of each pixel has a -100..-145 ADU transient -> drop 3. Lag-1 correlation is biased by -1/6 with 7
skips/pixel (pixel mean removed): summary.csv adds the corrected column. nV/rtHz assume the design gain 0.875 ADU/uV
(no dead channel on these boards). 1-7.5 MHz static band is below the anti-alias filter corner (FILTER=2): not a white floor.

Temperature 180 K (user, 2026-11-16; not in the headers). Channel thermal-noise bound (8/3 kT/gm, 20 kOhm load) 5.8 nV/rtHz
(4.9 at 130 K). Window slopes have opposite signs in the pedestal and signal windows -> settling, not sense-node leakage.
Static records: no ramp (|slope| <= 0.65 ADU/ms).
