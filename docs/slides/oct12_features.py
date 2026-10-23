"""
oct12_features.py -- every scalar trace feature (141 since 2026-10-23) of every board-101 trace of the 2026-10-12 packet
tests: the slides and lvl1/feature_study.py use it.  Traces with quality problems are skipped (as in processing).

5 setting groups (reference; VDD -20.5 V; V low rails + TG; all H low rails 6.2 V; H3_A_L 6.2 V) x 3 repeats x 4 amplifiers.
Each acquisition traces one amplifier (FILTER selects it); its data file is the non-empty RJ45ADCDATA<c>.
Writes docs/slides/data/oct12_trace_features.csv.

    .venv/bin/python docs/slides/oct12_features.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from decode_trace import decode_trace  # noqa: E402
from trace_features import trace_features  # noqa: E402
from trace_regions import split_regions  # noqa: E402

RUN = ROOT / "data" / "2026-10-12-PA03_PQ04_Packet_Tests"


def group(m: dict) -> str:
    if m["VDD1"] < -20:
        return "VDD −20.5 V"
    if m["V1_A_L"] < 7.6:
        return "V low rails + TG"
    if m["H1_A_L"] < 6.5:
        return "all H low rails"
    if m["H3_A_L"] < 6.5:
        return "H3_A_L only"
    return "reference"


rows = []
for meta_path in sorted(RUN.glob("trace_*_101_*.meta")):
    m = json.loads(meta_path.read_text())
    data = max(meta_path.parent.glob(meta_path.stem + ".RJ45ADCDATA*"), key=lambda p: p.stat().st_size)
    ch = int(data.suffix[-1])
    t = decode_trace(data)
    if t.quality_problems():                       # as process_packet.py: such a trace is not usable
        print(meta_path.stem[-18:], "ch", ch, group(m), "SKIPPED:", t.quality_problems(), flush=True)
        continue
    f = trace_features(t, meta=m, regions=split_regions(t, m))
    rows.append({"file": meta_path.stem, "channel": ch, "group": group(m), "NCOLPRE": m.get("NCOLPRE"), **f.scalars})
    print(meta_path.stem[-18:], "ch", ch, group(m), flush=True)
df = pd.DataFrame(rows)
df["repeat"] = df.groupby(["group", "channel"]).cumcount() + 1
out = Path(__file__).parent / "data" / "oct12_trace_features.csv"
df.to_csv(out, index=False)
print("wrote", out, df.shape)
