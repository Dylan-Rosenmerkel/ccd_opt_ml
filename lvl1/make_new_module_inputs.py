"""
make_new_module_inputs.py -- task C, new CCD: copy a model_inputs folder with the board-102 rows renamed to packets of
their own (``<id>@102``), so ``train_lvl1.py --board 101 102 --holdout-packets <the @102 ids>`` trains every model once
on board 101 and scores it on board 102 (the module it never saw), with the packet state of a board-101 row computed
from board 101 alone.  Writes the held-out ids to <out>/board102_packets.txt.

    .venv/bin/python lvl1/make_new_module_inputs.py processed_v2_trainD/model_inputs processed_v2_trainD/model_inputs_b102
"""

import sys
from pathlib import Path

import numpy as np

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
ids = set()
for f in sorted(src.glob("*.npz")):
    z = dict(np.load(f, allow_pickle=True))
    if "board" in z and "packet_id" in z and len(z["board"]) == len(z["packet_id"]):
        pid = np.asarray(z["packet_id"]).astype(object)
        b = np.asarray(z["board"]) == 102
        pid[b] = [f"{p}@102" for p in pid[b]]
        ids |= set(pid[b])
        z["packet_id"] = pid.astype(str)
    np.savez(out / f.name, **z)
(out / "board102_packets.txt").write_text("\n".join(sorted(ids)) + "\n")
print(f"{out}: {len(ids)} board-102 packet ids")
