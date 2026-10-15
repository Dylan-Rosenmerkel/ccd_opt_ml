"""
Build lvl1/lvl1_walkthrough.ipynb: level 1 step by step (take a packet by hand -> manifest -> processing -> per-model
input files -> training and testing every model without amplifier labels -> reading the report).

    .venv/bin/python lvl1/make_walkthrough.py
    .venv/bin/jupyter execute --inplace lvl1/lvl1_walkthrough.ipynb     (or run it in Jupyter, from lvl1/)

Real data: the 2026-10-12 packet tests (data/2026-10-12-PA03_PQ04_Packet_Tests; not in the repository) show the
manifest and processing steps; their binned images cannot be fitted by WADERs, so the TRAINING runs on synthetic
packets written in the same format (make_test_data.py).  Outputs go to lvl1/demo/ and lvl1/test_data/ (not committed).
"""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Level 1 walkthrough: predicting the image's noise parameters from the traces

**Level 1 question:** can σe (read noise), λ (dark charge) and the gain of an image be predicted from an amplifier's traces?
Packets are taken **by hand**, processed, and every model is trained and tested **offline**. Nothing here talks to the hardware.

| Step | Tool | Section |
|---|---|---|
| take a packet by hand | `daq_scripts/manual_packet.sh` | 1 |
| find its files, read its settings | `src/make_manual_manifest.py` | 2 |
| traces → features and curves; image → labels, λ row by row | `src/process_packet.py`, `src/image_rows.py` | 3, 4 |
| one ready-to-train file per model family | `src/ml/real_inputs.py` (called by `process_packet.py`) | 3 |
| train and test every model, report | `lvl1/train_lvl1.py` | 5, 6 |

Two rules hold throughout:
* **No amplifier labels.** No model is told which amplifier it sees; it sees that amplifier's own traces. Board and channel are used only to score.
* **One trace → one example.** A packet read by 4 amplifiers gives 4 examples (100 packets = 400). A packet's 4 examples always stay together
  in the same test fold.

Real labelled packets do not exist yet. Sections 2–3 use the real **2026-10-12 packet tests** (if the data are on this computer). Sections 5–6 train on
**synthetic packets** written in exactly the same format (`lvl1/make_test_data.py`): their numbers check the pipeline and compare methods,
not the real CCD.
""")

code(r"""
import json, subprocess, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "lvl1" else Path.cwd().resolve()
SRC, PY = ROOT / "src", sys.executable
sys.path.insert(0, str(SRC))
DEMO = ROOT / "lvl1" / "demo"
RUN_1012 = ROOT / "data" / "2026-10-12-PA03_PQ04_Packet_Tests"
def sh(args, cwd=ROOT, tail=None):
    out = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    text = "\n".join(l for l in (out.stdout + out.stderr).splitlines() if "RuntimeWarning" not in l)
    print("\n".join(text.splitlines()[-tail:]) if tail else text)
print("repository:", ROOT, "| 10-12 data present:", RUN_1012.exists())
""")

md(r"""
## 1. Taking a packet by hand

`daq_scripts/manual_packet.sh` queues the group's own `cdaq qadd` commands. Edit its SETTINGS block, then:

* run it with `DRY_RUN=1` first: it prints every command and sends nothing;
* run it without `DRY_RUN` to check the values again, show the plan, ask, and then queue.

Two things make a manual packet processable:
* **a unique name**, `PACKET_ID` (default `m<date>_<time>`): every trace is called `<PACKET_ID>_ref<set>` or `<PACKET_ID>_new<set>`;
* **a packet record**, `<RECORD_DIR>/<PACKET_ID>.json`: the name, the requested changes, and the trace and image settings.

The safety checks (unknown names, HIGH ≥ LOW, the group's voltage rules, the ACM hardware ranges) are unchanged.
""")

code(r"""
import os
out = subprocess.run(["bash", "daq_scripts/manual_packet.sh"], cwd=ROOT, capture_output=True, text=True,
                     env={**os.environ, "DRY_RUN": "1", "PACKET_ID": "m0001"})
lines = out.stdout.splitlines()
print("\n".join(lines[:16]))
print("   ...")
print("\n".join(l for l in lines if "m0001_new1" in l)[:400])
print("   ...")
print("\n".join(lines[-3:]))
""")

md(r"""
The record written when the commands are queued looks like this (it is written only by a real run, not by `DRY_RUN`):

```json
{"record_version": 1, "packet_id": "m0001", "scan_name": "manual", "queued_at": "2026-10-15T10:14:06-0400",
 "changes": ["H1_A_L=6.2", "H1_B_L=6.0", "H2_C_L=6.2", "H3_A_L=6.2", "H3_B_L=6.0"],
 "bcf": ["acm101=module_PA03.bcf", "acm102=ccd_PQ04.bcf"],
 "trace_prefix_ref": "m0001_ref", "trace_prefix_new": "m0001_new", "n_ref_sets": 5, "n_sets": 10,
 "trace_channels": ["0", "1", "2", "3"], "trace_args": "subroutine=AcquireImage, EXPOSURE=0, NROW=2, NCOL=5, ...",
 "flush_between": "set", "image_toml": "image_params/tc1/img.toml", "leader": 101}
```
""")

md(r"""
## 2. The manifest: the packet's files and settings

`src/make_manual_manifest.py <record> --run-dir <raw files>` writes the manifest that `process_packet.py` reads:

* **traces**: every `<PACKET_ID>_ref*` and `<PACKET_ID>_new*` file, with board (from the `.meta`), channel and set number;
* **image**: per board, the first image taken after the packet's last trace;
* **settings (θ)**: read from the `.meta` files, i.e. what the boards *report*. θ of the reference traces is `theta_ref`, and their difference is `delta_theta`;
* **checks**: every requested change is compared with the reported value. A change that did not arrive is a warning and sets the status to `check`.

The 2026-10-12 tests were taken before packet records existed. `--legacy-run` groups their files by identical settings instead, and
uses the first group's traces as the reference.
""")

code(r"""
if RUN_1012.exists():
    sh([PY, "src/make_manual_manifest.py", "--legacy-run", str(RUN_1012), "--board", "101", "--out", str(DEMO / "manifests")])
    m = json.loads(sorted((DEMO / "manifests").glob("*.json"))[1].read_text())
    print("packet:", m["packet_id"], "| status:", m["status"])
    print("traces:", len(m["traces"]), "at theta,", len(m["reference_traces"]), "reference; image:", Path(m["images"][0]["stem"]).name)
    print("settings that differ from the reference:", {k: round(v, 3) for k, v in m["delta_theta"]["101"].items() if abs(v) > 1e-6})
    print("trace and image read alike (NDCM, binning):", m["trace_matches_image"])
else:
    print("data/2026-10-12-PA03_PQ04_Packet_Tests is not on this computer: skipped")
""")

md(r"""
## 3. Processing: `dataset.csv` and one input file per model family

`python src/process_packet.py <manifest> ...` decodes every trace, checks it, and computes:
* the 85 trace features and 14 curves;
* for the image: the WADERs fit, the labels (σe, λ, gain with their errors), and λ **row by row** (section 4).

It then writes:
* **`dataset.csv`**: one row per **packet × board × channel**, i.e. one row per amplifier's traces;
* **`model_inputs/`**: one ready-to-train file per model family, since different models need different things.

Here one 10-12 packet is processed with `--skip-waders`: its 20×20-binned images cannot be fitted by WADERs, so there are no labels.
""")

code(r"""
proc = DEMO / "processed"
if RUN_1012.exists():
    mans = sorted((DEMO / "manifests").glob("*.json"))
    if not (proc / "dataset.csv").exists():
        sh([PY, "process_packet.py", *map(str, mans[:2]), "--skip-waders", "--out", str(proc)], cwd=SRC, tail=8)
    else:
        print("already processed:", proc)
    df = pd.read_csv(proc / "dataset.csv")
    df = df[df.board == 101]
    show = ["packet_id", "board", "channel", "amplifier", "trace.n_usable", "ref.n_usable", "trace.skip: CDS rms (detrended)", "image.NDCM"]
    display(df[[c for c in show if c in df]])
""")

code(r"""
from ml import real_inputs
if (proc / "dataset.csv").exists():
    if not (proc / "model_inputs" / "physics.npz").exists():
        real_inputs.build_all(proc)
    rows = []
    for p in sorted((proc / "model_inputs").glob("*.npz")):
        d = real_inputs.load(p)
        rows.append({"file": p.name, "models": real_inputs.FAMILIES[d["family"]].split(":")[0],
                     "contents": ", ".join(f"{k} {tuple(v.shape)}" for k, v in d.items()
                                           if isinstance(v, np.ndarray) and v.ndim > 1)})
    display(pd.DataFrame(rows))
""")

md(r"""
Every labelled file has the same **common part**:
* the ids `packet_id`, `board`, `channel`, `amplifier`, used for grouping and scoring, never as inputs;
* `Y`: log σe, log λ, log gain;
* `Y_err`: their relative errors from WADERs;
* `label_ok`, `fake`;
* the settings `theta`.

The rest depends on the family. For example, `physics.npz` adds the Allan curve that model A fits, and `pretrain.npz` holds every
usable trace's curves with no labels at all.

Nothing in these files is fitted (no scaling, no gap filling). Each model's preparation happens inside its training folds.
""")

md(r"""
## 4. λ row by row: dark current and CIC

With EXPOSURE 0, rows read later wait longer and collect more dark current:

$$\lambda_r = \mathrm{CIC} + \mathrm{DC}\,x_r,\qquad x_r = \frac{r+\tfrac12}{R/2}.$$

* **Each row** (`image_rows.row_lambda`): λ_r is fitted by maximum likelihood, with WADERs' gain and noise fixed, from that row's active pixels.
* **The line** (`image_rows.line_fit`): a weighted straight line through the R rows gives DC and CIC, with their 2×2 covariance.

The two estimates are strongly anti-correlated. The joint model (`physics_joint`) uses that covariance.

Below the fit is tried on simulated rows with known DC and CIC.
""")

code(r"""
import image_rows as ir
rng = np.random.default_rng(0); R, C, g, s, dc, cic = 16, 844, 1.3, 0.25, 0.02, 0.01
xr = (np.arange(R) + .5) / (R / 2); lam = cic + dc * xr
img = np.array([100 + g * rng.poisson(l, C) + s * rng.normal(size=C) for l in lam])
fit = [ir.row_lambda(row, g, s) for row in img]
L, E = np.array([f[0] for f in fit]), np.array([f[1] for f in fit])
line = ir.line_fit(L, E)
plt.figure(figsize=(6.5, 3.3))
plt.errorbar(range(R), L, E, fmt="ko", ms=4, label="fitted λ per row")
plt.plot(range(R), lam, "C0", label="truth")
plt.plot(range(R), line["lambda_cic"] + line["lambda_dark"] * xr, "C3--", label="line fit")
plt.xlabel("row (readout order)"); plt.ylabel("λ [e⁻/pixel]"); plt.legend(); plt.tight_layout(); plt.show()
corr = line["lambda_dark_cic_cov"] / (line["lambda_dark_err"] * line["lambda_cic_err"])
print(f"DC  = {line['lambda_dark']:.4f} ± {line['lambda_dark_err']:.4f}  (truth {dc})")
print(f"CIC = {line['lambda_cic']:.4f} ± {line['lambda_cic_err']:.4f}  (truth {cic});  correlation {corr:.2f}")
""")

md(r"""
**To check on the first real image:** DC should come out positive. If it is negative on every image, FITS row 0 is the *last* row read:
set `[rows] reverse = true` in `configs/processing.toml`.
""")

md(r"""
## 5. Training and testing every model: `lvl1/train_lvl1.py`

One command trains and tests every level-1 model by **packet-grouped cross-validation**:
* the packets are split into 5 folds;
* each fold is predicted by models trained on the other four.

| Model | What it is |
|---|---|
| `trace_physics` | no learning: noise from the trace's Allan curve ÷ the training mean gain (the baseline) |
| `physics` | **A (default)**: trace physics + learned gain and correction, GPs on settings, packet state and own features |
| `physics_joint` | A with λ from the joint dark-current / CIC fit |
| `sets` | E: A with the 10 trace sets combined by a learned set network |
| `ridge`, `trees`, `gp`, `nn` | the first-round models on settings + own trace and reference features |
| `cnn` | 1D CNN on the trace and reference curves |
| `pretrained` | D: CNN pre-trained on every unlabelled trace, then fine-tuned |
| `deep_kernel` | C: GP on the frozen pre-trained encoder's output, no amplifier term |

Here it runs on 100 synthetic packets (400 examples) with `--quick` (short training; the full run takes longer).
""")

code(r"""
TEST = ROOT / "lvl1" / "test_data"
if not (TEST / "model_inputs" / "physics.npz").exists():
    sh([PY, "lvl1/make_test_data.py", "--packets", "100", "--unlabeled", "300", "--out", str(TEST)], tail=3)
RES = DEMO / "lvl1_results_quick"
sh([PY, "lvl1/train_lvl1.py", "--inputs", str(TEST / "model_inputs"), "--quick", "--out", str(RES)])
""")

md(r"""
## 6. Reading the report

For each model and each target the report gives:

| Score | Meaning | Good |
|---|---|---|
| error | typical relative error, RMS of log(predicted/true) | close to the label error |
| bias | mean of log(predicted/true) | ≈ 0 |
| shape | error after removing each amplifier's own mean error | ≈ error or below |
| rank | within each amplifier: is the order of the true values reproduced? | > 0.9 |
| cover 68/95 | fraction of truths within ±1σ / ±1.96σ | 0.68 / 0.95 |
| calibrated | the same after conformal calibration | 0.68 / 0.95 |
| factor | how much to scale the error bars for 68 % | ≈ 1 |
| label error | the labels' own median error | — |
""")

code(r"""
rep = json.loads((RES / "report.json").read_text())
rows = []
for model, r in rep["results"].items():
    if "skipped" in r:
        continue
    for t, s in r["scores"].items():
        rows.append({"model": model, "target": t, "error %": 100 * s["error"], "bias %": 100 * s["bias"], "shape %": 100 * s["shape"],
                     "rank": s["rank"], "cover68": s["coverage68"], "cover95": s["coverage95"],
                     "calibrated68": s["calibrated_coverage68"], "factor": s["factor68"], "label error %": 100 * (s["label_error"] or np.nan)})
table = pd.DataFrame(rows)
for t in ("sigma_e", "lambda", "gain"):
    print(t)
    display(table[table.target == t].drop(columns="target").set_index("model").round(2))
""")

code(r"""
pred = pd.read_csv(RES / "predictions.csv")
amp = pred["board"].astype(str) + "_ch" + pred["channel"].astype(str)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, t in zip(axes, ("sigma_e", "lambda", "gain")):
    for a in sorted(amp.unique()):
        sel = amp == a
        ax.errorbar(np.exp(pred.loc[sel, f"true log {t}"]), np.exp(pred.loc[sel, f"physics: log {t}"]),
                    yerr=np.exp(pred.loc[sel, f"physics: log {t}"]) * pred.loc[sel, f"physics: sd {t}"], fmt="o", ms=3, alpha=0.6, label=a)
    lim = [np.exp(pred[f"true log {t}"]).min(), np.exp(pred[f"true log {t}"]).max()]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(f"true {t}"); ax.set_ylabel(f"predicted {t} (model A)")
axes[0].legend(fontsize=8, title="amplifier (colour only)")
plt.tight_layout(); plt.show()
""")

md(r"""
**What this shows (synthetic packets):**
* **Trace physics alone** is far off in *level*, because without a label the gain is unknown. It still *orders* the settings well within each amplifier.
* **Model A** learns the gain and a small correction. It brings σe, λ and the gain close to the labels' own errors.
* **The models that learn everything** from features or curves need more packets to get there.

The amplifiers sit on different lines because their gains and noise differ. A model given **no amplifier label** still places each example on its own amplifier's line, from that amplifier's traces.

**On real data** the same command runs on `processed/model_inputs`. A suggested level-1 pass mark, to agree on:
* the best label-free model predicts σe on held-out packets within about twice the label error;
* it ranks σe correctly within each amplifier (rank > 0.9);
* its calibrated error bars cover 68 % / 95 %.
""")

md(r"""
## 7. Checklist for the first real packets

* **Manifest:** `make_manual_manifest.py` reports status `ok`; every requested change arrived; `trace_matches_image` is true (same NDCM and binning).
* **WADERs:** labels without flags for the amplifiers used. ch2's gain cannot be measured from a 1×1 image yet, so its labels will usually be flagged.
* **Row order:** `label.lambda_dark` is positive (otherwise set `[rows] reverse = true`).
* **Scale:** at least ~10 packets before the cross-validation scores mean much (5 folds of 2 packets). The physics model is useful from ~15 packets in the synthetic tests.
* **Run** `train_lvl1.py --inputs processed/model_inputs` and compare `trace_physics` with `physics`. The difference is what the image sees that the trace alone does not.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent / "lvl1_walkthrough.ipynb"
nbf.write(nb, out)
print("wrote", out)
