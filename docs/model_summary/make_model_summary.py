"""
make_model_summary.py -- the one-page PDF "Level-1 models at a glance" (docs/model_summary/model_summary.pdf).

    .venv/bin/python docs/model_summary/make_model_summary.py

Laid out with PyMuPDF's Story (HTML + CSS -> PDF).  The numbers are the synthetic results at 41 packets
(docs/slides/data/model_errors_41packets.csv, mean of 3 datasets) and the GP-prior comparison of
docs/context/2026-10-19_session_context.md (section 4b).  Only the title and the table: every model and what it does.
"""
from pathlib import Path

import pandas as pd
import pymupdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
err = pd.read_csv(ROOT / "docs" / "slides" / "data" / "model_errors_41packets.csv").groupby("model")[["sigma_e", "lambda", "gain"]].mean()


def e(m: str) -> str:
    r = err.loc[m]
    return f"{r.sigma_e:.1f}/{r['lambda']:.1f}/{r.gain:.1f}"


ROWS = [
    ("Baselines: no trace at all", None),
    ("mean", "nothing",
     "Predicts the average of the training labels for every example, with the training spread as its error bar. Any model worth "
     "using must beat it; the report's “vs mean” score divides every model's error by this one.", "mean"),
    ("settings", "settings only",
     "A Gaussian process on the operating settings alone: no trace and no amplifier label. It measures how much the settings explain "
     "by themselves; a trace model that cannot beat it has not shown that the traces add anything.", "settings"),
    ("Standard regressors: the settings + the 85 hand-made trace numbers (own traces and the nominal reference traces)", None),
    ("ridge", "85 numbers",
     "Bayesian linear regression: a weighted sum of the inputs, with a prior that keeps the weights small, and an error bar. Simple, fast "
     "and hard to fool; a sanity check on the fancier models.", "ridge"),
    ("trees", "85 numbers",
     "Gradient-boosted decision trees: a few hundred small if-then trees, each fitting what the previous ones got wrong; the error bar comes "
     "from extra fits of the 16 % and 84 % quantiles. Handles thresholds and kinks and ignores useless inputs, but cannot extrapolate.", "trees"),
    ("gp", "85 numbers",
     "Gaussian process: predicts from the training examples whose inputs are most similar, with one fitted length scale per input "
     "(how much that input matters). Smooth predictions with honest error bars, which level 2 needs; can over-fit with many inputs and few examples.", "gp"),
    ("nn", "85 numbers",
     "An ensemble of small neural networks (two hidden layers of 64); each outputs a value and an error bar, and their disagreement is added "
     "to the uncertainty. Flexible, but needs more data than the first packets will give.", "nn"),
    ("Physics-based: the main models", None),
    ("trace_physics", "Allan curve",
     "No learning. Fits σ(N)<sup>2</sup> = a<sup>2</sup>/N + b<sup>2</sup> to the trace's Allan deviation (a = noise of one skip, b = noise floor) "
     "and evaluates it at the image's N = 500 skips; divides by the average training gain. Shows how far pure physics gets.", "trace_physics"),
    ("physics (A), the default", "Allan curve + settings + numbers",
     "Starts from trace_physics and adds Gaussian processes that learn the gain and a correction for what the trace physics misses "
     "(e.g. noise only the image sees). Training packets far off the rest are down-weighted, so one bad packet cannot bend the correction. "
     "λ comes from a SAAS GP, a Gaussian process whose prior switches off inputs the data do not need.", "physics"),
    ("physics_joint", "as A + λ row by row",
     "As A, but λ is built from the image's row-by-row dark charge: dark current and clock-induced charge (CIC) are fitted together by one "
     "two-output Gaussian process that knows their correlated errors (SAAS prior). Best λ so far.", "physics_joint"),
    ("sets (E)", "each trace set",
     "As A, but instead of averaging the repeated trace sets first, a small network reads every set's Allan curve and learns how much to "
     "trust each (attention). Robust to damaged sets and intermittent pickup. Best σe so far.", "sets"),
    ("Networks that learn their own features from the 14 trace curves", None),
    ("cnn", "14 curves + settings",
     "A 1D convolutional network that reads the curves directly (Allan deviation, spectra, autocorrelation, waveform shapes) and learns "
     "its own features; an ensemble gives the error bar. Can find effects the hand-made numbers miss, given enough data.", "cnn"),
    ("pretrained (D)", "14 curves + settings",
     "The same kind of network, first pre-trained on unlabelled traces (reference traces need no image) with self-made tasks: fill in "
     "masked parts of the curves, predict the settings from the curves, and a contrastive task. Then fine-tuned on the labelled packets.", "pretrained"),
    ("deep_kernel (C)", "pre-trained summary + settings",
     "A Gaussian process that measures similarity on the pre-trained network's summary of the curves (kept fixed) plus the settings: "
     "learned features with Gaussian-process error bars.", "deep_kernel"),
    ("Reference only", None),
    ("physics_hier (A+B)", "as A + told the amplifier",
     "Model A on hierarchical Gaussian processes told the module and amplifier: shared trends plus per-amplifier offsets. Not a candidate: "
     "comparing it with A measures what amplifier labels would add.", "physics_hier"),
]


def table() -> str:
    out = ['<table><tr><th width="20%">Model <i>(looks at)</i></th><th width="65%">What it does</th>'
           '<th width="15%">σe/λ/gain error [%]</th></tr>']
    for row in ROWS:
        if row[1] is None:
            out.append(f'<tr><td colspan="3" class="grp">{row[0]}</td></tr>')
        else:
            name, looks, what, key = row
            out.append(f"<tr><td><b>{name}</b><br/><i>{looks}</i></td><td>{what}</td><td class='num'>{e(key)}</td></tr>")
    out.append("</table>")
    return "".join(out)


HTML = f"""
<h1>Level-1 models at a glance</h1>
<p class="sub">ccd_opt_ml · each model predicts an amplifier's read noise σe, dark charge λ and gain from its trace (no amplifier labels,
except the reference). Last column: typical error on synthetic packets (toy CCD world), 41 packets, mean of 3 datasets, packet-grouped
5-fold cross-validation; it compares methods, not the real CCD.</p>
{table()}
"""

CSS = """
* { font-family: sans-serif; }
body { font-size: 8.8pt; line-height: 1.22; color: #111; }
h1 { font-size: 15pt; margin: 0 0 1pt 0; }
h2 { font-size: 11pt; margin: 7pt 0 2pt 0; border-bottom: 1px solid #111; }
p { margin: 0 0 4pt 0; text-align: justify; }
p.sub { color: #555; margin-bottom: 6pt; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0 2pt 0; }
th { background-color: #e6e6e6; text-align: left; padding: 2pt 3pt; border: 0.5px solid #999; }
td { padding: 1.6pt 3pt; border: 0.5px solid #bbb; vertical-align: top; }
td.grp { background-color: #f3f3f3; font-weight: bold; font-style: italic; }
td.num { text-align: center; }
"""


def main():
    out = HERE / "model_summary.pdf"
    story = pymupdf.Story(html=HTML, user_css=CSS)
    writer = pymupdf.DocumentWriter(str(out))
    page = pymupdf.paper_rect("letter")
    where = page + (32, 28, -32, -28)
    pages = 0
    more = True
    while more:
        dev = writer.begin_page(page)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
        pages += 1
    writer.close()
    print(f"wrote {out} ({pages} page{'s' if pages > 1 else ''})")


if __name__ == "__main__":
    main()
