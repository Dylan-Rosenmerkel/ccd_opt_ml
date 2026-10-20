"""
make_model_summary.py -- the one-page PDF "Level-1 models at a glance" (docs/model_summary/model_summary.pdf).

    .venv/bin/python docs/model_summary/make_model_summary.py

Laid out with PyMuPDF's Story (HTML + CSS -> PDF).  The numbers are the synthetic results at 41 packets
(docs/slides/data/model_errors_41packets.csv, mean of 3 datasets) and the GP-prior comparison of
docs/context/2026-10-19_session_context.md (section 4b).
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
    ("Baselines", None),
    ("mean", "nothing", "Always predicts the training average. Any useful model must beat it.", "mean"),
    ("settings", "settings only", "Gaussian process on the settings, no trace. A trace model that cannot beat it has not shown the traces help.", "settings"),
    ("Standard regressors (settings + 85 trace numbers of the amplifier's own and reference traces)", None),
    ("ridge", "85 numbers", "Bayesian linear fit with an error bar; simple and hard to fool.", "ridge"),
    ("trees", "85 numbers", "Gradient-boosted decision trees: thresholds and kinks.", "trees"),
    ("gp", "85 numbers", "Gaussian process: smooth prediction from similar examples, with an honest error bar.", "gp"),
    ("nn", "85 numbers", "Small ensemble of neural networks; their disagreement is the error bar.", "nn"),
    ("Physics-based (the main models)", None),
    ("trace_physics", "Allan curve", "No learning: fits σ(N)<sup>2</sup> = a<sup>2</sup>/N + b<sup>2</sup> to the trace's Allan deviation to predict the noise of the image's N = 500-skip average; gain = training mean.", "trace_physics"),
    ("physics (A), default", "Allan curve + numbers", "trace_physics + GPs that learn the gain and a correction for what the trace physics misses (outlier packets down-weighted). λ by a <b>SAAS GP</b>.", "physics"),
    ("physics_joint", "as A + λ per row", "As A, but λ = dark current + CIC fitted jointly from the image's row-by-row λ (GP with the SAAS prior).", "physics_joint"),
    ("sets (E)", "each trace set", "As A, but looks at every repeated trace set and learns which to trust: robust to damaged sets.", "sets"),
    ("Networks that learn their own features (the 14 curves)", None),
    ("cnn", "14 curves", "1D convolutional network on the curves (Allan deviation, spectra, waveform shapes).", "cnn"),
    ("pretrained (D)", "14 curves", "The CNN, first pre-trained on unlabelled traces (reference traces need no image), then fine-tuned.", "pretrained"),
    ("deep_kernel (C)", "14 curves", "Gaussian process on the pre-trained network's summary of the curves: proper error bars.", "deep_kernel"),
    ("Reference only", None),
    ("physics_hier (A+B)", "as A + amplifier", "A told the module and amplifier (hierarchical GPs): measures what amplifier labels would add.", "physics_hier"),
]


def table() -> str:
    out = ['<table><tr><th width="22%">Model <i>(looks at)</i></th><th width="58%">What it does</th>'
           '<th width="20%">σe / λ / gain error [%]</th></tr>']
    for row in ROWS:
        if row[1] is None:
            out.append(f'<tr><td colspan="3" class="grp">{row[0]}</td></tr>')
        else:
            name, looks, what, key = row
            out.append(f"<tr><td><b>{name}</b> <i>({looks})</i></td><td>{what}</td><td class='num'>{e(key)}</td></tr>")
    out.append("</table>")
    return "".join(out)


HTML = f"""
<h1>Level-1 models at a glance</h1>
<p class="sub">ccd_opt_ml · module PA03, board 101 · October 2026</p>
<p><b>Task.</b> From an amplifier's short readout trace (and the settings) predict what a WADERs fit of the image will give:
read noise <b>σe</b>, dark charge <b>λ</b> and <b>gain</b>. One packet = 4 amplifiers = 4 examples. No model is told which amplifier it sees
(except the reference). <b>Testing:</b> 5-fold cross-validation <i>grouped by packet</i> (a packet's 4 amplifiers are always held out together);
scores: error (typical % off), error ÷ that of guessing the average, rank within an amplifier, calibrated coverage of the error bars.
Error column: synthetic packets (toy CCD world), 41 packets, mean of 3 datasets; it compares methods, not the real CCD.</p>
{table()}
<h2>What is a SAAS GP?</h2>
<p><b>Gaussian process (GP).</b> Predicts a value at new settings as a weighted average of the training examples, weighted by how similar
their inputs are, and returns an error bar. Similarity is the kernel k(x, x′) = s<sup>2</sup> exp(−½ Σ<sub>i</sub> (x<sub>i</sub> − x′<sub>i</sub>)<sup>2</sup> / ℓ<sub>i</sub><sup>2</sup>),
with one <b>length scale ℓ<sub>i</sub></b> per input: a short ℓ<sub>i</sub> means the input matters, a long one that it is ignored.</p>
<p><b>The problem.</b> Our GPs see 60–320 inputs but only 40–400 examples. Fitting every ℓ<sub>i</sub> freely gives more knobs than data:
the GP latches onto chance patterns.</p>
<p><b>SAAS</b> (sparse axis-aligned subspace; Eriksson &amp; Jankowiak, UAI 2021, arXiv:2103.00349) puts a <b>sparsity prior</b> on the
inverse squared length scales: ρ<sub>i</sub> = 1/ℓ<sub>i</sub><sup>2</sup> ~ HalfCauchy(τ), with a global scale τ ~ HalfCauchy(0.1).
Most ρ<sub>i</sub> are pulled to ~0 (input switched off); the heavy tail lets the few inputs the data really need stay on.
Instead of one best fit, the hyperparameters are <b>sampled</b> (NUTS, Markov chain Monte Carlo) and the predictions averaged, so the
error bar includes the uncertainty about which inputs matter. A by-product: the inputs it keeps are a feature-relevance ranking.</p>
<p><b>In this project</b> (src/ml/bogp.py, BoTorch <i>SaasFullyBayesianSingleTaskGP</i>): <b>physics</b> fits λ with a SAAS GP (short chains:
96 warm-up / 48 samples, ~75 s per run at 100 packets); <b>physics_joint</b> uses the same prior fitted by maximum a posteriori (no sampling, no extra time).
Synthetic λ error at 13 / 41 / 100 packets: physics 19.1 → 15.1, 10.9 → 7.4, 4.9 → 3.9 %; physics_joint 17.3 → 16.9, 5.2 → 4.6, 4.6 → 4.4 %.
σe and gain unchanged. Not adopted: SAAS or the dimension-scaled prior for the plain <i>gp</i> (no gain there).</p>
"""

CSS = """
* { font-family: sans-serif; }
body { font-size: 8.8pt; line-height: 1.2; color: #111; }
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
    where = page + (36, 32, -36, -32)
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
