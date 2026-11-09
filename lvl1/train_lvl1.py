"""
train_lvl1.py -- LEVEL 1: can the image's noise parameters be predicted from the traces?  One command.

Every model is trained and tested on the processed packets, WITHOUT amplifier labels: a model sees an amplifier's
own traces (its trace and reference features or curves) and the settings, never which amplifier it is.  The one
exception is ``physics_hier`` (A+B), a REFERENCE that is told the module and amplifier: comparing it with ``physics``
measures what amplifier labels add on the real CCD.  It is marked "(labels)" in every report.  One
training example = one amplifier's traces in one packet, with that amplifier's labels from the image (a packet
read by 4 amplifiers gives 4 examples).

Testing is by packet-grouped cross-validation: the packets are split into K folds; each fold's packets are
predicted by models trained on the other folds only.  The 4 examples of a packet always stay in the same fold
(they share the CCD's state at that moment; splitting them would make the scores look better than they are).
Packets taken at identical settings (the nominal repeats, the replicates) also share a fold (``--group-by setting``,
the default), so a test packet's twin is never in training.

Reported per model and per target (read noise sigma_e, dark charge lambda, gain):

* error     RMS of log(predicted / true), i.e. the typical relative error (0.05 = 5 %);
* bias      mean of log(predicted / true);
* shape     the error after removing each amplifier's own mean error (is the dependence right, even if one
            amplifier's level is off?);
* rank      Spearman correlation between predicted and true values within each amplifier (is the ORDER right?);
* coverage  fraction of truths within the predicted +-1 sigma (should be 0.68) and +-1.96 sigma (0.95);
* cal. cov. the same after conformal calibration, with each fold's factor taken from the OTHER folds (honest);
* factor    by how much the error bars must be scaled for 68 % coverage (1 = calibrated);
* label     the median error of the labels themselves (WADERs): no model can be judged below this;
* CRPS      continuous ranked probability score of the Gaussian prediction in log space (lower is better);
* worst amp the calibrated 68 % coverage of the worst-covered amplifier.

Stronger validation (docs/literature/landscape_grounded.md, H1; lvl1/README.md, "The report"): packet-bootstrap
intervals of every score (resampling the out-of-fold predictions, no refitting), a paired comparison of every model
with ``--paired-ref`` (default physics) on the same resamples, an optional prospective test (``--holdout-session C``:
train once on the other packets, score the held-out ones) and an optional leave-one-amplifier-out score (``--loao``).

Feature sets (docs/trace_features_and_tiers.md): ``--feature-tiers`` chooses the trace features every model sees (tier
labels, or ``auto`` = MODEL_TIERS, which always includes the protected estimators P); ``--tier-ablation`` trains every
model on several sets; ``--feature-ladder`` feeds every model more and more features in LABEL-FREE orders (the evidence
tiers; a coverage order that adds the least redundant feature next; random orders as the control; one physics group at a
time) and reports error against the number of features.

The amplifier and board are used ONLY to group rows for these scores, never as model inputs.

    python lvl1/train_lvl1.py --inputs <processed>/model_inputs [--models physics gp cnn ...] [--board 101]
                              [--folds 5] [--out DIR] [--quick] [--group-by setting|packet] [--n-boot 1000]
                              [--paired-ref physics] [--holdout-session C | --holdout-packets ID ...] [--loao]
                              [--feature-tiers auto | --tier-ablation [SET ...] | --feature-ladder [tiers cover random groups]]

Baselines (no trace at all; a model is only useful if it beats them):
    mean            [features]  the training mean of each target, with the training spread as its error bar
    settings        [features]  Gaussian process on the settings alone (no trace, no amplifier label)

Models (family file in brackets; see lvl1/README.md):
    trace_physics   [physics]   no learning: noise from the trace's Allan curve, the training mean gain
    physics         [physics]   A: trace physics + learned gain and correction (the default model); lambda by a SAAS GP
    physics_joint   [physics]   A with the joint dark-current / CIC fit of lambda (needs per-row lambda); SAAS prior
    physics_hier    [physics]   A+B, REFERENCE WITH LABELS: A on hierarchical GPs told the module and amplifier
    sets            [sets]      E: A with the trace sets combined by a learned set network (lambda by SAAS, as physics)
    ridge           [features]  Bayesian ridge on settings + own trace + own reference features
    trees           [features]  gradient-boosted trees, same inputs
    gp              [features]  Gaussian process, same inputs
    nn              [features]  neural-network ensemble, same inputs
    cnn             [curves]    1D CNN on the trace and reference curves + settings
    pretrained      [curves + pretrain]  D: CNN whose encoder was pre-trained on every unlabeled trace, fine-tuned
    deep_kernel     [curves + pretrain]  C: GP on the pre-trained encoder's embedding (frozen), no amplifier term

GP-prior variants (not run by default; the literature review's item 3, src/ml/bogp.py):
    gp_dim          [features]  gp with BoTorch's dimension-scaled length-scale prior (Hvarfner et al. 2024)
    gp_saas         [features]  gp as a SAAS GP (sparsity prior, NUTS sampling; Eriksson & Jankowiak 2021)
    physics_dim     [physics]   physics (A) with its GPs replaced by the dimension-scaled-prior GP
    physics_saas    [physics]   physics (A) with its GPs replaced by SAAS GPs
    physics_gp_lambda [physics]  physics (A) as before 2026-10-19 (lambda by the same GP as gain and correction)
    physics_joint_weak   [physics]  physics_joint before 2026-10-20 (weak length-scale prior instead of SAAS)
    physics_joint_select [physics]  physics_joint on the inputs a SAAS GP (NUTS) on log lambda finds relevant
    physics_saasmap_lambda  [physics]  physics with lambda by our GP with the SAAS prior fitted by MAP (fast, weaker)
    physics_saasfull_lambda [physics]  physics with lambda by the SAAS GP with full-length NUTS chains (slow)
    sets_gp_lambda  [sets]      sets as before 2026-10-23 (lambda by the weak-prior GP instead of the SAAS GP)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scipy.stats import spearmanr  # noqa: E402

from ml import models, real_inputs, structured as st  # noqa: E402
from ml.evaluate import bootstrap_weights, crps_gaussian, grouped_folds, percentile_interval, setting_groups  # noqa: E402

TARGETS = ("sigma_e", "lambda", "gain")
ALL_MODELS = ("mean", "settings", "trace_physics", "physics", "physics_joint", "physics_hier", "sets", "ridge", "trees", "gp", "nn", "cnn", "pretrained", "deep_kernel")
#: GP-prior variants: run only when named in --models
EXTRA_MODELS = ("gp_dim", "gp_saas", "physics_dim", "physics_saas", "physics_gp_lambda", "physics_joint_weak", "physics_joint_select", "physics_saasmap_lambda", "physics_saasfull_lambda", "sets_gp_lambda",
                "physics_white", "trace_physics_white", "settings_amp", "settings_amp_trees", "physics_cnn", "physics_cnn_only")
#: models that use no trace features at all (they keep their plain names in every run)
NO_FEATURES = ("mean", "trace_physics", "trace_physics_white", "settings_amp", "settings_amp_trees")
#: baselines that see no trace: every report compares the models with them
BASELINES = ("mean", "settings")
#: models told which module and amplifier they see (references only; see the module docstring)
USES_LABELS = ("physics_hier", "settings_amp", "settings_amp_trees")
FAMILY = {"mean": "features", "settings": "features", "settings_amp": "features", "settings_amp_trees": "features", "trace_physics": "physics", "trace_physics_white": "physics", "physics_white": "physics", "physics": "physics", "physics_joint": "physics", "physics_hier": "physics", "sets": "sets",
          "ridge": "features", "trees": "features", "gp": "features", "nn": "features",
          "cnn": "curves", "pretrained": "curves", "deep_kernel": "curves",
          "gp_dim": "features", "gp_saas": "features", "physics_dim": "physics", "physics_saas": "physics", "physics_gp_lambda": "physics", "sets_gp_lambda": "sets",
          "physics_joint_weak": "physics", "physics_joint_select": "physics",
          "physics_saasmap_lambda": "physics", "physics_saasfull_lambda": "physics",
          "physics_cnn": "physics", "physics_cnn_only": "physics"}
MAX_NAN_FRACTION = 0.2
#: the evidence tiers of every trace feature (lvl1/feature_study.py); see docs/trace_features_and_tiers.md
TIER_FILE = ROOT / "lvl1" / "feature_study" / "evidence_tiers.json"
#: per-model default feature sets for --feature-tiers auto (from the synthetic tolerance test, lvl1/feature_study/README.md):
#: the plain GP collapses with 20 uninformative columns, nn and ridge degrade, trees are immune.  P (guard a) = the
#: protected physical estimators, given to every model that takes trace features: the label-free evidence tiers can put
#: the most direct sigma_e and gain estimators in B (lvl1/feature_study/README.md, "Caveat").  The plain gp gets P only:
#: in the synthetic feature ladder (lvl1/feature_study/ladder_synthetic) it collapsed beyond 15-25 features in every order.
#: gp_dim stayed flat up to all 137 features (best sigma_e at P+A1); gp_saas kept sigma_e and gain but its lambda got
#: worse with every step (19 % at P, 30 % with all), so it gets P only.
MODEL_TIERS = {"gp": "P", "gp_dim": "P+A1", "gp_saas": "P", "nn": "P+A", "ridge": "P+A", "settings": "none",
               "mean": "none", "trees": "all"}
DEFAULT_MODEL_TIERS = "P+A+B"
#: the feature sets tried by --tier-ablation (by default); "none" = settings only, "impl0" = the original 85 features
ABLATION_SETS = ["none", "P", "P+A1", "P+A", "P+A+B", "all", "impl0"]
#: --feature-ladder: label-free orders in which the features are fed to every model (none of them uses the labels)
LADDER_ORDERS = ("tiers", "cover", "random", "groups")
LADDER_STEPS = (15, 25, 40, 60, 90)                     # feature counts of the cover and random ladders (plus P and all)


def load_tiers(path: Path = TIER_FILE) -> dict[str, set[str]]:
    """Feature name -> its tier labels: evidence tiers from ``path`` (A, A1, A2, B, C) and implementation tiers
    ("impl0", "impl1", "impl2", from trace_features.FEATURES)."""
    labels: dict[str, set[str]] = {}
    if Path(path).exists():
        for tier, names in json.loads(Path(path).read_text()).items():
            for n in names:
                labels.setdefault(n, set()).add(tier)
    try:
        import trace_features as tf
        for n, f in tf.FEATURES.items():
            labels.setdefault(n, set()).add(f"impl{f.tier}")
    except Exception:                                        # noqa: BLE001 - tiers by file only
        pass
    for n, ls in labels.items():                             # A = A1 + A2
        if ls & {"A1", "A2"}:
            ls.add("A")
    return labels


def cover_order(X: np.ndarray, names, first) -> list[str]:
    """A label-free order of the features that covers the feature space evenly: ``first`` (the protected set), then
    repeatedly the feature with the most variance NOT explained by those already taken (column-pivoted QR of the
    standardized features, after projecting out ``first``).  Mostly-missing or constant columns go last."""
    names = [str(n) for n in names]
    X = np.asarray(X, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = (np.isnan(X).mean(0) <= MAX_NAN_FRACTION) & (np.nanstd(X, 0) > 0)
        Z = np.where(np.isnan(X), np.nanmedian(X, 0), X)
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    head = [i for i, n in enumerate(names) if n in set(first)]
    rest = [i for i in range(len(names)) if i not in head and ok[i]]
    R = Z[:, rest]
    if head:
        B = Z[:, head]
        R = R - B @ np.linalg.lstsq(B, R, rcond=None)[0]
    from scipy.linalg import qr
    piv = qr(R, mode="r", pivoting=True)[1] if R.size else np.array([], int)
    order = head + [rest[i] for i in piv] + [i for i in range(len(names)) if i not in head and not ok[i]]
    return [names[i] for i in order]


def ladder_sets(names, X, tiers: dict, orders, steps=LADDER_STEPS, n_random: int = 3, seed: int = 0) -> list[tuple[str, str, int]]:
    """The feature sets of --feature-ladder, as (order, set label, number of features); the sets of the cover and random
    ladders are added to ``tiers`` as labels "ladder:<order>:<k>".  None of the orders uses the labels:

    * tiers   none -> P -> P+A1 -> P+A -> P+A+B -> all (the evidence tiers, cumulative)
    * cover   P, then the cover order (cover_order) cut at ``steps`` features, then all
    * random  P, then ``n_random`` random orders of the rest cut at ``steps`` (the control: does the order matter?)
    * groups  P + one physics group at a time (fit, skip, noise, ...), not nested: which family adds what"""
    names = [str(n) for n in names]
    P = [n for n in names if "P" in tiers.get(n, set())]
    count = lambda spec: int(feature_mask(names, spec, tiers).sum())
    out = []
    ks = [k for k in steps if len(P) < k < len(names)]
    def label_prefixes(order_name, order):
        for k in ks:
            lab = f"ladder:{order_name}:{k}"
            for n in order[:k]:
                tiers.setdefault(n, set()).add(lab)
            out.append((order_name.split(".")[0], lab, k))
    for o in orders:
        if o == "tiers":
            out += [("tiers", spec, count(spec)) for spec in ("none", "P", "P+A1", "P+A", "P+A+B", "all")]
        elif o == "cover":
            out.append(("cover", "P", len(P)))
            label_prefixes("cover", cover_order(X, names, P))
            out.append(("cover", "all", len(names)))
        elif o == "random":
            rng = np.random.default_rng(seed)
            out.append(("random", "P", len(P)))
            rest = [n for n in names if n not in P]
            for r in range(n_random):
                label_prefixes(f"random.{r + 1}", P + list(rng.permutation(rest)))
            out.append(("random", "all", len(names)))
        elif o == "groups":
            import trace_features as tf
            for g in sorted({tf.FEATURES[n].group for n in names if n in tf.FEATURES}):
                lab = f"group:{g}"
                for n in names:
                    if n in tf.FEATURES and tf.FEATURES[n].group == g:
                        tiers.setdefault(n, set()).add(lab)
                out.append(("groups", f"P+{lab}", count(f"P+{lab}")))
        else:
            raise ValueError(f"unknown ladder order {o!r} (choose from {LADDER_ORDERS})")
    return out


def ladder_tables(results: dict, ladder: list[tuple[str, str, int]], plot_path: Path | None = None) -> list[str]:
    """Error against the number of features, per order, model and target; random orders are averaged (mean, and the
    spread over orders).  The suggested set per model and target: the smallest step whose error is within one standard
    error (half the 68 % bootstrap interval) of that model's best step (the one-standard-error rule)."""
    models = list(dict.fromkeys(split_key(k)[0] for k in results if split_key(k)[1] is not None))
    def err(m, lab, t):
        r = results.get(f"{m} [{lab}]", {})
        return 100 * r["scores"][t]["error"] if "scores" in r else np.nan
    def se(m, lab, t):
        b = results.get(f"{m} [{lab}]", {}).get("bootstrap", {}).get(t, {}).get("error", {}).get("68")
        return 100 * (b[1] - b[0]) / 2 if b else np.nan
    lines = ["Every model is fed more and more features in label-free orders (same folds): `tiers` adds the evidence "
             "tiers one by one, `cover` adds the feature that is least redundant with those already taken, `random` "
             "is the control (mean of the random orders, their spread in brackets), `groups` adds one physics group to P. "
             "Cells: error %. Suggested set: the smallest step within one standard error of the model's best step.", ""]
    curves = {}
    for o in dict.fromkeys(o for o, _, _ in ladder):
        steps = [(lab, k) for oo, lab, k in ladder if oo == o]
        if o == "random":                                   # merge the random orders by k
            ks = sorted({k for _, k in steps})
            by_k = {k: [lab for lab, kk in steps if kk == k] for k in ks}
        else:
            ks = [k for _, k in steps]
            by_k = None
        heads = [lab.replace("ladder:", "").split(":")[-1] if lab.startswith("ladder:") else lab for lab, _ in steps] if by_k is None else [str(k) for k in ks]
        lines += [f"### order `{o}`", ""]
        for t in TARGETS:
            lines += [f"**{t}**", "", "| model | " + " | ".join(f"{h} ({k})" if by_k is None else h for h, k in zip(heads, ks)) + " | suggested |",
                      "|---|" + "---|" * (len(ks) + 1)]
            for m in models:
                if by_k is None:
                    e = [err(m, lab, t) for lab, _ in steps]
                    u = [se(m, lab, t) for lab, _ in steps]
                    cells = [f"{x:.1f}" if np.isfinite(x) else "–" for x in e]
                    labs = [lab for lab, _ in steps]
                else:
                    E = [[err(m, lab, t) for lab in by_k[k]] for k in ks]
                    e = [np.nanmean(v) if np.isfinite(v).any() else np.nan for v in E]
                    u = [np.nanmean([se(m, lab, t) for lab in by_k[k]]) for k in ks]
                    cells = [f"{x:.1f}" + (f" ({np.nanstd(v):.1f})" if np.isfinite(v).sum() > 1 else "") if np.isfinite(x) else "–" for x, v in zip(e, E)]
                    labs = [f"{k} features" for k in ks]
                e, u = np.array(e, float), np.array(u, float)
                pick = "–"
                if np.isfinite(e).any() and o != "groups":        # the groups are not nested: no "smallest" step
                    b = int(np.nanargmin(e))
                    thr = e[b] + (u[b] if np.isfinite(u[b]) else 0)
                    pick = labs[int(np.flatnonzero(np.isfinite(e) & (e <= thr))[0])]
                lines.append(f"| {m} | " + " | ".join(cells) + f" | `{pick}` |")
                curves[(o, m, t)] = (np.array(ks), e)
            lines.append("")
    if plot_path is not None and curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        orders = [o for o in dict.fromkeys(o for o, _, _ in ladder) if o != "groups"]
        fig, axes = plt.subplots(len(models), len(TARGETS), figsize=(4.2 * len(TARGETS), 2.8 * len(models)), squeeze=False)
        for i, m in enumerate(models):
            for j, t in enumerate(TARGETS):
                ax = axes[i, j]
                for o in orders:
                    if (o, m, t) in curves:
                        k, e = curves[(o, m, t)]
                        ax.plot(np.maximum(k, 0.5), e, "o-", ms=3, label=o, ls="--" if o == "random" else "-")
                ax.set_xscale("log"); ax.set_title(f"{m}: {t}", fontsize=9)
                if i == len(models) - 1:
                    ax.set_xlabel("trace features (0 = settings only)")
                if j == 0:
                    ax.set_ylabel("error %")
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(plot_path, dpi=110); plt.close(fig)
        lines += [f"![feature ladder]({plot_path.name})", ""]
    return lines


def feature_mask(names, spec: str, tiers: dict[str, set[str]]) -> np.ndarray:
    """Columns of ``names`` selected by ``spec``: "all", "none", or tier labels joined by "+" (e.g. "A+B", "A1",
    "impl0+impl1").  A trace feature without a tier label is kept only by "all".  The level-1 strip features ("l1: ...",
    inputs model_inputs_l1 / model_inputs_both) have no tiers: every spec but "none" keeps them."""
    names = [str(n) for n in names]
    if spec == "all":
        return np.ones(len(names), bool)
    if spec == "none":
        return np.zeros(len(names), bool)
    want = set(spec.split("+"))
    return np.array([bool(tiers.get(n, set()) & want) or n.startswith("l1: ") for n in names])


#: NUTS settings of the SAAS GP that fits lambda in ``physics`` (short chains: almost all of the full chains' gain)
SAAS_LAMBDA_KW = dict(warmup=96, num_samples=48, thinning=8)


# --------------------------------------------------------------------------- #
# inputs: everything fitted here uses the TRAINING rows of the fold only        #
# --------------------------------------------------------------------------- #


def prep_columns(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep columns with at most 20 % missing and some variation in the training rows; fill gaps with the training median."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nan_frac = np.isnan(train).mean(0)
        sd = np.nanstd(train, 0)
        keep = (nan_frac <= MAX_NAN_FRACTION) & np.isfinite(sd) & (sd > 0)
        med = np.nanmedian(train[:, keep], 0)
    fill = lambda a: np.where(np.isnan(a[:, keep]), med, a[:, keep])
    return fill(train), fill(test), keep


def packet_state(packet_id: np.ndarray, X_ref: np.ndarray, amp: np.ndarray | None = None) -> np.ndarray:
    """Per row: the mean over its packet's rows of the reference features, each amplifier's own average removed first
    (as StructuredPrep.state_vector), so that a packet missing one amplifier's row does not get a state shifted by that
    amplifier's signature.  The amplifier only centres the features (no labels, no amplifier identity in the output)."""
    X = np.asarray(X_ref, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if amp is not None:
            X = X.copy()
            for a in np.unique(amp):
                X[amp == a] -= np.nanmean(X[amp == a], 0)
        out = np.full_like(X, np.nan)
        for p in np.unique(packet_id):
            rows = packet_id == p
            out[rows] = np.nanmean(X[rows], 0)
    return out


def scalars(d: dict, with_features: bool, with_state: bool) -> np.ndarray:
    """The scalar input block of a family: settings, optionally the own-trace and reference features, optionally the
    packet state."""
    blocks = [d["theta"]]
    if with_features:
        blocks += [d["X_trace"], d["X_ref"]]
    if with_state:
        amp = np.array([f"{b}_ch{c}" for b, c in zip(d["board"], d["channel"])]) if "board" in d and "channel" in d else None
        blocks.append(packet_state(d["packet_id"], d["X_ref"], amp))
    return np.hstack(blocks)


def fill_sets(sets: np.ndarray, n_sets: np.ndarray, rng) -> np.ndarray:
    """Rows with fewer trace sets than the maximum: the missing slots are filled with copies of existing sets."""
    out = sets.copy()
    for i in range(len(out)):
        ok = np.flatnonzero(np.isfinite(out[i]).all(1))
        missing = np.flatnonzero(~np.isfinite(out[i]).all(1))
        if len(ok) and len(missing):
            out[i, missing] = out[i, rng.choice(ok, len(missing))]
    return out


# --------------------------------------------------------------------------- #
# models: fit on the training rows, return (mean, std) for the test rows        #
# --------------------------------------------------------------------------- #


class Runner:
    """Holds the loaded files and the shared pre-trained encoder; ``run(name, train, test)``."""

    def __init__(self, files: dict, rows: np.ndarray, quick: bool, tiers: dict | None = None):
        self.files, self.rows, self.quick = files, rows, quick
        self.pretrainer = None
        self.tiers = tiers or {}
        self.feature_spec = "all"                            # which trace features the models see (--feature-tiers)
        self.exclude: set[str] = set()                       # "own: <name>" / "ref: <name>" never seen (--exclude-features)
        self.no_settings = False                             # --no-settings: the models see the traces only, never theta
        d = files[next(iter(files))]
        self.Y, self.Y_err = d["Y"][rows], d["Y_err"][rows]
        # integer module / amplifier ids (for the labelled reference only)
        boards, keys = d["board"][rows], np.array([f"{b}_ch{c}" for b, c in zip(d["board"][rows], d["channel"][rows])])
        self.module_id = np.unique(boards, return_inverse=True)[1]
        self.amp_id = np.unique(keys, return_inverse=True)[1]

    #: arrays that are not one-per-example (names, block lengths)
    NOT_PER_ROW = ("allan_ns", "theta_names", "feature_names", "curve_names", "targets")

    def f(self, family):
        """The family's arrays restricted to the selected examples, and to the trace features of ``feature_spec``."""
        d = self.files[family]
        n = len(d["packet_id"])
        out = {k: (v[self.rows] if k not in self.NOT_PER_ROW and isinstance(v, np.ndarray) and v.ndim and len(v) == n else v)
               for k, v in d.items()}
        if self.feature_spec != "all" and "feature_names" in out and "X_trace" in out:
            keep = feature_mask(out["feature_names"], self.feature_spec, self.tiers)
            out["X_trace"], out["X_ref"] = out["X_trace"][:, keep], out["X_ref"][:, keep]
            out["feature_names"] = np.asarray(out["feature_names"])[keep]
        if self.no_settings and "theta" in out and np.asarray(out["theta"]).size:
            out["theta"] = np.full(np.shape(out["theta"]), np.nan)      # all-NaN: prep_columns drops every setting
        if self.exclude and "feature_names" in out and "X_trace" in out:
            # an excluded column becomes all-NaN, so prep_columns drops it (own and reference blocks separately)
            for key, prefix in (("X_trace", "own: "), ("X_ref", "ref: ")):
                drop = np.array([prefix + str(n) in self.exclude for n in out["feature_names"]])
                if drop.any():
                    out[key] = np.array(out[key], float, copy=True)
                    out[key][:, drop] = np.nan
        return out

    # -- the physics family -------------------------------------------------------
    def _physics_X(self, d, train, test):
        xs = scalars(d, True, True)
        xtr, xte, _ = prep_columns(xs[train], xs[test])
        # the Allan curves -> (a, b) of sigma_A(n)^2 = a^2/n + b^2, fitted only up to n = n_skips / 5 (the n = 160 point
        # of a 500-skip trace comes from 3 blocks and is biased low, structured.allan_fit); a missing point is filled
        # with the median of the same n over all rows (features only, no labels), not with one value for every n
        allan = np.asarray(d["allan"], float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            allan = np.where(np.isnan(allan), np.nanmedian(allan, axis=0), allan)
        a, b = st.allan_fit(allan, d["allan_ns"], max_n=self._allan_max_n(d))
        # the trace's white noise per skip (for noise_source "white"; NaN where the feature is missing).  Like the Allan
        # curve it is the physics term, not an input, so it comes from the family file before any feature selection.
        full = self.files["physics"]
        names = list(full["feature_names"]) if "feature_names" in full else []
        # (the level-1 source has no trace feature: the strip's own white noise per skip takes its place)
        wname = next((n for n in (st.WHITE_FEATURE, "l1: white noise per skip (ADU)") if n in names), None)
        white = (np.asarray(full["X_trace"], float)[self.rows][:, names.index(wname)]
                 if wname else np.full(len(a), np.nan))
        def part(idx, x):
            X = {"x": x, "a": a[idx], "b": b[idx], "white": white[idx], "n_skips": d["n_skips"][idx], "y_var": self.Y_err[idx] ** 2,
                 # module and amplifier ids: used ONLY by the labelled reference physics_hier (kind "hier")
                 "module": self.module_id[idx], "amp": self.amp_id[idx]}
            if "lambda_parts" in d:
                X["lambda_parts"], X["lambda_cov"] = d["lambda_parts"][idx], d["lambda_cov"][idx]
            return X
        return part(train, xtr), part(test, xte)

    def trace_physics(self, train, test, noise_source="allan"):
        """No learning: the trace physics of sigma_e in ADU divided by the training mean gain; lambda and gain are
        the training means."""
        d = self.f("physics")
        Xtr, Xte = self._physics_X(d, train, test)
        A = st.PhysicsStructured(noise_source=noise_source)
        g, lam = self.Y[train, 2].mean(), self.Y[train, 1].mean()
        m = np.column_stack([A.physics_log_sigma_adu(Xte) - g, np.full(len(test), lam), np.full(len(test), g)])
        s = np.column_stack([np.full(len(test), self.Y[train, 2].std()), np.full(len(test), self.Y[train, 1].std()),
                             np.full(len(test), self.Y[train, 2].std())])
        return m, s

    @staticmethod
    def _allan_max_n(d) -> float | None:
        """The largest Allan n used in the fit, or None for all.  Only the detrended Allan curve is biased low at
        n > n_skips / 5 (structured.allan_fit); real data use the overlapping, common-pattern-removed curve
        (st.REAL_ALLAN, no polynomial removed), which needs no cut, as in StructuredPrep, and the synthetic curves are
        unbiased.  So no cut is made: cutting the synthetic n = 160 point raised the physics sigma_e error 5.3 -> 5.8 %
        (2026-10-23).  Kept as one place to change if a detrended curve is ever used on real data."""
        return None

    def physics(self, train, test, joint=False, **joint_kw):
        """A (physics-structured outputs); lambda by the SAAS GP, or by the joint dark-current / CIC model if
        ``joint``."""
        d = self.f("physics")
        Xtr, Xte = self._physics_X(d, train, test)
        if joint:
            # (a part fitted at or below zero is fine: the joint fit clips it at 5 % of lambda, in log space)
            if "lambda_parts" not in Xtr or not (np.isfinite(Xtr["lambda_parts"]).all() and (Xtr["lambda_parts"].sum(1) > 0).all()):
                raise SkipModel("no per-row lambda (dark current / CIC) for every training row")
            model = st.PhysicsStructured(lambda_model="joint", lambda_kw=self._saas_kw(), **joint_kw)
        else:
            # lambda by a SAAS GP (sparsity prior, sampled with short NUTS chains): on synthetic data 19 -> 15 %,
            # 11 -> 7.4 %, 4.9 -> 3.9 % at 13 / 41 / 100 packets, sigma_e and gain unchanged, ~1 min per run at 100
            # packets (full chains: 13.5 / 7.3 / 3.9 % but 19 min; 2026-10-20; physics_gp_lambda is the previous version)
            model = st.PhysicsStructured(lambda_kind="saas", lambda_kw=SAAS_LAMBDA_KW)
        return model.fit(Xtr, self.Y[train]).predict(Xte)

    def trace_physics_white(self, train, test):
        """trace_physics with the white-noise term (white noise per skip / sqrt(N)) instead of the Allan fit."""
        return self.trace_physics(train, test, noise_source="white")

    def physics_white(self, train, test):
        """physics with the white-noise trace term instead of the Allan fit (PhysicsStructured noise_source "white";
        added 2026-11-03 after the first real packets)."""
        d = self.f("physics")
        Xtr, Xte = self._physics_X(d, train, test)
        model = st.PhysicsStructured(lambda_kind="saas", lambda_kw=SAAS_LAMBDA_KW, noise_source="white")
        return model.fit(Xtr, self.Y[train]).predict(Xte)

    def _embedding(self) -> np.ndarray:
        """Per row: the pre-trained CNN encoder's 16 numbers for the own curves, for the reference curves, and their
        difference (48 columns; the encoder never saw a label)."""
        pre = self._pretrained()
        d = self.f("curves")
        own = pre.encode(np.nan_to_num(d["curves_theta"]))
        ref = pre.encode(np.nan_to_num(d["curves_ref"]))
        return np.hstack([own, ref, own - ref])

    def physics_cnn(self, train, test, only: bool = False):
        """physics_white with the pre-trained CNN encoder inside: its embedding of the own and reference curves is added
        to the inputs of the gain, correction and lambda GPs (only=True: the embedding replaces every other input).
        Added 2026-11-04."""
        d = self.f("physics")
        Xtr, Xte = self._physics_X(d, train, test)
        E = self._embedding()
        if only:
            Xtr["x"], Xte["x"] = E[train], E[test]
        else:
            Xtr["x"], Xte["x"] = np.hstack([Xtr["x"], E[train]]), np.hstack([Xte["x"], E[test]])
        model = st.PhysicsStructured(lambda_kind="saas", lambda_kw=SAAS_LAMBDA_KW, noise_source="white")
        return model.fit(Xtr, self.Y[train]).predict(Xte)

    def physics_cnn_only(self, train, test):
        """physics_cnn with only the CNN embedding as the GPs' inputs (no hand-made features, no packet state)."""
        return self.physics_cnn(train, test, only=True)

    def physics_saasmap_lambda(self, train, test):
        """physics with lambda by our GP with the SAAS prior fitted by MAP (no sampling: as fast as physics_gp_lambda)."""
        d = self.f("physics")
        Xtr, Xte = self._physics_X(d, train, test)
        return st.PhysicsStructured(lambda_kind="none", lambda_kw={"ls_prior": "saas"}).fit(Xtr, self.Y[train]).predict(Xte)

    def physics_saasfull_lambda(self, train, test):
        """physics with lambda by the SAAS GP sampled with full-length NUTS chains (slow: ~20x the default)."""
        d = self.f("physics")
        Xtr, Xte = self._physics_X(d, train, test)
        return st.PhysicsStructured(lambda_kind="saas", lambda_kw=dict(warmup=256, num_samples=128, thinning=16)).fit(Xtr, self.Y[train]).predict(Xte)

    def physics_gp_lambda(self, train, test):
        """physics before 2026-10-19: lambda by the same GP as the gain and the correction."""
        d = self.f("physics")
        Xtr, Xte = self._physics_X(d, train, test)
        return st.PhysicsStructured().fit(Xtr, self.Y[train]).predict(Xte)

    def _saas_kw(self):
        return dict(warmup=96, num_samples=48, thinning=8) if self.quick else dict(warmup=256, num_samples=128, thinning=16)

    def physics_dim(self, train, test):
        """physics with its GPs replaced by the dimension-scaled-prior GP."""
        d = self.f("physics")
        Xtr, Xte = self._physics_X(d, train, test)
        return st.PhysicsStructured(kind="dim").fit(Xtr, self.Y[train]).predict(Xte)

    def physics_saas(self, train, test):
        """physics with its GPs replaced by SAAS GPs."""
        d = self.f("physics")
        Xtr, Xte = self._physics_X(d, train, test)
        return st.PhysicsStructured(kind="saas", **self._saas_kw()).fit(Xtr, self.Y[train]).predict(Xte)

    def physics_joint(self, train, test):
        """physics with lambda from the joint dark-current / CIC model of the per-row line fits (SAAS length-scale
        prior)."""
        # the SAAS sparsity prior (MAP) on the joint model's length scales: lambda 17.3 -> 16.9 %, 5.2 -> 4.6 %,
        # 4.6 -> 4.4 % at 13 / 41 / 100 synthetic packets, at no extra cost (2026-10-20; physics_joint_weak: before)
        return self.physics(train, test, joint=True, joint_prior="saas")

    def physics_joint_weak(self, train, test):
        """physics_joint before 2026-10-20: the weak log-normal length-scale prior."""
        return self.physics(train, test, joint=True)

    def physics_joint_select(self, train, test):
        """physics_joint on the inputs that a SAAS GP on log lambda finds relevant (>= 10 % of the largest
        relevance)."""
        return self.physics(train, test, joint=True, joint_select=0.1)

    def physics_hier(self, train, test):
        """A+B: model A on hierarchical GPs with module and amplifier levels -- the labelled reference."""
        d = self.f("physics")
        Xtr, Xte = self._physics_X(d, train, test)
        return st.PhysicsStructured(kind="hier").fit(Xtr, self.Y[train]).predict(Xte)

    def sets(self, train, test, saas_lambda: bool = True):
        """E: the physics model with the trace sets combined by a learned set network; lambda by the same SAAS GP as
        ``physics`` (since 2026-10-23; ``sets_gp_lambda`` is the previous version with the weak-prior GP)."""
        from ml import setmodels
        d, ds = self.f("physics"), self.f("sets")
        Xtr, Xte = self._physics_X(d, train, test)
        S = fill_sets(ds["sets"], ds["n_sets"], np.random.default_rng(0))
        if not np.isfinite(S).all():
            raise SkipModel("some rows have no usable trace set")
        Xtr["sets"], Xte["sets"] = S[train], S[test]
        Xtr["packet"] = d["packet_id"][train]                        # early stopping of the set network by packet
        lam = dict(lambda_kind="saas", lambda_kw=SAAS_LAMBDA_KW) if saas_lambda else {}
        pool = dict(n_models=2, epochs=100) if self.quick else {}
        model = setmodels.SetStructured(combiner="learned", kind="none", allan_ns=ds["allan_ns"],
                                        allan_max_n=self._allan_max_n(d), pool_kw=pool, **lam)
        return model.fit(Xtr, self.Y[train]).predict(Xte)

    def sets_gp_lambda(self, train, test):
        """sets before 2026-10-23: lambda by the weak-prior GP."""
        return self.sets(train, test, saas_lambda=False)

    # -- the baselines (no trace) ----------------------------------------------------
    def mean(self, train, test):
        """Baseline: the training mean of every target, with the training spread as its error bar."""
        Y = self.Y[train]
        return np.tile(Y.mean(0), (len(test), 1)), np.tile(Y.std(0, ddof=1), (len(test), 1))

    def settings(self, train, test):
        """Baseline: a GP on the settings alone (no trace)."""
        th = self.f("features")["theta"]
        xtr, xte, keep = prep_columns(th[train], th[test])
        if not keep.any():                      # every training packet at the same settings: nothing to learn
            return self.mean(train, test)
        return models.GP(max_iter=60 if self.quick else 150).fit(xtr, self.Y[train]).predict(xte)

    def _settings_amp(self, train, test):
        """The settings and the amplifier's one-hot (no trace)."""
        th = self.f("features")["theta"]
        onehot = np.eye(self.amp_id.max() + 1)[self.amp_id]
        xtr, xte, _ = prep_columns(np.hstack([th[train], onehot[train]]), np.hstack([th[test], onehot[test]]))
        return xtr, xte

    def settings_amp(self, train, test):
        """Reference WITH labels, no trace: a GP on the settings and the amplifier's identity (one-hot).  A trace model
        that does not beat it has learned nothing from the traces beyond which amplifier they come from (2026-11-03)."""
        xtr, xte = self._settings_amp(train, test)
        return models.GP(max_iter=60 if self.quick else 150).fit(xtr, self.Y[train]).predict(xte)

    def settings_amp_trees(self, train, test):
        """settings_amp with gradient-boosted trees (the same learner as ``trees``)."""
        xtr, xte = self._settings_amp(train, test)
        return models.GBT(max_iter=100 if self.quick else 300).fit(xtr, self.Y[train]).predict(xte)

    # -- the features family --------------------------------------------------------
    def _features(self, train, test):
        d = self.f("features")
        xs = scalars(d, True, False)
        xtr, xte, _ = prep_columns(xs[train], xs[test])
        return xtr, xte

    def ridge(self, train, test):
        """Bayesian ridge on the settings and the trace features."""
        xtr, xte = self._features(train, test)
        return models.BayesRidge().fit(xtr, self.Y[train]).predict(xte)

    def trees(self, train, test):
        """Gradient-boosted trees on the settings and the trace features."""
        xtr, xte = self._features(train, test)
        return models.GBT(max_iter=100 if self.quick else 300).fit(xtr, self.Y[train]).predict(xte)

    def gp(self, train, test):
        """Plain GP (one length scale per input) on the settings and the trace features."""
        xtr, xte = self._features(train, test)
        return models.GP(max_iter=60 if self.quick else 150).fit(xtr, self.Y[train]).predict(xte)

    def gp_dim(self, train, test):
        """GP with the dimension-scaled length-scale prior (BoTorch) on the settings and the trace features."""
        from ml.bogp import BoTorchGP
        xtr, xte = self._features(train, test)
        return BoTorchGP("dim").fit(xtr, self.Y[train]).predict(xte)

    def gp_saas(self, train, test):
        """SAAS GP (sparsity prior, NUTS) on the settings and the trace features."""
        from ml.bogp import BoTorchGP
        xtr, xte = self._features(train, test)
        return BoTorchGP("saas", **self._saas_kw()).fit(xtr, self.Y[train]).predict(xte)

    def nn(self, train, test):
        """Ensemble of small networks (mean and variance) on the settings and the trace features."""
        xtr, xte = self._features(train, test)
        groups = self.f("features")["packet_id"][train]           # early stopping by packet
        return models.MLPEnsemble(n_models=2 if self.quick else 5, epochs=100 if self.quick else 600).fit(
            xtr, self.Y[train], groups=groups).predict(xte)

    # -- the curves family -------------------------------------------------------------
    def _curves(self, train, test, both=True):
        d = self.f("curves")
        c = np.concatenate([d["curves_theta"], d["curves_ref"]], 1) if both else d["curves_theta"]
        c = np.nan_to_num(c)
        sc = np.hstack([d["theta"], self.f("features")["X_ref"]]) if "features" in self.files else d["theta"]
        str_, ste, _ = prep_columns(sc[train], sc[test])
        return {"curves": c[train], "scalars": str_}, {"curves": c[test], "scalars": ste}

    def cnn(self, train, test):
        """1D CNN on the trace curves (own and reference) plus the settings and reference features."""
        Xtr, Xte = self._curves(train, test)
        groups = self.f("curves")["packet_id"][train]             # early stopping by packet
        return models.CurveCNN(n_models=2 if self.quick else 3, epochs=60 if self.quick else 400).fit(
            Xtr, self.Y[train], groups=groups).predict(Xte)

    def _pretrained(self):
        if self.pretrainer is None:
            from ml import deep
            if "pretrain" not in self.files:
                raise SkipModel("no pretrain.npz")
            p = self.files["pretrain"]
            keep = np.isin(p["board"], np.unique(self.files["curves"]["board"][self.rows]))
            c = np.nan_to_num(p["curves"][keep])
            th = np.nan_to_num(p["theta"][keep]) if p["theta"].size and not self.no_settings else None
            # unlabeled traces only: no label of any packet is used here
            self.pretrainer = deep.Pretrainer(epochs=10 if self.quick else 40).fit(c, th)
        return self.pretrainer

    def pretrained(self, train, test):
        """D: a network on the curve encoder pre-trained on unlabeled traces, plus the scalar inputs."""
        from ml import deep
        pre = self._pretrained()
        Xtr, Xte = self._curves(train, test, both=False)
        groups = self.f("curves")["packet_id"][train]             # early stopping by packet
        return deep.PretrainedCNN(pre, n_models=2 if self.quick else 3, epochs=60 if self.quick else 400).fit(
            Xtr, self.Y[train], groups=groups).predict(Xte)

    def deep_kernel(self, train, test):
        """C: a GP whose kernel works on the pre-trained curve encoding plus the scalar inputs."""
        from ml import deep
        pre = self._pretrained()
        Xtr, Xte = self._curves(train, test, both=False)
        model = deep.DeepKernelGP(pre, finetune=False, amp_term=False, steps=60 if self.quick else 250)
        return model.fit({"curves": Xtr["curves"], "x": Xtr["scalars"]}, self.Y[train]).predict({"curves": Xte["curves"], "x": Xte["scalars"]})


class SkipModel(Exception):
    """Raised by a model that cannot run on these inputs (e.g. a file or the per-row lambda is missing); the run is
    reported as skipped."""
    pass


# --------------------------------------------------------------------------- #
# scores                                                                        #
# --------------------------------------------------------------------------- #


def conformal_q(z: np.ndarray, c: float) -> float:
    """The c-quantile of |z| with the finite-sample conformal correction (rank ceil((n + 1) c))."""
    z = np.sort(z[np.isfinite(z)])
    if not len(z):
        return np.nan
    k = int(np.ceil((len(z) + 1) * c)) - 1
    return float(z[min(max(k, 0), len(z) - 1)])


def row_terms(Y, mean, std, folds=None, cal_z=None) -> list[dict]:
    """Per target, the per-row pieces every score is built from: residual r, ok, |z|, the calibrated 68 / 95 %
    coverage indicators and the CRPS.  The calibration is cross-conformal over ``folds`` (each fold's factor from
    the other folds) or, for a held-out test set, the factor of the out-of-fold |z| ``cal_z[t]`` of the CV."""
    out = []
    for t in range(len(TARGETS)):
        ok = np.isfinite(Y[:, t]) & np.isfinite(mean[:, t])
        r = mean[:, t] - Y[:, t]
        s = np.maximum(std[:, t], 1e-12)
        z = np.abs(r) / s
        cal68, cal95 = np.zeros(len(r), bool), np.zeros(len(r), bool)
        if cal_z is not None:
            cal68, cal95 = z <= conformal_q(cal_z[t], 0.68), z <= conformal_q(cal_z[t], 0.95)
        else:
            for test in folds:                                   # cross-conformal: each fold's factor from the others
                other = np.setdiff1d(np.flatnonzero(ok), test)
                cal68[test] = z[test] <= conformal_q(z[other], 0.68)
                cal95[test] = z[test] <= conformal_q(z[other], 0.95)
        crps = np.where(ok, crps_gaussian(np.nan_to_num(Y[:, t]), np.nan_to_num(mean[:, t]), s), np.nan)
        out.append({"ok": ok, "r": np.where(ok, r, 0.0), "z": np.where(ok, z, np.nan), "cal68": cal68, "cal95": cal95, "crps": crps})
    return out


def scores(Y, Y_err, mean, std, amp, folds, cal_z=None) -> dict:
    """Every score of one model's out-of-fold (or held-out) predictions, per target (see the module docstring)."""
    with warnings.catch_warnings():                          # Spearman of a near-constant prediction
        warnings.simplefilter("ignore")
        return _scores(Y, Y_err, mean, std, amp, folds, cal_z)


def _scores(Y, Y_err, mean, std, amp, folds, cal_z=None) -> dict:
    out = {}
    for t, (name, rt) in enumerate(zip(TARGETS, row_terms(Y, mean, std, folds, cal_z))):
        ok, r, z, cal68, cal95 = rt["ok"], rt["r"], rt["z"], rt["cal68"], rt["cal95"]
        shape = r.copy()
        for a in np.unique(amp):
            sel = (amp == a) & ok
            if sel.any():
                shape[sel] -= r[sel].mean()
        ranks = []
        for a in np.unique(amp):
            sel = (amp == a) & ok
            if sel.sum() >= 4 and np.std(Y[sel, t]) > 0 and np.std(mean[sel, t]) > 0:
                ranks.append(spearmanr(mean[sel, t], Y[sel, t])[0])
        by_amp = {str(a): {"calibrated_coverage68": float(np.mean(cal68[(amp == a) & ok])),
                           "calibrated_coverage95": float(np.mean(cal95[(amp == a) & ok])), "n": int(((amp == a) & ok).sum())}
                  for a in np.unique(amp) if ((amp == a) & ok).any()}
        out[name] = {"error": float(np.sqrt(np.mean(r[ok] ** 2))), "bias": float(np.mean(r[ok])),
                     "shape": float(np.sqrt(np.mean(shape[ok] ** 2))),
                     "rank": float(np.mean(ranks)) if ranks else None,
                     "coverage68": float(np.mean(z[ok] <= 1.0)), "coverage95": float(np.mean(z[ok] <= 1.96)),
                     "calibrated_coverage68": float(np.mean(cal68[ok])), "calibrated_coverage95": float(np.mean(cal95[ok])),
                     "factor68": conformal_q(z[ok], 0.68),
                     "label_error": float(np.nanmedian(Y_err[ok, t])) if np.isfinite(Y_err[ok, t]).any() else None,
                     "crps": float(np.mean(rt["crps"][ok])),
                     "worst_amp_calibrated_coverage68": min(v["calibrated_coverage68"] for v in by_amp.values()) if by_amp else None,
                     "worst_amp_calibrated_coverage95": min(v["calibrated_coverage95"] for v in by_amp.values()) if by_amp else None,
                     "by_amp": by_amp,
                     "n": int(ok.sum())}
    return out


# -- packet bootstrap (no refitting: the out-of-fold predictions are resampled by packet) ---------------------------

def _weighted_ranks(v: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Ranks of v in each resample (rows of W = how often each row was drawn), ties of the copies averaged."""
    o = np.argsort(v, kind="stable")
    Ws = W[:, o]
    R = np.empty_like(Ws)
    R[:, o] = np.cumsum(Ws, 1) - Ws + (Ws + 1) / 2
    return R


def weighted_spearman(x: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Spearman correlation of x and y in every resample (= spearmanr on the resampled rows; NaN below 4 rows)."""
    rx, ry = _weighted_ranks(x, W), _weighted_ranks(y, W)
    sw = W.sum(1, keepdims=True)
    with np.errstate(all="ignore"):
        dx, dy = rx - (W * rx).sum(1, keepdims=True) / sw, ry - (W * ry).sum(1, keepdims=True) / sw
        rho = (W * dx * dy).sum(1) / np.sqrt((W * dx ** 2).sum(1) * (W * dy ** 2).sum(1))
    rho[(W > 0).sum(1) < 4] = np.nan
    return rho


def bootstrap_scores(Y, mean, std, amp, folds, W, cal_z=None) -> list[dict]:
    """Per target, each score in every packet-bootstrap resample (arrays of length n_boot).  The calibrated coverage
    indicators are those of the full data (the conformal factor is not re-estimated per resample)."""
    amps = np.unique(amp)
    with warnings.catch_warnings():                          # resamples without a usable amplifier: NaN
        warnings.simplefilter("ignore")
        return _bootstrap_scores(Y, mean, std, amp, amps, row_terms(Y, mean, std, folds, cal_z), W)


def _bootstrap_scores(Y, mean, std, amp, amps, terms, W) -> list[dict]:
    out = []
    for t, rt in enumerate(terms):
        ok, r = rt["ok"], rt["r"]
        Wk = W * ok
        n = Wk.sum(1)
        with np.errstate(all="ignore"):
            mse = Wk @ r ** 2 / n
            shape_ss = Wk @ r ** 2
            ranks, cov68 = [], []
            for a in amps:
                sel = (amp == a) & ok
                Wa = Wk[:, sel]
                na = Wa.sum(1)
                shape_ss = shape_ss - np.where(na > 0, (Wa @ r[sel]) ** 2 / np.where(na > 0, na, 1), 0)
                cov68.append(np.where(na > 0, Wa @ rt["cal68"][sel] / na, np.nan))
                if sel.sum() >= 4 and np.std(Y[sel, t]) > 0 and np.std(mean[sel, t]) > 0:
                    ranks.append(weighted_spearman(mean[sel, t], Y[sel, t], Wa))
            out.append({"error": np.sqrt(mse), "mse": mse, "shape": np.sqrt(np.maximum(shape_ss, 0) / n),
                        "rank": np.nanmean(np.array(ranks), 0) if ranks else np.full(len(W), np.nan),
                        "calibrated_coverage68": Wk @ rt["cal68"] / n, "calibrated_coverage95": Wk @ rt["cal95"] / n,
                        "worst_amp_calibrated_coverage68": np.nanmin(np.array(cov68), 0),
                        "crps": Wk @ np.nan_to_num(rt["crps"]) / n})
    return out


BOOT_KEYS = ("error", "vs_mean", "shape", "rank", "calibrated_coverage68", "calibrated_coverage95",
             "worst_amp_calibrated_coverage68", "crps")


def bootstrap_summary(boots: dict, base: str | None = "mean") -> dict:
    """{model: {target: {score: {'68': [lo, hi], '95': [lo, hi]}}}} from the bootstrap arrays of every model;
    'vs_mean' is the ratio to the mean baseline's error in the same resample."""
    out = {}
    for model, per_t in boots.items():
        out[model] = {}
        for t, name in enumerate(TARGETS):
            b = dict(per_t[t])
            if base in boots and model != base:
                with np.errstate(all="ignore"):
                    b["vs_mean"] = b["error"] / boots[base][t]["error"]
            out[model][name] = {k: percentile_interval(b[k]) for k in BOOT_KEYS if k in b}
    return out


def paired_vs(preds: dict, Y, ref: str, W, sel=None) -> dict:
    """Paired comparison of every model with ``ref`` on the same packet-bootstrap resamples.

    Per target: the difference in RMS log error (model - ref; negative = the model is better) on the rows both
    predicted, its 68 / 95 % intervals, the mean difference in squared log error per row (the resampled quantity)
    and the fraction of resamples in which the model's error is below the reference's."""
    if ref not in preds:
        return {}
    sel = np.arange(len(Y)) if sel is None else sel
    out = {}
    for model, (m, _) in preds.items():
        if model == ref:
            continue
        out[model] = {}
        for t, name in enumerate(TARGETS):
            y, a, b = Y[sel, t], m[sel, t], preds[ref][0][sel, t]
            ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
            if not ok.any():
                continue
            d2 = np.where(ok, (a - y) ** 2 - (b - y) ** 2, 0.0)
            ra, rb = np.where(ok, (a - y) ** 2, 0.0), np.where(ok, (b - y) ** 2, 0.0)
            Wk = W * ok
            n = Wk.sum(1)
            with np.errstate(all="ignore"):
                dmse = Wk @ d2 / n
                derr = np.sqrt(Wk @ ra / n) - np.sqrt(Wk @ rb / n)
            out[model][name] = {"delta_error": float(np.sqrt(ra[ok].mean()) - np.sqrt(rb[ok].mean())),
                                "delta_error_interval": percentile_interval(derr),
                                "delta_sq_log_error": float(d2[ok].mean()), "delta_sq_log_error_interval": percentile_interval(dmse),
                                # identical predictions (e.g. physics_joint's sigma_e is physics'): no winner
                                "identical": bool(np.all(d2[ok] == 0)),
                                "p_better": None if np.all(d2[ok] == 0) else float(np.mean(dmse[np.isfinite(dmse)] < 0)),
                                "n_rows": int(ok.sum())}
    return out


# -- report ---------------------------------------------------------------------------------------------------------

def fmt(v, pct=True):
    """Format a score for the report: a percentage (pct) or a plain number; '–' for missing."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{100 * v:.1f} %" if pct else f"{v:.2f}"


def fmt_iv(iv, pct=True, level="68"):
    """'12.1–18.3' (percent) or '0.85–0.95' for an interval {'68': [lo, hi], ...}."""
    if not iv or not iv.get(level):
        return "–"
    lo, hi = iv[level]
    return f"{100 * lo:.1f}–{100 * hi:.1f}" if pct else f"{lo:.2f}–{hi:.2f}"


def fmt_pp(v):
    """Format a difference in percentage points with its sign; '–' for missing."""
    return "–" if v is None or not np.isfinite(v) else f"{100 * v:+.1f}"


def vs_mean(results: dict, model: str, target: str) -> float | None:
    """The model's error divided by the mean baseline's (None if the baseline was not run)."""
    base = results.get("mean", {}).get("scores", {}).get(target)
    if not base or not base["error"] or "scores" not in results.get(model, {}):
        return None
    return results[model]["scores"][target]["error"] / base["error"]


def _tag(model):
    return " (labels)" if model in USES_LABELS else " (baseline)" if model in BASELINES else ""


def _score_tables(results: dict, level: str = "##") -> list[str]:
    """The main table per target (as before, plus the error's 68 % interval, the worst amplifier and the CRPS)."""
    lines = []
    for name in TARGETS:
        lines += [f"{level} {name}", "",
                  "| model | error | 68 % CI | vs mean | bias | shape | rank | cover 68 / 95 | calibrated 68 / 95 | worst amp 68 | factor | CRPS | label error |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for model, res in results.items():
            if "skipped" in res:
                continue
            s = res["scores"][name]
            ci = res.get("bootstrap", {}).get(name, {}).get("error")
            lines.append(f"| {model}{_tag(model)} | {fmt(s['error'])} | {fmt_iv(ci)} | {fmt(vs_mean(results, model, name), False)} | {fmt(s['bias'])} | {fmt(s['shape'])} | {fmt(s['rank'], False)} | "
                         f"{fmt(s['coverage68'], False)} / {fmt(s['coverage95'], False)} | {fmt(s['calibrated_coverage68'], False)} / "
                         f"{fmt(s['calibrated_coverage95'], False)} | {fmt(s.get('worst_amp_calibrated_coverage68'), False)} | {fmt(s['factor68'], False)} | "
                         f"{fmt(s.get('crps'))} | {fmt(s['label_error'])} |")
        lines.append("")
    return lines


def _interval_tables(results: dict, level: str = "###") -> list[str]:
    lines = []
    for name in TARGETS:
        rows = [(m, r["bootstrap"][name]) for m, r in results.items() if "bootstrap" in r and name in r["bootstrap"]]
        if not rows:
            continue
        lines += [f"{level} {name}", "", "| model | error 68 % | error 95 % | vs mean 68 % | shape 68 % | rank 68 % | calibrated 68 (68 %) | worst amp 68 (68 %) | CRPS 68 % |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m, b in rows:
            lines.append(f"| {m}{_tag(m)} | {fmt_iv(b.get('error'))} | {fmt_iv(b.get('error'), level='95')} | {fmt_iv(b.get('vs_mean'), False)} | "
                         f"{fmt_iv(b.get('shape'))} | {fmt_iv(b.get('rank'), False)} | {fmt_iv(b.get('calibrated_coverage68'), False)} | "
                         f"{fmt_iv(b.get('worst_amp_calibrated_coverage68'), False)} | {fmt_iv(b.get('crps'))} |")
        lines.append("")
    return lines


def _paired_table(paired: dict, ref: str) -> list[str]:
    if not paired:
        return [f"(`{ref}` was not run, so there is no paired comparison.)", ""]
    head = "| model | " + " | ".join(f"{t}: Δ error [95 % CI] | P(better)" for t in TARGETS) + " |"
    lines = [head, "|---|" + "---|---|" * len(TARGETS)]
    for m, per_t in paired.items():
        cells = []
        for t in TARGETS:
            p = per_t.get(t)
            if not p:
                cells += ["–", "–"]
                continue
            if p.get("identical"):
                cells += [f"same as {ref}", "–"]
                continue
            iv = p["delta_error_interval"].get("95")
            ci = f" [{100 * iv[0]:+.1f}, {100 * iv[1]:+.1f}]" if iv else ""
            sig = " *" if iv and (iv[1] < 0 or iv[0] > 0) else ""
            cells += [f"{fmt_pp(p['delta_error'])}{ci}{sig}", f"{p['p_better']:.2f}"]
        lines.append(f"| {m}{_tag(m)} | " + " | ".join(cells) + " |")
    return lines + ["", f"Δ error: the model's error minus `{ref}`'s, in percentage points of log error (negative = better than `{ref}`); "
                        "\\* = the 95 % interval excludes 0.  P(better): fraction of packet-bootstrap resamples in which the model's "
                        "error is the lower.  Change the default model only on a difference marked \\*.", ""]


def report_md(results: dict, info: dict, holdout: dict | None = None, loao: dict | None = None) -> str:
    """The Markdown report: data and folds, the score tables, the paired comparisons, the prospective test, LOAO."""
    grouping = ("grouped by setting (packets at identical settings share a fold)" if info.get("group_by") == "setting"
                else "grouped by packet")
    lines = [f"# Level 1 results ({info['date']})", "",
             f"Inputs: `{info['inputs']}`; boards {info['boards']}; {info['n_rows']} examples from {info['n_packets']} packets; "
             f"{info['n_folds']}-fold cross-validation {grouping}.  No amplifier labels are model inputs.", ""]
    if info.get("n_groups") is not None and info.get("group_by") == "setting":
        merged = info.get("merged_groups") or []
        lines += [f"{info['n_groups']} setting groups for {info['n_cv_packets']} cross-validated packets"
                  + (": " + ", ".join(f"`{g}`" for g in merged) + " are each one group." if merged else " (no two packets share settings)."), ""]
    if info.get("holdout_packets"):
        lines += [f"Held out (not in the cross-validation, scored once at the end): {len(info['holdout_packets'])} packets "
                  f"({info.get('holdout_source', '')}).", ""]
    if "mean" in results and "scores" in results["mean"]:
        lines += ["**vs mean** is a model's error divided by that of the `mean` baseline (always predicting the training average): "
                  "below 1 the model has learned something; near 1 it has not, however small its error looks.  `settings` uses "
                  "the settings alone: a trace model that does not beat it has not shown that the traces help.", ""]
    nb = info.get("n_boot", 0)
    lines += [(f"**68 % CI**: the 16–84 % range of the error over {nb} packet-bootstrap resamples of the out-of-fold "
               "predictions (no refitting).  " if nb else "") + "**worst amp 68**: the calibrated 68 % coverage of the worst-covered amplifier.  **CRPS**: "
              "the continuous ranked probability score of the Gaussian prediction in log space (lower is better; it rewards "
              "accuracy and honest error bars together, and equals the mean absolute log error for a perfectly sharp prediction).", ""]
    lines += _score_tables(results)
    if any(m in USES_LABELS for m in results):
        lines += ["(labels): a reference model told the module and amplifier; every other model sees only each amplifier's own traces.", ""]
    ref = info.get("paired_ref", "physics")
    if nb:
        lines += [f"## Paired comparison vs {ref}", "", f"On the same {nb} packet-bootstrap resamples as the intervals.", ""]
        lines += _paired_table(info.get("_paired", {}), ref)
    if nb:
        lines += ["## Bootstrap intervals", "", "68 % = 16–84 % range, 95 % = 2.5–97.5 % range over packet-bootstrap resamples (errors and CRPS in %).  "
                  "With ~40 packets the intervals are rough (a bootstrap over few clusters tends to make them too narrow).", ""]
        lines += _interval_tables(results)
    if holdout:
        lines += ["## Prospective test (held-out packets)", "",
                  f"Every model trained ONCE on all cross-validated packets and scored on the {holdout['info']['n_packets']} held-out "
                  f"packets ({holdout['info']['n_rows']} examples) only.  The calibrated coverage uses the conformal factor of the "
                  "cross-validation (no information from the held-out packets).  Choose the model on the cross-validation above; "
                  "read this section once, as the test of that choice.", ""]
        if holdout["info"].get("twins"):
            lines += ["Held-out packets whose settings also appear in the training packets (included in the scores, but they are "
                      "not new settings): " + ", ".join(f"`{p}`" for p in holdout["info"]["twins"]) + ".", ""]
        lines += _score_tables(holdout["results"], "###")
        if nb:
            lines += [f"### Paired comparison vs {ref} (held-out packets)", ""] + _paired_table(holdout["paired"], ref)
    if loao:
        lines += ["## Leave one amplifier out (new-module indicator)", "",
                  f"Each of the {loao['n_folds']} amplifiers predicted by models trained on the OTHER amplifiers only, crossed with the "
                  f"{loao.get('n_packet_folds', '')} packet folds (the test packets are not in training either, so this costs amplifiers x folds fits).  "
                  "Level 1 always has the same amplifiers in training, so this is only an indicator of how a model would do on a "
                  "new module; it is expected to be poor.", "",
                  "| model | " + " | ".join(f"{t} error | {t} vs mean" for t in TARGETS) + " |", "|---|" + "---|---|" * len(TARGETS)]
        for m, r in loao["results"].items():
            if "skipped" in r:
                lines.append(f"| {m} | " + " | ".join(["–"] * 2 * len(TARGETS)) + " |")
                continue
            lines.append(f"| {m}{_tag(m)} | " + " | ".join(f"{fmt(r['scores'][t]['error'])} | {fmt(vs_mean(loao['results'], m, t), False)}" for t in TARGETS) + " |")
        lines.append("")
    skipped = {m: r["skipped"] for m, r in results.items() if "skipped" in r}
    if skipped:
        lines += ["## Skipped", ""] + [f"* {m}: {why}" for m, why in skipped.items()] + [""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main                                                                          #
# --------------------------------------------------------------------------- #


def matches_packet(packet_id: str, ids) -> bool:
    """True if packet_id is one of ids, or one of them followed by a non-digit suffix ('L1_33' matches 'L1_33',
    'L1_33b', 'L1_33_retake', but not 'L1_330')."""
    for p in ids:
        if packet_id == p or (packet_id.startswith(p) and not packet_id[len(p)].isdigit()):
            return True
    return False


def session_packets(plan_path: Path, session: str) -> list[str]:
    """The planned packet ids of one session (lvl1/packet_plan/plan.json)."""
    plan = json.loads(Path(plan_path).read_text())
    return [p["packet_id"] for p in plan["packets"] if str(p.get("session")) == session]


def run_model(runner, name, splits, spec: str | None = None):
    """(mean, std) for every row of each (train, test) split; NaN where a row is in no test set.  ``name`` may be a
    run key "model [feature set]"; ``spec`` (or the set in the key) chooses the trace features the model sees."""
    model, key_spec = split_key(name)
    spec = spec or key_spec or runner.feature_spec
    previous, runner.feature_spec = runner.feature_spec, spec
    mean, std = np.full(runner.Y.shape, np.nan), np.full(runner.Y.shape, np.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for train, test in splits:
                m, s = getattr(runner, model)(train, test)
                mean[test], std[test] = m, s
    finally:
        runner.feature_spec = previous
    return mean, std


def split_key(key: str) -> tuple[str, str | None]:
    """"gp [A+B]" -> ("gp", "A+B"); "gp" -> ("gp", None)."""
    if key.endswith("]") and " [" in key:
        model, spec = key[:-1].split(" [", 1)
        return model, spec
    return key, None


def ablation_tables(results: dict, sets: list[str], ref_set: str) -> list[str]:
    """Per target: rows = models, columns = feature sets: error % and the paired difference against the model's own
    ``ref_set`` run (percentage points; * = the 95 % interval excludes 0)."""
    models = list(dict.fromkeys(split_key(k)[0] for k in results if split_key(k)[1] is not None))
    lines = [f"Each model was trained with each feature set (same folds). Cells: error % and, in brackets, the paired "
             f"difference against the same model with `{ref_set}` (percentage points; negative = better; * = 95 % interval "
             "excludes 0).", ""]
    for t in TARGETS:
        lines += [f"### {t}", "", "| model | " + " | ".join(f"`{s}`" for s in sets) + " |", "|---|" + "---|" * len(sets)]
        for m in models:
            cells = []
            for s in sets:
                r = results.get(f"{m} [{s}]", {})
                if "scores" not in r:
                    cells.append("–")
                    continue
                cell = f"{100 * r['scores'][t]['error']:.1f}"
                p = r.get("paired_vs_own", {}).get(t)
                if p and not p.get("identical") and p.get("delta_error_interval", {}).get("95"):
                    lo, hi = p["delta_error_interval"]["95"]
                    star = "*" if (lo > 0 or hi < 0) else ""
                    cell += f" ({100 * p['delta_error']:+.1f}{star})"
                cells.append(cell)
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", required=True, help="the model_inputs directory written by process_packet.py")
    parser.add_argument("--models", nargs="*", default=list(ALL_MODELS), choices=ALL_MODELS + EXTRA_MODELS)
    parser.add_argument("--board", type=int, nargs="*", default=[101], help="boards whose amplifiers are used (default 101)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--group-by", choices=("setting", "packet"), default="setting",
                        help="CV groups: packets at identical settings together (default), or each packet alone")
    parser.add_argument("--setting-tol", type=float, default=0.01,
                        help="settings are 'identical' within this fraction of each setting's range over the packets (default 0.01)")
    parser.add_argument("--n-boot", type=int, default=1000, help="packet-bootstrap resamples for the intervals (default 1000; 0 = none)")
    parser.add_argument("--paired-ref", default="physics", help="reference model of the paired comparison (default physics)")
    parser.add_argument("--holdout-packets", nargs="*", default=[], help="packet ids held out of the CV and scored once at the end")
    parser.add_argument("--holdout-session", help="hold out the plan's packets of this session (e.g. C: L1_33..L1_43)")
    parser.add_argument("--plan", default=str(ROOT / "lvl1" / "packet_plan" / "plan.json"), help="the packet plan (for --holdout-session)")
    parser.add_argument("--loao", action="store_true", help="also score leave-one-amplifier-out (a new-module indicator)")
    parser.add_argument("--feature-tiers", default="all",
                        help="trace features the models see: 'all' (default), 'none' (settings only), 'auto' (per-model "
                             "defaults, MODEL_TIERS), or tier labels joined by '+': protected P, evidence A, A1, A2, B, C (lvl1/feature_study/"
                             "evidence_tiers.json) and implementation impl0, impl1, impl2 (e.g. A+B, impl0)")
    parser.add_argument("--tier-ablation", nargs="*", default=None, metavar="SET",
                        help="train every model with each of these feature sets (default: " + " ".join(ABLATION_SETS) +
                             ") and compare them, each model against its own --ablation-ref run")
    parser.add_argument("--ablation-ref", default=None, help="the feature set each model is compared against in the ablation "
                                                             "(default all; P with --feature-ladder)")
    parser.add_argument("--feature-ladder", nargs="*", default=None, metavar="ORDER", choices=LADDER_ORDERS,
                        help="feed every model more and more features in label-free orders (default: tiers cover random; "
                             "also groups) and report error against the number of features; see ladder_sets")
    parser.add_argument("--ladder-steps", type=int, nargs="+", default=list(LADDER_STEPS), help="feature counts of the cover and random ladders")
    parser.add_argument("--ladder-random", type=int, default=3, help="number of random orders (default 3)")
    parser.add_argument("--tier-file", default=str(TIER_FILE), help="evidence tiers (default lvl1/feature_study/evidence_tiers.json)")
    parser.add_argument("--no-settings", action="store_true",
                        help="the models never see the settings (theta), not even in pre-training: traces only")
    parser.add_argument("--exclude-amps", nargs="*", default=[], metavar="BOARD:CHANNEL",
                        help="amplifiers left out entirely (e.g. 101:2 = board 101 channel 2, WADERs EXT3)")
    parser.add_argument("--exclude-features", help="text file of trace features no model may see, one per line, as "
                                                   "'own: <name>' or 'ref: <name>' (e.g. lvl1/amp_fingerprint.py's lists)")
    parser.add_argument("--include-flagged", action="store_true", help="also use labels that WADERs flagged")
    parser.add_argument("--exclude-fractional", action="store_true",
                        help="leave out rows whose delayInt is not a whole number of ADC samples (fractional_window: the image's on-board CDS mixes N / N+1-sample windows, so its labels may be biased)")
    parser.add_argument("--allow-fake", action="store_true", help="also use FAKE (rehearsal) packets")
    parser.add_argument("--quick", action="store_true", help="short training (to test the pipeline)")
    parser.add_argument("--out", help="results directory (default: <processed>/lvl1_results/<date>)")
    args = parser.parse_args()

    inputs = Path(args.inputs)
    files = {p.stem: real_inputs.load(p) for p in sorted(inputs.glob("*.npz"))}
    base = files.get("physics") or files.get("features")
    if base is None:
        raise SystemExit(f"no physics.npz or features.npz in {inputs}")
    rows = np.isin(base["board"], args.board) & np.isfinite(base["Y"]).all(1)
    if not args.include_flagged:
        rows &= base["label_ok"]
    if not args.allow_fake:
        rows &= ~base["fake"]
    if args.exclude_fractional:
        if "fractional_window" not in base:
            raise SystemExit("--exclude-fractional: no fractional_window in the inputs (rebuild them with process_packet.py --rebuild-only)")
        rows &= ~base["fractional_window"].astype(bool)
    if "trace_ok" in base:                      # no usable trace at the packet's own settings: nothing to predict from
        rows &= base["trace_ok"].astype(bool)
    for bc in args.exclude_amps:
        b, c = (int(x) for x in bc.split(":"))
        rows &= ~((base["board"] == b) & (base["channel"] == c))
    rows = np.flatnonzero(rows)
    if len(rows) < 6:
        raise SystemExit(f"only {len(rows)} usable examples (boards {args.board}, labels ok, not fake): take more packets")
    packets = base["packet_id"][rows].astype(str)
    amp = np.array([f"{b}_ch{c}" for b, c in zip(base["board"][rows], base["channel"][rows])])

    # held-out packets (a prospective test): out of the cross-validation, scored once at the end
    hold_ids, hold_source = list(args.holdout_packets), "--holdout-packets"
    if args.holdout_session:
        hold_ids += session_packets(Path(args.plan), args.holdout_session)
        hold_source = f"session {args.holdout_session} of {Path(args.plan).name}" + (" and --holdout-packets" if args.holdout_packets else "")
    held = np.array([matches_packet(p, hold_ids) for p in packets]) if hold_ids else np.zeros(len(rows), bool)
    if hold_ids and not held.any():
        print(f"warning: no packet matches the held-out ids {hold_ids[:5]}{'...' if len(hold_ids) > 5 else ''}: nothing is held out")
    cv_idx, ho_idx = np.flatnonzero(~held), np.flatnonzero(held)
    if len(np.unique(packets[cv_idx])) < 2:
        raise SystemExit("fewer than 2 packets left for the cross-validation after the hold-out")

    # CV groups: identical settings share a group (the nominal repeats, the replicates), so a twin is never in training
    if args.group_by == "setting" and "theta" in base and np.asarray(base["theta"]).size:
        groups = setting_groups(base["theta"][rows], packets, args.setting_tol)
    else:
        if args.group_by == "setting":
            print("warning: no settings (theta) in the inputs: grouping by packet")
            args.group_by = "packet"
        groups = packets.copy()
    n_groups = len(np.unique(groups[cv_idx]))
    merged = sorted(g for g in np.unique(groups[cv_idx]) if len(np.unique(packets[cv_idx][groups[cv_idx] == g])) > 1)
    n_folds = min(args.folds, n_groups)
    folds = [cv_idx[f] for f in grouped_folds(groups[cv_idx], n_folds, seed=0)]
    cv_splits = [(np.setdiff1d(cv_idx, test), test) for test in folds]
    local = lambda idx: [np.searchsorted(cv_idx, f) for f in idx]             # fold indices within cv_idx
    tiers = load_tiers(Path(args.tier_file))
    ladder = None
    if args.feature_ladder is not None:
        if "features" not in files:
            raise SystemExit("--feature-ladder needs features.npz")
        orders = args.feature_ladder or ["tiers", "cover", "random"]
        ladder = ladder_sets(files["features"]["feature_names"], files["features"]["X_trace"][rows][cv_idx], tiers, orders,
                             args.ladder_steps, args.ladder_random)
        args.tier_ablation = list(dict.fromkeys(lab for _, lab, _ in ladder))
        args.ablation_ref = args.ablation_ref or "P"
    args.ablation_ref = args.ablation_ref or "all"
    runner = Runner(files, rows, args.quick, tiers)
    runner.no_settings = args.no_settings
    if args.exclude_features:
        runner.exclude = {ln.strip() for ln in Path(args.exclude_features).read_text().splitlines()
                          if ln.strip() and not ln.startswith("#")}
        print(f"excluded trace features: {len(runner.exclude)} ({args.exclude_features})")
    # the runs: one per model, or (ablation) one per model and feature set; key "model [set]" when a set is given
    if args.tier_ablation is not None:
        sets = args.tier_ablation or list(ABLATION_SETS)
        if args.ablation_ref not in sets:
            sets.append(args.ablation_ref)
        # the baselines without trace features keep their plain names (the vs-mean scores look for "mean")
        runs = [(m, m, "none") if m in NO_FEATURES else (f"{m} [{s}]", m, s) for m in args.models
                for s in ([None] if m in NO_FEATURES else sets)]
    elif args.feature_tiers == "auto":
        runs = [(m, m, "none") if m in NO_FEATURES else
                (f"{m} [{MODEL_TIERS.get(m, DEFAULT_MODEL_TIERS)}]", m, MODEL_TIERS.get(m, DEFAULT_MODEL_TIERS)) for m in args.models]
    else:
        runs = [(m, m, args.feature_tiers) for m in args.models]
    # the paired reference by model name: with feature sets in the run keys ("physics [P+A+B]") the plain name would
    # match nothing and the paired comparison would silently be empty
    keys = [name for name, _, _ in runs]
    if args.paired_ref not in keys:
        same = [k for k in keys if split_key(k)[0] == args.paired_ref]
        own_ref = f"{args.paired_ref} [{args.ablation_ref}]"
        if own_ref in same:
            args.paired_ref = own_ref
        elif len(same) == 1:
            args.paired_ref = same[0]
    if "features" in files:
        names = files["features"]["feature_names"]
        for spec in dict.fromkeys(s for _, _, s in runs):
            print(f"feature set {spec!r}: {int(feature_mask(names, spec, tiers).sum())} of {len(names)} trace features")
    out = Path(args.out) if args.out else inputs.parent / "lvl1_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    info = {"date": datetime.now().isoformat(timespec="seconds"), "inputs": str(inputs), "boards": args.board,
            "n_rows": int(len(cv_idx)), "n_packets": int(len(np.unique(packets[cv_idx]))), "n_folds": n_folds, "quick": args.quick,
            "models": args.models, "feature_tiers": args.feature_tiers, "tier_ablation": args.tier_ablation,
            "exclude_features": args.exclude_features, "exclude_amps": args.exclude_amps, "no_settings": args.no_settings,
            "feature_ladder": [list(x) for x in ladder] if ladder else None, "group_by": args.group_by, "setting_tol": args.setting_tol, "n_groups": n_groups,
            "n_cv_packets": int(len(np.unique(packets[cv_idx]))), "merged_groups": merged, "n_boot": args.n_boot,
            "paired_ref": args.paired_ref, "holdout_packets": sorted(np.unique(packets[ho_idx]).tolist()),
            "holdout_source": hold_source if len(ho_idx) else None}
    print(f"{len(cv_idx)} examples from {info['n_packets']} packets, boards {args.board}, {n_folds} folds (grouped by {args.group_by}"
          + (f": {n_groups} setting groups" + (f", merged: {', '.join(merged)}" if merged else ", none merged") if args.group_by == "setting" else "") + ")")
    if len(ho_idx):
        print(f"held out for the prospective test: {len(ho_idx)} examples from {len(info['holdout_packets'])} packets ({hold_source})")
    n_boot = max(args.n_boot, 0)
    W_cv = bootstrap_weights(packets[cv_idx], n_boot, seed=0) if n_boot else None
    sl = cv_idx
    results, preds, boots, cal_z = {}, {}, {}, {}
    # every run's feature set, so the prospective test and LOAO refit each model with the set it was scored with (a run
    # key without "[set]", e.g. under --feature-tiers P, does not carry it; before 2026-11-03 those refits used "all")
    run_spec = {name: spec for name, _, spec in runs}
    for name, model, spec in runs:
        if FAMILY[model] not in files:
            results[name] = {"skipped": f"{FAMILY[model]}.npz not found"}
            print(f"{name:14s} skipped: {results[name]['skipped']}")
            continue
        t0 = time.time()
        try:
            mean, std = run_model(runner, name, cv_splits, spec)
        except SkipModel as exc:
            results[name] = {"skipped": str(exc)}
            print(f"{name:14s} skipped: {exc}")
            continue
        sc = scores(runner.Y[sl], runner.Y_err[sl], mean[sl], std[sl], amp[sl], local(folds))
        results[name] = {"scores": sc, "seconds": time.time() - t0}
        preds[name] = (mean, std)
        cal_z[name] = [rt["z"] for rt in row_terms(runner.Y[sl], mean[sl], std[sl], local(folds))]
        if n_boot:
            boots[name] = bootstrap_scores(runner.Y[sl], mean[sl], std[sl], amp[sl], local(folds), W_cv)
            ci = {t: percentile_interval(boots[name][i]["error"])["68"] for i, t in enumerate(TARGETS)}
        ratio = lambda t: f", {vs_mean(results, name, t):.2f}x mean" if name != "mean" and vs_mean(results, name, t) is not None else ""
        iv = lambda t: f" [{100 * ci[t][0]:.1f}–{100 * ci[t][1]:.1f}]" if n_boot and ci[t] else ""
        print(f"{name + (' (labels)' if name in USES_LABELS else ''):14s} " + "  ".join(f"{t}: {100 * sc[t]['error']:5.1f} %{iv(t)} (rank {fmt(sc[t]['rank'], False)}, cover {sc[t]['coverage68']:.2f}{ratio(t)})"
                                          for t in TARGETS) + f"   {time.time() - t0:5.0f} s", flush=True)
    paired = {}
    if n_boot:
        for model, b in bootstrap_summary(boots).items():
            results[model]["bootstrap"] = b
        paired = paired_vs({k: (v[0][sl], v[1][sl]) for k, v in preds.items()}, runner.Y[sl], args.paired_ref, W_cv)
        for model, p in paired.items():
            results[model]["paired_vs_" + args.paired_ref] = p
        if paired:
            print(f"\npaired vs {args.paired_ref} (Δ error in percentage points [95 % CI], P(better)):")
            for model, p in paired.items():
                print(f"  {model:14s} " + "  ".join(f"{t}: {fmt_pp(p[t]['delta_error'])} [{fmt_pp(p[t]['delta_error_interval']['95'][0])}, "
                                                    f"{fmt_pp(p[t]['delta_error_interval']['95'][1])}] P={p[t]['p_better']:.2f}"
                                                    if not p[t]["identical"] else f"{t}: same"
                                                    for t in TARGETS if t in p and p[t]['delta_error_interval']['95']))
    info["_paired"] = paired
    if args.tier_ablation is not None and n_boot:
        print(f"\ntier ablation: each model vs its own [{args.ablation_ref}] run (Δ error in percentage points, * = 95 % CI excludes 0)")
        for m in args.models:
            ref = f"{m} [{args.ablation_ref}]"
            own = {k: (v[0][sl], v[1][sl]) for k, v in preds.items() if split_key(k)[0] == m}
            if ref not in own:
                continue
            for k, p in paired_vs(own, runner.Y[sl], ref, W_cv).items():
                results[k]["paired_vs_own"] = p
                print(f"  {k:24s} " + "  ".join(
                    f"{t}: {100 * p[t]['delta_error']:+.1f}{'*' if (p[t]['delta_error_interval']['95'][0] > 0 or p[t]['delta_error_interval']['95'][1] < 0) else ''}"
                    if t in p and not p[t]["identical"] and p[t]["delta_error_interval"]["95"] else f"{t}: same" for t in TARGETS))

    # the prospective test: train once on every cross-validated packet, score the held-out packets
    holdout = None
    if len(ho_idx):
        twins = sorted({p for p, g in zip(packets[ho_idx], groups[ho_idx]) if np.isin(g, groups[cv_idx])})
        h_res, h_preds, h_boots = {}, {}, {}
        W_ho = bootstrap_weights(packets[ho_idx], n_boot, seed=1) if n_boot else None
        print(f"\nprospective test on {len(info['holdout_packets'])} held-out packets"
              + (f" (setting twins in training: {', '.join(twins)})" if twins else ""))
        for name in preds:
            t0 = time.time()
            try:
                m, s = run_model(runner, name, [(cv_idx, ho_idx)], run_spec.get(name))
            except SkipModel as exc:
                h_res[name] = {"skipped": str(exc)}
                continue
            pm, ps = preds[name]
            pm[ho_idx], ps[ho_idx] = m[ho_idx], s[ho_idx]          # predictions.csv: CV rows out-of-fold, held-out rows from this fit
            h = ho_idx
            sc = scores(runner.Y[h], runner.Y_err[h], m[h], s[h], amp[h], None, cal_z=cal_z[name])
            h_res[name] = {"scores": sc, "seconds": time.time() - t0}
            h_preds[name] = (m[h], s[h])
            if n_boot:
                h_boots[name] = bootstrap_scores(runner.Y[h], m[h], s[h], amp[h], None, W_ho, cal_z=cal_z[name])
            print(f"{name:14s} " + "  ".join(f"{t}: {100 * sc[t]['error']:5.1f} %" for t in TARGETS) + f"   {time.time() - t0:5.0f} s", flush=True)
        h_paired = {}
        if n_boot:
            for model, b in bootstrap_summary(h_boots).items():
                h_res[model]["bootstrap"] = b
            h_paired = paired_vs(h_preds, runner.Y[ho_idx], args.paired_ref, W_ho)
        holdout = {"info": {"n_packets": len(info["holdout_packets"]), "n_rows": int(len(ho_idx)), "packets": info["holdout_packets"],
                            "twins": twins, "source": hold_source}, "results": h_res, "paired": h_paired}

    # leave one amplifier out (only an indicator for a new module)
    loao = None
    if args.loao:
        # crossed with the packet folds: the test amplifier's rows of one fold, trained on the OTHER amplifiers of the
        # OTHER folds (holding out the amplifier alone would leave the test packet's other amplifiers in training)
        amps = np.unique(amp[cv_idx])
        splits = [(np.setdiff1d(cv_idx[amp[cv_idx] != a], test), test[amp[test] == a]) for a in amps for test in folds]
        splits = [(tr, te) for tr, te in splits if len(te) and len(tr)]
        l_res, l_preds = {}, {}
        print(f"\nleave one amplifier out ({len(amps)} amplifiers x {len(folds)} packet folds)")
        for name in preds:
            try:
                m, s = run_model(runner, name, splits, run_spec.get(name))
            except Exception as exc:                                 # e.g. a model that needs every amplifier in training
                l_res[name] = {"skipped": f"{type(exc).__name__}: {exc}"}
                print(f"{name:14s} failed: {l_res[name]['skipped']}")
                continue
            l_res[name] = {"scores": scores(runner.Y[sl], runner.Y_err[sl], m[sl], s[sl], amp[sl], [np.searchsorted(cv_idx, t) for _, t in splits])}
            l_preds[name] = m
            print(f"{name:14s} " + "  ".join(f"{t}: {100 * l_res[name]['scores'][t]['error']:5.1f} %" for t in TARGETS), flush=True)
        loao = {"n_folds": len(amps), "n_packet_folds": len(folds), "results": l_res}

    info_json = {k: v for k, v in info.items() if not k.startswith("_")}
    (out / "report.json").write_text(json.dumps({"info": info_json, "results": results, "holdout": holdout, "leave_one_amplifier_out": loao}, indent=1))
    report = report_md(results, info, holdout, loao)
    if ladder:
        report += "\n## Feature ladder\n\n" + "\n".join(ladder_tables(results, ladder, out / "feature_ladder.png")) + "\n"
    elif args.tier_ablation is not None:
        report += "\n## Tier ablation\n\n" + "\n".join(ablation_tables(results, sets, args.ablation_ref)) + "\n"
    (out / "report.md").write_text(report)
    import pandas as pd
    table = {"packet_id": packets, "board": base["board"][rows], "channel": base["channel"][rows], "cv group": groups, "held out": held}
    for t, tn in enumerate(TARGETS):
        table[f"true log {tn}"] = runner.Y[:, t]
        table[f"label error {tn}"] = runner.Y_err[:, t]
        for name, (m, s) in preds.items():
            table[f"{name}: log {tn}"], table[f"{name}: sd {tn}"] = m[:, t], s[:, t]
        for name, m in (l_preds.items() if args.loao else []):       # leave one amplifier out (CV rows)
            table[f"loao {name}: log {tn}"] = m[:, t]
    pd.DataFrame(table).to_csv(out / "predictions.csv", index=False)
    print(f"\nwrote {out / 'report.md'}, report.json, predictions.csv")


if __name__ == "__main__":
    main()
