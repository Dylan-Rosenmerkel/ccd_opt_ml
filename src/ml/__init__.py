"""
ml -- models that learn the image quality (read noise, dark current, gain) of a skipper CCD from its
operating parameters and its traces, and propose better operating parameters.

Modules
-------
synthetic   a toy "CCD world" that makes packets in the format of dataset.csv (for testing only)
data        load dataset.csv (+ the trace curves) and build model inputs and targets
models      Bayesian ridge, gradient-boosted trees, Gaussian process, neural-network ensemble,
            1D CNN on the trace curves; all with fit / predict(mean, std)
evaluate    cross-validation grouped by packet, error, calibration, learning curves
optimize    Bayesian optimization: propose the next operating parameters
benchmark   the comparisons of docs/ml_models.tex, run on the synthetic world

Background and choices: docs/ml_models.tex (Overleaf copy in docs/overleaf_ml_models/).
"""
