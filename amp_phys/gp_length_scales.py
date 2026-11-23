exec(open("corr.py").read().split("names = [n for")[0])
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
X = Vg[[c for _, c in REP]].values; X = (X - X.mean(0)) / X.std(0)
for tname, y in [("ln DC", g.lnl.values), ("ln noise", g.lns.values)]:
    k = C(np.var(y), (1e-5, 1e2)) * Matern(np.ones(X.shape[1]) * 2.0, (0.05, 1e3), nu=2.5) + WhiteKernel(0.1 * np.var(y), (1e-6, 1e1))
    gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=2, random_state=0).fit(X, y - y.mean())
    ls = gp.kernel_.k1.k2.length_scale
    print(f"\n{tname}: sigma_f^2 {gp.kernel_.k1.k1.constant_value:.3f}  n^2 {gp.kernel_.k2.noise_level:.3f}  -lnp {-gp.log_marginal_likelihood_value_:.1f}")
    print(pd.Series(ls, index=[n for n, _ in REP]).sort_values().round(2).to_string())
