import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
f = np.logspace(0, np.log10(7.5e6), 300000); dt, T, tau = 13.8e-6, 32.6e-6, 10e-6
Sw, K = 1.221e-4, 3.93                      # board 101 ch1, one trace (sigma_1(tau) fit)
def sig(N):
    h = 4*np.sin(np.pi*f*dt)**2*np.sinc(f*tau)**2*(np.sin(np.pi*f*N*T)/(N*np.sin(np.pi*f*T)))**2
    return np.sqrt(np.trapezoid(Sw*h, f)), np.sqrt(np.trapezoid((Sw + K/f)*h, f))
Ns = np.unique(np.logspace(0, np.log10(3000), 30).astype(int))
W, A = np.array([sig(N) for N in Ns]).T
fig, ax = plt.subplots(figsize=(6.5, 4.3))
ax.loglog(Ns, A, "-", color="C0", label="predicted: white + 1/f (from the trace fit)")
ax.loglog(Ns, W, "--", color="gray", label="white part only")
ax.loglog([500], [0.147*1.39], "*", color="C3", ms=16, label="measured in images: 0.20 ADU")
s500 = sig(500)[1]; ax.annotate(f"σ₅₀₀ = {s500:.2f} ADU", (500, s500), (40, 0.12), arrowprops=dict(arrowstyle="->"), fontsize=12)
ax.set_xlabel("N, skips averaged"); ax.set_ylabel("noise of the N-skip pixel (ADU)")
ax.set_title("PA03 ch1: predicted noise vs skips"); ax.legend(fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[1] + "/sigma500.png"); print("s500", s500)
