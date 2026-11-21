import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = sys.argv[1]
plt.rcParams["mathtext.fontset"] = "cm"
E = {
 "sigma1": r"$\sigma_1=\frac{1}{\sqrt{2}}\,\mathrm{std}\left(\mathrm{CDS}_{j+1}-\mathrm{CDS}_{j}\right)$",
 "allan": r"$\sigma_A(N)=\sqrt{\frac{1}{2}\left\langle(\bar y_{k+1}-\bar y_k)^2\right\rangle}$",
 "H": r"$\sigma_1^2=\int_0^\infty S(f)\,|H(f)|^2\,df,\qquad |H|^2=4\sin^2(\pi f\Delta t)\;\mathrm{sinc}^2(f\tau)\;2\sin^2(\pi f T)$",
 "fit": r"$S(f)=S_w+\frac{K}{f}\quad\Rightarrow\quad \sigma_1^2(\tau)=S_w\,I_w(\tau)+K\,I_f(\tau)$",
 "kvg": r"$K_{V_g}=\frac{K}{(G\,A)^2}$",
 "mcw": r"$S_{V_g}(f)=\frac{q^2\,k_BT\,\lambda_t\,N_t}{W\!L\;C_{ox}^2}\,\frac{1}{f}$",
 "nt": r"$N_t=\frac{K_{V_g}\;W\!L\;C_{ox}^2}{q^2\,k_BT\,\lambda_t}$",
 "ck": r"$C_k=q\,\frac{N}{\Delta V_k}=\frac{q}{g}\left|\frac{\partial y}{\partial V_k}\right|$",
 "split": r"$\Delta=|S|\,\mathrm{Var}(S)-|S_L|\,\mathrm{Var}(S_L)-|S_R|\,\mathrm{Var}(S_R)$",
 "forest": r"$\hat y(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x)$",
 "err": r"$\mathrm{error}=100\,\sqrt{\frac{1}{n}\sum_i\left(\ln\hat\sigma_{e,i}-\ln\sigma_{e,i}\right)^2}\ \%$",
 "kernel": r"$k(x,x')=s^2\left(1+\sqrt{5}\,r+\frac{5}{3}r^2\right)e^{-\sqrt{5}\,r},\qquad r^2=\sum_j\frac{(x_j-x'_j)^2}{\ell_j^2}$",
 "post": r"$m(x_*)=\mu+k_*^{\top}K_y^{-1}(y-\mu),\qquad s^2(x_*)=k(x_*,x_*)-k_*^{\top}K_y^{-1}k_*$",
 "lml": r"$\ln p(y)=-\frac{1}{2}(y-\mu)^{\top}K_y^{-1}(y-\mu)-\frac{1}{2}\ln|K_y|-\frac{n}{2}\ln 2\pi$",
 "J": r"$J=\ln\lambda\quad(\mathrm{dark\ charge,\ lower\ is\ better})$",
 "mt": r"$J_a(x)=\mu+f(x)+c_a+g_a(x)+\varepsilon$",
 "cov": r"$\mathrm{Cov}\left[J_a(x),J_{a'}(x')\right]=s^2M(x,x')\left[\rho+(1-\rho)\,\delta_{aa'}\right]+c^2\,\delta_{aa'}+n^2\,\delta_{\mathrm{same\ image}}$",
 "ei": r"$\mathrm{EI}(x)=(b-m)\,\Phi(z)+s\,\varphi(z),\qquad z=\frac{b-m}{s}$",
 "gap": r"$\Delta_k=100\,\left(\min_{r\leq k}J_r-J^{\star}\right)\ \%$",
}
for k, s in ({} if len(sys.argv) > 2 else E).items():
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, s, fontsize=20)
    fig.savefig(f"{OUT}/{k}.png", dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)
print(len(E))
E2 = {
 "reg": r"$y=\beta_0+\sum_k \beta_k\,V_k$",
 "ck2": r"$C_k=q\,\frac{\beta_k}{g}$",
 "lov": r"$L_{ov}\leq\frac{C_{RG}}{C_{ox}\,W_{RG}}$",
 "cox": r"$C_{ox}=\frac{\varepsilon_0}{t_{ox}/3.9+t_{n}/7.5}$",
 "iwif": r"$I_w(\tau)=\int_0^\infty |H|^2\,df,\qquad I_f(\tau)=\int_0^\infty \frac{|H|^2}{f}\,df$",
 "s1int": r"$\sigma_1^2=\int_0^\infty S(f)\,|H(f)|^2\,df$",
 "Sf": r"$S(f)=S_w+\frac{K}{f}$",
}
for k, s in E2.items():
    fig = plt.figure(figsize=(0.1, 0.1)); fig.text(0, 0, s, fontsize=20)
    fig.savefig(f"{OUT}/{k}.png", dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True); plt.close(fig)
