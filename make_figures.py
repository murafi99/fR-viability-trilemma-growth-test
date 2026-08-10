"""
Regenerates Figures 1-6 exactly as described in the paper, from the source
models/growth/background code in this repo (not copied images).

Run:  python -m src.make_figures
Output: figures/*.png
"""
import numpy as np
import matplotlib.pyplot as plt
from .models import model_A, model_B, hu_sawicki_n1
from .growth import fsigma8_of_z, solve_growth
from .background import designer_R

Om0, OL0 = 0.3, 0.7
OUT = "figures"


def fig1_model_A():
    R = np.linspace(1e-3, 8, 400)
    f, fp, fpp = model_A(R, lam=0.7, Rc=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    axes[0].plot(R, f, lw=2)
    axes[0].plot(R, R, "--", color="gray", label="GR (f = R)")
    axes[0].set_xlabel(r"$R/R_c$"); axes[0].set_ylabel(r"$f(R)$")
    axes[0].legend(fontsize=8); axes[0].set_title("(a)")
    axes[1].plot(R, fp, color="crimson", lw=2)
    axes[1].axhline(0, color="gray", lw=0.7)
    axes[1].set_xlabel(r"$R/R_c$"); axes[1].set_ylabel(r"$f'(R)$")
    axes[1].set_title(r"(b) no-ghost: $f'>0$ $\checkmark$")
    axes[2].plot(R, fpp, color="green", lw=2)
    axes[2].set_xlabel(r"$R/R_c$"); axes[2].set_ylabel(r"$f''(R)$")
    axes[2].set_title(r"(c) no-tachyon: $f''>0$ $\checkmark$")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_modelA.png", dpi=150)
    plt.close(fig)


def fig2_model_B():
    R = np.linspace(1e-3, 8, 400)
    f, fp, fpp = model_B(R, lam=0.7, Rc=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    axes[0].plot(R, f, lw=2)
    axes[0].plot(R, R, "--", color="gray", label="GR (f = R)")
    axes[0].set_xlabel(r"$R/R_c$"); axes[0].set_ylabel(r"$f(R)$")
    axes[0].legend(fontsize=8); axes[0].set_title("(a)")
    axes[1].plot(R, fp, color="crimson", lw=2)
    axes[1].set_xlabel(r"$R/R_c$"); axes[1].set_ylabel(r"$f'(R)$")
    axes[1].set_title(r"(b) $f'(R)\to1$ at $R=0$ $\checkmark$")
    axes[2].plot(R, fpp, color="green", lw=2)
    axes[2].axhline(0, color="gray", lw=0.7)
    neg = fpp < 0
    if neg.any():
        axes[2].axvspan(R[neg][0], R[neg][-1], color="pink", alpha=0.4)
    axes[2].set_xlabel(r"$R/R_c$"); axes[2].set_ylabel(r"$f''(R)$")
    axes[2].set_title(r"(c) $f''<0$ near $R=0$  X")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_modelB.png", dpi=150)
    plt.close(fig)


def fig3_geff_k(c1=0.2, c2=1.0):
    k = np.logspace(-3, np.log10(3), 200)
    a = 1.0
    R = designer_R(a, Om0, OL0)
    f, fp, fpp = hu_sawicki_n1(R, c1, c2, Om0)
    M2 = fp / (3 * fpp)
    kappa2 = (k / a) ** 2
    Geff = (1.0 / fp) * (1 + (1.0 / 3.0) * kappa2 / (kappa2 + M2))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(k, Geff, color="darkgreen", lw=2)
    ax.axhline(1.0 / fp, ls="--", color="gray", label="super-Compton limit")
    ax.axhline((4 / 3) / fp, ls=":", color="gray", label="sub-Compton limit (4/3)/f'")
    ax.axvspan(0.01, 0.2, color="orange", alpha=0.2, label="typical linear/quasi-linear RSD range")
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel(r"$G_{\rm eff}(k,a{=}1)/G$")
    ax.set_title(f"Scale-dependent effective Newton constant\n(c1={c1}, c2={c2})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_geff_k.png", dpi=150)
    plt.close(fig)


def fig4_fsigma8_k(c1=0.2, c2=1.0):
    k_vals = np.logspace(-3, np.log10(3), 8)
    vals = []
    for k in k_vals:
        def model_func(R):
            return hu_sawicki_n1(R, c1, c2, Om0)
        v = fsigma8_of_z(Om0, OL0, model_func, 0.811, [0.0], k=k, model_func=model_func)
        vals.append(v[0])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(k_vals, vals, "o-", color="purple")
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel(r"$f\sigma_8(z{=}0,k)$")
    ax.set_title("Scale dependence of the growth-rate observable")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_fsigma8_k.png", dpi=150)
    plt.close(fig)


def fig5_growth_5pt(df):
    z = df["z_eff"].values[:5]
    data = df["fsigma8"].values[:5]
    err = df["sigma_eff"].values[:5]
    z_line = np.linspace(0, 0.3, 100)

    def fprime_GR(R):
        return R, np.ones_like(R), np.zeros_like(R)
    fid = fsigma8_of_z(Om0, OL0, fprime_GR, 0.811, z_line)

    def fp_A(R):
        return model_A(R, lam=-0.39, Rc=50.0)
    best = fsigma8_of_z(Om0, OL0, fp_A, 0.811, z_line)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(z, data, yerr=err, fmt="o", capsize=3, label="compiled data")
    ax.plot(z_line, fid, "--", color="gray", label=r"Planck-$\Lambda$CDM ($\sigma_{8,0}=0.811$)")
    ax.plot(z_line, best, color="crimson", label=r"best-fit Model A")
    ax.set_xlabel("redshift z"); ax.set_ylabel(r"$f\sigma_8(z)$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_growth5pt.png", dpi=150)
    plt.close(fig)


def fig6_growth_8pt(df, c1_best=1e-3, c2_best=1.92, s8_amp=0.764):
    z = df["z_eff"].values
    data = df["fsigma8"].values
    err = df["sigma_eff"].values
    z_line = np.linspace(0.0, 1.6, 100)

    def fprime_GR(R):
        return R, np.ones_like(R), np.zeros_like(R)
    fid = fsigma8_of_z(Om0, OL0, fprime_GR, 0.811, z_line)
    amp = fsigma8_of_z(Om0, OL0, fprime_GR, s8_amp, z_line)

    def model_func(R):
        return hu_sawicki_n1(R, c1_best, c2_best, Om0)
    hs = fsigma8_of_z(Om0, OL0, model_func, 0.811, z_line, model_func=model_func)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(z, data, yerr=err, fmt="ko", capsize=3, label=f"compiled data (N={len(z)})")
    ax.plot(z_line, fid, "--", color="gray", label=r"fiducial $\Lambda$CDM ($\sigma_{8,0}=0.811$)")
    ax.plot(z_line, amp, color="crimson", label=rf"$\Lambda$CDM + free amplitude ($\sigma_{{8,0}}={s8_amp}$)")
    ax.plot(z_line, hs, ":", color="navy", lw=2, label="best-fit Hu-Sawicki (n=1)  ~ GR")
    ax.set_xlabel("redshift z"); ax.set_ylabel(r"$f\sigma_8(z)$")
    ax.set_title("Extended growth-rate compilation, z = 0.02 - 1.52")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_growth8pt.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import pandas as pd
    import os
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv("data/growth_rate_compilation.csv")

    fig1_model_A()
    fig2_model_B()
    fig3_geff_k()
    fig4_fsigma8_k()
    fig5_growth_5pt(df)
    fig6_growth_8pt(df)
    print("All figures written to", OUT)
