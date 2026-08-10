"""
Interactive versions of Figures 1-4, using ipywidgets sliders so parameters
(lambda, Rc, c1, c2) can be explored live inside a Jupyter notebook.

The heavy lifting is in the *_core() functions, which take plain numbers and
return a matplotlib Figure -- these are plain functions you can call and test
without ipywidgets installed at all. The interactive_*() functions just wrap
them in ipywidgets.interact for live sliders.

Usage (inside a Jupyter notebook):

    from src.interactive import interactive_model, interactive_geff, interactive_fit

    interactive_model()   # sliders: model (A/B), lambda, Rc
    interactive_geff()    # sliders: c1, c2
    interactive_fit()     # sliders: c1, c2 -- overlaid on the real 8-point data
"""
import numpy as np
import matplotlib.pyplot as plt
from .models import model_A, model_B, hu_sawicki_n1
from .background import designer_R
from .growth import fsigma8_of_z

Om0, OL0 = 0.3, 0.7


# ------------------------------------------------------------- core plotters
def model_viability_core(model="A", lam=0.7, Rc=1.0):
    """Same 3-panel viability plot as Figures 1-2, parameterized live."""
    R = np.linspace(1e-3, 8 * Rc, 400)
    f, fp, fpp = (model_A if model == "A" else model_B)(R, lam, Rc)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    axes[0].plot(R, f, lw=2)
    axes[0].plot(R, R, "--", color="gray", label="GR (f = R)")
    axes[0].set_xlabel("R"); axes[0].set_ylabel("f(R)")
    axes[0].legend(fontsize=8); axes[0].set_title("(a)")

    ok1 = np.all(fp > 0)
    axes[1].plot(R, fp, color="crimson", lw=2)
    axes[1].axhline(0, color="gray", lw=0.7)
    axes[1].set_xlabel("R"); axes[1].set_ylabel("f'(R)")
    axes[1].set_title(f"(b) no-ghost: f'>0  {'OK' if ok1 else 'FAILS'}")

    ok2 = np.all(fpp >= -1e-9)
    axes[2].plot(R, fpp, color="green", lw=2)
    axes[2].axhline(0, color="gray", lw=0.7)
    neg = fpp < 0
    if neg.any():
        axes[2].axvspan(R[neg][0], R[neg][-1], color="pink", alpha=0.4)
    axes[2].set_xlabel("R"); axes[2].set_ylabel("f''(R)")
    axes[2].set_title(f"(c) no-tachyon: f''>=0  {'OK' if ok2 else 'FAILS'}")

    fig.suptitle(f"Model {model}:  lambda={lam:.3f}, Rc={Rc:.3f}", y=1.05)
    fig.tight_layout()
    return fig


def geff_core(c1=0.2, c2=1.0):
    """Same plot as Figure 3, parameterized live."""
    k = np.logspace(-3, np.log10(3), 200)
    a = 1.0
    R = designer_R(a, Om0, OL0)
    f, fp, fpp = hu_sawicki_n1(R, c1, c2, Om0)
    fpp_safe = fpp if abs(fpp) > 1e-30 else 1e-30
    M2 = fp / (3 * fpp_safe)
    kappa2 = (k / a) ** 2
    Geff = (1.0 / fp) * (1 + (1.0 / 3.0) * kappa2 / (kappa2 + M2))

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(k, Geff, color="darkgreen", lw=2)
    ax.axhline(1.0 / fp, ls="--", color="gray", label="super-Compton limit")
    ax.axhline((4 / 3) / fp, ls=":", color="gray", label="sub-Compton limit (4/3)/f'")
    ax.axvspan(0.01, 0.2, color="orange", alpha=0.2, label="typical RSD survey range")
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel(r"$G_{\rm eff}(k,a{=}1)/G$")
    ax.set_title(f"c1={c1:.3f}, c2={c2:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fit_overlay_core(c1=0.001, c2=1.92, data=None):
    """Overlay a live Hu-Sawicki(c1,c2) curve on the real 8-point compilation (Figure 6 style)."""
    import pandas as pd
    if data is None:
        data = pd.read_csv("data/growth_rate_compilation.csv")
    z = data["z_eff"].values
    d = data["fsigma8"].values
    e = data["sigma_eff"].values
    z_line = np.linspace(0.0, 1.6, 100)

    def fprime_GR(R):
        return R, np.ones_like(R), np.zeros_like(R)

    def model_func(R):
        return hu_sawicki_n1(R, c1, c2, Om0)

    fid = fsigma8_of_z(Om0, OL0, fprime_GR, 0.811, z_line)
    hs = fsigma8_of_z(Om0, OL0, model_func, 0.811, z_line, model_func=model_func)
    chi2 = float(np.sum(((hs[np.searchsorted(z_line, z)] - d) / e) ** 2)) if len(z_line) else None

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(z, d, yerr=e, fmt="ko", capsize=3, label=f"compiled data (N={len(z)})")
    ax.plot(z_line, fid, "--", color="gray", label="fiducial LCDM (sigma8=0.811)")
    ax.plot(z_line, hs, color="navy", lw=2, label=f"Hu-Sawicki: c1={c1:.4f}, c2={c2:.2f}")
    ax.set_xlabel("redshift z"); ax.set_ylabel("fsigma8(z)")
    ax.set_title("Drag the sliders -- try to beat the amplitude-rescaling fit's chi2 = 11.03")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------- ipywidgets wrappers
def interactive_model():
    """Live sliders for Model A/B: lambda, Rc. Run inside Jupyter."""
    import ipywidgets as widgets
    from ipywidgets import interact

    interact(
        model_viability_core,
        model=widgets.Dropdown(options=["A", "B"], value="A", description="Model"),
        lam=widgets.FloatSlider(value=0.7, min=0.05, max=0.95, step=0.05, description="lambda"),
        Rc=widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description="Rc"),
    )


def interactive_geff():
    """Live sliders for the Hu-Sawicki G_eff(k,a) transition: c1, c2."""
    import ipywidgets as widgets
    from ipywidgets import interact

    interact(
        geff_core,
        c1=widgets.FloatSlider(value=0.2, min=0.01, max=0.9, step=0.01, description="c1"),
        c2=widgets.FloatSlider(value=1.0, min=0.05, max=10.0, step=0.05, description="c2"),
    )


def interactive_fit():
    """Live sliders letting you try to beat the paper's amplitude-rescaling fit by hand."""
    import ipywidgets as widgets
    from ipywidgets import interact
    import pandas as pd

    data = pd.read_csv("data/growth_rate_compilation.csv")
    interact(
        lambda c1, c2: fit_overlay_core(c1, c2, data),
        c1=widgets.FloatLogSlider(value=0.001, base=10, min=-4, max=-0.05, step=0.05, description="c1"),
        c2=widgets.FloatSlider(value=1.92, min=0.05, max=50.0, step=0.1, description="c2"),
    )


if __name__ == "__main__":
    # Smoke test of the core plotting functions (no ipywidgets needed).
    import matplotlib
    matplotlib.use("Agg")
    fig1 = model_viability_core("A", 0.7, 1.0)
    fig2 = geff_core(0.2, 1.0)
    fig3 = fit_overlay_core(0.001, 1.92)
    print("Core plotting functions ran without error:", fig1 is not None, fig2 is not None, fig3 is not None)
