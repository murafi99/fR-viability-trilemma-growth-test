"""
chi^2 / AIC / BIC fits to the compiled fsigma8(z) data (Table 2), and the
curvature-amplitude degeneracy forecast (Table 4).

Run as a script to regenerate results/table3_model_comparison.csv and
results/table4_degeneracy_forecast.csv.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .growth import fsigma8_of_z
from .models import model_A, hu_sawicki_n1

Om0, OL0 = 0.3, 0.7
SIGMA8_FID = 0.811


def load_data(path="data/growth_rate_compilation.csv"):
    return pd.read_csv(path)


def chi2(model_vals, data_vals, errs):
    return float(np.sum(((model_vals - data_vals) / errs) ** 2))


def aic(chi2_val, k):
    return chi2_val + 2 * k


def bic(chi2_val, k, n):
    return chi2_val + k * np.log(n)


# --------------------------------------------------------------- fits
def fit_amplitude_only(z, data, err):
    def neg(params):
        (s8,) = params

        def fprime_GR(R):
            return R, np.ones_like(R), np.zeros_like(R)

        model = fsigma8_of_z(Om0, OL0, fprime_GR, s8, z)
        return chi2(model, data, err)

    res = minimize(neg, x0=[0.76], method="Nelder-Mead")
    return res.x[0], res.fun


def fit_hu_sawicki(z, data, err):
    def neg(params):
        c1, c2 = params
        c1 = np.clip(c1, 1e-4, 0.95)
        c2 = np.clip(c2, 1e-3, 50)

        def model_func(R):
            return hu_sawicki_n1(R, c1, c2, Om0)

        model = fsigma8_of_z(Om0, OL0, model_func, SIGMA8_FID, z, model_func=model_func)
        return chi2(model, data, err)

    res = minimize(neg, x0=[0.1, 1.0], method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 400})
    c1_clipped = float(np.clip(res.x[0], 1e-4, 0.95))
    c2_clipped = float(np.clip(res.x[1], 1e-3, 50))
    return (c1_clipped, c2_clipped), res.fun


def run_table3(df, outpath="results/table3_model_comparison.csv"):
    z, data, err = df["z_eff"].values, df["fsigma8"].values, df["sigma_eff"].values
    n = len(z)

    def fprime_GR(R):
        return R, np.ones_like(R), np.zeros_like(R)

    fid_model = fsigma8_of_z(Om0, OL0, fprime_GR, SIGMA8_FID, z)
    chi2_fid = chi2(fid_model, data, err)

    s8_best, chi2_amp = fit_amplitude_only(z, data, err)
    (c1_best, c2_best), chi2_hs = fit_hu_sawicki(z, data, err)

    rows = [
        {"model": "Fiducial LCDM", "k": 0, "chi2": chi2_fid, "AIC": aic(chi2_fid, 0),
         "BIC": bic(chi2_fid, 0, n), "params": f"sigma8_0={SIGMA8_FID} (fixed)"},
        {"model": "LCDM + amplitude", "k": 1, "chi2": chi2_amp, "AIC": aic(chi2_amp, 1),
         "BIC": bic(chi2_amp, 1, n), "params": f"sigma8_0={s8_best:.3f}"},
        {"model": "Hu-Sawicki n=1", "k": 2, "chi2": chi2_hs, "AIC": aic(chi2_hs, 2),
         "BIC": bic(chi2_hs, 2, n), "params": f"c1={c1_best:.3f}, c2={c2_best:.2f}"},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(outpath, index=False)
    return out


def run_table4(outpath="results/table4_degeneracy_forecast.csv", s8_amp=0.764):
    """
    Search the Hu-Sawicki (c1,c2) plane for the point that best mimics the
    amplitude-rescaled LCDM curve over z in [0.02, 1.6], then compare out to z=3.
    """
    z_fit = np.linspace(0.02, 1.6, 12)

    def fprime_GR(R):
        return R, np.ones_like(R), np.zeros_like(R)

    target = fsigma8_of_z(Om0, OL0, fprime_GR, s8_amp, z_fit)

    def neg(params):
        c1, c2 = np.clip(params, [1e-4, 1e-3], [0.95, 50])

        def model_func(R):
            return hu_sawicki_n1(R, c1, c2, Om0)

        model = fsigma8_of_z(Om0, OL0, model_func, SIGMA8_FID, z_fit, model_func=model_func)
        return float(np.sum((model - target) ** 2))

    res = minimize(neg, x0=[0.2, 1.0], method="Nelder-Mead",
                    options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 600})
    c1_best = float(np.clip(res.x[0], 1e-4, 0.95))
    c2_best = float(np.clip(res.x[1], 1e-3, 50))

    def model_func(R):
        return hu_sawicki_n1(R, c1_best, c2_best, Om0)

    z_report = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    hs_curve = fsigma8_of_z(Om0, OL0, model_func, SIGMA8_FID, z_report, model_func=model_func)
    amp_curve = fsigma8_of_z(Om0, OL0, fprime_GR, s8_amp, z_report)
    frac_diff = (hs_curve - amp_curve) / amp_curve * 100

    out = pd.DataFrame({
        "z": z_report, "hu_sawicki_best_mimic": hs_curve,
        "amplitude_rescaling": amp_curve, "fractional_diff_pct": frac_diff,
    })
    out.to_csv(outpath, index=False)
    return out, (c1_best, c2_best)


if __name__ == "__main__":
    df = load_data()
    t3 = run_table3(df)
    print(t3.to_string(index=False))
    t4, best = run_table4()
    print("\nBest-mimicking (c1, c2):", best)
    print(t4.to_string(index=False))
