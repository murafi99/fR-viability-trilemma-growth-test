"""
f(R) model definitions used throughout the paper.

- Model A:  f_A(R) = R - lambda*Rc*tanh(R/Rc)                (Eq. 3.1)
- Model B:  f_B(R) = R - lambda*Rc*tanh(R^2/Rc^2)             (Eq. 3.2)
- Hu-Sawicki n=1: f(R) = R - m2*c1*(R/m2) / (1 + c2*R/m2)     (Eq. 8.1)
                  m2 = Omega_m0 * H0^2

All functions return (f, f', f'') so viability conditions N1-N4
(Table 1) can be checked directly.
"""
import numpy as np


# ---------------------------------------------------------------- Model A
def model_A(R, lam, Rc):
    R = np.asarray(R, dtype=float)
    x = R / Rc
    f = R - lam * Rc * np.tanh(x)
    fp = 1 - lam / np.cosh(x) ** 2
    fpp = (2 * lam / Rc) * np.tanh(x) / np.cosh(x) ** 2
    return f, fp, fpp


# ---------------------------------------------------------------- Model B
def model_B(R, lam, Rc):
    R = np.asarray(R, dtype=float)
    x2 = (R / Rc) ** 2
    f = R - lam * Rc * np.tanh(x2)
    fp = 1 - (2 * lam * R / Rc) / np.cosh(x2) ** 2
    fpp = (2 * lam / Rc**3) * (4 * R**2 * np.tanh(x2) - Rc**2) / np.cosh(x2) ** 2
    return f, fp, fpp


# ---------------------------------------------------------- Hu-Sawicki n=1
def hu_sawicki_n1(R, c1, c2, Om0, H0=1.0):
    """Eq. (8.1)-(8.2). m2 = Omega_m0 * H0^2 is physically fixed, not free."""
    R = np.asarray(R, dtype=float)
    m2 = Om0 * H0**2
    denom = 1 + c2 * R / m2
    f = R - m2 * c1 * (R / m2) / denom
    fp = 1 - c1 / denom**2
    fpp = (2 * c1 * c2 / m2) / denom**3
    return f, fp, fpp


def check_viability(fp, fpp, f, R, Lambda=None):
    """N1 (no ghost), N2 (no tachyon), and (optionally) N3 residual R-2*Lambda."""
    n1 = np.all(fp > 0)
    n2 = np.all(fpp >= -1e-12)
    out = {"N1_no_ghost": bool(n1), "N2_no_tachyon": bool(n2)}
    if Lambda is not None:
        out["N3_residual_at_Rmax"] = float((f - (R - 2 * Lambda))[-1])
    return out


if __name__ == "__main__":
    R = np.linspace(1e-3, 8, 400)
    fA, fAp, fApp = model_A(R, lam=0.7, Rc=1.0)
    fB, fBp, fBpp = model_B(R, lam=0.7, Rc=1.0)
    print("Model A viability:", check_viability(fAp, fApp, fA, R))
    print("Model B viability:", check_viability(fBp, fBpp, fB, R))
