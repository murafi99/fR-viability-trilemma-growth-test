"""
Background expansion history.

- designer_R(a):    Eq. (2.4), R(a) evaluated on the standard LCDM solution
                     ("designer" approximation used through Section 3/8/9).
- exact_ode_rhs:     Eq. (5.2), the closed second-order ODE for y(N) = E^2(N)
                     obtained without approximating R.
- forward_shoot():   Integrates Eq. (5.2) forward from matter domination and
                     measures the growth rate of the instability reported in
                     Section 5.2 (~150x per e-fold).
"""
import numpy as np
from scipy.integrate import solve_ivp
from .models import hu_sawicki_n1


def designer_R(a, Om0, OL0, H0=1.0):
    """Eq. (2.4): R(a) = 3 H0^2 [Om0 a^-3 + 4 OL0]."""
    return 3 * H0**2 * (Om0 * a ** (-3) + 4 * OL0)


def _fR_derivs(R, c1, c2, Om0, H0=1.0):
    f, fp, fpp = hu_sawicki_n1(R, c1, c2, Om0, H0)
    return f, fp, fpp


def exact_ode_rhs(N, state, c1, c2, Om0, H0=1.0):
    """
    Eq. (5.2): 9 y f'' y'' = 3 f' y - Om0 a^-3 - 0.5*(f' R - f) + 36 y y' f''
    with R = 12y + 3y',  a = exp(N).
    state = [y, y']  ->  returns [y', y'']
    """
    y, yp = state
    a = np.exp(N)
    R = 12 * y + 3 * yp
    f, fp, fpp = _fR_derivs(R, c1, c2, Om0, H0)
    if abs(fpp) < 1e-30:
        fpp = 1e-30
    rhs = 3 * fp * y - Om0 * a ** (-3) - 0.5 * (fp * R - f) + 36 * y * yp * fpp
    ypp = rhs / (9 * y * fpp)
    return [yp, ypp]


def lcdm_y(N, Om0, OL0):
    a = np.exp(N)
    return Om0 * a ** (-3) + OL0


def forward_shoot(c1=0.1, c2=0.5, Om0=0.3, OL0=0.7, N0=-9.0, Nf=-6.0, n_points=400):
    """
    Reproduce the forward-shooting instability of Section 5.2.
    Starts deep in matter domination with LCDM designer initial conditions
    and integrates forward; returns the fractional deviation |dy/y| vs N.
    """
    y0 = lcdm_y(N0, Om0, OL0)
    # y'(N) for LCDM: d/dN [Om0 a^-3 + OL0] = -3 Om0 a^-3
    yp0 = -3 * Om0 * np.exp(N0) ** (-3)

    sol = solve_ivp(
        exact_ode_rhs, [N0, Nf], [y0, yp0], args=(c1, c2, Om0, H0 := 1.0),
        method="Radau", dense_output=True, rtol=1e-10, atol=1e-12,
        t_eval=np.linspace(N0, Nf, n_points),
    )
    y_num = sol.y[0]
    y_lcdm = lcdm_y(sol.t, Om0, OL0)
    frac_dev = np.abs((y_num - y_lcdm) / y_lcdm)
    return sol.t, frac_dev, sol.success


if __name__ == "__main__":
    N, dev, ok = forward_shoot()
    print("Integration success:", ok)
    print("Fractional deviation at start / end:", dev[0], dev[-1])
    # per-e-fold growth factor, avoiding the first ~zero point
    nz = dev > 1e-14
    if nz.sum() > 1:
        idx = np.where(nz)[0]
        growth_per_efold = (dev[idx[-1]] / dev[idx[0]]) ** (1 / (N[idx[-1]] - N[idx[0]]))
        print(f"Approx growth factor per e-fold: {growth_per_efold:.1f}x")
