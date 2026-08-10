"""
Linear growth of structure, delta(a), and f*sigma8(z).

- Geff_over_G_superCompton(R, model):        G_eff/G = 1/f'(R)             (Sec. 2.4)
- Geff_over_G_scale_dependent(R, k, a, ...): full k-dependent form         (Eq. 6.1)
- solve_growth(...):                          integrates Eq. (2.8) or the
                                               scale-dependent analogue for delta(N)
- fsigma8(...):                               converts delta(N) to the observable
"""
import numpy as np
from scipy.integrate import solve_ivp
from .background import designer_R


def Om_of_a(a, Om0, OL0):
    return Om0 * a ** (-3) / (Om0 * a ** (-3) + OL0)


def growth_rhs_superCompton(N, state, Om0, OL0, fprime_func, H0=1.0):
    """Eq. (2.8): delta'' + (2 - 1.5 Om(a)) delta' - 1.5 Om(a) (Geff/G) delta = 0."""
    delta, ddelta = state
    a = np.exp(N)
    Om = Om_of_a(a, Om0, OL0)
    R = designer_R(a, Om0, OL0, H0)
    _, fp, _ = fprime_func(R)
    Geff_over_G = 1.0 / fp
    dddelta = -(2 - 1.5 * Om) * ddelta + 1.5 * Om * Geff_over_G * delta
    return [ddelta, dddelta]


def growth_rhs_scale_dependent(N, state, Om0, OL0, model_func, k, H0=1.0):
    """Uses the full G_eff(k,a) of Eq. (6.1) instead of the super-Compton limit."""
    delta, ddelta = state
    a = np.exp(N)
    Om = Om_of_a(a, Om0, OL0)
    R = designer_R(a, Om0, OL0, H0)
    f, fp, fpp = model_func(R)
    fpp_safe = fpp if abs(fpp) > 1e-30 else 1e-30
    M2 = fp / (3 * fpp_safe)
    kappa2 = (k / a) ** 2
    Geff_over_G = (1.0 / fp) * (1 + (1.0 / 3.0) * kappa2 / (kappa2 + M2))
    dddelta = -(2 - 1.5 * Om) * ddelta + 1.5 * Om * Geff_over_G * delta
    return [ddelta, dddelta]


def solve_growth(Om0, OL0, fprime_func, N_i=-6.0, N_f=0.0, k=None, model_func=None,
                  H0=1.0, n_points=500):
    """
    Matter-dominated initial conditions: delta ~ a, delta' ~ delta (growing mode).
    If k is given, uses the scale-dependent equation with model_func(R)->(f,fp,fpp);
    otherwise uses the super-Compton limit with fprime_func(R)->(f,fp,fpp).
    """
    a_i = np.exp(N_i)
    delta_i = a_i
    ddelta_i = delta_i  # d(delta)/dN = delta in matter domination

    if k is None:
        args = (Om0, OL0, fprime_func, H0)
        rhs = growth_rhs_superCompton
    else:
        args = (Om0, OL0, model_func, k, H0)
        rhs = growth_rhs_scale_dependent

    sol = solve_ivp(rhs, [N_i, N_f], [delta_i, ddelta_i], args=args,
                     method="RK45", rtol=1e-9, atol=1e-12,
                     t_eval=np.linspace(N_i, N_f, n_points))
    N = sol.t
    delta = sol.y[0]
    ddelta = sol.y[1]
    f_growth = ddelta / delta  # dlnD/dlna
    return N, delta, f_growth


def fsigma8_of_z(Om0, OL0, fprime_func, sigma8_0, z_targets, k=None, model_func=None, H0=1.0):
    """Evaluate f(z)*sigma8(z) at requested redshifts, normalizing D(a=1)=1."""
    N, delta, f_growth = solve_growth(Om0, OL0, fprime_func, k=k, model_func=model_func, H0=H0)
    D_norm = delta / delta[-1]  # normalize so D(z=0) = 1
    sigma8_of_a = sigma8_0 * D_norm
    a_grid = np.exp(N)
    z_grid = 1 / a_grid - 1

    out = []
    for z in z_targets:
        idx = np.argmin(np.abs(z_grid - z))
        out.append(f_growth[idx] * sigma8_of_a[idx])
    return np.array(out)


if __name__ == "__main__":
    from .models import hu_sawicki_n1

    Om0, OL0 = 0.3, 0.7

    def fprime_GR(R):
        return R, np.ones_like(R), np.zeros_like(R)

    z = np.array([0.02, 0.15, 0.5, 1.0, 1.52])
    vals = fsigma8_of_z(Om0, OL0, fprime_GR, sigma8_0=0.811, z_targets=z)
    print("GR f*sigma8 at z =", z, "->", vals)
