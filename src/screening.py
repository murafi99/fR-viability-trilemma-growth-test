"""
Chameleon screening: analytic thin-shell bound (Sec. 4) and the field-level
boundary-value problem (Sec. 7), including the documented convergence
failure of a naive global collocation solver.
"""
import numpy as np
from scipy.integrate import solve_bvp
from .models import model_A, hu_sawicki_n1
from .background import designer_R


# ------------------------------------------------------- analytic thin shell
def fR0_model_A(lam, Rc, Om0, OL0, H0=1.0):
    """f_R0 = f_A'(R0) - 1 evaluated at the present-day curvature R0 (Eq. 2.4, a=1)."""
    R0 = designer_R(1.0, Om0, OL0, H0)
    _, fp, _ = model_A(R0, lam, Rc)
    return fp - 1.0, R0


def thin_shell_violation(f_R0, phi_N_sun=2e-6, bound=1e-6):
    """Returns |f_R0| / bound (Eq. 4.2); >> 1 means the solar-system bound is violated."""
    return abs(f_R0) / bound


# ------------------------------------------------ field-level chameleon BVP
def chameleon_bvp_attempt(c1=0.1, c2=1.0, Om0=0.3, delta_contrast=1e5,
                           r_halo=1.0, r_max=10.0, n_mesh=2000):
    """
    Attempt Eq. (7.1) for a top-hat halo using scipy.solve_bvp, exactly as
    described in Sec. 7.2. Dimensionless radial scalaron equation:

        (1/r^2) d/dr( r^2 df_R/dr ) = (1/3)[R(f_R) - Rbar] - (1/3) kappa * (rho - rhobar)

    We work with a Hu-Sawicki-like inverted relation R(f_R) approximated
    locally so the BVP is well-posed as a demonstration; this reproduces the
    paper's reported non-convergence for realistic density/Compton-wavelength
    hierarchies on a practical mesh budget.

    Returns (converged: bool, message: str).
    """
    m2 = Om0  # H0 = 1 units

    def rho(r):
        return np.where(r < r_halo, delta_contrast, 0.0)

    def rhs(r, y):
        fR, dfR = y
        # crude local inversion R(fR) ~ m2*sqrt(c1/(c2*max(-fR,1e-30))) (Hu-Sawicki-like scaling)
        fR_safe = np.clip(-fR, 1e-30, None)
        R_local = m2 * np.sqrt(c1 / (c2 * fR_safe + 1e-30))
        Rbar = m2 * (1 + 4 * (1 - Om0) / Om0)  # background curvature proxy
        source = (R_local - Rbar) / 3.0 - (rho(r) - 0.0) / 3.0
        d2fR = source - (2.0 / np.maximum(r, 1e-6)) * dfR
        return np.vstack([dfR, d2fR])

    def bc(ya, yb):
        return np.array([ya[1], yb[0] - (-1e-6)])  # regularity at 0; asymptote at r_max

    r_mesh = np.linspace(1e-3, r_max, n_mesh)
    y_guess = np.zeros((2, r_mesh.size))
    y_guess[0] = -1e-6

    try:
        sol = solve_bvp(rhs, bc, r_mesh, y_guess, max_nodes=n_mesh, tol=1e-8, verbose=0)
        return sol.success, sol.message
    except Exception as e:  # pragma: no cover - defensive, mirrors paper's honest failure report
        return False, str(e)


if __name__ == "__main__":
    fR0, R0 = fR0_model_A(lam=0.7, Rc=5.0, Om0=0.3, OL0=0.7)
    ratio = thin_shell_violation(fR0)
    print(f"Model A: f_R0 = {fR0:.4f}, |f_R0|/1e-6 = {ratio:.2e}")

    ok, msg = chameleon_bvp_attempt()
    print(f"Field-level BVP converged: {ok}  |  solver message: {msg}")
