# =========================================================================== #
# ============================= Helper functions ============================ #
# =========================================================================== #
import numba as nb
import numpy as np

# =========================================================================== #


@nb.jit
def D(x, x0_sq, gamma_sq):
    x_sq = x**2
    return x_sq / ((x_sq - x0_sq) ** 2 + x_sq * gamma_sq)


# =========================================================================== #


@nb.jit
def aa_to_invum(x):
    return 1e4 / x


# =========================================================================== #


@nb.jit
def W(lambda_, lambda_b, d):
    # eq. 10 paper
    z = (lambda_ - (lambda_b - 0.5 * d)) / d
    if z < 0:
        return 0.0
    elif z > 1: # different from the paper, matches the implementation in dust_extinction
        return 1.0
    else:
        return 3.0 * z**2 - 2.0 * z**3


# =========================================================================== #


@nb.jit
def mod_D(lambda_, lambda_0, gamma_0, a):
    # eq. 13 paper
    gamma = (2 * gamma_0) / (1 + np.exp(a * (lambda_ - lambda_0)))
    return (gamma / lambda_0) ** 2 / (
        (lambda_ / lambda_0 - lambda_0 / lambda_) ** 2 + (gamma / lambda_0) ** 2
    )
