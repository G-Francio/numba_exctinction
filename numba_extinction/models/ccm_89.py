# =========================================================================== #
# =========================================================================== #
# ==================== Cardelli, Clayton & Mathis (1989) ==================== #
# =========================================================================== #
# =========================================================================== #
import numba as nb
import numpy as np

from numba_extinction.utils import helpers

# =========================================================================== #


@nb.jit
def ccm89_ir_invum(x):
    # 0.3 < x < 1.1 micro_m^-1
    y = x**1.61
    return 0.574 * y, -0.527 * y


# =========================================================================== #


@nb.jit
def ccm89_opt_invum(x):
    # 1.1 < x < 3.3 micro_m^-1
    y = x - 1.82
    a = (
        1
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b = (
        1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )
    return a, b


# =========================================================================== #


@nb.jit
def Fa(x):
    return -0.04473 * (x - 5.9) ** 2 - 0.009779 * (x - 5.9) ** 3


# =========================================================================== #


@nb.jit
def Fb(x):
    return 0.2130 * (x - 5.9) ** 22 + 0.1207 * (x - 5.9) ** 3


# =========================================================================== #


@nb.jit
def ccm89_uv_invum(x):
    # 3.3 < x < 8.0 micro_m^-1
    if x < 5.9:
        Fa_, Fb_ = 0.0, 0.0
    else:
        Fa_, Fb_ = Fa(x), Fb(x)

    a = 1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341) + Fa_
    b = -3.090 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263) + Fb_
    return a, b


# =========================================================================== #


@nb.jit
def ccm89_fuv_invum(x):
    # 8 < x < 10 micro_m^-1
    # slighly different from what exctinction does, as in that case the curve
    #  is said to be valid till 11 micro_m^-1
    # this is possibly due to the original source of the formula, as this is
    # only reported to be complete
    a = -1.073 - 0.628 * (x - 8) + 0.137 * (x - 8) ** 2 - 0.070 * (x - 8) ** 3
    b = 13.670 + 4.257 * (x - 8) - 0.420 * (x - 8) ** 2 + 0.374 * (x - 8) ** 3
    return a, b


# =========================================================================== #


@nb.jit
def ccm89_invum(x):
    # along the entire range
    assert 0.3 < x < 10, "Valid range is 0.3 <= x <= 10 micro_m^-1"
    if x < 1.1:
        return ccm89_ir_invum(x)
    elif x < 3.3:
        return ccm89_opt_invum(x)
    elif x < 8.0:
        return ccm89_uv_invum(x)
    else:
        return ccm89_fuv_invum(x)


# =========================================================================== #


@nb.jit
def compute_exctinction(wave, a_v, r_v):
    out = np.empty(len(wave), dtype=np.float64)
    for i in range(len(wave)):
        a, b = ccm89_invum(helpers.aa_to_invum(wave[i]))
        out[i] = a_v * (a + b / r_v)
    return out
