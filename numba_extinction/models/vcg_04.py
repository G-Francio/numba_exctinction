# =========================================================================== #
# =========================================================================== #
# ============================= O'Donnell (1994) ============================ #
# =========================================================================== #
# =========================================================================== #
import numba as nb
import numpy as np

from numba_extinction.models import ccm_89
from numba_extinction.utils import helpers


@nb.jit()
def Fa(x):
    return -0.0077 * (x - 5.9) ** 2 - 0.0003 * (x - 5.9) ** 3


# =========================================================================== #


@nb.jit()
def Fb(x):
    return 0.2060 * (x - 5.9) ** 2 - 0.0550 * (x - 5.9) ** 3


# =========================================================================== #


@nb.jit
def vcg04_opt_invum(x):
    if x < 5.9:
        Fa_, Fb_ = 0.0, 0.0
    else:
        Fa_, Fb_ = Fa(x), Fb(x)

    a = 1.808 - 0.215 * x - 0.134 / ((x - 4.558) ** 2 + 0.566) + Fa_
    b = -2.350 + 1.403 * x + 1.103 / ((x - 4.587) ** 2 + 0.263) + Fb_
    return a, b


# =========================================================================== #


@nb.jit
def vcg04_invum(x):
    # along the entire range
    assert 0.3 < x < 10, "Valid range is 0.3 <= x <= 10 micro_m^-1"
    if x < 1.1:
        return ccm_89.ccm89_ir_invum(x)
    elif x < 3.3:
        return ccm_89.ccm89_opt_invum(x)
    elif x < 8.0:
        return vcg04_opt_invum(x)
    else:
        return ccm_89.ccm89_fuv_invum(x)


# =========================================================================== #


@nb.jit
def compute_exctinction(wave, a_v, r_v):
    out = np.empty(len(wave), dtype=np.float64)
    for i in range(len(wave)):
        a, b = vcg04_invum(helpers.aa_to_invum(wave[i]))
        out[i] = a_v * (a + b / r_v)
    return out
