# =========================================================================== #
# =========================================================================== #
# ============================= O'Donnell (1994) ============================ #
# =========================================================================== #
# =========================================================================== #
import numba as nb
import numpy as np

from numba_extinction.models import ccm_89
from numba_extinction.utils import helpers


@nb.jit
def od94_opt_invum(x):
    """od94 a, b parameters for 1.1 < x < 3.3 (optical)"""
    y = x - 1.82
    a = (
        1
        + 0.104 * y
        - 0.609 * y**2
        + 0.701 * y**3
        + 1.137 * y**4
        - 1.718 * y**5
        - 0.827 * y**6
        + 1.647 * y**7
        - 0.505 * y**8
    )
    b = (
        1.952 * y
        + 2.908 * y**2
        - 3.989 * y**3
        - 7.985 * y**4
        + 11.102 * y**5
        + 5.491 * y**6
        - 10.805 * y**7
        + 3.347 * y**8
    )

    return a, b


# =========================================================================== #


@nb.jit
def od94_invum(x):
    # along the entire range
    assert 0.3 < x < 10, "Valid range is 0.3 <= x <= 10 micro_m^-1"
    if x < 1.1:
        return ccm_89.ccm89_ir_invum(x)
    elif x < 3.3:
        return od94_opt_invum(x)
    elif x < 8.0:
        return ccm_89.ccm89_uv_invum(x)
    else:
        return ccm_89.ccm89_fuv_invum(x)


# =========================================================================== #


@nb.jit
def compute_od94_exctinction(wave, a_v, r_v):
    out = np.empty(len(wave), dtype=np.float64)
    for i in range(len(wave)):
        a, b = od94_invum(helpers.aa_to_invum(wave[i]))
        out[i] = a_v * (a + b / r_v)
    return out