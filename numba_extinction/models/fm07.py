# =========================================================================== #
# =========================================================================== #
# ======================== Fitzpatrick & Massa (2007) ======================= #
# =========================================================================== #
# =========================================================================== #
import numba as nb
import numpy as np

import numba_extinction.utils.cubic_spline as cs
from numba_extinction.utils import helpers

# =========================================================================== #

FM07_params = np.array(
    [
        4.592,
        0.922,
        -0.175,
        0.807,
        2.991,
        0.319,
        6.097,
        3.1,
        4.592**2,
        0.922**2,
    ]
)


# =========================================================================== #


@nb.jit
def F_FM07(x, th=6.097):
    # defaults to the actual threshold for lack of a better option
    if x < th:
        return 0.0
    else:
        return (x - th) ** 2


# =========================================================================== #


@nb.jit
def k_FM07(x, params):
    return (
        params[2]
        + params[3] * x
        + params[4] * helpers.D(x, params[8], params[9])
        + params[5] * F_FM07(x, params[6])
    )


# =========================================================================== #


@nb.jit
def FM07_uv_invum(x, a_v, params):
    # according to the IDL library, valid from 0.1 to 3.5 microns
    k = k_FM07(x, params)
    return a_v * (1.0 + k / params[7])


# =========================================================================== #

FM07_x_knots = np.array(
    [
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
        1.0e4 / 5530.0,
        1.0e4 / 4000.0,
        1.0e4 / 3300.0,
        1.0e4 / 2700.0,
        1.0e4 / 2600.0,
    ]
)

# =========================================================================== #


# here we just need to compute things once, keep in a function to keep things tidy
@nb.jit
def FM07_y_knots(r_v, params):
    return np.array(
        [
            -r_v,
            # probably don't need all of these decimal places, oh well
            (-0.83 + 0.63 * r_v) * 0.07802065930635074 - r_v,
            (-0.83 + 0.63 * r_v) * 0.27932178451805495 - r_v,
            (-0.83 + 0.63 * r_v) * 0.58899651431966620 - r_v,
            (-0.83 + 0.63 * r_v) * 1.00000000000000000 - r_v,
            0.0,
            1.322,
            2.055,
            k_FM07(1.0e4 / 2700, params),
            k_FM07(1.0e4 / 2600, params),
        ]
    )


# =========================================================================== #


@nb.jit(parallel=True)
def compute_exctinction(wave, a_v, r_v, params, x_knots, y_knots, a, b):
    x = helpers.aa_to_invum(wave)
    out = np.empty(len(x), dtype=np.float64)

    for i in nb.prange(len(x)):
        if x[i] < 1e4 / 2700:
            out[i] = (
                a_v / r_v * (cs.cubic_y_w_coeff(x_knots, x[i], y_knots, a, b) + r_v)
            )
        else:
            out[i] = FM07_uv_invum(x[i], a_v, params)

    return out
