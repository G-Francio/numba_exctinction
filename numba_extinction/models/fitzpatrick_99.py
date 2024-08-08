# =========================================================================== #
# =========================================================================== #
# ============================ Fitzpatrick (1999) =========================== #
# =========================================================================== #
# =========================================================================== #

# Quoted literature:
#  - FM90: https://articles.adsabs.harvard.edu/pdf/1990ApJS...72..163F
#     ^ provides theoretical bkg and is what Fi99 also uses
#  - Fi99: https://iopscience.iop.org/article/10.1086/316293/pdf
#     ^ parameters were taken from here, see appendix


# =========================================================================== #
# ================================ Constants ================================ #
# =========================================================================== #
import numba as nb
import numpy as np

import numba_extinction.utils.cubic_spline as cs
from numba_extinction.utils import helpers

# perfomance reasons, adds a factor of two to execution time...
Fi99_params = np.array(
    [
        4.596,
        0.99,
        3.23,
        0.41,
        5.9,
        4.596**2,
        0.99**2,
    ]
)


# =========================================================================== #

# see paper, table 3
Fi99_x_knots = np.array(
    [
        0.0,
        1.0e4 / 26500.0,
        1.0e4 / 12200.0,
        1.0e4 / 6000.0,
        1.0e4 / 5470.0,
        1.0e4 / 4670.0,
        1.0e4 / 4110.0,
        1.0e4 / 2700.0,  # Use the functional form to compute this
        1.0e4 / 2600.0,  # Use the functional form to compute this
    ]
)


# =========================================================================== #


@nb.jit
def Fi99_y_knots(r_v, params):
    # For the reasoning behind this formulae see page 11
    # However, following the exctinction package, we
    #  simply match the IDL routine so these will be different
    #  from the original paper.
    # In extinction.py there is a 3.1 subtracted everywhere
    #  that is used to match the fact that in the IDL library
    #  the equivalent of k_ here has a + 3.1.
    # For consistency we follow extinction.py
    return np.array(
        [
            +0.0 - r_v,
            +0.264690 * r_v / 3.1 - r_v,
            +0.829250 * r_v / 3.1 - r_v,
            -0.422809 + 1.00270 * r_v + 2.13572e-04 * r_v**2 - r_v,
            -0.051354 + 1.00216 * r_v - 7.35778e-05 * r_v**2 - r_v,
            +0.700127 + 1.00184 * r_v - 3.32598e-05 * r_v**2 - r_v,
            # 1.01707 * r_v - r_v = (1.01707 - 1) * r_v = 0.01707 * r_v
            1.19456
            + 0.01707 * r_v
            - 5.46959e-03 * r_v**2
            + 7.97809e-04 * r_v**3
            - 4.45636e-05 * r_v**4,
            k_Fi99(1.0e4 / 2700.0, r_v, params),
            k_Fi99(1.0e4 / 2600.0, r_v, params),
        ]
    )


# =========================================================================== #
# ================================ Functions ================================ #
# =========================================================================== #


@nb.jit
def F_Fi99(x, th=5.9):
    # see FM90 for this 5.9, eq. 4 in the referece above
    # defaults to this number for lack of a better option
    if x < th:
        return 0.0
    else:
        return 0.5392 * (x - th) ** 2 + 0.05644 * (x - th) ** 3


# =========================================================================== #


@nb.jit
def k_Fi99(x, r_v, params):
    c2 = -0.824 + 4.717 / r_v
    c1 = 2.030 - 3.007 * c2

    return (
        c1
        + c2 * x
        + params[2] * helpers.D(x, params[5], params[6])
        + params[3] * F_Fi99(x, params[4])
    )


# =========================================================================== #


@nb.jit
def Fi99_uv_invum(x, a_v, r_v, params):
    # according to the IDL library, valid from 0.1 to 3.5 microns
    k = k_Fi99(x, r_v, params)
    return a_v * (1.0 + k / r_v)


# =========================================================================== #


@nb.jit
def compute_exctinction(wave, a_v, r_v, params, x_knots, y_knots, a, b):
    x = helpers.aa_to_invum(wave)
    out = np.empty(len(x), dtype=np.float64)

    for i in range(len(x)):
        if x[i] < 1e4 / 2700:
            out[i] = (
                a_v / r_v * (cs.cubic_y_w_coeff(x_knots, x[i], y_knots, a, b) + r_v)
            )
        else:
            out[i] = Fi99_uv_invum(x[i], a_v, r_v, params)

    return out
