import numba as nb
import numpy as np

from numba_extinction.utils import helpers

# =========================================================================== #


Go23_uv_a_params = np.array(
    [
        4.6,
        0.99,
        0.81297,
        0.2775,
        1.06295,
        0.11303,
        5.9,
        0.0,  # useless, just a placeholder
        4.6**2,
        0.99**2,
    ]
)

# =========================================================================== #


Go23_uv_b_params = np.array(
    [
        4.6,
        0.99,
        -2.97868,
        1.89808,
        3.10334,
        0.65484,
        5.9,
        0.0,  # useless, just a placeholder
        4.6**2,
        0.99**2,
    ]
)

# =========================================================================== #


Go23_opt_a_params = np.array(
    [
        -0.35848,
        0.71220,
        0.08746,
        -0.05403,
        0.00674,
        0.03893,
        2.288**2,
        0.243**2,
        0.02965,
        2.054**2,
        0.179**2,
        0.01747,
        1.587**2,
        0.243**2,
    ]
)
# =========================================================================== #


Go23_opt_b_params = np.array(
    [
        0.12345,
        -2.68335,
        2.01901,
        -0.39299,
        0.03355,
        0.18453,
        2.288**2,
        0.243**2,
        0.19728,
        2.054**2,
        0.179**2,
        0.17130,
        1.587**2,
        0.243**2,
    ]
)

# =========================================================================== #

Go23_ir_a_params = np.array(
    [
        0.38526,
        1.68467,
        4.30578,
        4.78338,
        0.78791,
        0.06652,
        9.8434,
        2.21205,
        -0.24703,
        0.0267,
        19.58294,  # 19.258294 -> this is what is reported in the paper
        17.0,
        -0.27,
    ]
)

# =========================================================================== #

Go23_ir_b_params = np.array(
    [
        -1.01251,
        -1.06099,
    ]
)


# =========================================================================== #


@nb.jit
def F_Go23(x, th=5.9):
    # see FM90 for this 5.9, eq. 4 in the referece above
    # defaults to this number for lack of a better option
    if x < th:
        return 0.0
    else:
        return 0.5392 * (x - th) ** 2 + 0.05644 * (x - th) ** 3


# =========================================================================== #


@nb.jit
def k_Go23(x, params):
    return (
        params[2]
        + params[3] * x
        + params[4] * helpers.D(x, params[8], params[9])
        + params[5] * F_Go23(x, params[6])
    )


# =========================================================================== #


@nb.jit
def uv_inum(x, params_a, params_b):
    return k_Go23(x, params_a), k_Go23(x, params_b)


# =========================================================================== #


@nb.jit
def opt_inum(x, params_a, params_b):
    a = (
        params_a[0]
        + params_a[1] * x
        + params_a[2] * x**2
        + params_a[3] * x**3
        + params_a[4] * x**4
        + params_a[5] * helpers.D(x, params_a[6], params_a[7]) * params_a[7]
        + params_a[8] * helpers.D(x, params_a[9], params_a[10]) * params_a[10]
        + params_a[11] * helpers.D(x, params_a[12], params_a[13]) * params_a[13]
    )

    b = (
        params_b[0]
        + params_b[1] * x
        + params_b[2] * x**2
        + params_b[3] * x**3
        + params_b[4] * x**4
        + params_b[5] * helpers.D(x, params_b[6], params_b[7]) * params_b[7]
        + params_b[8] * helpers.D(x, params_b[9], params_b[10]) * params_b[10]
        + params_b[11] * helpers.D(x, params_b[12], params_b[13]) * params_b[13]
    )
    return a, b


# =========================================================================== #


@nb.jit
def ir(lambda_, params_a, params_b):
    w = helpers.W(lambda_, params_a[2], params_a[3])
    a = (
        params_a[0]
        * (
            lambda_ ** (-params_a[1]) * (1 - w)
            + params_a[2] ** (-params_a[1])
            * (lambda_ / params_a[2]) ** (-params_a[4])
            * w
        )
        + params_a[5] * helpers.mod_D(lambda_, params_a[6], params_a[7], params_a[8])
        + params_a[9] * helpers.mod_D(lambda_, params_a[10], params_a[11], params_a[12])
    )
    b = (
        params_b[0] * lambda_ ** (params_b[1])
    )  # slighly different from the paper, but needed to match the other package
    return a, b


# =========================================================================== #


@nb.jit
def Go23_inum(lambda_, x):
    assert 0.0912 < lambda_ < 32, "Valid range is 0.0912 <= x <= 32 micro_m^-1"

    if lambda_ < 0.3:
        return uv_inum(x, Go23_uv_a_params, Go23_uv_b_params)
    elif lambda_ < 0.33:
        w_1 = helpers.W(lambda_, 0.315, 0.03)
        uv_a, uv_b = uv_inum(x, Go23_uv_a_params, Go23_uv_b_params)
        opt_a, opt_b = opt_inum(x, Go23_opt_a_params, Go23_opt_b_params)
        return (1 - w_1) * uv_a + w_1 * opt_a, (1 - w_1) * uv_b + w_1 * opt_b
    elif lambda_ < 0.9:
        return opt_inum(x, Go23_opt_a_params, Go23_opt_b_params)
    elif lambda_ < 1.1:
        w_2 = helpers.W(lambda_, 1.0, 0.2)
        opt_a, opt_b = opt_inum(x, Go23_opt_a_params, Go23_opt_b_params)
        ir_a, ir_b = ir(lambda_, Go23_ir_a_params, Go23_ir_b_params)
        return (1 - w_2) * opt_a + w_2 * ir_a, (1 - w_2) * opt_b + w_2 * ir_b
    elif lambda_ < 32:
        return ir(lambda_, Go23_ir_a_params, Go23_ir_b_params)


# =========================================================================== #


@nb.jit(parallel=True)
def compute_exctinction(wave, a_v, r_v):
    out = np.empty(len(wave), dtype=np.float64)

    # This is a bit of a mix between wave and inverse wave, so we do the conversion here
    #  once and for all - note that here lambda is in microns from the previous call,
    #  for all the other formulas, so we convert to AA
    x = helpers.aa_to_invum(wave * 1e4)

    for i in nb.prange(len(wave)):
        a, b = Go23_inum(wave[i], x[i])
        out[i] = a_v * (a + b * (1 / r_v - 1 / 3.1))
    return out


# =========================================================================== #


@nb.jit(parallel=True)
def compute_exctinction_many(wave, a_v, r_v):
    out = np.empty(wave.shape, dtype=np.float64)
    for i in nb.prange(wave.shape[0]):
        out[i] = compute_exctinction(wave[i], a_v[i], r_v)

    return out
