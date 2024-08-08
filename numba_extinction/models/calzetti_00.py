# =========================================================================== #
# =========================================================================== #
# ============================== Calzetti (2000) ============================ #
# =========================================================================== #
# =========================================================================== #
import numba as nb
import numpy as np

from numba_extinction.utils import helpers


@nb.jit
def calzetti00k_uv_invum(x):
    # 0.12 microns < wave < 0.63 microns, see eq. 4 of paper
    return 2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3)


@nb.jit
def calzetti00k_ir_invum(x):
    # 0.63 microns < wave < 2.20 microns, see eq. 4 of paper
    return 2.659 * (1.040 * x - 1.857)


@nb.jit
def calzetti00_invum(x, r_v):
    if x > 1.0 / 0.63:
        k = calzetti00k_uv_invum(x)
    else:
        k = calzetti00k_ir_invum(x)

    return 1.0 + k / r_v


@nb.jit
def compute_extinction(wave, a_v, r_v):
    out = np.empty(len(wave), dtype=np.float64)
    i_wave = helpers.aa_to_invum(wave)
    for i in range(len(wave)):
        out[i] = a_v * calzetti00_invum(i_wave[i], r_v)
    return out
