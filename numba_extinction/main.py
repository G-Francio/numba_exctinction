import timeit

import extinction as original_ext
import matplotlib.pyplot as plt
import numpy as np
from astropy import units
from dust_extinction.parameter_averages import G23

import numba_extinction.ne as numba_ext

original_ext

if __name__ == "__main__":
    # check that extinction.pip and this version give the same output
    # this was run with all the available function
    wave = np.arange(11000, 320000, 0.1) * units.AA
    n_rep = 250

    a_v = 1.0
    r_v = 3.1

    # ref = original_ext.fm07(wave, a_v)

    ref = G23(Rv=r_v)(wave) * a_v
    upd = numba_ext.Go23(wave, a_v, r_v)

    print(f"[Info] Maximum difference: {np.max(ref - upd):.3e}")
    print(
        f"[Info] Time taken for the original implementation: {timeit.timeit('G23(Rv=r_v)(wave) * a_v', number = 100, globals=globals()):.3f}"
    )
    print(
        f"[Info] Time taken for the numba version: {timeit.timeit('upd = numba_ext.Go23(wave, a_v, r_v)', number = 100, globals=globals()):.3f}"
    )

    # make plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6 / 1.61), layout="constrained")

    ax.plot(wave, ref, label="Reference PKG")
    ax.plot(wave, upd, label="This PKG")

    ax.set_xlabel("Wavelength [AA]")
    ax.set_ylabel(r"A$_{v}$")

    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.show()
