# numba_extinction
Numba'd version of the [`extinction`](https://github.com/kbarbary/extinction) package. Most of this was developed before finding the much-more-up-to-date [`dust_extinction`](https://github.com/karllark/dust_extinction).

The exctinction curves reproduce exactly (up to some e-15 due to float math) the curves produced by `extinction`. In addition, an equivalent implementation of the UV to IR extinction curve by [Gordon et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...950...86G/abstract) was implemented, and checked against `dust_extinction` for consistency.

I encourage you to check both packages out! `dust_exctinction` in particular is much more fleshed out and complete than the this package. Eventually I might try to implement all of the curves available there, but that is if I'll have time in the future.

### Install instruction
- clone the repository: `git clone git@github.com:G-Francio/numba_exctinction.git`
- install as local package: `pip install -e .`

#### Requirements
Optional packages are only used for consistency checks and visualisation purposes.
- numba
- astropy
- scipy
- matplotlib [optional]
- extinction [optional]
- dust_exctinction [optional]

### Links:
- [extinction](https://github.com/kbarbary/extinction)
- [dust_extinction](https://github.com/karllark/dust_extinction)
