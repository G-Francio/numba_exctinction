# Rewriting of the extinction package using python
# Performace improvements are achived through numba
# maybe not as fast but I understand much better what
# is going on

# Based on https://github.com/kbarbary/extinction and amply rewritten

# TODO: find a way to put \lambda in documentation

import warnings

import numba as nb
from astropy import units

from numba_extinction.models import calzetti_00 as C00_
from numba_extinction.models import ccm_89 as CCM89_
from numba_extinction.models import fitzpatrick_99 as Fi99_
from numba_extinction.models import fm07 as FM07_
from numba_extinction.models import gordon_23 as Go23_
from numba_extinction.models import odonnell_94 as OD94_
from numba_extinction.utils import cubic_spline as cs

# =========================================================================== #
# =========================================================================== #
#  No need to over optimise this, numba takes care of it - make it readable!  #
# =========================================================================== #
# =========================================================================== #


# =========================================================================== #
# ========================== Convenience functions ========================== #
# =========================================================================== #


@nb.jit
def redden(extinction, flux, inplace=False):
    """redden(extinction, flux, inplace=False)

    Convenience function to apply extinction to flux values (i.e., redden).
    It simply performs ``flux * 10**(-0.4 * extinction)``:
    flux is decreased (for positive extinction values).

    Extinction and flux should be broadcastable.

    :param extinction: Extinction in magnitude.
    :type extinction: numpy.ndarray
    :param flux: Flux values.
    :type flux: numpy.ndarray
    :param inplace: Modify in place, defaults to False
    :type inplace: bool, optional
    :return: Reddeded flux (copy or modification of in-memory object)
    :rtype: numpy.ndarray
    """
    fact = 10.0 ** (-0.4 * extinction)

    if inplace:
        flux *= fact
        return flux
    else:
        return flux * fact


# =========================================================================== #


@nb.jit
def deredden(extinction, flux, inplace=False):
    """deredden(extinction, flux, inplace=False)

    Convenience function to remove extinction to flux values (i.e., deredden).
    It simply performs ``flux * 10**(0.4 * extinction)``:
    flux is increased (for positive extinction values).

    Extinction and flux should be broadcastable.

    :param extinction: Extinction in magnitude.
    :type extinction: numpy.ndarray
    :param flux: Flux values.
    :type flux: numpy.ndarray
    :param inplace: Modify in place, defaults to False
    :type inplace: bool, optional
    :return: Reddeded flux (copy or modification of in-memory object)
    :rtype: numpy.ndarray
    """
    fact = 10.0 ** (0.4 * extinction)

    if inplace:
        flux *= fact
        return flux
    else:
        return flux * fact


# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


@units.quantity_input
def CCM89(wave: units.AA, a_v, r_v):
    """ccm89(wave: units.AA, a_v, r_v):

    Cardelli, Clayton & Mathis (1989) extinction function, see page 5
    of the pdf at https://articles.adsabs.harvard.edu/pdf/1989ApJ...345..245C.

    # TODO Check this
    The claimed validity is 1250 Angstroms to 3.3 microns.

    Parameters
    ----------
    wave : numpy.ndarray (1-d)
        Wavelengths or wavenumbers.
    a_v : float
        Scaling parameter, A_V: extinction in magnitudes at characteristic
        V band wavelength.
    r_v : float
        Ratio of total to selective extinction, A_V / E(B-V).

    Returns
    -------
    Extinction in magnitudes at each input wavelength.

    Notes
    -----
    In Cardelli, Clayton & Mathis (1989) the mean
    R_V-dependent extinction law, is parameterized as

    .. math::

       <A(\\lambda)/A_V> = a(x) + b(x) / R_V

    where the coefficients a(x) and b(x) are functions of
    wavelength. At a wavelength of approximately 5494.5 angstroms (a
    characteristic wavelength for the V band), a(x) = 1 and b(x) = 0,
    so that A(5494.5 angstroms) = A_V. This function returns

    .. math::

       A(\\lambda) = A_V (a(x) + b(x) / R_V)

    References
    ----------
    .. [1] Cardelli, J. A., Clayton, G. C., & Mathis, J. S. 1989, ApJ, 345, 245

    """
    if not wave.unit.is_equivalent(units.AA):
        raise ValueError(
            "Invalid units. Dispersion array should be convertible"
            "to `astropy.units.AA`"
        )

    return CCM89_.compute_ccm89_exctinction(wave.to(units.AA).value, a_v, r_v)


# =========================================================================== #


@units.quantity_input
def OD94(wave: units.AA, a_v, r_v):
    """odonnell94(wave, a_v, r_v, unit='aa', out=None)

    O'Donnell (1994) extinction function.

    Like Cardelli, Clayton, & Mathis (1989) [1]_ but using the O'Donnell
    (1994) [2]_ optical coefficients between 3030 A and 9091 A.

    Parameters
    ----------
    wave : numpy.ndarray (1-d)
        Wavelengths or wavenumbers.
    a_v : float
        Scaling parameter, A_V: extinction in magnitudes at characteristic
        V band wavelength.
    r_v : float
        Ratio of total to selective extinction, A_V / E(B-V).
    unit : {'aa', 'invum'}, optional
        Unit of wave: 'aa' (Angstroms) or 'invum' (inverse microns).
    out : np.ndarray, optional
        If specified, store output values in this array.

    Returns
    -------
    Extinction in magnitudes at each input wavelength.

    Notes
    -----
    This function matches the Goddard IDL astrolib routine CCM_UNRED.
    From the documentation for that routine:

    1. The CCM curve shows good agreement with the Savage & Mathis (1979)
       [3]_ ultraviolet curve shortward of 1400 A, but is probably
       preferable between 1200 and 1400 A.

    2. Curve is extrapolated between 912 and 1000 A as suggested by
       Longo et al. (1989) [4]_

    3. Valencic et al. (2004) [5]_ revise the ultraviolet CCM
       curve (3.3 -- 8.0 um^-1).    But since their revised curve does
       not connect smoothly with longer and shorter wavelengths, it is
       not included here.

    References
    ----------
    .. [1] Cardelli, J. A., Clayton, G. C., & Mathis, J. S. 1989, ApJ, 345, 245
    .. [2] O'Donnell, J. E. 1994, ApJ, 422, 158O
    .. [3] Savage & Mathis 1979, ARA&A, 17, 73
    .. [4] Longo et al. 1989, ApJ, 339,474
    .. [5] Valencic et al. 2004, ApJ, 616, 912

    """
    if not wave.unit.is_equivalent(units.AA):
        raise ValueError(
            "Invalid units. Dispersion array should be convertible"
            "to `astropy.units.AA`"
        )

    return OD94_.compute_od94_exctinction(wave.to(units.AA).value, a_v, r_v)


# =========================================================================== #


@units.quantity_input
def Fi99(wave: units.AA, a_v, r_v=3.1):
    """fitzpatrick99(wave, a_v, r_v=3.1, unit='aa')

    Fitzpatrick (1999) dust extinction function.

    Fitzpatrick (1999) [1]_ model which relies on the parametrization
    of Fitzpatrick & Massa (1990) [2]_ in the UV (below 2700 A) and
    spline fitting in the optical and IR. This function is defined
    from 910 A to 6 microns, but note the claimed validity goes down
    only to 1150 A. The optical spline points are not taken from F99
    Table 4, but rather updated versions from E. Fitzpatrick (this
    matches the Goddard IDL astrolib routine FM_UNRED).


    Parameters
    ----------
    wave : numpy.ndarray (1-d)
        Input wavelengths or wavenumbers (see units).
    a_v : float
        Total V-band extinction in magnitudes.
    r_v : float
        Ratio of total to selective extinction, A_V / E(B-V).
    unit : {'aa', 'invum'}, optional
        Wavelength units: Angstroms or inverse microns.

    Returns
    -------
    Extinction in magnitudes at each input wavelength.

    References
    ----------
    .. [1] Fitzpatrick, E. L. 1999, PASP, 111, 63
    .. [2] Fitzpatrick, E. L. & Massa, D. 1990, ApJS, 72, 163
    """
    # make sure the units are correct
    if not wave.unit.is_equivalent(units.AA):
        raise ValueError(
            "Invalid units. Dispersion array should be convertible"
            "to `astropy.units.AA`"
        )

    params = Fi99_.Fi99_params
    x_knots = Fi99_.Fi99_x_knots

    y_knots = Fi99_.Fi99_y_knots(r_v, params)
    a, b = cs.cubic_coef(x_knots, y_knots)

    return Fi99_.compute_Fi99_exctinction(
        wave.to(units.AA).value,
        a_v,
        r_v,
        params,
        x_knots,
        y_knots,
        a,
        b,
    )


# =========================================================================== #


@units.quantity_input
def C00(wave: units.AA, a_v, r_v):
    """calzetti00(wave, a_v, r_v, unit='aa', out=None)

    Calzetti (2000) extinction function.

    Calzetti et al. (2000, ApJ 533, 682) developed a recipe for
    dereddening the spectra of galaxies where massive stars dominate the
    radiation output, valid between 0.12 to 2.2 microns. They estimate
    :math:`R_V = 4.05 \\pm 0.80` from optical-IR observations of
    4 starburst galaxies.

    Parameters
    ----------
    wave : numpy.ndarray (1-d)
        Wavelengths or wavenumbers.
    a_v : float
        Scaling parameter, A_V: extinction in magnitudes at characteristic
        V band  wavelength.
    r_v : float
        Ratio of total to selective extinction, A_V / E(B-V).
    unit : {'aa', 'invum'}, optional
        Unit of wave: 'aa' (Angstroms) or 'invum' (inverse microns).
    out : np.ndarray, optional
        If specified, store output values in this array.

    Returns
    -------
    Extinction in magnitudes at each input wavelength.

    """
    if not wave.unit.is_equivalent(units.AA):
        raise ValueError(
            "Invalid units. Dispersion array should be convertible"
            "to `astropy.units.AA`"
        )

    return C00_.compute_calzetti00_extinction(wave.to(units.AA).value, a_v, r_v)


# =========================================================================== #


@units.quantity_input
def FM07(wave: units.AA, a_v, r_v=None):
    """fm07(wave, a_v, unit='aa')

    Fitzpatrick & Massa (2007) extinction model for R_V = 3.1.

    The Fitzpatrick & Massa (2007) [1]_ model, which has a slightly
    different functional form from that of Fitzpatrick (1999) [3]_
    (`extinction_f99`). Fitzpatrick & Massa (2007) claim it is
    preferable, although it is unclear if signficantly so (Gordon et
    al. 2009 [2]_). Defined from 910 A to 6 microns.

    Parameters
    ----------
    wave : numpy.ndarray (1-d)
        Wavelengths or wavenumbers.
    a_v : float
        Scaling parameter, A_V: extinction in magnitudes at characteristic
        V band  wavelength.
    unit : {'aa', 'invum'}, optional
        Unit of wave: 'aa' (Angstroms) or 'invum' (inverse microns).

    References
    ----------
    .. [1] Fitzpatrick, E. L. & Massa, D. 2007, ApJ, 663, 320
    .. [2] Gordon, K. D., Cartledge, S., & Clayton, G. C. 2009, ApJ, 705, 1320
    .. [3] Fitzpatrick, E. L. 1999, PASP, 111, 63

    """

    # make sure the units are correct
    if not wave.unit.is_equivalent(units.AA):
        raise ValueError(
            "Invalid units. Dispersion array should be convertible"
            "to `astropy.units.AA`"
        )

    if r_v is not None:
        warnings.warn(
            "[Info] FM07 assumes r_v = 3.1, overwriting the r_v set by the user.",
            RuntimeWarning,
        )

    r_v = FM07_.FM07_params[7]

    # might as well hardcode this if I use two different functions
    params = FM07_.FM07_params
    x_knots = FM07_.FM07_x_knots
    y_knots = FM07_.FM07_y_knots(r_v, params)
    a, b = cs.cubic_coef(x_knots, y_knots)

    return FM07_.compute_FM07_exctinction(
        wave.to(units.AA).value,
        a_v,
        r_v,
        params,
        x_knots,
        y_knots,
        a,
        b,
    )


# =========================================================================== #


@units.quantity_input
def Go23(wave: units.AA, a_v, r_v):
    """Go23(wave: units.AA, a_v, r_v)

    Gordon et al. (2023) Milky Way R(V) dependent model. See
    10.3847/1538-4357/accb59 (Gordon K. D. et al 2023, ApJ, 950, 86).

    Parameters
    ----------
    wave : numpy.ndarray (1-d)
        Wavelengths or wavenumbers.
    a_v : float
        Scaling parameter, A_V: extinction in magnitudes at characteristic
        V band  wavelength.
    unit : {'aa', 'invum'}, optional
        Unit of wave: 'aa' (Angstroms) or 'invum' (inverse microns).

    """

    # make sure the units are correct
    if not wave.unit.is_equivalent(units.AA):
        raise ValueError(
            "Invalid units. Dispersion array should be convertible"
            "to `astropy.units.AA`"
        )

    return Go23_.compute_Go23_exctinction(wave.to(units.micron).value, a_v, r_v)
