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
from numba_extinction.models import vcg_04 as VCG04_
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
    """CCM89(wave: units.AA, a_v, r_v)

    Cardelli, Clayton & Mathis (1989, ApJ, 345, 245) extinction function.

    :param wave: Dispersion array
    :type wave: numpy.ndarray [units.AA]
    :param a_v: V band extinction, in magnitudes.
    :type a_v: float
    :param r_v: Ratio of total to selective extinction, A_V / E(B-V)
    :type r_v: float
    :raises ValueError: Raises error if the units of wave cannot be converted to Angstrom.
    :return: Extinction in magnitudes at each input wavelength.
    :rtype: numpy.ndarray
    """
    if not wave.unit.is_equivalent(units.AA):
        raise ValueError(
            "Invalid units. Dispersion array should be convertible"
            "to `astropy.units.AA`"
        )

    return CCM89_.compute_exctinction(wave.to(units.AA).value, a_v, r_v)


# =========================================================================== #


@units.quantity_input
def OD94(wave: units.AA, a_v, r_v):
    """OD94(wave: units.AA, a_v, r_v)

    O'Donnell (1994, ApJ, 422, 158O) extinction function.
    The functional shape is the same as CCM89, but with updated optical
    coefficients.

    See also the revised version in Valencic et al. (2004, ApJ, 616, 912),
    implemented as VCG04 just below.

    :param wave: Dispersion array
    :type wave: numpy.ndarray [units.AA]
    :param a_v: V band extinction, in magnitudes.
    :type a_v: float
    :param r_v: Ratio of total to selective extinction, A_V / E(B-V)
    :type r_v: float
    :raises ValueError: Raises error if the units of wave cannot be converted
    to Angstrom.
    :return: Extinction in magnitudes at each input wavelength.
    :rtype: numpy.ndarray
    """
    if not wave.unit.is_equivalent(units.AA):
        raise ValueError(
            "Invalid units. Dispersion array should be convertible"
            "to `astropy.units.AA`"
        )

    return OD94_.compute_exctinction(wave.to(units.AA).value, a_v, r_v)


# =========================================================================== #


@units.quantity_input
def Fi99(wave: units.AA, a_v, r_v=3.1):
    """Fi99(wave: units.AA, a_v, r_v=3.1)

    Fitzpatrick (1999, PASP, 111, 63) dust extinction curve.

    This curve relies on the parametrisation and the method developed in
    Fitzpatrick & Massa (1990, ApJS, 72, 163). Following the original package,
    the implementation is designed to reproduce the NASA Goddart IDL function.

    :param wave: Dispersion array
    :type wave: numpy.ndarray [units.AA]
    :param a_v: V band extinction, in magnitudes.
    :type a_v: float
    :param r_v: Ratio of total to selective extinction, A_V / E(B-V)
    :type r_v: float
    :raises ValueError: Raises error if the units of wave cannot be converted
    to Angstrom.
    :return: Extinction in magnitudes at each input wavelength.
    :rtype: numpy.ndarray

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

    return Fi99_.compute_exctinction(
        wave.to(units.AA).value, a_v, r_v, params, x_knots, y_knots, a, b
    )


# =========================================================================== #


@units.quantity_input
def C00(wave: units.AA, a_v, r_v):
    """C00(wave: units.AA, a_v, r_v)

    Calzetti (2000, ApJ 533, 682) extinction function.

    :param wave: Dispersion array
    :type wave: numpy.ndarray [units.AA]
    :param a_v: V band extinction, in magnitudes.
    :type a_v: float
    :param r_v: Ratio of total to selective extinction, A_V / E(B-V)
    :type r_v: float
    :raises ValueError: Raises error if the units of wave cannot be converted
    to Angstrom.
    :return: Extinction in magnitudes at each input wavelength.
    :rtype: numpy.ndarray

    """
    if not wave.unit.is_equivalent(units.AA):
        raise ValueError(
            "Invalid units. Dispersion array should be convertible"
            "to `astropy.units.AA`"
        )

    return C00_.compute_extinction(wave.to(units.AA).value, a_v, r_v)


# =========================================================================== #


@units.quantity_input
def VCG04(wave: units.AA, a_v, r_v):
    """VCG04(wave: units.AA, a_v, r_v)

    Valencic, Clayton & Gordon (2004, ApJ, 616, 912) extinction curve.
    The functional shape is the same as Od94, but with updated  UV coefficients.
    As far as I understand, OD94 is not taken into account in this case.

    :param wave: Dispersion array
    :type wave: numpy.ndarray [units.AA]
    :param a_v: V band extinction, in magnitudes.
    :type a_v: float
    :param r_v: Ratio of total to selective extinction, A_V / E(B-V)
    :type r_v: float
    :raises ValueError: Raises error if the units of wave cannot be converted
    to Angstrom.
    :return: Extinction in magnitudes at each input wavelength.
    :rtype: numpy.ndarray
    """
    if not wave.unit.is_equivalent(units.AA):
        raise ValueError(
            "Invalid units. Dispersion array should be convertible"
            "to `astropy.units.AA`"
        )

    return VCG04_.compute_exctinction(wave.to(units.AA).value, a_v, r_v)


# =========================================================================== #


@units.quantity_input
def FM07(wave: units.AA, a_v, r_v=None):
    """FM07(wave: units.AA, a_v, r_v=None)

    Fitzpatrick & Massa (2007) extinction model only for R_V = 3.1.

    The Fitzpatrick & Massa (2007, ApJ, 663, 320) model, which has a slightly
    different functional form from that of Fitzpatrick (`Fi99`, 1999, PASP,
    111, 63). Defined from 910 A to 6 microns.

    :param wave: Dispersion array
    :type wave: numpy.ndarray [units.AA]
    :param a_v: V band extinction, in magnitudes.
    :type a_v: float
    :param r_v: Ratio of total to selective extinction, A_V / E(B-V)
    :type r_v: float
    :raises ValueError: Raises error if the units of wave cannot be converted
    to Angstrom.
    :return: Extinction in magnitudes at each input wavelength.
    :rtype: numpy.ndarray

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

    return FM07_.compute_exctinction(
        wave.to(units.AA).value, a_v, r_v, params, x_knots, y_knots, a, b
    )


# =========================================================================== #


@units.quantity_input
def Go23(wave: units.AA, a_v, r_v):
    """Go23(wave: units.AA, a_v, r_v)

    Gordon et al. (2023, ApJ, 950, 86) Milky Way R(V) dependent model.

    :param wave: Dispersion array
    :type wave: numpy.ndarray [units.AA]
    :param a_v: V band extinction, in magnitudes.
    :type a_v: float
    :param r_v: Ratio of total to selective extinction, A_V / E(B-V)
    :type r_v: float
    :raises ValueError: Raises error if the units of wave cannot be converted
    to Angstrom.
    :return: Extinction in magnitudes at each input wavelength.
    :rtype: numpy.ndarray
    """

    # make sure the units are correct
    if not wave.unit.is_equivalent(units.AA):
        raise ValueError(
            "Invalid units. Dispersion array should be convertible"
            "to `astropy.units.AA`"
        )

    if wave.ndim == 1:
        return Go23_.compute_exctinction(wave.to(units.micron).value, a_v, r_v)
    else:
        return Go23_.compute_exctinction_many(wave.to(units.micron).value, a_v, r_v)
