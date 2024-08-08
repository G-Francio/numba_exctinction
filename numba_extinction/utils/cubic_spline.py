# Largely taken from https://github.com/hwsnell/Cubic-Spline, accessed on 
#  2024/08/01, licensed under MIT License

# Original copyright statement

# Copyright (c) 2023 hwsnell

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from numba import guvectorize, jit, njit


@jit()
def sparse_solver(a, b, c, d):
    nf = len(a)
    for it in range(1, nf):
        mc = a[it] / b[it - 1]
        b[it] = b[it] - mc * c[it - 1]
        d[it] = d[it] - mc * d[it - 1]

    xc = a
    xc[-1] = d[-1] / b[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (d[il] - c[il] * xc[il + 1]) / b[il]

    return xc


# =========================================================================== #


@guvectorize(["void(float64[:],float64[:],float64[:],float64[:])"], "(n),(n)->(n),(n)")
def cubic_coef_solver(x, y, coef_a, coef_b):
    d = np.zeros(len(x))
    d[0] = 3 * (y[1] - y[0]) / (x[1] - x[0]) ** 2
    d[-1] = 3 * (y[-1] - y[-2]) / (x[-1] - x[-2]) ** 2
    d[1:-1] = 3 * (
        (y[1:-1] - y[:-2]) / (x[1:-1] - x[:-2]) ** 2
        + (y[2:] - y[1:-1]) / (x[2:] - x[1:-1]) ** 2
    )

    b = np.zeros_like(d)
    b[0] = 2 / (x[1] - x[0])
    b[-1] = 2 / (x[-1] - x[-2])
    b[1:-1] = 2 * (1 / (x[1:-1] - x[:-2]) + 1 / (x[2:] - x[1:-1]))

    a = np.zeros_like(d)
    a[1:] = 1 / (x[1:] - x[:-1])

    c = np.zeros_like(d)
    c[:-1] = 1 / (x[1:] - x[:-1])

    k = sparse_solver(a, b, c, d)

    for i in range(len(k) - 1):
        coef_a[i] = k[i] * (x[i + 1] - x[i]) - (y[i + 1] - y[i])
        coef_b[i] = -k[i + 1] * (x[i + 1] - x[i]) + (y[i + 1] - y[i])


# =========================================================================== #
# ========================= Actual spline functions ========================= #
# =========================================================================== #


def cubic_coef(x, y):
    """Calculates cubic spline coefficients

    The restrictions on the shape of x and y are the same as those on x and xq in cubic_index

    Parameters
    ----------
    x  : array (n) or (s,n), NumPy array of observed, increasing x values
    y  : array (n) or (s,n), NumPy array of observed y values

    Returns
    ----------
    a  : array (n-1) or (s,n-1), NumPy array of coefficients
    b  : array (n-1) or (s,n-1), NumPy array of coefficients

    """
    coef_a, coef_b = cubic_coef_solver(x, y)
    return coef_a[:-1], coef_b[:-1]


# =========================================================================== #


@njit
def cubic_y_w_coeff(x, xq, y, a, b):
    """Efficiently solves for interpolated value of a point given spline coefficients

    Parameters
    ----------
    x   : array (n), NumPy array of observed, increasing values
    xq  : scalar, query point of interest
    y   : array (n), NumPy array of y values corresponding to x
    a   : array (n-1), NumPy coefficient array from cubic_coef
    b   : array (n-1), NumPy coefficient array from cubic_coef

    Returns
    ----------
    yq  : scalar, interpolated value for the query point xq

    """
    xi = 0
    x_high = x[1]
    while xi < x.shape[0] - 2:
        if x_high >= xq:
            break
        xi += 1
        x_high = x[xi + 1]

    t = (xq - x[xi]) / (x[xi + 1] - x[xi])
    yq = (1 - t) * y[xi] + t * y[xi + 1] + t * (1 - t) * ((1 - t) * a[xi] + t * b[xi])

    return yq


# =========================================================================== #
