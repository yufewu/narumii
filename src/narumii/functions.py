import numpy as np
from numpy.typing import NDArray
from scipy.special import jv
from scipy.constants import pi


# Functions used in heteronulcear dipolar coupling measurements
def redor_bessel(max_order):

    def calculator(t, d):
        if max_order > 5:
            pass  # for raising error

        z = t * 2**0.5 * d

        predict = 1 - jv(0, z)**2

        for order in range(1,max_order):
            predict = predict + 2 / (16*order**2-1) * jv(order, z)**2
    
        return predict
    return calculator


def redor_parabola(t, d):
    predict = 1.067 * t**2 * d**2
    return predict


def redor_threehalf(t, d):

    z = t * 2**0.5 * d

    predict = 1 - 2**0.5 * pi / 8 * (jv(0.25, z)*jv(-0.25, z)+jv(0.25, 3*z)*jv(-0.25, 3*z))
    
    return predict


def reapdor_threehalf(t, d):
    # https://doi.org/10.1016/j.pnmrs.2005.08.004

    predict = 0.7 - 0.7*np.exp(-(1.82*t*d)**2)
    
    return predict

def reapdor_fivehalf(t, d):
    # https://doi.org/10.1016/j.pnmrs.2005.08.004

    predict = 0.83 - 0.63*np.exp(-(3*t*d)**2) - 0.2*np.exp(-(0.7*t*d)**2)
    
    return predict


def reapdor_PB_natural_abundance(t, r):

    predict = 0.60 - 0.60*np.exp(-(27.2*t)**2*r**(-6))
    
    return predict


# Functions used in homonulcear dipolar coupling measurements
def ctdrenar(theta, z):

    return 6/5 * z * (1+np.cos(2*theta*pi/180))  # z = (d*t)^2


def drenar_parabola(t, d):

    return 0.86*pi*pi/15*d*d*t*t


def codex(t, k, m):

    s = (1-1/m)*np.exp(-k*t) + 1/m
    
    return s


# Functions used in relaxation measurements
def expdec(t, a, b, R):
    return a + b*np.exp(-R*t)


def expdec2(t, a, b1, b2, R1, R2):

    return a + b1*np.exp(-R1*t) + b2*np.exp(-R2*t)


def satrec(t, a, b, R):

    return a - b*np.exp(-R*t)


# Other common mathematical functions
def linear(
    x: NDArray[np.floating] | np.floating, 
    a: float, 
    b: float
    ) -> NDArray[np.floating] | np.floating:
    """
    Linear function: f(x) = a*x + b

    Parameters
    -----
    x: array_like
        Independent variable.
    a: float
        Slope of the line.
    b: float
        Y-intercept.

    Returns
    ------
    f(x): array_like
        Computed linear function values.
    """
    return a*x+b


def parabola(
    x: NDArray[np.floating] | np.floating, 
    a: float, 
    x0: float, 
    c: float
    ) -> NDArray[np.floating] | np.floating:
    """
    Parabolic function: f(x) = a*(x - x0)^2 + c

    Parameters
    -----
    x: array_like
        Independent variable.
    a: float
        Coefficient for the quadratic term.
    x0: float
        Vertex position along x-axis.
    c: float
        Vertical offset.
    
    Returns
    ------
    f(x): array_like
        Computed parabolic function values.
    """
    return a * (x - x0)**2 + c


def compute_rmsd(
    data1: NDArray[np.floating], 
    data2: NDArray[np.floating], 
    weights: NDArray[np.floating] | np.floating = 1):
    """
    Compute the root mean square deviation (RMSD) between two datasets with optional weighting.
    
    Parameters
    -----
    data1: array_like
        First dataset.
    data2: array_like
        Second dataset.
    weights (optional): array_like
        Weights for each data point.

    Returns
    ------
    rmsd: float
        Root mean square deviation between data1 and data2 weighted by weights.
    """

    rmsd = np.sqrt(np.mean(weights * (data1 - data2) ** 2))

    return rmsd


def compute_ssd(
    data1: NDArray[np.floating], 
    data2: NDArray[np.floating], 
    weights: NDArray[np.floating] | np.floating = 1):
    """
    Compute the sum squared deviations (SSD) between two datasets with optional weighting.
    
    Parameters
    -----
    data1: array_like
        First dataset.
    data2: array_like
        Second dataset.
    weights (optional): array_like
        Weights for each data point.

    Returns
    ------
    ssd: float
        Sum squared deviations between data1 and data2 weighted by weights.
    """

    ssd = np.sum(weights * (data1 - data2) ** 2)

    return ssd