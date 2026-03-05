import numpy as np
from typing import Callable
from scipy.special import jv
from scipy.constants import pi


# Functions used in heteronulcear dipolar coupling measurements
def redor_bessel(max_order: int) -> Callable:

    def calculator(t: np.ndarray, b: float) -> np.ndarray:
        if max_order > 5:
            raise ValueError("max_order cannot exceed 5")

        z = t * 2**0.5 * b

        predict = 1 - jv(0, z)**2

        for order in range(1,max_order):
            predict = predict + 2 / (16*order**2-1) * jv(order, z)**2
    
        return predict
    return calculator


def redor_parabola(t: np.ndarray, b: float) -> np.ndarray:
    predict = 1.067 * t**2 * b**2
    return predict


def redor_threehalf(t: np.ndarray, b: float) -> np.ndarray:

    z = t * 2**0.5 * b

    predict = 1 - 2**0.5 * pi / 8 * (jv(0.25, z)*jv(-0.25, z)+jv(0.25, 3*z)*jv(-0.25, 3*z))
    
    return predict


def reapdor_threehalf(t: np.ndarray, b: float) -> np.ndarray:
    # https://doi.org/10.1016/j.pnmrs.2005.08.004

    predict = 0.7 - 0.7*np.exp(-(1.82*t*b)**2)
    
    return predict

def reapdor_fivehalf(t: np.ndarray, b: float) -> np.ndarray:
    # https://doi.org/10.1016/j.pnmrs.2005.08.004

    predict = 0.83 - 0.63*np.exp(-(3*t*b)**2) - 0.2*np.exp(-(0.7*t*b)**2)
    
    return predict


def reapdor_PB_natural_abundance(t: np.ndarray, r: float) -> np.ndarray:

    predict = 0.60 - 0.60*np.exp(-(27.2*t)**2*r**(-6))
    
    return predict


# Functions used in homonulcear dipolar coupling measurements
def ctdrenar(theta: float, z: float) -> float:

    return 6/5 * z * (1+np.cos(2*theta*pi/180))  # z = (d*t)^2


def drenar_parabola(t: np.ndarray, b: float) -> np.ndarray:

    return 0.86*pi*pi/15*b*b*t*t


def codex(t: np.ndarray, k: float, m: float) -> np.ndarray:

    s = (1-1/m)*np.exp(-k*t) + 1/m
    
    return s


# Functions used in relaxation measurements
def expdec(t: np.ndarray, a: float, b: float, R: float) -> np.ndarray:
    return a + b*np.exp(-R*t)


def expdec2(t: np.ndarray, a: float, b1: float, b2: float, R1: float, R2: float) -> np.ndarray:

    return a + b1*np.exp(-R1*t) + b2*np.exp(-R2*t)


def satrec(t: np.ndarray, a: float, b: float, R: float) -> np.ndarray:

    return a - b*np.exp(-R*t)


# Other common mathematical functions
def linear(
    x: np.ndarray, 
    a: float, 
    b: float
    ) -> np.ndarray:
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
    x: np.ndarray, 
    a: float, 
    x0: float, 
    c: float
    ) -> np.ndarray:
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
    data1: np.ndarray, 
    data2: np.ndarray, 
    weights: np.ndarray | float = 1):
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
    data1: np.ndarray, 
    data2: np.ndarray, 
    weights: np.ndarray | float = 1):
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