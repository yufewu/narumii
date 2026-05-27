import numpy as np
from scipy.special import jv
from scipy.constants import pi

from typing import Callable


# Functions used in heteronulcear dipolar coupling measurements
def redor_bessel(max_order: int) -> Callable:
    """
    Return a Bessel series with a given maximum order for analytical prediction of REDOR curves: 

    $$
    \\frac{\\Delta S}{S_0} = 1 - \\left[J_0\\left(\\sqrt{2}\\lambda_n\\right)\\right]^2 + 2\\sum^\\infty_{k=1}\\frac{1}{16k^2-1}\\left[J_k\\left(\\sqrt{2}\\lambda_n\\right)\\right]^2
    $$

    where $\\lambda_n = nD\\tau$ is the product of the number of rotor cycles $n$, the dipolar coupling strength $D$, and the rotor period $\\tau$. 

    References
    ----------
    Mueller, K.T. (1995). Analytical Solutions for the Time Evolution of Dipolar-Dephasing NMR Signals. \
        Journal of Magnetic Resonance Series A, 113, 81-93. https://doi.org/10.1006/jmra.1995.1059

    Parameters
    ----------
    max_order: int
        Maximum order of the Bessel series expansion. Must not exceed 5.

    Returns
    -------
    func: Callable
        A function that calculates the REDOR difference curve using Bessel functions.
        The function signature is: 'calculator(t, b) -> np.ndarray'
        where t is the evolution time and b is the effective dipolar coupling constant.

    Raises
    ------
    ValueError
        If max_order exceeds 5.

    Examples
    --------
    >>> redor_func = redor_bessel(max_order=5)
    >>> t = np.linspace(0, 10, 100)
    >>> difference = redor_func(t, 2.5)  # b = 2.5 kHz
    """
    def func(t: np.ndarray, b: float) -> np.ndarray:
        if max_order > 5:
            raise ValueError("max_order cannot be larger than 5")

        z = t * 2**0.5 * b

        result = 1 - jv(0, z)**2

        for order in range(1,max_order):
            result = result + 2 / (16*order**2-1) * jv(order, z)**2
    
        return result
    return func


def redor_parabola(t: np.ndarray, b: float) -> np.ndarray:
    """
    Calculate the REDOR difference curve using parabolic approximation.

    Parameters
    ----------
    t: np.ndarray
        Evolution time, in ms.
    b: float
        Effective dipolar coupling constant, in kHz.

    Returns
    -------
    predict: np.ndarray
        Predicted REDOR difference values (1 - S'/S₀).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 100)
    >>> difference = redor_parabola(t, 2.5)  # b = 2.5 kHz
    """
    predict = 1.067 * t**2 * b**2
    return predict


def redor_threehalf(t: np.ndarray, b: float) -> np.ndarray:
    """
    Analytical calculation for the REDOR difference in an I=1/2, S=3/2 spin system: 

    $$
    \\frac{\\Delta S}{S_0} = 1 - \\frac{1}{2}\\left[\\frac{\\sqrt{2}\\pi}{4}J_{1/4}(\\sqrt{2}\\lambda_n)J_{-1/4}(\\sqrt{2}\\lambda_n) + \\frac{\\sqrt{2}\\pi}{4}J_{1/4}(3\\sqrt{2}\\lambda_n)J_{-1/4}(3\\sqrt{2}\\lambda_n)\\right]
    $$

    where $\\lambda_n = nD\\tau$ is the product of the number of rotor cycles $n$, the dipolar coupling strength $D$, and the rotor period $\\tau$. 

    References
    ----------
    Gullion, T., Vega, A.J. (2005). Measuring heteronuclear dipolar couplings for I=1/2, S>1/2 spin pairs by REDOR and REAPDOR NMR. \
        Progress in Nuclear Magnetic Resonance Spectroscopy, 47, 123-136. https://doi.org/10.1016/j.pnmrs.2005.08.004

    Parameters
    ----------
    t: np.ndarray
        Evolution time, in ms.
    b: float
        Effective dipolar coupling constant, in kHz.

    Returns
    -------
    predict: np.ndarray
        Predicted REDOR difference values (1 - S'/S₀).

    Examples
    --------
    >>> t = np.linspace(0, 10, 100)
    >>> difference = redor_threehalf(t, 2.5)  # b = 2.5 kHz
    """
    z = t * 2**0.5 * b

    predict = 1 - 2**0.5 * pi / 8 * (jv(0.25, z)*jv(-0.25, z)+jv(0.25, 3*z)*jv(-0.25, 3*z))
    
    return predict


def reapdor_threehalf(t: np.ndarray, b: float) -> np.ndarray:
    """
    Analytical calculation for the REAPDOR difference in an I=1/2, S=3/2 spin system: 

    $$
    \\frac{\\Delta S}{S_0} = 0.7(1 - \\exp[-(1.82\\lambda_n)^2])
    $$

    where $\\lambda_n = nD\\tau$ is the product of the number of rotor cycles $n$, the dipolar coupling strength $D$, and the rotor period $\\tau$. 

    References
    ----------
    Gullion, T., Vega, A.J. (2005). Measuring heteronuclear dipolar couplings for I=1/2, S>1/2 spin pairs by REDOR and REAPDOR NMR. \
        Progress in Nuclear Magnetic Resonance Spectroscopy, 47, 123-136. https://doi.org/10.1016/j.pnmrs.2005.08.004

    Parameters
    ----------
    t: np.ndarray
        Evolution time, in ms.
    b: float
        Effective dipolar coupling constant, in kHz.

    Returns
    -------
    predict: np.ndarray
        Predicted REAPDOR difference values.

    Examples
    --------
    >>> t = np.linspace(0, 10, 100)
    >>> difference = reapdor_threehalf(t, 2.5)  # b = 2.5 kHz
    """
    predict = 0.7 - 0.7*np.exp(-(1.82*t*b)**2)
    
    return predict

def reapdor_fivehalf(t: np.ndarray, b: float) -> np.ndarray:
    """
    Analytical calculation for the REAPDOR difference in an I=1/2, S=5/2 spin system: 

    $$
    \\frac{\\Delta S}{S_0} = 0.63(1 - \\exp[-(3.0\\lambda_n)^2]) + 0.2(1 - \\exp[-(0.7\\lambda_n)^2])
    $$

    where $\\lambda_n = nD\\tau$ is the product of the number of rotor cycles $n$, the dipolar coupling strength $D$, and the rotor period $\\tau$. 

    References
    ----------
    Gullion, T., Vega, A.J. (2005). Measuring heteronuclear dipolar couplings for I=1/2, S>1/2 spin pairs by REDOR and REAPDOR NMR. \
        Progress in Nuclear Magnetic Resonance Spectroscopy, 47, 123-136. https://doi.org/10.1016/j.pnmrs.2005.08.004

    Parameters
    ----------
    t: np.ndarray
        Evolution time, in ms.
    b: float
        Effective dipolar coupling constant, in kHz.

    Returns
    -------
    predict: np.ndarray
        Predicted REAPDOR difference values.

    Examples
    --------
    >>> t = np.linspace(0, 10, 100)
    >>> difference = reapdor_fivehalf(t, 2.5)  # b = 2.5 kHz
    """
    predict = 0.83 - 0.63*np.exp(-(3*t*b)**2) - 0.2*np.exp(-(0.7*t*b)**2)
    
    return predict


def reapdor_PB_natural_abundance(t: np.ndarray, r: float) -> np.ndarray:
    """
    Analytical calculation for the REAPDOR difference between an I=1/2 and S=B11 (in natural abundance) spin system: 

    $$
    \\frac{\\Delta S}{S_0} = 0.60(1 - \\exp[-(16.9t)^2r^{-6}])
    $$

    References
    ----------
    Nimerovsky, E. & Goldbourt, A. (2012). Distance measurements between boron and carbon at natural abundance using \
        magic angle spinning REAPDOR NMR and a universal curve. Phys. Chem. Chem. Phys., 14, 13437-13443. \
        https://doi.org/10.1039/C2CP41851G

    Parameters
    ----------
    t: np.ndarray
        Evolution time, in ms.
    r: float
        Internuclear distance, in Å.

    Returns
    -------
    predict: np.ndarray
        Predicted REAPDOR difference values.

    Examples
    --------
    >>> t = np.linspace(0, 10, 100)
    >>> difference = reapdor_PB_natural_abundance(t, 3.0)  # r = 3.0 Å
    """
    predict = 0.60 - 0.60*np.exp(-(16.9*t)**2*r**(-6))
    
    return predict


# Functions used in homonulcear dipolar coupling measurements
def ctdrenar(theta: np.ndarray, z: float) -> np.ndarray:
    """
    Analytical approximation to BaBa-xy16 CT-DRENAR.

    References
    ----------
    Ren, J. & Eckert, H. (2015). Measurement of homonuclear magnetic dipole-dipole interactions 
    in multiple 1/2-spin systems using constant-time DQ-DRENAR NMR. Journal of Magnetic Resonance, 260(1), 46-53.
    https://doi.org/10.1016/j.jmr.2015.08.022

    Parameters
    ----------
    theta: np.ndarray
        Phase angles, in degrees.
    z: float
        Dephasing parameter (Dt)², where D is the dipolar coupling constant and t is dephasing time, the units should match
        so that z is dimentionless.

    Returns
    -------
    intensity: np.ndarray
        Normalized DQ intensity in CT-DRENAR experiments. 

    Examples
    --------
    >>> theta = np.linspace(0, 180, 100)
    >>> z = 0.5
    >>> intensity = ctdrenar(theta, z)
    """
    return 6/5 * z * (1+np.cos(2*theta*pi/180))  # z = (d*t)^2


def drenar_postc7(t: np.ndarray, b: float) -> np.ndarray:
    """
    Analytical approximation to POST-C7 VT-DRENAR.

    References
    ----------
    Ren, J. & Eckert, H. (2013). DQ-DRENAR: A new NMR technique to measure siteresolved magnetic dipole-dipole interactions in 
    multispin-1/2 systems: Theory and validation on crystalline phosphates. Journal of Chemical Physics, 138, 164201.
    https://doi.org/10.1063/1.4801634

    Parameters
    ----------
    t: np.ndarray
        Dephasing time, in ms.
    b: float
        Effective dipolar coupling constant, in kHz.

    Returns
    -------
    intensity: np.ndarray
        Predicted DQ intensity.

    Examples
    --------
    >>> t = np.linspace(0, 10, 100)
    >>> intensity = drenar_parabola(t, 2.5)  # b = 2.5 kHz
    """
    return 0.86*pi*pi/15 * (b*t)**2


def drenar_babaxy16(t: np.ndarray, b: float) -> np.ndarray:
    """
    Analytical approximation to BaBa-xy16 VT-DRENAR.

    References
    ----------
    Ren, J. & Eckert, H. (2013). DQ-DRENAR with back-to-back (BABA) excitation: Measuring homonuclear dipole-dipole 
    interactions in multiple spin-1/2 systems. Solid State Nuclear Magnetic Resonance, 71, 11-18.
    https://doi.org/10.1016/j.ssnmr.2015.10.007

    Parameters
    ----------
    t: np.ndarray
        Dephasing time, in ms.
    b: float
        Effective dipolar coupling constant, in kHz.

    Returns
    -------
    intensity: np.ndarray
        Predicted DQ intensity.

    Examples
    --------
    >>> t = np.linspace(0, 10, 100)
    >>> intensity = drenar_parabola(t, 2.5)  # b = 2.5 kHz
    """
    return 12/5 * (b*t)**2


def codex(t: np.ndarray, k: float, m: float) -> np.ndarray:
    """
    Calculate the magnetization in CODEX (COherence Decay Experiment) pulse sequence.

    Parameters
    ----------
    t: np.ndarray
        Evolution time, in ms.
    k: float
        Decay rate constant.
    m: float
        Magnetization asymptotic parameter.

    Returns
    -------
    s: np.ndarray
        Predicted magnetization values.

    Examples
    --------
    >>> t = np.linspace(0, 10, 100)
    >>> s = codex(t, 0.5, 0.3)  # k = 0.5, m = 0.3
    """
    s = (1-1/m)*np.exp(-k*t) + 1/m
    
    return s


# Functions used in relaxation measurements
def expdec(t: np.ndarray, a: float, b: float, R: float) -> np.ndarray:
    """
    Calculate exponential decay with baseline offset (single exponential).

    Parameters
    ----------
    t: np.ndarray
        Time, in ms.
    a: float
        Baseline/offset value.
    b: float
        Amplitude of the decaying component.
    R: float
        Relaxation rate, in ms⁻¹.

    Returns
    -------
    signal: np.ndarray
        Predicted signal values following: a + b*exp(-R*t).

    Examples
    --------
    >>> t = np.linspace(0, 100, 100)
    >>> signal = expdec(t, 0.1, 0.9, 0.02)  # a=0.1, b=0.9, R=0.02 ms⁻¹
    """
    return a + b*np.exp(-R*t)


def expdec2(t: np.ndarray, a: float, b1: float, b2: float, R1: float, R2: float) -> np.ndarray:
    """
    Calculate exponential decay with baseline offset (double exponential).

    Parameters
    ----------
    t: np.ndarray
        Time, in ms.
    a: float
        Baseline/offset value.
    b1: float
        Amplitude of the first decaying component.
    b2: float
        Amplitude of the second decaying component.
    R1: float
        Relaxation rate of the first component, in ms⁻¹.
    R2: float
        Relaxation rate of the second component, in ms⁻¹.

    Returns
    -------
    signal: np.ndarray
        Predicted signal values following: a + b₁*exp(-R₁*t) + b₂*exp(-R₂*t).

    Examples
    --------
    >>> t = np.linspace(0, 100, 100)
    >>> signal = expdec2(t, 0.1, 0.5, 0.4, 0.01, 0.05)  # two exponential components
    """
    return a + b1*np.exp(-R1*t) + b2*np.exp(-R2*t)


def satrec(t: np.ndarray, a: float, b: float, R: float) -> np.ndarray:
    """
    Calculate saturation recovery signal (inverse exponential).

    Parameters
    ----------
    t: np.ndarray
        Delay time, in ms.
    a: float
        Equilibrium magnetization value.
    b: float
        Amplitude of recovery.
    R: float
        Recovery rate, in ms⁻¹.

    Returns
    -------
    signal: np.ndarray
        Predicted signal values following: a - b*exp(-R*t).

    Notes
    -----
    This function models the recovery of magnetization following saturation in
    saturation-recovery experiments, particularly useful for T₁ relaxation measurements.

    Examples
    --------
    >>> t = np.linspace(0, 100, 100)
    >>> signal = satrec(t, 1.0, 1.0, 0.02)  # equilibrium at 1.0, recovery rate 0.02 ms⁻¹
    """
    return a - b*np.exp(-R*t)


# Other common mathematical functions
def linear(
    x: np.ndarray, 
    a: float, 
    b: float
    ) -> np.ndarray:
    """
    Linear function: f(x) = a*x + b.

    Parameters
    ----------
    x: np.ndarray
        Independent variable.
    a: float
        Slope of the line.
    b: float
        Y-intercept.

    Returns
    -------
    f_x: np.ndarray
        Computed linear function values.

    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = linear(x, 2.0, 3.0)  # y = 2*x + 3
    """
    return a*x+b


def parabola(
    x: np.ndarray, 
    a: float, 
    x0: float, 
    c: float
    ) -> np.ndarray:
    """
    Parabolic function: f(x) = a*(x - x0)² + c.

    Parameters
    ----------
    x: np.ndarray
        Independent variable.
    a: float
        Coefficient for the quadratic term.
    x0: float
        Vertex position along x-axis.
    c: float
        Vertical offset.
    
    Returns
    -------
    f_x: np.ndarray
        Computed parabolic function values.

    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = parabola(x, 1.0, 5.0, 2.0)  # vertex at (5, 2)
    """
    return a * (x - x0)**2 + c


def compute_rmsd(
    data1: np.ndarray, 
    data2: np.ndarray, 
    weights: np.ndarray | float = 1
    ) -> float:
    """
    Compute the root mean square deviation (RMSD) between two datasets with optional weighting.
    
    Parameters
    ----------
    data1: np.ndarray
        First dataset.
    data2: np.ndarray
        Second dataset.
    weights: np.ndarray or float, optional
        Weights for each data point. Default is 1 (uniform weighting).

    Returns
    -------
    rmsd: float
        Root mean square deviation between data1 and data2 weighted by weights.

    Examples
    --------
    >>> rmsd = compute_rmsd(data1, data2)
    >>> rmsd_weighted = compute_rmsd(data1, data2, weights=weights)
    """
    rmsd = np.sqrt(np.mean(weights * (data1 - data2) ** 2))

    return rmsd


def compute_ssd(
    data1: np.ndarray, 
    data2: np.ndarray, 
    weights: np.ndarray | float = 1
    ) -> float:
    """
    Compute the sum squared deviations (SSD) between two datasets with optional weighting.
    
    Parameters
    ----------
    data1: np.ndarray
        First dataset.
    data2: np.ndarray
        Second dataset.
    weights: np.ndarray or float, optional
        Weights for each data point. Default is 1 (uniform weighting).

    Returns
    -------
    ssd: float
        Sum squared deviations between data1 and data2 weighted by weights.

    Examples
    --------
    >>> ssd = compute_ssd(data1, data2)
    >>> ssd_weighted = compute_ssd(data1, data2, weights=weights)
    """
    ssd = np.sum(weights * (data1 - data2) ** 2)

    return ssd