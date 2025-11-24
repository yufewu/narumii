# Functions used in heteronulcear dipolar coupling measurements
def redor_bessel(max_order):
    from scipy.special import jv

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
    from scipy.special import jv
    from scipy.constants import pi

    z = t * 2**0.5 * d

    predict = 1 - 2**0.5 * pi / 8 * (jv(0.25, z)*jv(-0.25, z)+jv(0.25, 3*z)*jv(-0.25, 3*z))
    
    return predict


def reapdor_threehalf(t, d):
    # https://doi.org/10.1016/j.pnmrs.2005.08.004
    from numpy import exp

    predict = 0.7 - 0.7*exp(-(1.82*t*d)**2)
    
    return predict

def reapdor_fivehalf(t, d):
    # https://doi.org/10.1016/j.pnmrs.2005.08.004
    from numpy import exp

    predict = 0.83 - 0.63*exp(-(3*t*d)**2) - 0.2*exp(-(0.7*t*d)**2)
    
    return predict


def reapdor_PB_natural_abundance(t, r):
    from numpy import exp

    predict = 0.60 - 0.60*exp(-(27.2*t)**2*r**(-6))
    
    return predict


# Functions used in homonulcear dipolar coupling measurements
def ctdrenar(theta, z):
    from numpy import cos
    from scipy.constants import pi

    return 6/5 * z * (1+cos(2*theta*pi/180))  # z = (d*t)^2


def drenar_parabola(t, d):
    from scipy.constants import pi

    return 0.86*pi*pi/15*d*d*t*t


def codex(t, k, m):
    from numpy import exp

    s = (1-1/m)*exp(-k*t) + 1/m
    
    return s


# Functions used in relaxation measurements
def expdec(t, a, b, R):
    from numpy import exp
    return a + b*exp(-R*t)


def expdec2(t, a, b1, b2, R1, R2):
    from numpy import exp

    return a + b1*exp(-R1*t) + b2*exp(-R2*t)


def satrec(t, a, b, R):
    from numpy import exp

    return a - b*exp(-R*t)


# Other common mathematical functions
def linear(x, a, b):
    
    return a*x+b