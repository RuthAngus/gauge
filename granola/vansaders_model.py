"""
Implementation of the van Saders (2016) gyrochronology model.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci


def Km(cw, R, M, L, P_phot):
    """
    Calculate the Kawaler coefficient.
    params:
    ------
    cw: (float)
        Centrifugal correction. Since stars usually rotate fairly slowly this
        is usually set to 1.
    R: (float)
        Stellar radius in Solar radii.
    M: (float)
        Stellar mass in Solar masses.
    L: (float)
        Luminosity in Solar luminosities.
    P_phot: (float)
        Pressure at the photosphere in Solar units. 0.868 mb for the Sun.
    """

    return cw * R**3.1 * M**-.22 * L**.56 * P_phot**.44


def dJdt(P, tau_cz, f_k, w_crit, Ro_crit, cw, R, M, L, P_phot):
    """
    Calculate the change in angular momentum over time.
    params:
    ------
    P: (float)
        Rotation period, days. Time dependent
    tau_cz: (float)
        The convective overturn timescale (days).
    f_k: (float)
        A constant factor for scaling.
    w_crit: (float)
        The critical angular frequency radians/s.
    Ro_crit: (float)
        The critical Rossby number at which magnetic braking ceases.
        Rossby number = the ratio of convective overturn timescale to
        rotation period.
    cw: (float)
        Centrifugal correction. Since stars usually rotate fairly slowly this
        is usually set to 1.
    R: (float)
        Stellar radius in Solar radii.
    M: (float)
        Stellar mass in Solar masses.
    L: (float)
        Luminosity in Solar luminosities.
    P_phot: (float)
        Pressure at the photosphere in Solar units. 0.868 mb for the Sun.
    """
    tau_czs = 30
    Ro = tau_cz / P  # Rossby number
    ws = 2*np.pi*(1/(26*24*3600))
    w = 2*np.pi*(1./(P*24*3600))

    if w_crit <= w*tau_cz/tau_czs and Ro <= Ro_crit:
        return f_k * Km(cw, R, M, L, P_phot) * w * (w_crit/ws)**2
    elif w*tau_cz/tau_czs < w_crit and Ro <= Ro_crit:
        return f_k * Km(cw, R, M, L, P_phot) * w * (w/ws*tau_cs/tau_czs)**2
    elif Ro_crit < Ro:
        return 0


def dtdJ(P, *args):
    tau_cz, f_k, w_crit, Ro_crit, cw, R, M, L, P_phot = args
    return 1./dJdt(P, tau_cz, f_k, w_crit, Ro_crit, cw, R, M, L, P_phot)


def J(R, M, w):
    return R**2 * M * w


if __name__ == "__main__":
    period = 26
    tau_cz = 30
    f_k, cw, w_crit, Ro_crit = 1, 1, 1, 1
    R, M, L, P_phot = 1, 1, 1, 1
    print(dJdt(period, tau_cz, f_k, w_crit, Ro_crit, cw, R, M, L, P_phot))
    args = [tau_cz, f_k, w_crit, Ro_crit, cw, R, M, L, P_phot]

    """ Current J """
    J_now = J(1, 1, 2*np.pi*(1./(26*24*3600)))
    print(J_now)
    t = sci.quad(dtdJ, 0, J_now, args=args)
    print(t)
