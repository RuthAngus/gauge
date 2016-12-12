# Age-rotation relations.
import numpy as np
from teff_bv import teff2bv


class gyro_age(object):

    def __init__(self, p, teff=None, feh=None, logg=None, bv=None):

        self.p = p
        if not bv:
            self.bv = teff2bv(teff, feh, logg)
        self.teff = teff
        self.feh = feh
        self.logg = logg

    def barnes07(self, par=[.4, .31, .55, .45]):
        a, b, c, n = par
        return 10**((np.log10(self.p) - np.log10(a) -
                     b*np.log10(self.bv - c)) / n) / 1000.

    def barnes10(self, par):
        return par

    def matt12(self, par):
        return par

    def vansaders16(self, par):
        return par
