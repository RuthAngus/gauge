# Age-rotation relations.
import numpy as np
from teff_bv import teff2bv


class gyro_age(object):

    def __init__(self, p, teff=None, feh=None, logg=None, bv=None):

        self.p = p
        self.bv = bv
        if not self.bv:
            self.bv = teff2bv(teff, logg, feh)
        self.teff = teff
        self.feh = feh
        self.logg = logg

    def barnes07(self, par=[.7725, .601, .4, .5189]):
        a, b, c, n = par
        return (self.p/(a*(self.bv - c)**b))**(1./n)*1e-3

    def barnes10(self, par):
        return par

    def matt12(self, par):
        return par

    def vansaders16(self, par):
        return par

if __name__ == "__main__":
    ga = gyro_age(26, teff=5778, feh=0., logg=4.44)
    ga = gyro_age(26, bv=.65)
    print(ga.barnes07())
