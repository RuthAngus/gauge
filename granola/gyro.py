"""
Various age-rotation relations.
"""

import pandas as pd
from teff_bv import teff2bv
import scipy.spatial as sps


class gyro_age(object):

    def __init__(self, p, teff=None, feh=None, logg=None, bv=None):

        self.p = p
        self.bv = bv
        if not self.bv:
            self.bv = teff2bv(teff, logg, feh)
        self.teff = teff
        self.feh = feh
        self.logg = logg
        self.df = pd.read_csv("skeleton3_run_000.out", skiprows=172)

    def barnes07(self, version):
        if version == "barnes":
            par = [.7725, .601, .4, .5189]
        elif version == "mh":
            par = [.407, .325, .495, .566]
        elif version == "angus":
            par = [.4, .31, .45, .55]
        a, b, c, n = par
        return (self.p/(a*(self.bv - c)**b))**(1./n)*1e-3

    def vansaders16(self, par):
        """
        Calculate an age from a luminosity, temperature and rotation period.
        Uses a KDtree to find the nearest point in Jen van Saders'
        evolutionary tracks.
        Currently has only solar metallicity.
        To do:
        Use interpolation instead of nearest.
        Include other metallicities.

        parameters:
        ----------
        par: (array_like)
            The temperature and period of a star.
        returns:
        -------
        age: (float)
            Age in Gyr.
        """
        m = (3000 < 10**self.df.log_Teff_K.values) & \
            (10**self.df.log_Teff_K.values < 8000) & (self.df.Age_Gyr > .1)
        teff = 10**self.df.log_Teff_K.values[m]
        P = self.df.Prot_days.values[m]
        age = self.df.Age_Gyr.values[m]
        data = np.vstack((teff, P)).T
        tree = sps.cKDTree(data)
        dist, index = tree.query(par, 1)
        return age[index]


if __name__ == "__main__":
    ga = gyro_age(26, teff=5778, feh=0., logg=4.44)
    ga = gyro_age(26, bv=.65)
    print(ga.barnes07())
