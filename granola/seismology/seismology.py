# Plot rotation / bv / age.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool

import os

import kplr

import gprot_fit as gp
import kepler_data as kd


def get_lc(id, KPLR_DIR):
    """
    Downloads the kplr light curve and loads x, y and yerr.
    """
    kid = str(int(id)).zfill(9)
    client = kplr.API()
    star = client.star(kid)
    star.get_light_curves(fetch=True, short_cadence=False)
    x, y, yerr = kd.load_kepler_data(os.path.join(KPLR_DIR, "{}".format(kid)))
    return x, y, yerr


def measure_periods(kic, KPLR_DIR):

    print("KepID = ", kic)
    x, y, yerr = get_lc(kic, KPLR_DIR)

    # Use GProtation to get probabilistic rotation periods.
    GP = gp.fit(x, y, yerr, kic)
    GP.gp_fit(nsets=2, p_max=np.log(100))


def parallel(i):
    DATA_DIR = "/export/bbq2/angusr/granola/granola/data"
    KPLR_DIR = "/home/angusr/.kplr/data/lightcurves"

    # Metcalfe data.
#     data = pd.read_csv(os.path.join(DATA_DIR, "metcalfe.csv"))

    # Silva Aguirre data.
    data = pd.read_csv(os.path.join(DATA_DIR, "silva_aguirre.csv"))

    kics = data.KIC
    measure_periods(kics[i], KPLR_DIR)


if __name__ == "__main__":

    nmetcalfe, nsilva = 17, 30

    pool = Pool()
    results = pool.map(parallel, range(nsilva))
