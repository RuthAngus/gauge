# Plot rotation / bv / age.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool

import os
import sys

import kplr

import gprot_fit as gp
import emcee2_gprot_fit as e2
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
    emcee2 = True
    if emcee2:
        e2.gp_fit(x, y, yerr, kic, "results", p_max=np.log(100))
    else:
        GP = gp.fit(x, y, yerr, kic)
        GP.gp_fit(burnin=2000, nwalkers=16, nruns=10, full_run=1000, nsets=2,
                  p_max=np.log(100))


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
    dataset = str(sys.argv[1])

    pool = Pool()
    if dataset == "metcalfe"
        results = pool.map(parallel, range(nmetcalfe))
    else:
        results = pool.map(parallel, range(nsilva))
