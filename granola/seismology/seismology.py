# Plot rotation / bv / age.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os

import kplr

from gprot_fit import fit
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


if __name__ == "__main__":

    # load van saders table
    DATA_DIR = "/Users/ruthangus/projects/granola/granola/data"
    DATA_DIR = "/export/bbq2/angusr/granola/granola/data"
    data = pd.read_csv(os.path.join(DATA_DIR, "vanSaders.txt"))
    kics = data.KIC

    KPLR_DIR = "/Users/ruthangus/.kplr/data/lightcurves"
    KPLR_DIR = "/home/angusr/.kplr/data/lightcurves"
    for kic in kics:
        print("KepID = ", kic)
        x, y, yerr = get_lc(kic, KPLR_DIR)

        # Use GProtation to get probabilistic rotation periods.
        gp = fit(x, y, yerr, kic)
