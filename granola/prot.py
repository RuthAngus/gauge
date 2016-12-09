# Measure ACF rotation periods for each star in KTGAS.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import h5py

import kplr

import simple_acf as sa
import kepler_data as kd


def get_lc(id, KPLR_DIR="/Users/ruthangus/.kplr/data/lightcurves"):
    """
    Downloads the kplr light curve and loads x, y and yerr.
    """
    kid = str(int(id)).zfill(9)
    client = kplr.API()
    star = client.star(kid)
    star.get_light_curves(fetch=True, short_cadence=False)
    x, y, yerr = kd.load_kepler_data(os.path.join(KPLR_DIR, "{}".format(kid)))
    return x, y, yerr


def get_period_samples(x, y, nsamps=1000, ndays=200, pfunc=.1):
    """
    Measure period with simple_acf using first ndays of data to reduce time.
    Generate samples from a Gaussian distribution centered on the period with
    a made up stdev.

    param x: (array)
        The time array.
    param y: (array)
        The flux array.
    param nsamps: (int)
        The number of samples.
    param ndays: (int)
        The number of days to use for the ACF period measurement.
    param pfunc: (float)
        The fractional uncertainty on the period to be used as the Gaussian
        stdev during the sampling.
    Returns rotation period (days) and period samples.
    """
    period, acf, lags, rvar = sa.simple_acf(x[:ndays], y[:ndays])
    period_samps = period*pfunc*np.random.randn(nsamps) + period
    return period, period_samps


if __name__ == "__main__":

    # load kepler ids
    DATA_DIR = "/Users/ruthangus/projects/gauge/gauge/data"
    data = pd.read_csv(os.path.join(DATA_DIR, "kic_tgas.csv"))

    for i in range(1):
        x, y, yerr = get_lc(data.kepid.values[i])
        period, samples = get_period_samples(x, y)

    # save the samples somehow.
    f = h5py.File(os.path.join(RESULTS_DIR, "{0}.h5".format(id)), "w")
    data = f.create_dataset(id)

    # calculate age samples.
