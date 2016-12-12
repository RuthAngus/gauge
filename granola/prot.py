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

    DATA_DIR = "/Users/ruthangus/projects/granola/granola/data"
    RESULTS_DIR = "results"
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # load KIC-TGAS
    data = pd.read_csv(os.path.join(DATA_DIR, "kic_tgas.csv"))

    # cut on temperature and logg
    m = (6250 < data.teff.values) * (4 < data.logg.values)
    data = data.iloc[m]

    fp = h5py.File(os.path.join(RESULTS_DIR, "acf_period_samples.h5"), "w")
    # fa = h5py.File(os.path.join(RESULTS_DIR, "age_samples.h5"), "w")
    for i, kic in enumerate(data.kepid.values[:100]):
        print(kic, i, "of", len(data.kepid.values[:100]))
        x, y, yerr = get_lc(kic)
        period, samples = get_period_samples(x, y)

        # save samples
        pdata = fp.create_dataset(str(kic), np.shape(samples))
        pdata[:] = samples

    # with h5py.File("results/acf_period_samples.h5", "r") as f:
    #     data = f["{}".format(data.kepid.values[1])][...]
    # print(data)

