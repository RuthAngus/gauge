# Make a plot of age vs J_z for Kepler-TGAS.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

import os

from gyro import gyro_age
from actions import action


def get_properties(kic, data):
    """
    Calculate age and J_z samples for a kic_tgas star.
    """
    i = data.kepid.values == kic

    with h5py.File(os.path.join(RESULTS_DIR, "acf_period_samples.h5"),
                   "r") as f:
        p_samps = f["{}".format(kic)][...]

    # Generate all the samples.
    teff_samps, feh_samps, logg_samps, ra_samps, dec_samps, d_samps, \
        pmra_samps, pmdec_samps, v_samps = gen_sample_set(data, i,
                                                          len(p_samps))

    # Calculate age samples.
    ga = gyro_age(p_samps, teff=teff_samps, feh=feh_samps, logg=logg_samps)
    age_samps = ga.barnes07()

    # Calculate J_z samples.
    J_z = action(ra_samps, dec_samps, d_samps, pmra_samps, pmdec_samps,
                 v_samps)


def gen_sample_set(d, i, N):
    """
    Generate all the samples needed for this analysis by sampling from the
    (Gaussian assumed) posteriors.
    """

    # Generate the non-covariant samples.
    teff_samps = gen_samps(N, d.teff.values[i], d.teff_err1.values[i],
                           d.teff_err2.values[i])
    feh_samps = gen_samps(N, d.feh.values[i], d.feh_err1.values[i],
                          d.feh_err2.values[i])
    logg_samps = gen_samps(N, d.logg.values[i], d.logg_err1.values[i],
                           d.logg_err2.values[i])
    d_samps = 1./gen_samps(N, d.tgas_parallax.values[i],
                           d.tgas_parallax_error.values[i])
    v_samps = gen_samps(N, 0., 0.)

    # Generate the covariant samples.
    ra_dec_samps = gen_multivariate_samps(N, np.array([0, 0, 0]),
                                          np.array([.1, .1, .1]),
                                          np.array([.05, .01, .2]))
    ra_samps, dec_samps, _ = ra_dec_samps
    plt.clf()
    plt.plot(ra_samps, dec_samps, "k.")
    plt.savefig("test")
    assert 0

    # ra_dec_samps = gen_multivariate_samps(N, d.tgas_ra.values[i][0],
    #                                       d.tgas_dec.values[i][0],
    #                                       d.tgas_ra_error[i][0],
    #                                       d.tgas_dec_error[i][0],
    #                                       d.tgas_ra_dec_corr[i][0])
    # ra_samps, dec_samps = ra_dec_samps
    # pm_samps = gen_multivariate_samps(N, d.tgas_pmra.values[i][0],
    #                                   d.tgas_pmdec.values[i][0],
    #                                   d.tgas_pmra_error[i][0],
    #                                   d.tgas_pmdec_error[i][0],
    #                                   d.tgas_ra_dec_corr[i][0])
    # pmra_samps, pmdec_samps = pm_samps

    return teff_samps, feh_samps, logg_samps, ra_samps, dec_samps, d_samps, \
        pmra_samps, pmdec_samps, v_samps


def gen_samps(N, mu, e1, e2=None):
    """
    Generate teff, feh, logg, etc samples.
    e1 is either a standard deviation or a covariance matrix.
    """
    if not e2:
        return e1 * np.random.randn(N) + mu
    return .5*(e1 + e2) * np.random.randn(N) + mu


def gen_multivariate_samps(N, mus, var, covar):
    """
    Generate the covariant samples.
    """
    # Construct the covariance matrix
    c = np.vstack((np.ones(len(var)), np.array([np.ones(len(var)) * covar for
                                                i in range(len(var)-1)]))).T
    C = c * c.T

    # Fill the diagonal elements with the variances.
    d = np.diag_indices_from(C)
    for i in range(len(mus)):
        C[d[0][i]][d[1][i]] = var[i]

    return np.random.multivariate_normal(mus, C, size=N).T


if __name__ == "__main__":

    # Load the rotation period samples.
    DATA_DIR = "/Users/ruthangus/projects/granola/granola/data"
    RESULTS_DIR = "results"

    # Load kic_tgas
    data = pd.read_csv(os.path.join(DATA_DIR, "kic_tgas.csv"))

    # cut on temperature and logg
    m = (6250 < data.teff.values) * (4 < data.logg.values)
    data = data.iloc[m]

    kic = data.kepid.values[0]
    print(kic)
    get_properties(kic, data)
