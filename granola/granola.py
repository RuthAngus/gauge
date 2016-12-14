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

    try:
        with h5py.File(os.path.join(RESULTS_DIR, "acf_period_samples.h5"),
                    "r") as f:
            p_samps = f["{}".format(kic)][...]
    except KeyError:
        return 0, [0, 0]

    # Generate all the samples.
    try:
        teff_samps, feh_samps, logg_samps, ra_samps, dec_samps, d_samps, \
            pmra_samps, pmdec_samps, plx_samps, v_samps = \
            gen_sample_set(data, i, len(p_samps))
    except KeyError:
        return 0, [0, 0]

    # Calculate age samples.
    ga = gyro_age(p_samps, teff=teff_samps, feh=feh_samps, logg=logg_samps)
    age_samps = ga.barnes07()

    # Calculate J_z samples.
    R_kpc, phi_rad, z_kpc, vR_kms, vT_kms, vz_kms, jR, lz, Jz = \
        action(ra_samps[0], dec_samps[0], d_samps[0], pmra_samps[0],
               pmdec_samps[0], v_samps[0])

    return age_samps[0], Jz


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

    # assign mean and stdev variables
    ra, ra_err = d.tgas_ra.values[i][0], d.tgas_ra_error.values[i][0]
    dec, dec_err = d.tgas_dec.values[i][0], d.tgas_dec_error.values[i][0]
    pmra, pmra_err = d.tgas_pmra.values[i][0], d.tgas_pmra_error.values[i][0]
    pmdec, pmdec_err = d.tgas_pmdec.values[i][0], d.tgas_pmdec.values[i][0]
    plx = d.tgas_parallax.values[i][0]
    plx_err = d.tgas_parallax_error.values[i][0]

    # assign covariance variables
    ra_dec = d.tgas_ra_dec_corr.values[i][0]
    ra_plx = d.tgas_ra_parallax_corr[i][0]
    ra_pmra = d.tgas_ra_pmra_corr.values[i][0]
    ra_pmdec = d.tgas_ra_pmdec_corr.values[i][0]
    dec_plx = d.tgas_dec_parallax_corr[i][0]
    dec_pmra = d.tgas_dec_pmra_corr.values[i][0]
    dec_pmdec = d.tgas_dec_pmdec_corr.values[i][0]
    plx_pmra = d.tgas_parallax_pmra_corr.values[i][0]
    plx_pmdec = d.tgas_parallax_pmdec_corr.values[i][0]
    pmra_pmdec = d.tgas_pmra_pmdec_corr.values[i][0]

    mus = np.array([ra, dec, pmra, pmdec, plx])
    C = np.matrix([[ra_err**2, ra_dec, ra_pmra, ra_pmdec, ra_plx],
                   [ra_dec, dec_err**2, dec_pmra, dec_pmdec, dec_plx],
                   [ra_pmra, dec_pmra, pmra_err**2, pmra_pmdec, plx_pmra],
                   [ra_pmdec, dec_pmdec, pmra_pmdec, pmdec_err**2, plx_pmdec],
                   [ra_plx, dec_plx, plx_pmra, plx_pmdec, plx_err**2]
                   ])
    corr_samps = np.random.multivariate_normal(mus, C, size=N).T
    ra_samps, dec_samps, pmra_samps, pmdec_samps, plx_samps = corr_samps

    return teff_samps, feh_samps, logg_samps, ra_samps, dec_samps, d_samps, \
        pmra_samps, pmdec_samps, plx_samps, v_samps


def gen_samps(N, mu, e1, e2=None):
    """
    Generate teff, feh, logg, etc samples.
    e1 is either a standard deviation or a covariance matrix.
    """
    if not e2:
        return e1 * np.random.randn(N) + mu
    return .5*(e1 + e2) * np.random.randn(N) + mu


def gen_multivariate_samps_general(N, mus, var, covar):
    """
    Generate the covariant samples for a matrix with symmetric covariances.
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

    plt.clf()
    for i, kic in enumerate(data.kepid.values):
        print(kic, i, "of", len(data.kepid.values))
        age, Jz = get_properties(kic, data)
        plt.plot(age, Jz[0], "k.")
    plt.xlabel("Age (Gyr)")
    plt.ylabel("Jz")
    plt.savefig("age_Jz")
