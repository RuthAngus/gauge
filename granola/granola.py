"""
Make a plot of age vs J_z for Kepler-TGAS.
Including functions for error propagation.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

import galpy

from gyro import gyro_age
from actions import action

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 18,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)



def get_properties(kic, data, RESULTS_DIR, period=None, period_err=None):
    """
    Calculate age and J_z samples for a kic_tgas star.
    """
    i = data.kepid.values == kic
    i = np.arange(len(data.kepid.values))[i][0]

    if period == None:
        try:  # if a rotation measurement has been made.
            with h5py.File(os.path.join(RESULTS_DIR, "acf_period_samples.h5"),
                        "r") as f:
                p_samps = f["{}".format(kic)][...]
            print("period samples found, period = ", np.mean(p_samps))
        except KeyError:
            return 0, [0, 0], 0, 0, 0, 0
    else:
        p_samps = period_err*np.random.randn(1000) + period

    # Generate all the samples.
    teff_samps, feh_samps, logg_samps, ra_samps, dec_samps, d_samps, \
        pmra_samps, pmdec_samps, plx_samps, v_samps = \
        gen_sample_set(data, i, len(p_samps))

    # Calculate age samples.
    ga = gyro_age(p_samps, teff=teff_samps, feh=feh_samps, logg=logg_samps)
    age_samps = ga.barnes07(version="barnes")

    # Calculate J_z samples.
    try:
        R_kpc, phi_rad, z_kpc, vR_kms, vT_kms, vz_kms, jR, lz, Jz = \
            action(ra_samps[0], dec_samps[0], d_samps[0], pmra_samps[0],
                pmdec_samps[0], v_samps[0])
    except galpy.actionAngle_src.actionAngle.UnboundError:
        return 0, [0, 0], 0, 0, 0, 0


    print("age = {0:.2f}".format(np.mean(age_samps)),
          "period = {0:.2f}".format(np.mean(p_samps)),
          "teff = {0:.1f}".format(np.mean(teff_samps)))
    print("Jz = {:.2}".format(Jz[0]), "\n")

    return age_samps[0], Jz, np.mean(p_samps), np.mean(teff_samps), \
        np.mean(logg_samps), np.mean(feh_samps)


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
    ra, ra_err = d.tgas_ra.values[i], d.tgas_ra_error.values[i]
    dec, dec_err = d.tgas_dec.values[i], d.tgas_dec_error.values[i]
    pmra, pmra_err = d.tgas_pmra.values[i], d.tgas_pmra_error.values[i]
    pmdec, pmdec_err = d.tgas_pmdec.values[i], d.tgas_pmdec.values[i]
    plx = d.tgas_parallax.values[i]
    plx_err = d.tgas_parallax_error.values[i]

    # assign covariance variables
    ra_dec = d.tgas_ra_dec_corr.values[i]
    ra_plx = d.tgas_ra_parallax_corr.values[i]
    ra_pmra = d.tgas_ra_pmra_corr.values[i]
    ra_pmdec = d.tgas_ra_pmdec_corr.values[i]
    dec_plx = d.tgas_dec_parallax_corr.values[i]
    dec_pmra = d.tgas_dec_pmra_corr.values[i]
    dec_pmdec = d.tgas_dec_pmdec_corr.values[i]
    plx_pmra = d.tgas_parallax_pmra_corr.values[i]
    plx_pmdec = d.tgas_parallax_pmdec_corr.values[i]
    pmra_pmdec = d.tgas_pmra_pmdec_corr.values[i]

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
    vs_tracks = pd.read_csv("skeleton3_run_000.out", skiprows=172)

    # Load kic_tgas
    d = pd.read_csv(os.path.join(DATA_DIR, "kic_tgas.csv"))

    # cut on temperature and logg
    m = (d.teff.values < 6250) & (4 < d.logg.values) & \
        np.isfinite(d.teff.values)
    data = d.iloc[m]

    clobber = True

    # Based on the ACF rotation periods, calculate posterior samples for
    # period, age, Jz, teff, etc.
    # If this has been done previously, load previous result.
    ages, Jzs, periods, teffs, loggs, fehs, kics = \
        [np.zeros((len(data.kepid.values))) for i in range(7)]
    for i, kic in enumerate(data.kepid.values):
        print(kic, i, "of", len(data.kepid.values))
        path = os.path.join(RESULTS_DIR, "{}.csv".format(int(kic)))

        if os.path.exists(path) and not clobber:
            df = pd.read_csv(path)
            age, Jz, period = df.age.values, df.Jz.values, df.period.values
            teff, logg, feh = df.teff.values, df.logg.values, df.feh.values

        else:
            print("Calculating properties...")
            age, Jz, period, teff, logg, feh = get_properties(kic, data,
                                                              RESULTS_DIR)
            dic = {"KIC": [int(kic)], "age": [age], "Jz": [Jz[0]],
                   "period": [period], "teff": [teff], "logg": [logg],
                   "feh": [feh]}
            df = pd.DataFrame(dic)
            df.to_csv(path)
        ages[i], Jzs[i], periods[i], teffs[i], loggs[i], fehs[i], kics[i] = \
            float(age), float(Jz[0]), float(period), float(teff), float(logg), \
            float(feh), kic
        if np.isfinite(teffs[i]):
            ga = gyro_age(period, teff=teffs[i])
            ages[i] = ga.vansaders16(vs_tracks)
        else:
            ages[i] = np.nan

    full_dic = {"KIC": kics, "age": ages, "Jz": Jzs, "period": periods,
                "teff": teffs, "logg": loggs, "feh": fehs}
    full_df = pd.DataFrame(full_dic)
    full_df.to_csv("ages_and_actions_vansaders.csv")

    m = ages < 14
    plt.clf()
    # plt.scatter(ages, Jzs, c=teffs, vmin=5000, vmax=max(teffs))
    plt.scatter(ages[m], Jzs[m], c=periods[m])
    plt.xlim(0, 14)
    plt.colorbar()
    plt.xlabel("Age (Gyr)")
    plt.ylabel("Jz")
    plt.ylim(0, 50)
    plt.savefig("age_Jz")

    plt.clf()
    plt.plot(periods[m], Jzs[m], "k.")
    # plt.xlim(0, 14)
    # plt.colorbar()
    plt.xlabel("$P_{\mathrm{rot}} \mathrm{(Days)}$")
    plt.ylabel("$J_z$")
    # plt.ylim(0, 50)
    plt.savefig("period_Jz")
