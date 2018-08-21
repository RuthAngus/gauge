"""
Calculate the actions for a star.
"""
import numpy as np
import pandas as pd
import galpy
from actions import action


def gen_sample_set(d, i, N, rv, rverr):
    """
    Generate all the samples needed for this analysis by sampling from the
    (Gaussian assumed) posteriors.
    parameters:
    ----------
    d: (pandas.dataframe)
        The TGAS dataframe.
    i: (int)
        The index of the star in the database.
    N: (int)
        The number of samples to generate.
    rv: (array)
        The RV array.
    rverr: (array)
        The RV uncertainty array.
    """

    def gen_samps(N, mu, e):
        return e * np.random.randn(N) + mu

    # Generate the non-covariant samples.
    d_samps = 1./gen_samps(N, d.tgas_parallax.values[i],
                           d.tgas_parallax_error.values[i])
    v_samps = gen_samps(N, rv, rverr)

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

    return ra_samps, dec_samps, d_samps, pmra_samps, pmdec_samps, plx_samps, \
        v_samps


def calc_actions(df, rv, rv_err):
    """
    Calculate the actions and their uncertainties using Monte Carlo.
    parameters:
    -----------
    df: (pandas.dataframe)
        The dataframe containing TGAS parameters.
    rv: (array)
        The RV array.
    rv_err: (array)
        The RV uncertainty array.
    """
    nstars = len(df.tgas_source_id.values)
    r, p, z, vr, vt, vz, jr, lz, jz = [np.zeros(nstars) for i in range(9)]
    for i, _ in enumerate(df.tgas_source_id.values):
        print(i, "of", nstars)
        r[i], p[i], z[i], vr[i], vt[i], vz[i], jr[i], lz[i], jz[i] = \
            action(df.ra.values[i], df.dec.values[i],
                   1./df.tgas_parallax.values[i], df.tgas_pmra.values[i],
                   df.tgas_pmdec.values[i], rv[i])
    return jr, lz, jz


def calc_action_errs(df, rv, rverr, nsamps):
    """
    Calculates the actions and the uncertainties on the actions using Monte
    Carlo.
    parameters:
    ----------
    df: (pandas.DataFrame)
        The TGAS dataframe.
    rv: (array)
        The array of rvs.
    rverr: (array)
        The array of rv uncertainties.
    nsamps: (int)
        The number of samples to use for the Monte Carlo calculation.
        Already gets slow with 100.
    returns:
    --------
    actions: (2d array) (nstars, 3)
        Array of the three actions (J_r, L_z, J_z)
    action_errp: (2d array) (nstars, 3)
        Upper action confidence interval uncertainties.
    action_errm: (2d array) (nstars, 3)
        Lower action confidence interval uncertainties.
    """
    r, p, z, vr, vt, vz, jr, l, j, jrerrm, lerrm, jerrm, jrerrp, lerrp, \
        jerrp = [np.zeros(len(df.tgas_source_id.values)) for i in range(15)]

    nstars = len(df.tgas_source_id.values)
    action_samps = np.zeros((nstars, nsamps, 9))  # 9 = nparameters
    for i, _ in enumerate(df.tgas_source_id.values):
        print(i, "of", nstars)

        # Loop over stars
        ra_samps, dec_samps, d_samps, pmra_samps, pmdec_samps, plx_samps, \
            v_samps = gen_sample_set(df, i, nsamps, rv[i], rverr[i])

        # Loop over samples
        for j in range(nsamps):
            action_samps[i, j, :] = action(ra_samps[j], dec_samps[j],
                                           d_samps[j], pmra_samps[j],
                                           pmdec_samps[j], v_samps[j])
        meds = np.median(action_samps, axis=1)
        upper = np.percentile(action_samps, 84, axis=1)
        lower = np.percentile(action_samps, 16, axis=1)
        errps, errms = upper - meds, meds - lower

        jzs, jzerrps, jzerrms = meds[:, -1], errps[:, -1], errms[:, -1]
        lzs, lzerrps, lzerrms = meds[:, -2], errps[:, -2], errms[:, -2]
        jrs, jrerrps, jrerrms = meds[:, -3], errps[:, -3], errms[:, -3]
    return np.vstack((jrs, lzs, jzs)).T, \
        np.vstack((jrerrps, lzerrps, jzerrps)).T, \
        np.vstack((jrerrms, lzerrms, jzerrms)).T


if __name__ == "__main__":

    # print(action(np.array([19, 19]), np.array([30, 30]), np.array([10, 10]),
    #              np.array([.1, .1]), np.array([.1, .1]),
    #              np.array([100, 100])))
    print(action(19, 30, 10, .1, .1, 100))
    assert 0
    df = pd.read_csv("data/kic_tgas.csv")
    rvs = np.zeros(len(df.tgas_source_id.values))
    rverrs = np.zeros_like(rvs)
    print(calc_action_errs(df.iloc[:5], rvs, rverrs, 100))
    # jR, lz, Jz = calc_actions(df, rvs, rverrs)
