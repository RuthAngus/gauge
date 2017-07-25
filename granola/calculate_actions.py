"""
Calculate the actions for a star.
"""
from granola import gen_sample_set, gen_samps


def calc_actions(df, rv, rv_err):
    """
    Calculate the actions and their uncertainties using Monte Carlo.
    parameters:
    -----------
    df: (pandas.dataframe)
        The dataframe containing TGAS parameters.
    """


def gen_sample_set(d, i, N, rv, rverr):
    """
    Generate all the samples needed for this analysis by sampling from the
    (Gaussian assumed) posteriors.
    """

    def gen_samps(N, mu, e):
        return e1 * np.random.randn(N) + mu

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
