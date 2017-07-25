"""
Calculate the actions for a star.
"""
import numpy as np
from granola import gen_sample_set, gen_samps
import matplotlib.pyplot as pl


def calc_actions(df, rv, rv_err):
    """
    Calculate the actions and their uncertainties using Monte Carlo.
    parameters:
    -----------
    df: (pandas.dataframe)
        The dataframe containing TGAS parameters.
    """


def gen_multivariate_samps_general(N, mus, var, covar):
    """
    Generate the covariant samples for a matrix with symmetric covariances.
    parameters:
    ----------
    N: (int)
        The number of samples to draw.
    mus: (array-like)
        The array of means.
    var: (array-like)
        The array of variances.
    covar: (array-like)
        The array of covariances.
    """
    # Construct the covariance matrix
    c = np.vstack((np.ones(len(var)), np.array([np.ones(len(var)) * covar for
                                                i in range(len(var)-1)]))).T
    print(np.shape(c))
    C = c * c.T
    print(np.shape(C))

    # Fill the diagonal elements with the variances.
    d = np.diag_indices_from(C)
    for i in range(len(mus)):
        C[d[0][i]][d[1][i]] = var[i]

    return np.random.multivariate_normal(mus, C, size=N).T


if __name__ == "__main__":
    samps = gen_multivariate_samps_general(10000, np.ones(2), np.ones(2)*.5,
                                           np.ones(2)*.3)
    print(np.shape(samps))
    # print(samps)

    pl.clf()
    pl.plot(samps[0, :], samps[1, :], "k.")
    pl.savefig("test")
