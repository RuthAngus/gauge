"""
Test the code for calculating actions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from calculate_actions import calc_actions, calc_action_errs, gen_sample_set


def test_gen_sample_set(nsamps):
    df = pd.read_csv("data/kic_tgas.csv")
    rvs = np.zeros(len(df.tgas_source_id.values))
    rverrs = np.zeros_like(rvs)
    samps = gen_sample_set(df, 0, nsamps, rvs, rverrs)
    assert np.shape(samps) == (7, nsamps)


def test_calc_action_errs(nsamps):
    df = pd.read_csv("data/kic_tgas.csv")
    rvs = np.zeros(len(df.tgas_source_id.values))
    rverrs = np.zeros_like(rvs)
    action_errs = calc_action_errs(df.iloc[:3], rvs[:3], rverrs[:3], nsamps)
    print(np.shape(action_errs))
    pl.clf()
    pl.plot(df.pmra.values[:3], action_errs[:, ]


if __name__ == "__main__":
    # test_gen_sample_set(1000)
    test_calc_action_errs(10)
