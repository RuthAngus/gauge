# Make a plot of age vs J_z for Kepler-TGAS.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

import os

from gyro import gyro_age
from actions import action
from granola import get_properties

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 13,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def calc_dispersion(age, jz, nbins):
    hist_age, bins = np.histogram(age, nbins)  # make histogram
    dispersions, Ns = [], []
    m = age < bins[0]
    dispersions.append(RMS(jz[m]))
    Ns.append(len(age[m]))
    for i in range(len(bins)-1):
        m = (bins[0] < age) * (age < bins[i+1])
        if len(age[m]):
            dispersions.append(RMS(jz[m]))
            Ns.append(len(age[m]))
    return bins, np.array(dispersions), np.array(Ns)


def RMS(x):
    return (np.median(x**2))**.5


def dispersion(ages, Jzs, minage, maxage):
    """
    Dispersion in a single bin.
    """
    m = (minage < ages) * (ages < maxage)
    return RMS(Jzs[m]), len(ages[m])


def x_and_y(ages, Jzs):
    xs = np.linspace(min(ages), max(ages), 1000)
    ys = []
    for x in xs:
        y, N = dispersion(ages, Jzs, x-.5, x+.5)
        ys.append(y)
    return xs, ys


if __name__ == "__main__":

    DATA_DIR = "/Users/ruthangus/granola/granola/data"
    d = pd.read_csv("ages_and_actions.csv")
    # d = pd.read_csv("ages_and_actions_vansaders.csv")
    m = (d.age.values > 0) * (d.age.values < 14)
    df = d.iloc[m]

    ages, dispersions, Ns = calc_dispersion(df.age.values, df.Jz.values, 8)
    d_err = dispersions / (2 * Ns - 2)**.5

    plt.clf()
    plt.errorbar(ages - .5*(ages[1] - ages[0]), np.array(dispersions),
                 yerr=d_err, fmt="k.", capsize=0, ms=.1)
    plt.step(ages, dispersions, color="k")
    plt.xlabel("$\mathrm{Age~Gyr}$")
    plt.ylabel("$\sigma J_z~(\mathrm{Kpc~kms}^{-1})$")
    plt.savefig("linear_age_dispersion.pdf")
    # plt.savefig("linear_age_dispersion_vansaders.pdf")

    m = np.log(df.age.values) > - 1
    lnages, dispersions, Ns = calc_dispersion(np.log10(df.age.values[m]),
                                      df.Jz.values[m], 8)
    d_err = dispersions / (2 * Ns - 2)**.5

    plt.clf()
    plt.errorbar(lnages - .5*(lnages[1] - lnages[0]), np.array(dispersions),
                 yerr=d_err, fmt="k.", capsize=0, ms=.1)
    plt.step(lnages, dispersions, color="k")
    plt.xlabel("$\log_{10}(\mathrm{Age,~Gyr})$")
    plt.ylabel("$\sigma J_z~(\mathrm{Kpc~kms}^{-1})$")
    # plt.xlim(-1, 2.6)
    plt.subplots_adjust(left=.15, bottom=.15)
    plt.savefig("log_age_dispersion.pdf")
    # plt.savefig("log_age_dispersion_vansaders.pdf")

    m = np.log(df.age.values) > - 1
    x, y = x_and_y(np.log(df.age.values[m]), df.Jz.values[m])

    plt.clf()
    plt.plot(x, y)
    plt.savefig("cont_age_dispersion.pdf")
    # plt.savefig("cont_age_dispersion_vansaders.pdf")


    """
    Plot vansaders model and barnes model on the same axes.
    """
    DATA_DIR = "/Users/ruthangus/granola/granola/data"
    d1 = pd.read_csv("ages_and_actions.csv")
    d2 = pd.read_csv("ages_and_actions_vansaders.csv")
    print(np.shape(d1), np.shape(d2))
    d = pd.merge(d1, d2, on="KIC", how="inner", suffixes=("", "_vs"))

    m1 = (d.age.values > 0) * (d.age.values < 14)
    d1 = d.iloc[m1]
    m2 = (d.age_vs.values > 0) * (d.age_vs.values < 14)
    d2 = d.iloc[m2]

    m1 = np.log(d1.age.values) > - 1
    lnages1, dispersions1, Ns1 = calc_dispersion(np.log10(d1.age.values[m1]),
                                                 d1.Jz.values[m1], 8)
    d_err1 = dispersions / (2 * Ns1 - 2)**.5

    m2 = (np.log(d2.age_vs.values) > - 1) #& (df2.Jz.values < max(df1.Jz.values))
    lnages2, dispersions2, Ns2 = calc_dispersion(np.log10(d2.age_vs.values[m2]),
                                                 d2.Jz_vs.values[m2], 8)
    # lnages2, dispersions2, Ns2 = calc_dispersion(np.log10(d2.age_vs.values),
                                                 # d2.Jz_vs.values, 8)
    d_err2 = dispersions2 / (2 * Ns2 - 2)**.5

    plt.clf()

    # Random stars.
    import random
    rdisps = np.zeros((1000, 9))
    for i in range(1000):
        rage, rdisp, rNs = calc_dispersion(np.random.choice(
                                           (d1.age.values[m1]),
                                        size=len(np.log10(d1.age.values[m1]))),
                                        d1.Jz.values[m1], 8)
        rd_err = rdisp / (2 * rNs - 2)**.5
        rdisps[i, :] = rdisp
        # plt.step(rage, rdisp, color="k", alpha=.01)

    # plt.errorbar(rage - .5*(rage[1] - rage[0]),
                # np.array(rdisp), yerr=rd_err, fmt=".", capsize=0,
                # ms=.1, color="k", alpha=.1)
    # plt.step(rage, rdisp, label=("$\mathrm{Random}$"), color="k", alpha=.1)
    rdisp_av = np.mean(rdisps, axis=0)
    rdisp_std = np.std(rdisps, axis=0)
    print(np.mean(d1.Jz.values))
    plt.axhline(np.mean(d1.Jz.values), color=".5", ls="--")
    # plt.fill_between(rage, rdisp_av-rdisp_std, rdisp_av+rdisp_std,
    #                  label="$\mathrm{Random}$", alpha=.2, color="k",
    #                  edgecolor=".9", linewidth=0)

    plt.errorbar(lnages1 - .5*(lnages1[1] - lnages1[0]),
                 np.array(dispersions1), yerr=d_err1, ms=5, fmt="o", capsize=0,
                 color="cornflowerblue", label="$\mathrm{Model}~1$")
    # plt.step(lnages1, dispersions1, label="$\mathrm{Model}~1$",
    #          color="cornflowerblue")

    plt.errorbar(lnages2 - .5*(lnages2[1] - lnages2[0]),
                 np.array(dispersions2), yerr=d_err2, ms=5, fmt="o", capsize=0,
                 color="orange", label="$\mathrm{Model}~2$")
    # plt.step(lnages2, dispersions2, label="$\mathrm{Model}~2$",
    #          color="orange")

    plt.legend(loc="lower right")
    # plt.xlim(-.2, 1)
    plt.xlabel("$\log_{10}(\mathrm{Age,~Gyr})$")
    plt.ylabel("$\sigma J_z~(\mathrm{Kpc~kms}^{-1})$")
    plt.subplots_adjust(left=.15, bottom=.15)
    # plt.ylim(.75, 1.4)
    plt.ylim(0, 1.4)

    dw = pd.read_csv("dwarf.txt")
    plt.plot(np.log10(dw.age.values), dw.jz.values, ".7", ls="--")

    plt.savefig("log_age_dispersion_both.pdf")
    plt.savefig("log_age_dispersion_both")

    plt.clf()
    plt.hist(d1.Jz.values[m1], 20, alpha=.5, edgecolor="k")
    plt.hist(d2.Jz_vs.values[m2], 1000, alpha=.5, edgecolor="k")
    plt.xlim(0, 100)
    plt.savefig("jz_hist_compare")
