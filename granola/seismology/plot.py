# Create a plot of age vs Jz.
# Create a plot of rotation period vs age.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import random

import os

import barnes_age as ba
import teff_bv as tb

plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 20,
           'xtick.labelsize': 15,
           'ytick.labelsize': 15,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def age_period(data, vs):
    # make a plot of rotation period vs asteroseismic age.
    plt.clf()
    a, ps, ages = [], [], []
    for i, kic in enumerate(data.KIC.values):
        m = data.KIC.values == int(kic)
        mv = vs.KIC.values == int(kic)
        in_vs = False
        if len(vs.KIC.values[mv]):  # is it in the van Saders catalogue?
            in_vs = True
        print(int(kic))
        # Find and load rotation period samples
        if os.path.exists(os.path.join(RESULTS_DIR, "{}.h5".format(int(kic)))):
            with h5py.File(os.path.join(RESULTS_DIR,
                           "{}.h5".format(int(kic))), "r") as f:
                samples = f["samples"][...]
                nw, ns, nd = np.shape(samples)
                psamps = random.sample(np.reshape(samples[:, :, 4], nw*ns),
                                       500)
                asamps = random.sample(np.reshape(samples[:, :, 0], nw*ns),
                                       500)
                p = np.median(psamps)
                ps.append(p)
                ages.append(data.age.values[m])
                a.append(np.median(asamps))
                perrp = np.percentile(psamps, 84) - p
                perrm = p - np.percentile(psamps, 16)
                plt.errorbar(data.age.values[m], np.exp(p), zorder=0,
                             yerr=perrm*np.exp(p), fmt="k.", capsize=0)
#                 if in_vs:
#                     plt.errorbar(data.age.values[m], np.exp(p),
#                                  yerr=perrm*np.exp(p), fmt=".",
#                                  color="HotPink", capsize=0)
    plt.scatter(ages, np.exp(ps), s=20, c=a, cmap="GnBu",
                edgecolor="", vmin=min(a), vmax=max(a), zorder=2)
    print(min(a), max(a))
    plt.colorbar()

    plt.xlabel("$\mathrm{Age~(Gyr)}$")
    plt.ylabel("$\mathrm{Period~(days)}$")
    plt.savefig("age_period")


def period_period(data, vs):
    # make a plot of rotation period vs asteroseismic age.
    plt.clf()
    for i, kic in enumerate(data.KIC.values):
        m = data.KIC.values == int(kic)
        mv = vs.KIC.values == int(kic)
        if len(vs.KIC.values[mv]):  # is it in the van Saders catalogue?
            print(kic)
            if os.path.exists(os.path.join(RESULTS_DIR,
                              "{}.h5".format(int(kic)))):
                with h5py.File(os.path.join(RESULTS_DIR,
                               "{}.h5".format(int(kic))), "r") as f:
                    samples = f["samples"][...]
                    nw, ns, nd = np.shape(samples)
                    psamps = random.sample(np.reshape(samples[:, :, 4],
                                           nw*ns), 500)
                    p = np.median(psamps)
                    perrp = np.percentile(psamps, 84) - p
                    perrm = p - np.percentile(psamps, 16)
                    xs = np.arange(0, 60, .1)
                    plt.plot(xs, xs, "k--")
#                     plt.errorbar(np.log(vs.period.values[mv]), p, yerr=perrm,
#                                  xerr=vs.period_err.values[mv]
#                                  /vs.period.values[mv], fmt="k.", capsize=0)

                    plt.errorbar(vs.period.values[mv], np.exp(p),
                                 yerr=perrm*np.exp(p),
                                 xerr=vs.period_err.values[mv], fmt="k.",
                                 capsize=0)

    plt.xlabel("$\mathrm{ACF~Period}$")
    plt.ylabel("$\mathrm{GP~Period}$")
    plt.savefig("period_period")


if __name__ == "__main__":
    DATA_DIR = "/export/bbq2/angusr/granola/granola/data"
    RESULTS_DIR = "/export/bbq2/angusr/granola/granola/seismology/results"

    # load van Saders catalogue
    vs = pd.read_csv(os.path.join(DATA_DIR, "vanSaders.txt"))

    # load metcalfe and silva catalogues and merge.
    met = pd.read_csv(os.path.join(DATA_DIR, "metcalfe.csv"))
    sil = pd.read_csv(os.path.join(DATA_DIR, "silva_aguirre.csv"))
    data = pd.merge(met, sil, on=["KIC", "age"], how="right")

    print(data.keys())
    period_period(data, vs)
    age_period(data, vs)
