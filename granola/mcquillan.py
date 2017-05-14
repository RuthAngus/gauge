# Calculate actions and make a df of mcquillan data.

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
           'legend.fontsize': 18,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


# Load the rotation period samples.
DATA_DIR = "/Users/ruthangus/projects/granola/granola/data"
RESULTS_DIR = "/Users/ruthangus/projects/granola/granola/mcquillan_results"

# cross match and load the mcquillan data.
d = pd.read_csv(os.path.join(DATA_DIR, "kplr_tgas_periods.csv"))

# cut on temperature and logg
m = (d.teff.values < 6250) * (4 < d.logg.values) * (d.prot.values > 0)
data = d.iloc[m]

# Calculate posterior samples for period, age, Jz, teff, etc.
ages, Jzs, periods, teffs, loggs, fehs, kics = \
    [np.zeros((len(data.kepid.values))) for i in range(7)]
for i, kic in enumerate(data.kepid.values):
    path = os.path.join(RESULTS_DIR, "{}.csv".format(int(kic)))
    if os.path.exists(path):
        df = pd.read_csv(path)
        age, Jz, period = df.age.values, df.Jz.values, df.period.values
        teff, logg, feh = df.teff.values, df.logg.values, df.feh.values
    else:
        print("Calculating properties...")
        age, Jz, period, teff, logg, feh = \
            get_properties(kic, data, RESULTS_DIR, data.prot.values[i],
                        data.prot_err.values[i])
        dic = {"KIC": [int(kic)], "age": [age], "Jz": [Jz[0]],
                "period": [period], "teff": [teff], "logg": [logg],
                "feh": [feh]}
        df = pd.DataFrame(dic)
        df.to_csv(path)
    ages[i], Jzs[i], periods[i], teffs[i], loggs[i], fehs[i], kics[i] = \
        float(age), float(Jz[0]), float(period), float(teff), \
        float(logg), float(feh), kic

full_dic = {"KIC": kics, "age": ages, "Jz": Jzs, "period": periods,
            "teff": teffs, "logg": loggs, "feh": fehs}
full_df = pd.DataFrame(full_dic)
full_df.to_csv("ages_and_actions.csv")

m = ages < 14
plt.clf()
plt.scatter(np.log(ages[m]), np.log(Jzs[m]), c=periods[m], cmap="GnBu_r",
            edgecolor=".7")
clb = plt.colorbar()
clb.ax.set_ylabel("$\mathrm{P}_{\mathrm{rot}}~\mathrm{(Days)}$")
plt.xlabel("$\ln(\mathrm{Age,~Gyr})$")
plt.ylabel("$\ln(J_z, \mathrm{kpc~kms}^{-1})$")
plt.savefig("age_Jz.pdf")


plt.clf()
plt.plot(np.log(periods[m]), np.log(Jzs[m]), "k.")
plt.xlabel("$\ln(\mathrm{P}_{\mathrm{rot}},~\mathrm{Days})$")
plt.ylabel("$\ln(J_z, \mathrm{kpc~kms}^{-1})$")
plt.subplots_adjust(bottom=.15)
plt.savefig("period_Jz.pdf")
