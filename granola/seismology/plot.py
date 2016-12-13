# Create a plot of age vs Jz.
# Create a plot of rotation period vs age.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os

DATA_DIR = "/export/bbq2/angusr/granola/granola/data"
RESULTS_DIR = "/export/bbq2/angusr/granola/granola/results"

# load metcalfe and silva catalogues and merge.
met = pd.read_csv(os.path.join(DATA_DIR, "metcalfe.csv"))
sil = pd.read_csv(os.path.join(DATA_DIR, "silva_aguirre.csv"))

print(met.keys())
print(sil.keys())
data = pd.merge(met, sil, on="KIC", how="right")
print(data.keys())

# load samples.
with h5py.File(os.path.join(RESULTS_DIR, "")
for i, kic in enumerate(data.KIC.values):


# calculate their ages.

# plot rotation period vs age
