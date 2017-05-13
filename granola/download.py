#  Downloading Kepler light curves

import os

import pandas as pd
import kplr

import kepler_data as kd


def get_lc(id, KPLR_DIR="/Users/ruthangus/.kplr/data/lightcurves"):
    """
    Downloads the kplr light curve and loads x, y and yerr.
    """
    kid = str(int(id)).zfill(9)
    path = os.path.join(KPLR_DIR, "{}".format(kid))
    if not os.path.exists(path):
        client = kplr.API()
        star = client.star(kid)
        print("Downloading LC...")
        star.get_light_curves(fetch=True, short_cadence=False)
        x, y, yerr = kd.load_kepler_data(os.path.join(KPLR_DIR,
                                                      "{}".format(kid)))
    else:
        x, y, yerr = kd.load_kepler_data(os.path.join(KPLR_DIR,
                                                      "{}".format(kid)))
    x -= x[0]
    return x, y, yerr


if __name__ == "__main__":

    DATA_DIR = "/Users/ruthangus/projects/granola/granola/data"

    # load KIC-TGAS
    data = pd.read_csv(os.path.join(DATA_DIR, "kic_tgas.csv"))

    # cut on temperature and logg
    m = (data.teff.values < 6250) * (4 < data.logg.values)
    data = data.iloc[m]

    for i, kic in enumerate(data.kepid.values[275:400]):
        print(kic, i, "of", len(data.kepid.values[275:400]))
        x, y, yerr = get_lc(kic)
