# The gauge
Calibrating MS ages using galactic dynamics with Gaia, Kepler and APOGEE.

Codez:
-----

Prep:
    Produce rotation period posterior samples of all Kepler/TGAS stars.

Data:
    Kepler/TGAS intersection. KIC + rotation periods converted to ages.
        Matt (2012)
        van Saders (2016)
        Barnes (2010)
        Barnes (2007)

Model:
    Age - Action relation of Gaia/APOGEE red giants.

Method:
    Use rotation period posterior samples to test and calibrate four gyro
    models + an intrinsic spread and binary sequence. Step one: Cross
    validate to determine the best model. Step two: evidence integral.
