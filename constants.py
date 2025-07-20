'''Store constants used throughout analysis.'''


# astrophysical constants
c = 299792458.0
G = 6.6743e-11
Msun = 1.9891e30
Tsun = Msun * G / c**3.
kpc = 3.085677581491367e+19
Mpc = 1.e3 * kpc
Tkpc = kpc / c

# times
year_months = 12.
year_days = 365.25
day_sec = 24. * 60. * 60.
year_sec = year_days * day_sec
us_sec = 1.e-6

