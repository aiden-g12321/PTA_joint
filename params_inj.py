'''Parameters injected into data and constants used in analysis.'''


import numpy as np
import jax.numpy as jnp
import jax.random as jr

# use double precision
from jax import config
config.update('jax_enable_x64', True)


# astrophysical constants
c = 299792458.0
G = 6.6743e-11
Msun = 1.9891e30
Tsun = Msun * G / c**3.
kpc = 3.085677581491367e+19
Mpc = 1.e3 * kpc
Tkpc = kpc / c

# number of pulsars
Np = 5

# number of frequency bins used in CW FFT model
Nf = 7
Na = 2 * Nf

# reference time to model CW signal
tref = 1.e9

# span of observations
Tspan_yr = 15.
Tspan = Tspan_yr * 365.25 * 24. * 3600.
Ntoas = int(Tspan_yr * 12)

# frequency bins
freqs = jnp.arange(1, Nf + 1) / Tspan

# observe ~monthly (+/- 2 days)
observation_seed = 0
observation_day_offset = jr.normal(jr.key(observation_seed), (Ntoas,)) * 2. * 24. * 3600.
# don't offset first and last observation
observation_day_offset = observation_day_offset.at[jnp.r_[0, -1]].set(jnp.zeros(2))
toas = jnp.array([jnp.linspace(tref, tref + Tspan, Ntoas, endpoint=True) \
                  + observation_day_offset
                  for _ in range(Np)])
MJDs = toas / 86400.

# Fourier design matrix
Fs = jnp.zeros((Np, Ntoas, Na))
for i in range(Np):
    for j in range(Nf):
        Fs = Fs.at[i, :, 2 * j].set(jnp.sin(2. * jnp.pi * freqs[j] * toas[i]))
        Fs = Fs.at[i, :, 2 * j + 1].set(jnp.cos(2. * jnp.pi * freqs[j] * toas[i]))

# white noise (EFAC only) in each pulsar
efac_seed = 1
efac_min = 0.5
efac_max = 3.0
efacs_inj = jr.uniform(key=jr.key(efac_seed),
                       shape=(Np,),
                       minval=efac_min,
                       maxval=efac_max)

# pulsar uncertainty
psr_uncertainty_us = 0.5
psr_uncertainty_s = psr_uncertainty_us * 1.e-6

# coninuous wave parameters
gwtheta_inj = 2 * jnp.pi / 5
gwphi_inj = 7 * jnp.pi / 4.
mc_inj = 10.**8.5
dist_inj = 1.0
fgw_inj = 4.e-9
phase0_inj = 0.
psi_inj = 0.
inc_inj = jnp.pi / 2.
log10_dist_inj = jnp.log10(dist_inj)
cosinc_inj = jnp.cos(inc_inj)
costheta_inj = jnp.cos(gwtheta_inj)
log10_mc_inj = jnp.log10(mc_inj)
log10_fgw_inj = jnp.log10(fgw_inj)
cw_params_inj = jnp.array([log10_mc_inj, log10_fgw_inj, cosinc_inj, psi_inj, 
                          log10_dist_inj, costheta_inj, gwphi_inj, phase0_inj])

# pulsar distance (kpc)
psr_dist_seed = 2
psr_dist_min = 0.1
psr_dist_max = 7.
psr_dists_inj = jr.uniform(key=jr.key(psr_dist_seed),
                           shape=(Np,),
                           minval=psr_dist_min,
                           maxval=psr_dist_max)
# set pulsar distance uncertainty to fixed value
psr_dists_stdev = jnp.array([0.2] * Np)

# pulsar positions
psr_pos_seed = 3
psr_pos_min = -1.
psr_pos_max = 1.
psr_pos_not_normal = jr.uniform(key=jr.key(psr_pos_seed),
                                shape=(Np, 3),
                                minval=psr_pos_min,
                                maxval=psr_pos_max)
# normalize pulsar positions
psr_pos = psr_pos_not_normal / jnp.sum(psr_pos_not_normal**2., axis=1)[:, None]


# compute injected pulsar phases from other CW / pulsar parameters
def get_psr_phase(cw_params, psr_position, psr_dist):

    # unpack CW parameters
    log10_mc, log10_fgw, cosinc, psi, log10_dist, costheta, gwphi, phase0 = cw_params

    # define sky location parameters
    singwtheta = jnp.sin(jnp.arccos(costheta))
    cosgwtheta = costheta
    singwphi = jnp.sin(gwphi)
    cosgwphi = jnp.cos(gwphi)
    omhat = jnp.array([-singwtheta * cosgwphi, -singwtheta * singwphi, -cosgwtheta])

    # store pulsar phase
    cosMu = -jnp.dot(omhat, psr_position)
    pphase = (1 + 256/5 * (10**log10_mc*Tsun)**(5/3) * (jnp.pi * 10**log10_fgw)**(8/3)
            * psr_dist*Tkpc*(1-cosMu)) ** (5/8) - 1
    pphase /= 32 * (10**log10_mc*Tsun)**(5/3) * (jnp.pi * 10**log10_fgw)**(5/3)
    psr_phase = -pphase%(2*jnp.pi)

    return psr_phase

# store injected pulsar phases
psr_phases_inj = jnp.array([get_psr_phase(cw_params_inj, psr_position, psr_dist)
                            for psr_position, psr_dist in zip(psr_pos, psr_dists_inj)])

# store injected parameters in one array
x_inj = jnp.concatenate((efacs_inj, cw_params_inj, psr_phases_inj, psr_dists_inj))

# parameter bounds
efac_mins = jnp.ones(Np) * efac_min
efac_maxs = jnp.ones(Np) * efac_max
psr_dist_mins = jnp.ones(Np) * psr_dist_min
psr_dist_maxs = jnp.ones(Np) * psr_dist_max
psr_phase_mins = jnp.zeros(Np)
psr_phase_maxs = jnp.ones(Np) * 2. * jnp.pi
cw_mins = jnp.array([7., -10., -1., -jnp.pi / 2., -1., -1., 0., -jnp.pi / 2.])
cw_maxs = jnp.array([10., -7.2, 1., jnp.pi / 2., 2., 1., 2. * jnp.pi, jnp.pi / 2.])
x_mins = jnp.concatenate((efac_mins, cw_mins, psr_phase_mins, psr_dist_mins))
x_maxs = jnp.concatenate((efac_maxs, cw_maxs, psr_phase_maxs, psr_dist_maxs))

# parameter labels
efac_labels = np.array([rf'$\mathrm{{EFAC}} \;\; [{i}]$' for i in np.arange(1, Np + 1)])
cw_labels = np.array([r'$\log_{10}(\mathcal{M}\,\,[M_\odot])$', r'$\log_{10}(f_{GW}\,\,[\text{Hz}])$', r'$\cos{\iota}$', r'$\psi$', 
                   r'$\log_{10}(D_{L}\,\,[\text{Mpc}])$', r'$\cos{\theta}$', r'$\phi$', r'$\Phi_0$'])
phase_labels = [rf'$\Phi$ [{i}]' for i in np.arange(1, Np + 1)]
dist_labels = [f'L [{i}]' for i in np.arange(1, Np + 1)]
x_labels = np.concatenate((efac_labels, cw_labels, phase_labels, dist_labels))

# indices to access parameter subsets from arrays
ndim = x_inj.shape[0]
efac_ndxs = jnp.r_[:Np]
cw_psr_ndxs = jnp.r_[Np : ndim]
cw_ndxs = jnp.r_[Np : Np + 8]
psr_ndxs = jnp.r_[Np + 8 : ndim]
psr_phases_ndxs = psr_ndxs[:Np]
psr_dist_ndxs = psr_ndxs[Np:]

