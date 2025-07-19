'''Model and parameter attributes used throughout analysis.'''


import numpy as np

from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr

# use double precision
from jax import config
config.update('jax_enable_x64', True)

import constants as c



###############################################################################################
######################################### PULSARS #############################################
###############################################################################################

# number of pulsars
Np = 5

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
psr_pos = psr_pos_not_normal / jnp.sqrt(jnp.sum(psr_pos_not_normal**2., axis=1)[:, None])



###############################################################################################
##################################### TOA OBSERVATIONS ########################################
###############################################################################################

# span of observations
Tspan_yr = 15.
Tspan = Tspan_yr * c.year_sec

# pulsar TOA uncertainty
psr_uncertainty_us = 0.5
psr_uncertainty_s = psr_uncertainty_us * 1.e-6

# reference time to start observations
tref = 1.e9

# observe TOAs ~monthly
Ntoas = int(Tspan_yr * c.year_months)
toas_no_offset = jnp.array([jnp.linspace(tref, tref + Tspan, Ntoas, endpoint=True)
                            for _ in range(Np)])

# offset TOA observations by ~couple days so not evenly spaced
toa_offset_seed = 0
day_offset = 2.
toa_offsets = jr.normal(jr.key(toa_offset_seed), (Np, Ntoas)) * day_offset * c.day_sec

# don't offset first and last TOA to preserve Tspan
toa_offsets = toa_offsets.at[:, jnp.r_[0, -1]].set(jnp.zeros((Np, 2)))

# make TOAs
toas = jnp.array([jnp.linspace(tref, tref + Tspan, Ntoas, endpoint=True) \
                  for _ in range(Np)]) + toa_offsets
MJDs = toas / c.day_sec

# frequency bins
Nf = 10
Na = 2 * Nf
freqs = jnp.arange(1, Nf + 1) / Tspan

# Fourier design matrix
Fs = jnp.zeros((Np, Ntoas, Na))
for i in range(Np):
    for j in range(Nf):
        Fs = Fs.at[i, :, 2 * j].set(jnp.sin(2. * jnp.pi * freqs[j] * toas[i]))
        Fs = Fs.at[i, :, 2 * j + 1].set(jnp.cos(2. * jnp.pi * freqs[j] * toas[i]))



###############################################################################################
######################################## WHITE NOISE ##########################################
###############################################################################################

# model white noise
model_wn = True

# EFAC parameter bounds
efac_min = 0.5
efac_max = 3.0

# randomly draw EFACs to inject in each pulsar
efac_seed = 1
efacs_inj = jr.uniform(key=jr.key(efac_seed),
                       shape=(Np,),
                       minval=efac_min,
                       maxval=efac_max)

# EFAC parameter labels
efac_labels = np.array([rf'EFAC$_{{{i}}}$' for i in range(1, Np + 1)])

# number of EFAC parameters
N_efac = Np

# white noise covariance matrix
Ns = jnp.array([jnp.eye(Ntoas) * psr_uncertainty_s**2.
                for _ in range(Np)])



###############################################################################################
#################################### INTRINSIC RED NOISE ######################################
###############################################################################################

# model intrinsic pulsar red noise
model_rn = True

# intrinsic pulsar red noise parameter bounds
rn_log_amp_min = -15.
rn_log_amp_max = -12.
rn_gamma_min = 2.
rn_gamma_max = 7.
rn_mins = jnp.array([rn_log_amp_min, rn_gamma_min] * Np)
rn_maxs = jnp.array([rn_log_amp_max, rn_gamma_max] * Np)

# randomly draw red noise parameters to inject in each pulsar
rn_seed = 2
rn_inj = jr.uniform(key=jr.key(rn_seed),
                    shape=(2 * Np,),
                    minval=rn_mins,
                    maxval=rn_maxs)

# intrinsic pulsar red noise parameter labels
rn_labels = np.array([rf'$\log_{{{10}}}\,A_{{{i // 2}}}$' if i % 2 == 0 else rf'$\gamma_{{{i // 2}}}$'
                      for i in range(2, 2 * Np + 2)])

# number of red noise parameters
N_rn = 2 * Np



###############################################################################################
################################ GRAVITATIONAL WAVE BACKGROUND ################################
###############################################################################################

# model gravitational wave background
model_gwb = True

# gravitational wave background parameter bounds
gwb_log_amp_min = -17.
gwb_log_amp_max = -12.
gwb_gamma_min = 2.
gwb_gamma_max = 7.
gwb_mins = jnp.array([gwb_log_amp_min, gwb_gamma_min])
gwb_maxs = jnp.array([gwb_log_amp_max, gwb_gamma_max])

# GWB parameters to inject
gwb_inj = jnp.array([-14., 13. / 3.])

# intrinsic pulsar red noise parameter labels
gwb_labels = np.array([r'$\log_{{{10}}}\,A_B$', r'$\gamma_B$'])

# number of gravitational wave background parameters
N_gwb = 2

# angles between pulsars
angles = np.zeros((Np, Np))
for i in range(Np):
    for j in range(i, Np):
        pos1 = psr_pos[i]
        pos2 = psr_pos[j]
        angles[i,j] = angles[j,i] = np.arccos(np.clip(np.dot(pos1, pos2), -1.0, 1.0))
angles = jnp.array(angles)

# Hellings-Downs weighting
alpha = np.zeros((Np, Np))
for i in range(Np):
    for j in range(Np):
        if i == j:
            alpha[i,j] = 1.
        else:
            ang = angles[i,j]
            beta = (1. - np.cos(ang)) / 2.
            alpha[i,j] = 1.5 * beta * np.log(beta) - 0.25 * beta + 0.5
alpha = jnp.array(alpha)
alpha_inv = jnp.linalg.inv(alpha)



###############################################################################################
#################################### FOURIER COEFFICIENTS #####################################
###############################################################################################

# bounds on Fourier coefficients
a_min = -100_000.
a_max = 100_000.

# Fourier coefficient labels
a_labels = np.array([[rf'$a^{{{j // 2}}}_{{{i}}}$' if j % 2 == 0 else rf'$b^{{{j // 2}}}_{{{i}}}$'
                      for j in range(2, Na + 2)] for i in range(1, Np + 1)]).flatten()

# total number of Fourier coefficients in PTA
Na_PTA = Np * Na

# diagonal of covariance matrix of Fourier coefficients using power-law
rho_scale = (c.year_sec ** 3.) / (12. * (jnp.pi ** 2.) * Tspan)
rho_scale1 = Tspan / c.year_sec
arr = jnp.repeat(jnp.arange(1, Nf + 1), 2)
arr /= rho_scale1
arr = jnp.array(arr)

@jit
def get_rho_diag(hyper_params):
    logAmp, gamma = hyper_params
    Amp = 10. ** logAmp
    return (Amp ** 2.) * rho_scale * (arr **  (-gamma))
vectorized_get_rho_diag = jit(vmap(get_rho_diag))

# get power law covariance matrix for Fourier coefficients from background
@jit
def get_phi_gwb(hypers_gwb):
    rhodiag = get_rho_diag(hypers_gwb)
    return alpha[:, :, None] * rhodiag[None, None, :]
fast_get_phi_gwb = jit(get_phi_gwb)

# get covariance matrix of coefficients for RN + GWB
rn_phi_inj = jnp.diag(jnp.array([get_rho_diag(rn_hypers)
                                 for rn_hypers in rn_inj.reshape((Np, 2))]).flatten())
gwb_phi_inj = jnp.kron(alpha, jnp.diag(get_rho_diag(gwb_inj)))
phi_inj = jnp.zeros((Na_PTA, Na_PTA))
if model_rn:
    phi_inj = phi_inj.at[:, :].add(rn_phi_inj)
if model_gwb:
    phi_inj = phi_inj.at[:, :].add(gwb_phi_inj)

# injected coefficients are drawn according to this covariance matrix
a_seed = 3
L_phi_inj = jnp.linalg.cholesky(phi_inj)
a_inj = L_phi_inj @ jr.normal(key=jr.key(a_seed), shape=(Na_PTA,))



###############################################################################################
####################################### CONTINUOUS WAVE #######################################
###############################################################################################

# model continuous wave
model_cw = True

# continuous wave parameter bounds
cw_mins = jnp.array([7., -10., -1., -jnp.pi / 2., -1., -1., 0., -jnp.pi / 2.])
cw_maxs = jnp.array([10., -7.2, 1., jnp.pi / 2., 2., 1., 2. * jnp.pi, jnp.pi / 2.])
psr_dist_mins = jnp.ones(Np) * psr_dist_min
psr_dist_maxs = jnp.ones(Np) * psr_dist_max
psr_phase_mins = jnp.zeros(Np)
psr_phase_maxs = jnp.ones(Np) * 2. * jnp.pi
cw_psr_mins = jnp.concatenate((cw_mins, psr_phase_mins, psr_dist_mins))
cw_psr_maxs = jnp.concatenate((cw_maxs, psr_phase_maxs, psr_dist_maxs))

# continous wave parameter labels
cw_labels = np.array([r'$\log_{10}(\mathcal{M}\,\,[M_\odot])$', r'$\log_{10}(f_{GW}\,\,[\text{Hz}])$',
                      r'$\cos{\iota}$', r'$\psi$', r'$\log_{10}(D_{L}\,\,[\text{Mpc}])$',
                      r'$\cos{\theta}$', r'$\phi$', r'$\Phi_0$'])
phase_labels = [rf'$\Phi$ [{i}]' for i in range(1, Np + 1)]
dist_labels = [f'L [{i}]' for i in range(1, Np + 1)]
cw_psr_labels = np.concatenate((cw_labels, phase_labels, dist_labels))

# coninuous wave parameters injected
gwtheta_inj = 2 * jnp.pi / 5
gwphi_inj = 7 * jnp.pi / 4.
mc_inj = 10.**8.8
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
cw_inj = jnp.array([log10_mc_inj, log10_fgw_inj, cosinc_inj, psi_inj, 
                    log10_dist_inj, costheta_inj, gwphi_inj, phase0_inj])

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
    pphase = (1 + 256/5 * (10**log10_mc*c.Tsun)**(5/3) * (jnp.pi * 10**log10_fgw)**(8/3)
            * psr_dist*c.Tkpc*(1-cosMu)) ** (5/8) - 1
    pphase /= 32 * (10**log10_mc*c.Tsun)**(5/3) * (jnp.pi * 10**log10_fgw)**(5/3)
    psr_phase = -pphase%(2*jnp.pi)

    return psr_phase

# injected pulsar phases
psr_phases_inj = jnp.array([get_psr_phase(cw_inj, psr_position, psr_dist)
                            for psr_position, psr_dist in zip(psr_pos, psr_dists_inj)])

# injected continuous wave and pulsar parameters in one array
cw_psr_inj = jnp.concatenate((cw_inj, psr_phases_inj, psr_dists_inj))

# number of continuous wave parameters
N_cw = 8
N_psr = 2 * Np
N_cw_psr = N_cw + N_psr

# sparse times used for FFT in CW model
sparse_toas_CW = jnp.array([jnp.linspace(toas[idx][0], toas[idx][-1], Na + 2, endpoint=False)
                            for idx in range(Np)])  # (Np, N_sparse)
Nsparse = sparse_toas_CW.shape[1]
freqs_forFFT = jnp.array([jnp.fft.fftfreq(Nsparse, Tspan / Nsparse)
                          for _ in range(Np)])

# antenna pattern for continuous waves
@jit
def create_gw_antenna_pattern(gwtheta, gwphi):
    '''
    Creates a continuous gravitational wave antenna pattern.

    param: gwtheta: theta coordinate for sky location of CW source
    param: gwphi: phi coordinate for sky location of CW source
    '''

    # use definition from Sesana et al 2010 and Ellis et al 2012
    sgwphi = jnp.sin(gwphi)
    cgwphi = jnp.cos(gwphi)
    sgwtheta = jnp.sin(gwtheta)
    cgwtheta = jnp.cos(gwtheta)

    mdotpos = sgwphi * psr_pos[:, 0] - cgwphi * psr_pos[:, 1]
    ndotpos = -cgwtheta * cgwphi * psr_pos[:, 0] - cgwtheta * sgwphi * psr_pos[:, 1] \
                + sgwtheta * psr_pos[:, 2]
    omhatdotpos = -sgwtheta * cgwphi * psr_pos[:, 0] - sgwtheta * sgwphi * psr_pos[:, 1] \
                    -cgwtheta * psr_pos[:, 2]

    fplus = 0.5 * (mdotpos ** 2 - ndotpos ** 2) / (1 + omhatdotpos)
    fcross = (mdotpos * ndotpos) / (1 + omhatdotpos)
    cosMu = -omhatdotpos

    return fplus, fcross, cosMu

# get signal due to continuous wave
@jit
def cw_delay(x_CW):
    '''
    Returns CW signal over sparse TOAs given CW parameters, pulsar parameters, and
    position of pulsar.
    
    params: x_CW: jax array of continuous wave parameters
    '''
    # unpack parameters
    log10_mc, log10_fgw, cos_inc, psi, log10_dist, cos_gwtheta, gwphi, phase0 = x_CW[:8]
    p_phases = x_CW[8 : 8 + Np]
    pdists = x_CW[8 + Np:]

    # convert units to time
    mc = 10 ** log10_mc * c.Tsun
    fgw = 10 ** log10_fgw
    # mc = jnp.power(10., log10_mc) * Tsun
    # fgw = jnp.power(10., log10_fgw)
    gwtheta = jnp.arccos(cos_gwtheta)
    inc = jnp.arccos(cos_inc)
    p_dists = pdists * c.kpc / c.c
    # dist = jnp.power(10., log10_dist) * Mpc / c
    dist = 10 ** log10_dist * c.Mpc / c.c

    # get antenna pattern funcs and cosMu
    # write function to get pos from theta,phi
    fplus, fcross, cosMu = create_gw_antenna_pattern(gwtheta, gwphi)

    # get pulsar time
    # toas_copy = jnp.copy(toas_input)
    # toas_copy -= tref
    toas_copy = sparse_toas_CW - tref
    tp = toas_copy - (p_dists*(1-cosMu))[:, None]

    # orbital frequency
    w0 = jnp.pi * fgw
    phase0 = phase0 / 2.0  # convert GW to orbital phase

    # calculate time dependent frequency at earth and pulsar
    mc53 = mc**(5./3.)
    w083 = w0**(8./3.)
    fac1 = 256./5. * mc53 * w083
    omega = w0 * (1. - fac1 * toas_copy)**(-3./8.)
    omega_p = w0 * (1. - fac1 * tp)**(-3./8.)
    omega_p0 = (w0 * (1. + fac1 * p_dists*(1-cosMu))**(-3./8.))[:, None]

    # calculate time dependent phase
    phase = phase0 + 1./32./mc53 * (w0**(-5./3.) - omega**(-5./3.))

    phase_p = (phase0 + p_phases[:, None]
                + 1./32./mc53 * (omega_p0**(-5./3.) - omega_p**(-5./3.)))

    # define time dependent coefficients
    inc_factor = -0.5 * (3. + jnp.cos(2. * inc))
    # At = -0.5*np.sin(2*phase)*(3+np.cos(2*inc))
    At = jnp.sin(2. * phase) * inc_factor
    Bt = 2. * jnp.cos(2. * phase) * cos_inc
    # At_p = -0.5*np.sin(2*phase_p)*(3+np.cos(2*inc))
    At_p = jnp.sin(2. * phase_p) * inc_factor
    Bt_p = 2. * jnp.cos(2. * phase_p) * cos_inc

    # now define time dependent amplitudes
    alpha = mc**(5./3.)/(dist*omega**(1./3.))
    alpha_p = mc**(5./3.)/(dist*omega_p**(1./3.))

    # define rplus and rcross
    c2psi = jnp.cos(2. * psi)
    s2psi = jnp.sin(2. * psi)
    rplus = alpha*(-At*c2psi+Bt*s2psi)
    rcross = alpha*(At*s2psi+Bt*c2psi)
    rplus_p = alpha_p*(-At_p*c2psi+Bt_p*s2psi)
    rcross_p = alpha_p*(At_p*s2psi+Bt_p*c2psi)

    # residuals
    res = fplus[:, None] * (rplus_p - rplus) + fcross[:, None] * (rcross_p - rcross)
    return res  # (Np, Nsparse)


# get Fourier coefficients for CW signal in all pulsars
@jit
def get_CW_coefficients(x_CW):
    '''
    Use FFT to get Fourier coefficients for CW residuals given CW parameters.

    params: x_CW: continuous wave parameters
    '''
    cw_residuals = cw_delay(x_CW)
    cw_fft = jnp.fft.fft(cw_residuals, n=None, axis=-1, norm=None)  # dim (Np, 2 * Nf + 2)
    # apply time shift to fft to set initial time
    cw_fft *= jnp.exp(-1.j * 2 * jnp.pi * freqs_forFFT * sparse_toas_CW[:, 0:1])
    
    # extract sine and cosine coefficients
    a_n = jnp.imag(cw_fft[:, :Nsparse // 2]) * (-2 / Nsparse)  # (Np, Nf + 1)
    b_n = jnp.real(cw_fft[:, :Nsparse // 2]) * (2 / Nsparse)  # (Np, Nf + 1)
    coeff = jnp.concatenate((a_n, b_n), axis=1).reshape((Np, 2, Nf + 1))\
                            .transpose((0, 2, 1)).reshape((Np, Na + 2))
    return coeff[:, 2:]  # remove DC



###############################################################################################
##################################### TIMING MODEL ############################################
###############################################################################################

# timing design matrix
Ms = jnp.array([jnp.vstack([jnp.ones(Ntoas),
                            toas[i],
                            toas[i]**2]).T
                for i in range(Np)])

# projection orthogonal to space of timing model
Rs = jnp.array([jnp.eye(Ntoas) - M @ jnp.linalg.inv(M.T @ M) @ M.T
                for M in Ms])

# white noise covariance matrix marginalized over timing model parameters
U_s = jnp.array([jnp.linalg.svd(M)[0] for M in Ms])
Gs = jnp.array([U[:, 3:] for U in U_s])
Ntinvs = jnp.array([G @ jnp.linalg.inv(G.T @ N @ G) @ G.T for G, N in zip(Gs, Ns)])



###############################################################################################
################################## PARAMETER ORDERING #########################################
###############################################################################################

# combine injected parameters and labels into arrays
last_ndx = 0
x_inj = []
x_mins = []
x_maxs = []
x_labels = []
if model_wn:
    x_inj.extend(list(efacs_inj))
    efac_mins = jnp.array([efac_min] * N_efac)
    efac_maxs = jnp.array([efac_max] * N_efac)
    x_mins.extend(list(efac_mins))
    x_maxs.extend(list(efac_maxs))
    x_labels.extend(list(efac_labels))
    efac_ndxs = jnp.r_[last_ndx : last_ndx + N_efac]
    last_ndx = efac_ndxs[-1] + 1
if model_rn:
    x_inj.extend(list(rn_inj))
    x_mins.extend(list(rn_mins))
    x_maxs.extend(list(rn_maxs))
    x_labels.extend(list(rn_labels))
    rn_ndxs = jnp.r_[last_ndx : last_ndx + N_rn]
    last_ndx = rn_ndxs[-1] + 1
if model_gwb:
    x_inj.extend(list(gwb_inj))
    x_mins.extend(list(gwb_mins))
    x_maxs.extend(list(gwb_maxs))
    x_labels.extend(list(gwb_labels))
    gwb_ndxs = jnp.r_[last_ndx : last_ndx + N_gwb]
    last_ndx = gwb_ndxs[-1] + 1
if model_cw:
    x_inj.extend(list(cw_psr_inj))
    x_mins.extend(list(cw_psr_mins))
    x_maxs.extend(list(cw_psr_maxs))
    x_labels.extend(list(cw_psr_labels))
    cw_psr_ndxs = jnp.r_[last_ndx : last_ndx + N_cw_psr]
    last_ndx = cw_psr_ndxs[-1] + 1
    cw_ndxs = cw_psr_ndxs[:N_cw]
    psr_phase_ndxs = cw_psr_ndxs[N_cw : N_cw + Np]
    psr_dist_ndxs = cw_psr_ndxs[-Np:]
if model_rn or model_gwb:
    x_inj.extend(list(a_inj))
    a_mins = jnp.array([a_min] * Na_PTA)
    a_maxs = jnp.array([a_max] * Na_PTA)
    x_mins.extend(list(a_mins))
    x_maxs.extend(list(a_maxs))
    x_labels.extend(list(a_labels))
    a_ndxs = jnp.r_[last_ndx : last_ndx + Na_PTA]
    last_ndx = a_ndxs[-1] + 1

# convert to jax or numpy arrays
x_inj = jnp.array(x_inj)
x_mins = jnp.array(x_mins)
x_maxs = jnp.array(x_maxs)
x_labels = np.array(x_labels)
ndim = x_inj.shape[0]

# check everything is lined up
assert ndim == last_ndx, 'model dimension mis-match'


