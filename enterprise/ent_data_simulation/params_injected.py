# parameters injected into data, and model specification


import numpy as np


# number of pulsars
num_psrs = 5


# number of Fourier modes to model
num_modes = 5
num_coeff = 2 * num_modes


# white noise
wn_model = False
efac_seed = 112
efac_inj = 1.0


# GWB
gwb_model = False
free_spectral = False
GWB_seed = 333
GWB_logamp_inj = -13.6
GWB_gamma_inj = 13. / 3.
GWB_hypers_inj = np.array([GWB_logamp_inj, GWB_gamma_inj])


# intrinsic pulsar red noise
rn_model = True
RN_seed = 2266
# RN_logamps_inj = np.array([-13.2, -13.3, -13.1, -13.2, -13.1, -13.6, -13.5, -13.6, -13.8, -14.1,
#                             -12.5, -13.3, -13.1, -13.2, -14.1, -12.5, -13.2, -13.1, -13.8, -14.1])[:num_psrs]
RN_logamps_inj = np.array([-13.5, -13.7, -14.1, -13.8, -14., -13.6, -13.5, -13.6, -13.8, -14.1,
                            -12.5, -13.6, -13.1, -13.2, -14.1, -12.5, -13.2, -13.1, -13.8, -14.1])[:num_psrs]
RN_gammas_inj = np.array([4.6, 3.6, 4.5, 4.2, 4.0, 4.1, 3.8, 1.5, 5.6, 3.5,
                          2.1, 4.8, 1.5, 2.6, 2.5, 3.1, 3.8, 1.5, 5.6, 3.5])[:num_psrs]
RN_hypers_inj = np.zeros(2 * num_psrs)
RN_hypers_inj[::2] = RN_logamps_inj
RN_hypers_inj[1::2] = RN_gammas_inj
RN_hypers_inj = np.array(RN_hypers_inj)


# CW parameter attributes
cw_model = True
gwtheta_inj = np.pi / 2
gwphi_inj = 2.8 * np.pi / 2.
mc_inj = 10.**8.9
dist_inj = 1.0
fgw_inj = 4.e-9
phase0_inj = np.pi / 4.
psi_inj = np.pi / 3.
inc_inj = np.pi / 10.
# inc_inj = np.pi / 3.
log10_dist_inj = np.log10(dist_inj)
cosinc_inj = np.cos(inc_inj)
costheta_inj = np.cos(gwtheta_inj)
log10_mc_inj = np.log10(mc_inj)
log10_fgw_inj = np.log10(fgw_inj)
CW_params_inj = np.array([log10_mc_inj, log10_fgw_inj, cosinc_inj, psi_inj, 
                          log10_dist_inj, costheta_inj, gwphi_inj, phase0_inj])

tref = 1e9
