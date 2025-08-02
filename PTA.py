'''Detector characteristics, model, and parameter attributes used throughout analysis.'''


import numpy as np
from functools import partial

from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr

# use double precision
from jax import config
config.update('jax_enable_x64', True)

import constants as c


'''Store detector characteristics, model specifications, and simulation methods.'''
class PTA:

    def __init__(self,
                 Np,
                 Tspan_yr,
                 Nf,
                 model_wn=True,
                 model_rn=True,
                 model_gwb=True,
                 model_cw=True,
                 gwb_free_spectral=False,
                 efacs_inj=None,
                 rn_inj=None,
                 gwb_power_law_inj=None,
                 cw_inj=None,
                 tref=1.e9,
                 seed=0):
        

        # random keys used to draw model parameters and detector characteristics
        self.seed = seed
        self.simulation_keys = jr.split(key=jr.key(self.seed), num=8)

        ###############################################################################################
        ######################################### PULSARS #############################################
        ###############################################################################################

        # number of pulsars
        self.Np = Np

        # pulsar distance (kpc)
        self.psr_dist_min = 0.1
        self.psr_dist_max = 6.0
        self.psr_dists_inj = jr.uniform(key=self.simulation_keys[0],
                                        shape=(self.Np,),
                                        minval=self.psr_dist_min,
                                        maxval=self.psr_dist_max)

        # set pulsar distance uncertainty to fixed value
        self.psr_dists_stdev = jnp.array([0.2] * Np)

        # pulsar positions
        phis = jr.uniform(key=self.simulation_keys[1],
                          shape=(self.Np,),
                          minval=0,
                          maxval=2*jnp.pi)
        cos_thetas = jr.uniform(key=self.simulation_keys[2],
                                shape=(self.Np,),
                                minval=-1.,
                                maxval=1.)
        thetas = jnp.arccos(cos_thetas)
        xs = jnp.sin(thetas) * jnp.cos(phis)
        ys = jnp.sin(thetas) * jnp.sin(phis)
        zs = jnp.cos(thetas)
        psr_pos_not_normalized = jnp.array([xs, ys, zs]).T

        # normalize pulsar positions
        normalization = jnp.sqrt(jnp.sum(psr_pos_not_normalized**2., axis=1)[:, None])
        self.psr_pos = psr_pos_not_normalized / normalization



        ###############################################################################################
        ##################################### TOA OBSERVATIONS ########################################
        ###############################################################################################

        # span of observations
        self.Tspan_yr = Tspan_yr
        self.Tspan = self.Tspan_yr * c.year_sec

        # pulsar TOA uncertainty
        self.psr_uncertainty_us = 0.5
        self.psr_uncertainty_s = self.psr_uncertainty_us * c.us_sec

        # reference time to start observations
        self.tref = tref

        # observe TOAs ~monthly
        self.Ntoas = int(self.Tspan_yr * c.year_months)
        self.toas_no_offset = jnp.array([jnp.linspace(self.tref, self.tref + self.Tspan,
                                                      self.Ntoas, endpoint=True)
                                         for _ in range(self.Np)])

        # offset TOA observations by ~couple days so not evenly spaced
        self.day_offset = 2.
        self.toa_offsets = jr.normal(self.simulation_keys[3],
                                     (self.Np, self.Ntoas)) * self.day_offset * c.day_sec

        # don't offset first and last TOA to preserve Tspan
        self.toa_offsets = self.toa_offsets.at[:, jnp.r_[0, -1]].set(jnp.zeros((self.Np, 2)))

        # make TOAs
        self.toas = jnp.array([jnp.linspace(self.tref, self.tref + self.Tspan,
                                            self.Ntoas, endpoint=True)
                               for _ in range(self.Np)]) + self.toa_offsets
        self.MJDs = self.toas / c.day_sec

        # frequency bins
        self.Nf = Nf
        self.Na = 2 * self.Nf
        self.freqs = jnp.arange(1, self.Nf + 1) / self.Tspan

        # total Fourier coefficients in entire PTA
        self.Na_PTA = self.Na * self.Np

        # Fourier design matrix
        self.Fs = jnp.zeros((self.Np, self.Ntoas, self.Na))
        for i in range(self.Np):
            for j in range(self.Nf):
                self.Fs = self.Fs.at[i, :, 2 * j].set(jnp.sin(2. * jnp.pi * \
                                                              self.freqs[j] * self.toas[i]))
                self.Fs = self.Fs.at[i, :, 2 * j + 1].set(jnp.cos(2. * jnp.pi * \
                                                                  self.freqs[j] * self.toas[i]))



        ###############################################################################################
        ######################################## WHITE NOISE ##########################################
        ###############################################################################################

        # # model white noise
        self.model_wn = model_wn
        # if self.model_wn:

        # EFAC parameter bounds
        self.efac_min = 0.5
        self.efac_max = 1.5

        # injected EFACs in each pulsar
        self.efacs_inj = efacs_inj
        if self.efacs_inj is None:
            # if not specified draw from uniform distribution
            self.efacs_inj = jr.uniform(key=self.simulation_keys[4],
                                        shape=(self.Np,),
                                        minval=self.efac_min,
                                        maxval=self.efac_max)

        # EFAC parameter labels
        self.efac_labels = np.array([rf'$EFAC_{{{i}}}$' for i in range(1, self.Np + 1)])

        # number of EFAC parameters
        self.N_efac = self.Np

        # white noise covariance matrix
        self.Ns = jnp.array([jnp.eye(self.Ntoas) * self.psr_uncertainty_s**2.
                             for _ in range(self.Np)])



        ###############################################################################################
        #################################### INTRINSIC RED NOISE ######################################
        ###############################################################################################

        # model intrinsic pulsar red noise
        self.model_rn = model_rn
        if self.model_rn:

            # intrinsic pulsar red noise parameter bounds for sampling
            self.rn_log_amp_min = -20.0
            self.rn_log_amp_max = -10.0
            self.rn_gamma_min = 0.1
            self.rn_gamma_max = 10.
            self.rn_mins = jnp.array([self.rn_log_amp_min, self.rn_gamma_min] * self.Np)
            self.rn_maxs = jnp.array([self.rn_log_amp_max, self.rn_gamma_max] * self.Np)
            # intrinsic pulsar red noise parameter bounds for injection
            self.rn_log_amp_min_inj = -15.0
            self.rn_log_amp_max_inj = -14.0
            self.rn_gamma_min_inj = 2.
            self.rn_gamma_max_inj = 7.
            self.rn_mins_inj = jnp.array([self.rn_log_amp_min_inj, self.rn_gamma_min_inj] * self.Np)
            self.rn_maxs_inj = jnp.array([self.rn_log_amp_max_inj, self.rn_gamma_max_inj] * self.Np)

            # intrinsic red noise parameters for each pulsar
            self.rn_inj = rn_inj
            if self.rn_inj is None:
                # draw from uniform distribution if not specified
                self.rn_inj = jr.uniform(key=self.simulation_keys[5],
                                        shape=(2 * self.Np,),
                                        minval=self.rn_mins_inj,
                                        maxval=self.rn_maxs_inj)

            # intrinsic pulsar red noise parameter labels
            self.rn_labels = np.array([rf'$\log_{{{10}}}\,A_{{{i // 2}}}$' if i % 2 == 0 else \
                                    rf'$\gamma_{{{i // 2}}}$'
                                    for i in range(2, 2 * self.Np + 2)])

            # number of red noise parameters
            self.N_rn = 2 * self.Np



        ###############################################################################################
        ################################ GRAVITATIONAL WAVE BACKGROUND ################################
        ###############################################################################################

        # model gravitational wave background
        self.model_gwb = model_gwb
        if self.model_gwb:

            # injected power law parameters
            self.gwb_power_law_inj = gwb_power_law_inj

            # free spectral or power law hyper-model
            self.gwb_free_spectral = gwb_free_spectral

            if self.gwb_free_spectral:

                # gravitational wave background power law parameter bounds
                self.gwb_log_rho_min = -20.
                self.gwb_log_rho_max = -8.
                self.gwb_mins = jnp.array([self.gwb_log_rho_min] * self.Nf)
                self.gwb_maxs = jnp.array([self.gwb_log_rho_max] * self.Nf)

                # injected GWB parameters
                self.gwb_power_law_inj = gwb_power_law_inj
                if self.gwb_power_law_inj is None:
                    self.gwb_power_law_inj = jnp.array([-14., 13. / 3.])
                
                # GWB injection defined below in Fourier coefficient section
                # self.gwb_inj = jnp.log10(self.get_rho_diag(self.gwb_power_law_inj)[::2])

                # intrinsic pulsar red noise parameter labels
                self.gwb_labels = np.array([rf'$\log_{{{10}}}\,\rho^{{{i}}}_B$'
                                            for i in range(1, self.Nf + 1)])

                # number of gravitational wave background parameters
                self.N_gwb = self.Nf
            
            else:

                # gravitational wave background free spectral parameter bounds
                self.gwb_log_amp_min = -17.
                self.gwb_log_amp_max = -12.
                self.gwb_gamma_min = 2.
                self.gwb_gamma_max = 7.
                self.gwb_mins = jnp.array([self.gwb_log_amp_min, self.gwb_gamma_min])
                self.gwb_maxs = jnp.array([self.gwb_log_amp_max, self.gwb_gamma_max])

                # injected GWB parameters
                self.gwb_inj = self.gwb_power_law_inj
                if self.gwb_inj is None:
                    self.gwb_inj = jnp.array([-14., 13. / 3.])
                    self.gwb_power_law_inj = self.gwb_inj

                # intrinsic pulsar red noise parameter labels
                self.gwb_labels = np.array([r'$\log_{{{10}}}\,A_B$', r'$\gamma_B$'])

                # number of gravitational wave background parameters
                self.N_gwb = 2

        # angles between pulsars
        self.angles = np.zeros((self.Np, self.Np))
        for i in range(self.Np):
            for j in range(i, self.Np):
                pos1 = self.psr_pos[i]
                pos2 = self.psr_pos[j]
                self.angles[i,j] = self.angles[j,i] = np.arccos(np.clip(np.dot(pos1, pos2), -1.0, 1.0))
        self.angles = jnp.array(self.angles)

        # Hellings-Downs weighting
        self.alpha = np.zeros((self.Np, self.Np))
        for i in range(self.Np):
            for j in range(self.Np):
                if i == j:
                    self.alpha[i,j] = 1.
                else:
                    ang = self.angles[i,j]
                    beta = (1. - np.cos(ang)) / 2.
                    self.alpha[i,j] = 1.5 * beta * np.log(beta) - 0.25 * beta + 0.5
        self.alpha = jnp.array(self.alpha)
        self.alpha_inv = jnp.linalg.inv(self.alpha)


        ###############################################################################################
        #################################### FOURIER COEFFICIENTS #####################################
        ###############################################################################################

        if self.model_rn or self.model_gwb:

            # bounds on Fourier coefficients
            self.a_min = -100_000.
            self.a_max = 100_000.

            # Fourier coefficient labels
            self.a_labels = np.array([[rf'$a^{{{j // 2}}}_{{{i}}}$' if j % 2 == 0 else \
                                    rf'$b^{{{j // 2}}}_{{{i}}}$'
                                    for j in range(2, self.Na + 2)]
                                    for i in range(1,self. Np + 1)]).flatten()


            # constants used in diagonal of covariance matrix for Fourier coefficients with power-law
            self.rho_scale = (c.year_sec ** 3.) / (12. * (jnp.pi ** 2.) * self.Tspan)
            self.rho_scale1 = self.Tspan / c.year_sec
            self.arr = jnp.repeat(jnp.arange(1, self.Nf + 1), 2)
            self.arr /= self.rho_scale1
            self.arr = jnp.array(self.arr)

            # vectorized diagonal of covariance matrix for Fourier coefficients
            self.vectorized_get_rho_diag = jit(vmap(self.get_rho_diag))

            # get covariance matrix of coefficients for RN + GWB
            phi_inj_flat = jnp.zeros((self.Na_PTA, self.Na_PTA))
            if self.model_rn:
                rn_rho_inj_diags = self.vectorized_get_rho_diag(self.rn_inj.reshape((self.Np, 2)))
                rn_phi_inj_flat = jnp.diag(rn_rho_inj_diags.flatten())
                phi_inj_flat = phi_inj_flat.at[:, :].add(rn_phi_inj_flat)
            if self.model_gwb:
                if self.gwb_free_spectral:
                    self.gwb_inj = jnp.log10(self.get_rho_diag(self.gwb_power_law_inj)[::2])
                else:
                    self.gwb_inj = self.gwb_power_law_inj
                gwb_rho_inj_diag = self.get_rho_diag(self.gwb_power_law_inj)
                gwb_phi_inj_flat = jnp.kron(self.alpha, jnp.diag(gwb_rho_inj_diag))
                phi_inj_flat = phi_inj_flat.at[:, :].add(gwb_phi_inj_flat)

            # injected coefficients are drawn according to this covariance matrix
            L_phi_inj = jnp.linalg.cholesky(phi_inj_flat)
            self.a_inj = L_phi_inj @ jr.normal(key=self.simulation_keys[6], shape=(self.Na_PTA,))



        ###############################################################################################
        ####################################### CONTINUOUS WAVE #######################################
        ###############################################################################################

        # model continuous wave
        self.model_cw = model_cw
        if self.model_cw:

            # continuous wave parameter bounds
            self.cw_mins = jnp.array([7., -10., -1., -jnp.pi / 2., -1., -1., 0., -jnp.pi / 2.])
            self.cw_maxs = jnp.array([10., -7.2, 1., jnp.pi / 2., 2., 1., 2. * jnp.pi, jnp.pi / 2.])
            self.psr_dist_mins = jnp.ones(self.Np) * self.psr_dist_min
            self.psr_dist_maxs = jnp.ones(self.Np) * self.psr_dist_max
            self.psr_phase_mins = jnp.zeros(self.Np)
            self.psr_phase_maxs = jnp.ones(self.Np) * 2. * jnp.pi
            self.cw_psr_mins = jnp.concatenate((self.cw_mins, self.psr_phase_mins, self.psr_dist_mins))
            self.cw_psr_maxs = jnp.concatenate((self.cw_maxs, self.psr_phase_maxs, self.psr_dist_maxs))

            # continous wave parameter labels
            self.cw_labels = np.array([r'$\log_{10}(\mathcal{M}\,\,[M_\odot])$',
                                    r'$\log_{10}(f_{GW}\,\,[\text{Hz}])$',
                                    r'$\cos{\iota}$', r'$\psi$', r'$\log_{10}(D_{L}\,\,[\text{Mpc}])$',
                                    r'$\cos{\theta}$', r'$\phi$', r'$\Phi_0$'])
            self.phase_labels = [rf'$\Phi$ [{i}]' for i in range(1, self.Np + 1)]
            self.dist_labels = [f'L [{i}]' for i in range(1, self.Np + 1)]
            self.cw_psr_labels = np.concatenate((self.cw_labels, self.phase_labels, self.dist_labels))    
            
            # injected CW parameters
            self.cw_inj = cw_inj
            if self.cw_inj is None:
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
                self.cw_inj = jnp.array([log10_mc_inj, log10_fgw_inj, cosinc_inj, psi_inj, 
                                        log10_dist_inj, costheta_inj, gwphi_inj, phase0_inj])

            # injected pulsar phases
            self.psr_phases_inj = jnp.array([self.get_psr_phase(self.cw_inj, psr_position, psr_dist)
                                            for psr_position, psr_dist in zip(self.psr_pos,
                                                                              self.psr_dists_inj)])

            # injected continuous wave and pulsar parameters in one array
            self.cw_psr_inj = jnp.concatenate((self.cw_inj, self.psr_phases_inj, self.psr_dists_inj))

            # number of continuous wave parameters
            self.N_cw = self.cw_inj.shape[0]
            self.N_psr = 2 * self.Np
            self.N_cw_psr = self.N_cw + self.N_psr

            # sparse times used for FFT in CW model
            self.sparse_toas_CW = jnp.array([jnp.linspace(self.toas[idx][0], self.toas[idx][-1],
                                                          self.Na + 2, endpoint=False)
                                        for idx in range(self.Np)])  # (Np, N_sparse)
            self.Nsparse = self.sparse_toas_CW.shape[1]
            self.freqs_forFFT = jnp.array([jnp.fft.fftfreq(self.Nsparse, self.Tspan / self.Nsparse)
                                           for _ in range(self.Np)])




        ###############################################################################################
        ##################################### TIMING MODEL ############################################
        ###############################################################################################

        # timing design matrix
        self.Ms = jnp.array([jnp.vstack([jnp.ones(self.Ntoas),
                                         self.toas[i],
                                         self.toas[i]**2.]).T
                             for i in range(self.Np)])

        # projection orthogonal to space of timing model
        self.Rs = jnp.array([jnp.eye(self.Ntoas) - M @ jnp.linalg.inv(M.T @ M) @ M.T
                             for M in self.Ms])

        # white noise covariance matrix marginalized over timing model parameters
        U_s = jnp.array([jnp.linalg.svd(M)[0] for M in self.Ms])
        self.Gs = jnp.array([U[:, 3:] for U in U_s])
        self.Ntinvs = jnp.array([G @ jnp.linalg.inv(G.T @ N @ G) @ G.T
                                 for G, N in zip(self.Gs, self.Ns)])



        ###############################################################################################
        ################################## PARAMETER ORDERING #########################################
        ###############################################################################################

        # combine injected parameters and labels into arrays
        last_ndx = 0
        x_inj = []
        x_mins = []
        x_maxs = []
        x_labels = []
        if self.model_wn:
            x_inj.extend(list(self.efacs_inj))
            self.efac_mins = jnp.array([self.efac_min] * self.N_efac)
            self.efac_maxs = jnp.array([self.efac_max] * self.N_efac)
            x_mins.extend(list(self.efac_mins))
            x_maxs.extend(list(self.efac_maxs))
            x_labels.extend(list(self.efac_labels))
            self.efac_ndxs = jnp.r_[last_ndx : last_ndx + self.N_efac]
            last_ndx = self.efac_ndxs[-1] + 1
        if self.model_rn:
            x_inj.extend(list(self.rn_inj))
            x_mins.extend(list(self.rn_mins))
            x_maxs.extend(list(self.rn_maxs))
            x_labels.extend(list(self.rn_labels))
            self.rn_ndxs = jnp.r_[last_ndx : last_ndx + self.N_rn]
            last_ndx = self.rn_ndxs[-1] + 1
        if self.model_gwb:
            x_inj.extend(list(self.gwb_inj))
            x_mins.extend(list(self.gwb_mins))
            x_maxs.extend(list(self.gwb_maxs))
            x_labels.extend(list(self.gwb_labels))
            self.gwb_ndxs = jnp.r_[last_ndx : last_ndx + self.N_gwb]
            last_ndx = self.gwb_ndxs[-1] + 1
        if self.model_cw:
            x_inj.extend(list(self.cw_psr_inj))
            x_mins.extend(list(self.cw_psr_mins))
            x_maxs.extend(list(self.cw_psr_maxs))
            x_labels.extend(list(self.cw_psr_labels))
            self.cw_psr_ndxs = jnp.r_[last_ndx : last_ndx + self.N_cw_psr]
            last_ndx = self.cw_psr_ndxs[-1] + 1
            self.cw_ndxs = self.cw_psr_ndxs[:self.N_cw]
            self.psr_phase_ndxs = self.cw_psr_ndxs[self.N_cw : self.N_cw + self.Np]
            self.psr_dist_ndxs = self.cw_psr_ndxs[-self.Np:]
        if self.model_rn or self.model_gwb:
            x_inj.extend(list(self.a_inj))
            a_mins = jnp.array([self.a_min] * self.Na_PTA)
            a_maxs = jnp.array([self.a_max] * self.Na_PTA)
            x_mins.extend(list(a_mins))
            x_maxs.extend(list(a_maxs))
            x_labels.extend(list(self.a_labels))
            self.a_ndxs = jnp.r_[last_ndx : last_ndx + self.Na_PTA]
            last_ndx = self.a_ndxs[-1] + 1

        # convert to jax or numpy arrays
        self.x_inj = jnp.array(x_inj)
        self.x_mins = jnp.array(x_mins)
        self.x_maxs = jnp.array(x_maxs)
        self.x_labels = np.array(x_labels)
        self.ndim = self.x_inj.shape[0]

        # check everything is lined up
        assert self.ndim == last_ndx, 'model dimension mis-match'

        ###############################################################################################
        ################################## SIMULATE RESIDUALS #########################################
        ###############################################################################################
        
        # (projected into space orthogonal to timing model)
        self.residuals = self.sim_residuals()



    ###############################################################################################
    ############################ MODEL AND SIMULATION METHODS #####################################
    ###############################################################################################


    # covariance matrix for Fourier coefficients under power law
    @partial(jit, static_argnums=(0,))
    def get_rho_diag(self, hyper_params):
        logAmp, gamma = hyper_params
        Amp = 10. ** logAmp
        return (Amp ** 2.) * self.rho_scale * (self.arr **  (-gamma))
    

    # compute injected pulsar phases from other CW / pulsar parameters
    @partial(jit, static_argnums=(0,))
    def get_psr_phase(self, cw_params, psr_position, psr_dist):

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
    
    # antenna pattern for continuous waves
    @partial(jit, static_argnums=(0,))
    def create_gw_antenna_pattern(self, gwtheta, gwphi):
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

        mdotpos = sgwphi * self.psr_pos[:, 0] - cgwphi * self.psr_pos[:, 1]
        ndotpos = -cgwtheta * cgwphi * self.psr_pos[:, 0] - cgwtheta * sgwphi * self.psr_pos[:, 1] \
                    + sgwtheta * self.psr_pos[:, 2]
        omhatdotpos = -sgwtheta * cgwphi * self.psr_pos[:, 0] - sgwtheta * sgwphi * self.psr_pos[:, 1] \
                        -cgwtheta * self.psr_pos[:, 2]

        fplus = 0.5 * (mdotpos ** 2 - ndotpos ** 2) / (1 + omhatdotpos)
        fcross = (mdotpos * ndotpos) / (1 + omhatdotpos)
        cosMu = -omhatdotpos

        return fplus, fcross, cosMu

    # get signal due to continuous wave
    @partial(jit, static_argnums=(0,))
    def cw_delay(self, x_CW):
        '''
        Returns CW signal over sparse TOAs given CW parameters, pulsar parameters, and
        position of pulsar.
        
        params: x_CW: jax array of continuous wave parameters
        '''
        # unpack parameters
        log10_mc, log10_fgw, cos_inc, psi, log10_dist, cos_gwtheta, gwphi, phase0 = x_CW[:self.N_cw]
        p_phases = x_CW[self.N_cw : self.N_cw + self.Np]
        pdists = x_CW[self.N_cw + self.Np:]

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
        fplus, fcross, cosMu = self.create_gw_antenna_pattern(gwtheta, gwphi)

        # get pulsar time
        # toas_copy = jnp.copy(toas_input)
        # toas_copy -= tref
        toas_copy = self.sparse_toas_CW - self.tref
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
    @partial(jit, static_argnums=(0,))
    def get_CW_coefficients(self, x_CW):
        '''
        Use FFT to get Fourier coefficients for CW residuals given CW parameters.

        params: x_CW: continuous wave parameters
        '''
        cw_residuals = self.cw_delay(x_CW)
        cw_fft = jnp.fft.fft(cw_residuals, n=None, axis=-1, norm=None)  # dim (Np, 2 * Nf + 2)
        # apply time shift to fft to set initial time
        cw_fft *= jnp.exp(-1.j * 2 * jnp.pi * self.freqs_forFFT * self.sparse_toas_CW[:, 0:1])
        
        # extract sine and cosine coefficients
        a_n = jnp.imag(cw_fft[:, :self.Nsparse // 2]) * (-2 / self.Nsparse)  # (Np, Nf + 1)
        b_n = jnp.real(cw_fft[:, :self.Nsparse // 2]) * (2 / self.Nsparse)  # (Np, Nf + 1)
        coeff = jnp.concatenate((a_n, b_n), axis=1).reshape((self.Np, 2, self.Nf + 1))\
                                .transpose((0, 2, 1)).reshape((self.Np, self.Na + 2))
        return coeff[:, 2:]  # remove DC
    
    # simulate residuals
    @partial(jit, static_argnums=(0,))
    def sim_residuals(self):
        residuals = jnp.zeros((self.Np, self.Ntoas))

        # if self.model_wn:  # add white noise
        white_noise = jnp.array([jr.multivariate_normal(key=self.simulation_keys[7],
                                                        mean=jnp.zeros(self.Ntoas),
                                                        cov=N)
                                    for N in self.Ns])
        residuals = residuals.at[:, :].add(white_noise * self.efacs_inj[:, None])

        if self.model_rn or self.model_gwb:  # add red noise and/or gravitational wave background
            rn_gwb_residuals = jnp.matmul(self.Fs, self.a_inj.reshape((self.Np, self.Na))[..., None]).squeeze(-1)
            residuals = residuals.at[:, :].add(rn_gwb_residuals)

        if self.model_cw:  # add continuous wave
            a_cw_inj = self.get_CW_coefficients(self.x_inj[self.cw_psr_ndxs])
            cw_residuals = jnp.matmul(self.Fs, a_cw_inj[..., None]).squeeze(-1)
            residuals = residuals.at[:,:].add(cw_residuals)

        # fit timing model (quadratic)
        residuals_TM_fitted = jnp.array([R @ res for R, res in zip(self.Rs, residuals)])
        return residuals_TM_fitted




