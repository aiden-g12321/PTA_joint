'''Likelihood and prior density functions.'''


from jax import jit, vmap
from jax.lax import cond
import jax.numpy as jnp
import jax.scipy as js
from functools import partial
from PTA import PTA



class Likelihood:

    def __init__(self,
                 toas,
                 residuals,
                 Ntinvs,
                 Fs,
                 Fs_cw,
                 Ntoas,
                 Np,
                 Na,
                 Na_cw,
                 get_rho_diag,
                 alpha,
                 psr_dists_inj,
                 psr_dists_stdev,
                 x_mins,
                 x_maxs):
        
        self.toas = toas
        self.residuals = residuals
        self.Ntinvs = Ntinvs
        self.Fs = Fs
        self.Fs_cw = Fs_cw
        self.Ntoas = Ntoas
        self.Np = Np
        self.Na = Na
        self.Na_cw = Na_cw
        self.get_rho_diag = jit(get_rho_diag)
        self.vectorized_get_rho_diag = jit(vmap(self.get_rho_diag))
        self.alpha = alpha
        self.psr_dists_inj = psr_dists_inj
        self.psr_dists_stdev = psr_dists_stdev
        self.x_mins = x_mins
        self.x_maxs = x_maxs

        # constants for likelihood
        self.Us = jnp.array([res.T @ Ntinv @ res for res, Ntinv in zip(self.residuals,
                                                                       self.Ntinvs)])
        self.Vs = jnp.array([F.T @ Ntinv @ res for F, Ntinv, res in zip(self.Fs,
                                                                        self.Ntinvs,
                                                                        self.residuals)])
        self.Ws = jnp.array([F.T @ Ntinv @ F for F, Ntinv in zip(self.Fs,
                                                                 self.Ntinvs)])
        self.Vs_cw = jnp.array([F.T @ Ntinv @ res for F, Ntinv, res in zip(self.Fs_cw,
                                                                        self.Ntinvs,
                                                                        self.residuals)])
        self.Ws_cw = jnp.array([F.T @ Ntinv @ F for F, Ntinv in zip(self.Fs_cw,
                                                                 self.Ntinvs)])
        self.Ws_tilde = jnp.array([F.T @ Ntinv @ F_cw for F, Ntinv, F_cw in zip(self.Fs,
                                                                 self.Ntinvs, self.Fs_cw)])

        # likelihood vectorized over pulsars
        self.vectorized_lnlike_per_psr = jit(vmap(self.lnlike_per_psr, in_axes=(0, 0, 0, 0, 0)))
        self.vectorized_lnlike_per_psr_cwbasis = jit(vmap(self.lnlike_per_psr_cwbasis,
                                                          in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0)))

    # likelihood per pulsar
    @partial(jit, static_argnums=(0,))
    def lnlike_per_psr(self, efac, a, U, V, W):
        return -0.5 * (U + a.T @ W @ a - 2 * jnp.inner(a, V)) / efac**2. - (self.Ntoas) * jnp.log(efac)

    # likelihood for all pulsars
    @partial(jit, static_argnums=(0,))
    def lnlike(self, efacs, a):
        a_stacked = a.reshape((self.Np, self.Na))
        return jnp.sum(self.vectorized_lnlike_per_psr(efacs, a_stacked, self.Us, self.Vs, self.Ws))
    
    # likelihood per pulsar when CW uses different basis
    @partial(jit, static_argnums=(0,))
    def lnlike_per_psr_cwbasis(self, efac, a_rn_gwb, a_cw, U, V, W, V_cw, W_cw, W_tilde):
        lnlike_val = U + a_rn_gwb.T @ W @ a_rn_gwb - 2 * jnp.inner(a_rn_gwb, V)
        lnlike_val += -2. * jnp.inner(a_cw, V_cw) + 2. * a_rn_gwb.T @ W_tilde @ a_cw + a_cw.T @ W_cw @ a_cw
        lnlike_val *= -0.5 / efac**2.
        lnlike_val +=  -(self.Ntoas) * jnp.log(efac)
        return lnlike_val
    
    # likelihood for all pulsars when CW uses different basis
    @partial(jit, static_argnums=(0,))
    def lnlike_cwbasis(self, efacs, a_rn_gwb, a_cw):
        a_rn_gwb_stacked = a_rn_gwb.reshape((self.Np, self.Na))
        a_cw_stacked = a_cw.reshape((self.Np, self.Na_cw))
        return jnp.sum(self.vectorized_lnlike_per_psr_cwbasis(efacs,
                                                              a_rn_gwb_stacked,
                                                              a_cw_stacked,
                                                              self.Us,
                                                              self.Vs,
                                                              self.Ws,
                                                              self.Vs_cw,
                                                              self.Ws_cw,
                                                              self.Ws_tilde))

    # prior on Fourier coefficients
    @partial(jit, static_argnums=(0,))
    def a_lnprior_rn(self, rn_hypers, a):
        # reshape parameters
        rn_hypers_stacked = rn_hypers.reshape((self.Np, 2))
        a_stacked = a.reshape((self.Np, self.Na))

        # covariance matrix for Fourier coefficients
        phi = jnp.zeros((self.Na, self.Np, self.Np))
        phi = phi.at[:, jnp.arange(self.Np), jnp.arange(self.Np)].\
              set(self.vectorized_get_rho_diag(rn_hypers_stacked).T)

        # prior contribution
        phi_chol_factors = vmap(lambda x: js.linalg.cho_factor(x, lower=True))(phi)
        phiinvs = vmap(lambda cf: js.linalg.cho_solve((cf[0], True),
                                                      jnp.identity(cf[0].shape[0])))(phi_chol_factors)
        philogdets = 2 * jnp.sum(jnp.log(jnp.diagonal(phi_chol_factors[0], axis1=1, axis2=2)), axis=1)
        ln_prior_val = -0.5 * jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a_stacked.T, phiinvs))
        ln_prior_val += -0.5 * jnp.sum(philogdets)
        return ln_prior_val
    

    @partial(jit, static_argnums=(0,))
    def a_lnprior_gwb(self, gwb_hypers, a):
        # reshape parameters
        a_stacked = a.reshape((self.Np, self.Na))

        # covariance matrix for Fourier coefficients
        phi = self.get_rho_diag(gwb_hypers)[:, None, None] * self.alpha[None, :, :]

        # prior contribution
        phi_chol_factors = vmap(lambda x: js.linalg.cho_factor(x, lower=True))(phi)
        phiinvs = vmap(lambda cf: js.linalg.cho_solve((cf[0], True),
                                                    jnp.identity(cf[0].shape[0])))(phi_chol_factors)
        philogdets = 2 * jnp.sum(jnp.log(jnp.diagonal(phi_chol_factors[0], axis1=1, axis2=2)), axis=1)
        ln_prior_val = -0.5 * jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a_stacked.T, phiinvs))
        ln_prior_val += -0.5 * jnp.sum(philogdets)
        return ln_prior_val
    

    @partial(jit, static_argnums=(0,))
    def a_lnprior_gwb_free_spectral(self, gwb_hypers, a):
        # reshape parameters
        a_stacked = a.reshape((self.Np, self.Na))

        # covariance matrix for Fourier coefficients
        phi = jnp.repeat(10. ** gwb_hypers, 2)[:, None, None] * self.alpha[None, :, :]

        # prior contribution
        phi_chol_factors = vmap(lambda x: js.linalg.cho_factor(x, lower=True))(phi)
        phiinvs = vmap(lambda cf: js.linalg.cho_solve((cf[0], True),
                                                    jnp.identity(cf[0].shape[0])))(phi_chol_factors)
        philogdets = 2 * jnp.sum(jnp.log(jnp.diagonal(phi_chol_factors[0], axis1=1, axis2=2)), axis=1)
        ln_prior_val = -0.5 * jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a_stacked.T, phiinvs))
        ln_prior_val += -0.5 * jnp.sum(philogdets)
        return ln_prior_val
    

    @partial(jit, static_argnums=(0,))
    def a_lnprior_rn_gwb(self, rn_hypers, gwb_hypers, a):
        # reshape parameters
        rn_hypers_stacked = rn_hypers.reshape((self.Np, 2))
        a_stacked = a.reshape((self.Np, self.Na))

        # covariance matrix for Fourier coefficients
        rn_phi = jnp.zeros((self.Na, self.Np, self.Np))
        rn_phi = rn_phi.at[:, jnp.arange(self.Np), jnp.arange(self.Np)].\
                        set(self.vectorized_get_rho_diag(rn_hypers_stacked).T)
        gwb_phi = self.get_rho_diag(gwb_hypers)[:, None, None] * self.alpha[None, :, :]
        phi = rn_phi + gwb_phi

        # prior contribution
        phi_chol_factors = vmap(lambda x: js.linalg.cho_factor(x, lower=True))(phi)
        phiinvs = vmap(lambda cf: js.linalg.cho_solve((cf[0], True),
                                                      jnp.identity(cf[0].shape[0])))(phi_chol_factors)
        philogdets = 2 * jnp.sum(jnp.log(jnp.diagonal(phi_chol_factors[0], axis1=1, axis2=2)), axis=1)
        ln_prior_val = -0.5 * jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a_stacked.T, phiinvs))
        ln_prior_val += -0.5 * jnp.sum(philogdets)
        return ln_prior_val
    
    @partial(jit, static_argnums=(0,))
    def a_lnprior_rn_gwb_free_spectral(self, rn_hypers, gwb_hypers, a):
        # reshape parameters
        rn_hypers_stacked = rn_hypers.reshape((self.Np, 2))
        a_stacked = a.reshape((self.Np, self.Na))

        # covariance matrix for Fourier coefficients
        rn_phi = jnp.zeros((self.Na, self.Np, self.Np))
        rn_phi = rn_phi.at[:, jnp.arange(self.Np), jnp.arange(self.Np)].\
                        set(self.vectorized_get_rho_diag(rn_hypers_stacked).T)
        gwb_phi = jnp.repeat(10. ** gwb_hypers, 2)[:, None, None] * self.alpha[None, :, :]
        phi = rn_phi + gwb_phi

        # prior contribution
        phi_chol_factors = vmap(lambda x: js.linalg.cho_factor(x, lower=True))(phi)
        phiinvs = vmap(lambda cf: js.linalg.cho_solve((cf[0], True),
                                                      jnp.identity(cf[0].shape[0])))(phi_chol_factors)
        philogdets = 2 * jnp.sum(jnp.log(jnp.diagonal(phi_chol_factors[0], axis1=1, axis2=2)), axis=1)
        ln_prior_val = -0.5 * jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a_stacked.T, phiinvs))
        ln_prior_val += -0.5 * jnp.sum(philogdets)
        return ln_prior_val


    # normal prior on pulsar distances
    @partial(jit, static_argnums=(0,))
    def psr_dist_lnprior(self, psr_distances):
        return jnp.sum(js.stats.norm.logpdf(x=psr_distances,
                                            loc=self.psr_dists_inj,
                                            scale=self.psr_dists_stdev))

    # uniform prior on all parameters
    @partial(jit, static_argnums=(0,))
    def uniform_lnprior(self, x):
        out_of_bounds = jnp.logical_or(jnp.any(x < self.x_mins),
                                    jnp.any(x > self.x_maxs))
        def out_of_bounds_case():
            return -jnp.inf
        def in_bounds_case():
            return 0.0
        return cond(out_of_bounds, out_of_bounds_case, in_bounds_case)
    


# make Likelihood object with PTA object
def get_likelihood_obj(pta_obj):
    l = Likelihood(pta_obj.toas,
                   pta_obj.residuals,
                   pta_obj.Ntinvs,
                   pta_obj.Fs,
                   pta_obj.Fs_cw,
                   pta_obj.Ntoas,
                   pta_obj.Np,
                   pta_obj.Na,
                   pta_obj.Na_cw,
                   pta_obj.get_rho_diag,
                   pta_obj.alpha,
                   pta_obj.psr_dists_inj,
                   pta_obj.psr_dists_stdev,
                   pta_obj.x_mins,
                   pta_obj.x_maxs)
    return l


