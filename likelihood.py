'''Likelihood and prior density functions.'''


from jax import jit, vmap
from jax.lax import cond
import jax.numpy as jnp
import jax.scipy as js
import models as m
from sim_data import residuals



###############################################################################################
####################################### LIKELIHOOD ############################################
###############################################################################################

# constants for likelihood
Us = jnp.array([res.T @ Ntinv @ res for res, Ntinv in zip(residuals, m.Ntinvs)])
Vs = jnp.array([F.T @ Ntinv @ res for F, Ntinv, res in zip(m.Fs, m.Ntinvs, residuals)])
Ws = jnp.array([F.T @ Ntinv @ F for F, Ntinv in zip(m.Fs, m.Ntinvs)])

# likelihood per pulsar
@jit
def lnlike_per_psr(efac, a, U, V, W):
    return -0.5 * (U + a.T @ W @ a - 2 * jnp.inner(a, V)) / efac**2. - m.Ntoas * jnp.log(efac)
vectorized_lnlike_per_psr = jit(vmap(lnlike_per_psr, in_axes=(0, 0, 0, 0, 0)))

# likelihood for all pulsars
@jit
def lnlike(efacs, a):
    a_stacked = a.reshape((m.Np, m.Na))
    return jnp.sum(vectorized_lnlike_per_psr(efacs, a_stacked, Us, Vs, Ws))



###############################################################################################
######################################### PRIORS ##############################################
###############################################################################################


# prior on Fourier coefficients
@jit
def a_lnprior_rn(rn_hypers, a):
    # reshape parameters
    rn_hypers_stacked = rn_hypers.reshape((m.Np, 2))
    a_stacked = a.reshape((m.Np, m.Na))

    # covariance matrix for Fourier coefficients
    phi = jnp.zeros((m.Na, m.Np, m.Np))
    phi = phi.at[:, jnp.arange(m.Np), jnp.arange(m.Np)].set(m.vectorized_get_rho_diag(rn_hypers_stacked).T)

    # prior contribution
    phi_chol_factors = vmap(lambda x: js.linalg.cho_factor(x, lower=True))(phi)
    phiinvs = vmap(lambda cf: js.linalg.cho_solve((cf[0], True), jnp.identity(cf[0].shape[0])))(phi_chol_factors)
    philogdets = 2 * jnp.sum(jnp.log(jnp.diagonal(phi_chol_factors[0], axis1=1, axis2=2)), axis=1)
    ln_prior_val = -0.5 * jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a_stacked.T, phiinvs))
    ln_prior_val += -0.5 * jnp.sum(philogdets)
    return ln_prior_val

@jit
def a_lnprior_gwb(gwb_hypers, a):
    # reshape parameters
    a_stacked = a.reshape((m.Np, m.Na))

    # covariance matrix for Fourier coefficients
    phi = m.get_rho_diag(gwb_hypers)[:, None, None] * m.alpha[None, :, :]

    # prior contribution
    phi_chol_factors = vmap(lambda x: js.linalg.cho_factor(x, lower=True))(phi)
    phiinvs = vmap(lambda cf: js.linalg.cho_solve((cf[0], True), jnp.identity(cf[0].shape[0])))(phi_chol_factors)
    philogdets = 2 * jnp.sum(jnp.log(jnp.diagonal(phi_chol_factors[0], axis1=1, axis2=2)), axis=1)
    ln_prior_val = -0.5 * jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a_stacked.T, phiinvs))
    ln_prior_val += -0.5 * jnp.sum(philogdets)
    return ln_prior_val

@jit
def a_lnprior_rn_gwb(rn_hypers, gwb_hypers, a):
    # reshape parameters
    rn_hypers_stacked = rn_hypers.reshape((m.Np, 2))
    a_stacked = a.reshape((m.Np, m.Na))

    # covariance matrix for Fourier coefficients
    rn_phi = jnp.zeros((m.Na, m.Np, m.Np))
    rn_phi = rn_phi.at[:, jnp.arange(m.Np), jnp.arange(m.Np)].set(m.vectorized_get_rho_diag(rn_hypers_stacked).T)
    gwb_phi = m.get_rho_diag(gwb_hypers)[:, None, None] * m.alpha[None, :, :]
    phi = rn_phi + gwb_phi

    # prior contribution
    phi_chol_factors = vmap(lambda x: js.linalg.cho_factor(x, lower=True))(phi)
    phiinvs = vmap(lambda cf: js.linalg.cho_solve((cf[0], True), jnp.identity(cf[0].shape[0])))(phi_chol_factors)
    philogdets = 2 * jnp.sum(jnp.log(jnp.diagonal(phi_chol_factors[0], axis1=1, axis2=2)), axis=1)
    ln_prior_val = -0.5 * jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a_stacked.T, phiinvs))
    ln_prior_val += -0.5 * jnp.sum(philogdets)
    return ln_prior_val


# normal prior on pulsar distances
@jit
def psr_dist_lnprior(psr_distances):
    return jnp.sum(js.stats.norm.logpdf(psr_distances, m.psr_dists_inj, m.psr_dists_stdev))

# uniform prior on all parameters
@jit
def uniform_lnprior(x):
    out_of_bounds = jnp.logical_or(jnp.any(x < m.x_mins),
                                   jnp.any(x > m.x_maxs))
    def out_of_bounds_case():
        return -jnp.inf
    def in_bounds_case():
        return 0.0
    return cond(out_of_bounds, out_of_bounds_case, in_bounds_case)


