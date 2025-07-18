'''Simulate data.'''


import numpy as np
import jax.numpy as jnp
import jax.random as jr
import params_inj as p
import cw_model as cw


# use double precision
from jax import config
config.update('jax_enable_x64', True)


# make residuals
a_cw_inj = cw.get_CW_coefficients(p.x_inj[p.cw_psr_ndxs])
residuals = jnp.matmul(p.Fs, a_cw_inj[..., None]).squeeze(-1)

# timing design matrix
Ms = jnp.array([jnp.vstack([jnp.ones(p.Ntoas),
                            p.toas[0],
                            p.toas[0]**2]).T
                for _ in range(p.Np)])

# projection orthogonal to space of timing model
Rs = jnp.array([jnp.eye(p.Ntoas) - M @ jnp.linalg.inv(M.T @ M) @ M.T
               for M in Ms])

# white noise covariance matrix
Ns = jnp.array([jnp.eye(p.Ntoas) * p.psr_uncertainty_s**2.
                for _ in range(p.Np)])

# white noise covariance matrix marginalized over timing model parameters
U_s = jnp.array([jnp.linalg.svd(M)[0] for M in Ms])
Gs = jnp.array([U[:, 3:] for U in U_s])
Ntinvs = jnp.array([G @ jnp.linalg.inv(G.T @ N @ G) @ G.T for G, N in zip(Gs, Ns)])

# add white noise to residuals
white_noise = jnp.array([jr.multivariate_normal(key=jr.key(p.efac_seed),
                                                mean=jnp.zeros(p.Ntoas),
                                                cov=N)
                         for N in Ns]) * p.efacs_inj[:, None]
residuals = residuals.at[:].add(white_noise)

# fit timing model
residuals = jnp.array([R @ res for R, res in zip(Rs, residuals)])

# save data in dictionary
data = {}
data['toas'] = p.toas
data['MJDs'] = p.MJDs
data['residuals'] = residuals
data['Fs'] = p.Fs
data['Rs'] = Rs
data['Ntinvs'] = Ntinvs

np.savez_compressed("data.npz", **data)

