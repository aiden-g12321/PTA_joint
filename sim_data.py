'''Simulate PTA dataset.'''


import jax.numpy as jnp
import jax.random as jr
import models as m


# make residuals
residuals = jnp.zeros((m.Np, m.Ntoas))

if m.model_wn:  # add white noise
    white_noise = jnp.array([jr.multivariate_normal(key=jr.key(m.efac_seed + 1),
                                                    mean=jnp.zeros(m.Ntoas),
                                                    cov=N)
                             for N in m.Ns]) * m.efacs_inj[:, None]
    residuals = residuals.at[:, :].add(white_noise)

if m.model_rn or m.model_gwb:  # add red noise and/or gravitational wave background
    rn_gwb_residuals = jnp.matmul(m.Fs, m.a_inj.reshape((m.Np, m.Na))[..., None]).squeeze(-1)
    residuals = residuals.at[:, :].add(rn_gwb_residuals)

if m.model_cw:  # add continuous wave
    a_cw_inj = m.get_CW_coefficients(m.x_inj[m.cw_psr_ndxs])
    cw_residuals = jnp.matmul(m.Fs, a_cw_inj[..., None]).squeeze(-1)
    residuals = residuals.at[:,:].add(cw_residuals)

# fit timing model (quadratic)
residuals = jnp.array([R @ res for R, res in zip(m.Rs, residuals)])


