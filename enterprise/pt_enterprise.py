import pickle
import shutil

import numpy as np

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc


# load Enterprise pulsar objects
with open('ent_data_simulation/data/enterprise_pulsars.pkl', 'rb') as f:
    psrs = pickle.load(f)

# load Enterprise PTA object
with open('ent_data_simulation/data/pta.pkl', 'rb') as f:
    pta = pickle.load(f)

for psr in psrs:
    print(psr.name)


# load injected parameters
data_dict = np.load('ent_data_simulation/data/data_dict.npz', allow_pickle=True)
x_inj_dict = data_dict['x_inj'].item()


print(pta.summary())


def get_dict(params):
    return {param_name: param for param_name, param in zip(pta.param_names, params)}

# likelihood function
def get_lnlike(params):
    return pta.get_lnlikelihood(get_dict(params))

# prior function
def get_lnprior(params):
    return pta.get_lnprior(get_dict(params))

# posterior
def get_lnpost(params):
    return get_lnprior(params) + get_lnlike(params)


# store injected parameters in array
x_inj = []
for param_name in pta.param_names:
    print(f'{param_name} = {x_inj_dict[param_name]}')
    if param_name[-6:] == 'p_dist':  # pulsar distances are normalized
        x_inj.append(0.)
    else:
        x_inj.append(x_inj_dict[param_name])
x_inj = np.array(x_inj)


# set up the sampler:
ndim = len(x_inj)
cov = np.eye(ndim) * 5.0
# cov[-4] = 0.0001
outDir = 'enterprise_chains'

sampler = ptmcmc(ndim=ndim,
                 logl=get_lnlike,
                 logp=get_lnprior,
                 cov=cov, 
                 outDir=outDir,
                 resume=False)

# do MCMC
num_samples = int(1e6)
sampler.sample(p0=x_inj,
               Niter=num_samples,
               # ladder=np.round(1.3**np.arange(4), 2),
               # writeHotChains=True,
               DEweight=20,
               SCAMweight=20,
               AMweight=5,
               # Tskip=300
               )


