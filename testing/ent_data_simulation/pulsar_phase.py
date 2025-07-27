import numpy as np
import scipy.constants as sc
from params_injected import num_psrs


# constants
SOLAR2S = sc.G / sc.c**3 * 1.98855e30
KPC2S = sc.parsec / sc.c * 1e3

# compute injected pulsar phases
def get_psr_phase(cw_params, psr_position, psr_dist):

    # unpack CW parameters
    log10_mc, log10_fgw, cosinc, psi, log10_dist, costheta, gwphi, phase0 = cw_params

    # define sky location parameters
    singwtheta = np.sin(np.arccos(costheta))
    cosgwtheta = costheta
    singwphi = np.sin(gwphi)
    cosgwphi = np.cos(gwphi)
    omhat = np.array([-singwtheta * cosgwphi, -singwtheta * singwphi, -cosgwtheta])

    # store pulsar phase
    cosMu = -np.dot(omhat, psr_position)
    pphase = (1 + 256/5 * (10**log10_mc*SOLAR2S)**(5/3) * (np.pi * 10**log10_fgw)**(8/3)
            * psr_dist*KPC2S*(1-cosMu)) ** (5/8) - 1
    pphase /= 32 * (10**log10_mc*SOLAR2S)**(5/3) * (np.pi * 10**log10_fgw)**(5/3)
    psr_phase = -pphase%(2*np.pi)

    return psr_phase

