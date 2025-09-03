'''Class to store samples from MCMC and make plots.'''


import numpy as np
from jax import jit, vmap
import matplotlib.pyplot as plt
import pandas as pd
from chainconsumer import ChainConsumer, Chain, Truth
from emcee.autocorr import integrated_time



class Samples:

    def __init__(self, samples_arr, labels, x_inj=None, lnpost_func=None, jax=True):
        self.samples = samples_arr
        self.labels = labels
        self.x_inj = x_inj
        self.num_samples = self.samples.shape[0]
        self.ndim = self.samples.shape[1]
        
        self.lnpost_func = lnpost_func
        if self.lnpost_func is None:
            self.lnpost_func = lambda x: 0.
        
        self.jax = jax
        if self.jax:
            self.vectorized_lnpost_func = jit(vmap(self.lnpost_func))

    
        # data frame object for corner plots
        self.samples_df = pd.DataFrame(self.samples, columns=self.labels)

        # dictionary object
        self.samples_dict = {name: samps for name, samps in zip(self.labels, self.samples.T)}

    
    # plot posterior values over samples
    def plt_posterior_vals(self, burnin=0, plt_inj=True, legend=False):
        if self.jax:
            lnpost_vals = self.vectorized_lnpost_func(self.samples[burnin:])
        else:
            lnpost_vals = np.array([self.lnpost_func(x) for x in self.samples[burnin:]])
        plt.plot(lnpost_vals, label='samples')
        plt.xlabel('MCMC iteration')
        plt.ylabel('ln(posterior)')
        if self.x_inj is not None and plt_inj:
            plt.axhline(self.lnpost_func(self.x_inj), color='C1', label='injection')
        if legend:
            plt.legend()
        plt.show()

    # trace plot
    def trace_plt(self, burnin=0, plt_inj=True, legend=False, param_ndxs=None):
        if param_ndxs is None:
            param_ndxs = np.arange(self.ndim)
        for i in param_ndxs:
            plt.plot(self.samples[burnin:, i], color=f'C{i}', alpha=0.5)
            if self.x_inj is not None and plt_inj:
                plt.axhline(self.x_inj[i], color=f'C{i}', alpha=0.8, label=self.labels[i])
        if legend:
            plt.legend()
        plt.xlabel('MCMC iteration')
        plt.ylabel('parameter values')
        plt.show()


    # plot auto-correlations
    def plt_auto_corr(self, burnin=0, label_x_axis=False):
        auto_corrs = np.array([integrated_time(self.samples[burnin:, i])[0]
                               for i in range(self.ndim)])
        if label_x_axis:
            plt.bar(self.labels, auto_corrs)    
        else:
            plt.bar(np.arange(self.ndim), auto_corrs)
            plt.xlabel('parameter index')
        plt.ylabel('auto-correlation')
        plt.show()

    # corner plot
    def corner_plt(self, param_ndxs, burnin=0, thin=1, other_samples=None,
                   name1='samples', name2='samples2', **kwargs):
        c = ChainConsumer()
        c.add_chain(Chain(samples=self.samples_df.iloc[burnin::thin, param_ndxs],
                          name=name1, **kwargs))
        if self.x_inj is not None:
            c.add_truth(Truth(location={name: val for name, val in zip(self.labels, self.x_inj)}))
        if other_samples is not None:
            other_df = pd.DataFrame(other_samples, columns=self.labels)
            c.add_chain(Chain(samples=other_df.iloc[::thin, param_ndxs], name=name2))
        fig = c.plotter.plot()

