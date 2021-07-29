# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import corner

#%%

def plot_lnlikelihoods(lnlikelihood_lists, saved_figure_name='', **kwargs):
    plt.xlabel("Step")
    plt.ylabel("ln likelihood")
    number_of_steps = len(lnlikelihood_lists[0])
    for lnlikelihood in lnlikelihood_lists:
        plt.plot([k for k in range(number_of_steps)], lnlikelihood, **kwargs)
    if saved_figure_name:
        plt.savefig(saved_figure_name)

def plot_acceptance_fraction(acceptance_fraction_list, saved_figure_name='', **kwargs):
    plt.xlabel("Walker")
    plt.ylabel("Acceptance fraction")
    number_of_walkers = len(acceptance_fraction_list)
    plt.plot([k for k in range(number_of_walkers)], acceptance_fraction_list, **kwargs)
    if saved_figure_name:
        plt.savefig(saved_figure_name)
    
def plot_corner_plot(chains, variable_labels,**kwargs):
    number_of_dimensions = len(variable_labels)
    samples = chains.reshape((-1,number_of_dimensions))
    fig = corner.corner(samples,labels=variable_labels,**kwargs)

#%%
if __name__ == '__main__':
    pass
    
