# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as optim
from multiprocessing import Pool
import emcee
import time

from GalaxyDynamicsFromVc.datahandling import read_data_from_1810_09466
from GalaxyDynamicsFromVc.galaxymodel import Galactic_model

#%%

class Variable:
    def __init__(self, name, value=1.0, position_in_list=None, prior_function=None):
        self._name = name
        if position_in_list is not None:
            assert isinstance(position_in_list, int), "position_in_list must be an int!"
        self._position_in_list = position_in_list
        self.value = value
        self._prior_function = self._set_prior_function(prior_function)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, x):
        self._value = x

    @property
    def position_in_list(self):
        if self._position_in_list is None:
            print("position_in_list is not defined!")
        return self._position_in_list
    def set_position_in_list(self, position):
        assert isinstance(position, int), "position must be an int!"
        self._position_in_list = position
    
    @property
    def prior_function(self):
        return self._prior_function
    
    def _set_prior_function(self, prior_function):
        if prior_function is None:
            return
        try:
            prior_function(1.0)
        except TypeError:
            print("prior_function should accept only 1 positional argument: the float value to be evaluated")
        assert isinstance(prior_function(1.0), float), "return from prior_function(value) must be a float!"
        return prior_function
    
    def get_prior(self, variable_value):
        if self.prior_function is None:
            print("Error: prior function is not defined for this variable!")
        return self.prior_function(variable_value)

class Prior:
    def __init__(self, flat_min=-np.infty, flat_max=np.infty, Gaussian_mean=0, Gaussian_sigma=np.infty):
        self.flat_min = flat_min
        self.flat_max = flat_max
        self.Gaussian_mean = Gaussian_mean
        self.Gaussian_sigma = Gaussian_sigma
        
    @property
    def flat_min(self):
        return self._flat_min
    @flat_min.setter
    def flat_min(self, value):
        self._flat_min = value

    @property
    def flat_max(self):
        return self._flat_max
    @flat_max.setter
    def flat_max(self, value):
        self._flat_max = value
        
    @property
    def Gaussian_mean(self):
        return self._Gaussian_mean
    @Gaussian_mean.setter
    def Gaussian_mean(self, value):
        self._Gaussian_mean = value
        
    @property
    def Gaussian_sigma(self):
        return self._Gaussian_sigma
    @Gaussian_sigma.setter
    def Gaussian_sigma(self, value):
        self._Gaussian_sigma = value

    def ln_function(self, value):
        if self.flat_min < value < self.flat_max:
            if not np.isfinite(self.Gaussian_sigma):
                return 0.0
            else:
                normalisation = np.log(1.0/(np.abs(self.Gaussian_mean)*np.sqrt(2.0*np.pi)))
                return normalisation - 0.5*((value-self.Gaussian_mean)/self.Gaussian_sigma)**2
        return -np.infty

class Analysis:
    def __init__(self, list_of_variables, galaxy_model_creator,
                 R_sun=Variable('R_sun',prior_function=Prior(Gaussian_mean=8.122, Gaussian_sigma=0.031).ln_function,value=8.122)):
        self._variables_list = self._add_Rsun_to_list_of_variables(list_of_variables, R_sun)
        self._variables_key_list = [var.name for var in self.variables_list]
        self._variables_dictionary = self._set_variables_dictionary()
        self._check_galaxy_model_creator(galaxy_model_creator)
        self._galaxy_model_creator = galaxy_model_creator
        self._data_lists = read_data_from_1810_09466()
        self._sampler = None

    @property
    def variables_list(self):
        return self._variables_list
    
    @property
    def variables_key_list(self):
        return self._variables_key_list

    @property
    def galaxy_model_creator(self):
        return self._galaxy_model_creator
    
    @property
    def sampler(self):
        return self._sampler
    
    def _add_Rsun_to_list_of_variables(self, list_of_variables, R_sun):
        _error_of_assertion_message = "Elements in list_of_variables must be instances of class Variable!"
        if isinstance(list_of_variables,Variable):
            list_of_variables = [list_of_variables]
        try:
            iter(list_of_variables)
        except TypeError:
            assert isinstance(list_of_variables,Variable), _error_of_assertion_message
            raise
        assert isinstance(list_of_variables[0],Variable), _error_of_assertion_message
        new_list = []
        new_list[:] = list_of_variables
        if R_sun is None:
            return new_list
        new_list.append(R_sun)
        return new_list

    def _set_variables_dictionary(self):
        d = {}
        k = 0
        for var in self.variables_list:
            var.set_position_in_list(k)
            d[var.name] = var
            k += 1
        return d
    
    def _check_galaxy_model_creator(self, galaxy_model_creator):
        d = self._variables_dictionary
        try:
            result = galaxy_model_creator(d)
        except KeyError:
            print("\nADVISE: Check that the dictionary input for galaxy_model_creator allows the same keys than your Variables in list_of_variables.\n")
            raise
        except TypeError:
            print("\nADVISE: Check that the your galaxy_model_creator accepts a dictionary as its only positional argument.\n")
            raise
        assert isinstance(result, Galactic_model), "galaxy_model_creator must return an instance of Galactic_model in galaxymodel module!"
    
    def set_galaxy_model(self, list_of_values):
        d = self._variables_dictionary
        l = list_of_values
        for key in d:
            d[key].value = l[d[key].position_in_list]
        return self.galaxy_model_creator(d)
    
    def get_ln_priors(self, list_of_values):
        d = self._variables_dictionary
        l = list_of_values
        prior = 0
        for var in self.variables_key_list:
            prior_function = d[var].prior_function
            index = d[var].position_in_list
            prior += prior_function(l[index])
        return prior
    
    def ln_likelihood(self, list_of_values):
        lp = self.get_ln_priors(list_of_values)
        if not np.isfinite(lp):
            return -np.infty
        GM = self.set_galaxy_model(list_of_values)
        chi2 = sum(((data[1] - GM.circular_velocity_km_s(data[0]))/(np.sqrt(
                ((data[2]+data[3])/2)**2 # statistical error in the data
                + (data[4])**2 # systematic error in the data
                )))**2 for data in self._data_lists)
        return lp -chi2/2
    
    def get_maximum_likelihood_variables(self):
        d = self._variables_dictionary
        initial_maximum = np.ones(len(self.variables_list))
        for var in self.variables_key_list:
            initial_maximum[d[var].position_in_list] = d[var].value
        loss_function_to_minimise = lambda args: -self.ln_likelihood(args)
        result = optim.minimize(loss_function_to_minimise, initial_maximum)
        initial_maximum = result["x"]
        return initial_maximum
    
    def compute_burntin(self, number_of_walkers, number_of_steps):
        number_of_dimensions = len(self.variables_list)
        initial_variables = self.get_maximum_likelihood_variables()
        walkers_position = [initial_variables + 0.05*np.random.randn(number_of_dimensions) for i in range(number_of_walkers)]
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(number_of_walkers, number_of_dimensions, self.ln_likelihood, pool=pool)
            start = time.time()
            walkers_after_burntin, prob, state = sampler.run_mcmc(walkers_position, number_of_steps)
            end = time.time()
            multi_time = end - start
            print("Burntin took {0:.1f} seconds".format(multi_time))
            sampler.reset()
        return walkers_after_burntin
    
    def compute_mcmc(self, number_of_walkers, number_of_steps, walkers_after_burntin):
        number_of_dimensions = len(self.variables_list)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(number_of_walkers, number_of_dimensions, self.ln_likelihood, pool=pool)
            start = time.time()
            sampler.run_mcmc(walkers_after_burntin, number_of_steps)
            end = time.time()
            multi_time = end - start
            print("MCMC took {0:.1f} seconds".format(multi_time))
        self._sampler = sampler
        
    def compute_mcmc_and_burntin(self, number_of_walkers, number_of_steps_burntin, number_of_steps_mcmc):
        walkers_after_burntin = self.compute_burntin(number_of_walkers,number_of_steps_burntin)
        self.compute_mcmc(number_of_walkers,number_of_steps_mcmc,walkers_after_burntin)
    


#%%
if __name__ == '__main__':
    pass
