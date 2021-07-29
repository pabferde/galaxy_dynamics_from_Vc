
import pytest
import numpy as np

from .. import analysisstatistics
from .. import galaxymodel

#%%
def test_class_Prior():
    
    def ln_Gaussian(x, mean, sigma):
        normalisation = np.log(1.0/(np.abs(mean)*np.sqrt(2.0*np.pi)))
        return normalisation - 0.5*((x-mean)/sigma)**2

    default_prior = analysisstatistics.Prior()
    assert default_prior.flat_min == -np.infty
    assert default_prior.flat_max == np.infty
    assert default_prior.Gaussian_mean == 0.0
    assert default_prior.Gaussian_sigma == np.infty
    
    test_flat_prior = analysisstatistics.Prior(flat_min=-1.0, flat_max=1.0)
    assert test_flat_prior.ln_function(0.0) == 0.0
    assert test_flat_prior.ln_function(-0.99) == 0.0
    assert test_flat_prior.ln_function(0.99) == 0.0
    assert test_flat_prior.ln_function(2.0) == -np.infty
    assert test_flat_prior.ln_function(-1.0) == -np.infty
    assert test_flat_prior.ln_function(1.0) == -np.infty
    
    mean = -1.1
    sigma = 2.3
    test_Gaussian_prior = analysisstatistics.Prior(Gaussian_mean=mean, Gaussian_sigma=sigma)
    assert test_Gaussian_prior.ln_function(0.0) == ln_Gaussian(0.0,mean,sigma)
    assert test_Gaussian_prior.ln_function(-999.0) == ln_Gaussian(-999.0,mean,sigma)
    assert test_Gaussian_prior.ln_function(3.0) == ln_Gaussian(3.0,mean,sigma)
    
    flat_min = -2.0
    flat_max = -0.7
    mean = 7.0
    sigma = 0.2
    test_Gaussian_prior_with_strict_bounds = analysisstatistics.Prior(flat_min=flat_min,flat_max=flat_max,Gaussian_mean=mean,Gaussian_sigma=sigma)
    assert test_Gaussian_prior_with_strict_bounds.ln_function(0.0) == -np.infty
    assert test_Gaussian_prior_with_strict_bounds.ln_function(-0.7) == -np.infty
    assert test_Gaussian_prior_with_strict_bounds.ln_function(-2.2) == -np.infty
    assert test_Gaussian_prior_with_strict_bounds.ln_function(-1.0) == ln_Gaussian(-1.0,mean,sigma)

#%%

@pytest.fixture
def test_variable_default():
    return analysisstatistics.Variable("test_default")

def test_class_Variable(test_variable_default):

    assert test_variable_default.name == "test_default"

    test_variable_default.value = 777.66
    assert test_variable_default.value == 777.66
    test_variable = analysisstatistics.Variable("test",value=123.5)
    assert test_variable.value == 123.5
    
    test_variable_default.set_position_in_list(12)
    assert test_variable_default.position_in_list == 12
    test_variable = analysisstatistics.Variable("test",position_in_list=7)
    assert test_variable.position_in_list == 7

    def test_prior(x,mu=1.0,sigma=1.0):
        return np.exp(-((x-mu)/sigma)**2)
    test_variable_prior = analysisstatistics.Variable("test",prior_function=test_prior)
    assert test_variable_prior.get_prior(-1.0) == test_prior(-1.0)
    assert test_variable_prior.prior_function == test_prior
    assert test_variable_prior.prior_function(0.0) == test_prior(0.0)
    
def test_class_Variable_undefined_position_in_list(capfd, test_variable_default):
    assert test_variable_default.position_in_list is None
    out, err = capfd.readouterr()
    assert "position_in_list is not defined!" in out
    
def test_class_Variable_ill_defined_position_in_list(test_variable_default):
    with pytest.raises(AssertionError):
        analysisstatistics.Variable("test",position_in_list=0.2)
    with pytest.raises(AssertionError):
        test_variable_default.set_position_in_list([7])

def test_class_Variable_undefined_prior_function(test_variable_default):
    with pytest.raises(TypeError):
        test_variable_default.get_prior(0.0)
        
def test_class_Variable_ill_defined_prior_function(test_variable_default):

    def test_prior_more_than_1_input(a,b):
        return 0.0
    with pytest.raises(TypeError):
        analysisstatistics.Variable("test",prior_function=test_prior_more_than_1_input)
    with pytest.raises(TypeError):
        test_variable_default._set_prior_function(test_prior_more_than_1_input)
 
    def test_prior_no_float_output(x):
        return [0]
    with pytest.raises(AssertionError):
        analysisstatistics.Variable("test",prior_function=test_prior_no_float_output)
    with pytest.raises(AssertionError):
        test_variable_default._set_prior_function(test_prior_no_float_output)

#%%
        
@pytest.fixture
def list_of_variables():
    return [analysisstatistics.Variable("x1"), analysisstatistics.Variable("x2")]
@pytest.fixture
def variables_dictionary(list_of_variables):
    d = {}
    k = 0
    for var in list_of_variables:
        var.set_position_in_list(k)
        d[var.name] = var
        k += 1
    return d
@pytest.fixture
def galaxy_model_creator(variables_dictionary):
    def make_galaxy_model_creator(dictionary=variables_dictionary):
        halo = galaxymodel.NFW()
        return galaxymodel.Galactic_model(halo)
    return make_galaxy_model_creator
        
def test_class_Analysis(list_of_variables, galaxy_model_creator):
    analysis_without_R_sun = analysisstatistics.Analysis(list_of_variables, galaxy_model_creator, R_sun=None)
    assert list_of_variables == analysis_without_R_sun.variables_list
    analysis_with_R_sun = analysisstatistics.Analysis(list_of_variables, galaxy_model_creator)
    assert list_of_variables == analysis_with_R_sun.variables_list[:-1]
    assert analysis_with_R_sun.variables_key_list == [var.name for var in list_of_variables]+["R_sun"]

def test_class_Analysis_ill_input_list_of_variables():
    with pytest.raises(AssertionError):
        analysisstatistics.Analysis(0,0)
    with pytest.raises(AssertionError):
        analysisstatistics.Analysis([0],0)

def test_class_Analysis_ill_input_galaxy_model_creator(capfd, list_of_variables):
    def galaxy_model_creator_wrong_output(dictionary):
        return 0
    def galaxy_model_creator_wrong_key_in_input_dict(dictionary):
        return dictionary["non_existing_key"]

    with pytest.raises(AssertionError):
        analysisstatistics.Analysis(list_of_variables,galaxy_model_creator_wrong_output)

    try:
        analysisstatistics.Analysis(list_of_variables,galaxy_model_creator_wrong_key_in_input_dict)
    except KeyError:
        out, err = capfd.readouterr()
        assert "ADVISE: Check that the dictionary input for galaxy_model_creator allows the same keys than your Variables in list_of_variables" in out
    except TypeError:
        out, err = capfd.readouterr()
        assert "ADVISE: Check that the your galaxy_model_creator accepts a dictionary as its only positional argument" in out
    
    
    
    
