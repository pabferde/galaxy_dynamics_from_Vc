# galaxy_dynamics_from_Vc
Python package to analyse a galactic model given circular velocity data. It contains a jupyter notebook with a working example.

This repository contains the Python package GalaxyDynamicsFromVc, which is intended for the inference of a galactic model's parameters, given circular velocity data. A modified version of this code has been used to obtain the results of the scientific publication <a href="https://doi.org/10.1088/1475-7516/2019/10/037" target="_blank">P.F. de Salas et al., JCAP 10 (2019) 037</a>, <a href="https://arxiv.org/abs/1906.06133" target="_blank">[arXiv:1906.06133]</a>.

The jupyter notebook goes through a working example.

## Install

Clone the repository
```
$ git clone https://github.com/pabferde/galaxy_dynamics_from_Vd.git
$ cd galaxy_dynamics_from_Vd
```

Install the requirements:

- numpy, scipy: for mathematical and optimisation purposes;
- emcee: for running the Markov chain Monte Carlo;
- matplotlib, corner: for data visualisation;
- pytest: for testing the code;
- jupyter: for accessing the example in the jupyter notebook.

The recommendation is to do it after creating a virtualenv and activating it:
```
# Optional (recommended): create a virtualenv venv and activate it
$ python3 -m venv venv
$ source venv/bin/activate
# Install requirements
$ python3 -m pip install -r requirements.txt
```

## Run the notebook with the example

To run the jupyter notebook, simply call it:
```
$ jupyter notebook Analysis-example.ipynb
```

## Test

The code includes a test unit that checks the correct performance of relevant parts of the code. To run the test unit, simply call pytest in the package folder:
```
$ pytest
```

