""" Demonstrates how to define a multivariate Gaussian Random Field,
sample a realization and plot it.

"""
import numpy as np
import torch
from meslas.means import ConstantMean
from meslas.covariance.covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.grid import Grid
from meslas.sampling import GRF

# Spatial Covariance.
matern_cov = Matern32(lmbda=0.1, sigma=1.0)

# Cross covariance.
cross_cov = UniformMixing(gamma0=0.9, sigmas=[np.sqrt(0.25), np.sqrt(0.6)])

covariance = FactorCovariance(matern_cov, cross_cov, n_out=2)

# Specify mean function
mean = ConstantMean([1.0, -2.0])

# Create the GRF.
myGRF = GRF(mean, covariance)

# Create a regular square grid in 2 dims.
# Number of repsones.
dim = 2
my_grid = Grid(100, dim)

# Sample all components at all locations.
sample = myGRF.sample_grid(my_grid)

# Plot.
from meslas.plotting import plot_2d_slice
plot_2d_slice(sample)
