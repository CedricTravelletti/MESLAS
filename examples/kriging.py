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


# Dimension of the response.
n_out = 2

# Spatial Covariance.
matern_cov = Matern32(lmbda=0.1, sigma=1.0)

# Cross covariance.
cross_cov = UniformMixing(gamma0=0.9, sigmas=[np.sqrt(0.25), np.sqrt(0.6)])

covariance = FactorCovariance(matern_cov, cross_cov, n_out=n_out)

# Specify mean function
mean = ConstantMean([1.0, 0])

# Create the GRF.
myGRF = GRF(mean, covariance)

# Create a regular square grid in 2 dims.
# Number of repsones.
dim = 2
my_grid = Grid(100, dim)

# Observe some data.
S_y = torch.tensor([[0.2, 0.1], [0.2, 0.2], [0.2, 0.3],
        [0.2, 0.4], [0.2, 0.5], [0.2, 0.6],
        [0.2, 0.7], [0.2, 0.8], [0.2, 0.9],
        [0.2, 1.0]])
L_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0 ,0 ,0])
y = torch.tensor(10*[-6])

krig_mean, krig_mean_1d, K_cond = myGRF.krig_grid(my_grid, S_y, L_y, y,
        noise_std=0.05,
        compute_post_cov=True)

# Plot.
from meslas.plotting import plot_2d_slice, plot_krig_slice
plot_krig_slice(krig_mean, S_y, L_y)

# Sample from the posterior.
from torch.distributions.multivariate_normal import MultivariateNormal
distr = MultivariateNormal(loc=krig_mean_1d, covariance_matrix=K_cond)
sample = distr.sample()

# Reshape to a regular grid.
grid_sample = my_grid.isotopic_vector_to_grid(sample, n_out)
# plot_2d_slice(grid_sample)
plot_krig_slice(grid_sample, S_y, L_y)
