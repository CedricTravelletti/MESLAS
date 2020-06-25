""" Demonstrates how to define a multivariate Gaussian Random Field,
sample a realization and plot it.

"""
import numpy as np
import torch
from meslas.means import ConstantMean
from meslas.covariance.spatial_covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.geometry.grid import SquareGrid
from meslas.sampling import GRF
from meslas.excursion import coverage_fct_fixed_location


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
my_grid = SquareGrid(100, dim)

# Observe some data.
S_y = torch.tensor([[0.2, 0.1], [0.2, 0.2], [0.2, 0.3],
        [0.2, 0.4], [0.2, 0.5], [0.2, 0.6],
        [0.2, 0.7], [0.2, 0.8], [0.2, 0.9], [0.2, 1.0],
        [0.6, 0.5]])
L_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0 ,0 ,0, 0])
y = torch.tensor(11*[-6])

mu_cond_grid, mu_cond_list, mu_cond_iso , K_cond_list, K_cond_iso = myGRF.krig_grid(
        my_grid, S_y, L_y, y,
        noise_std=0.05,
        compute_post_cov=True)

# Plot.
from meslas.plotting import plot_2d_slice, plot_krig_slice
plot_krig_slice(mu_cond_grid, S_y, L_y)

# Sample from the posterior.
from torch.distributions.multivariate_normal import MultivariateNormal
distr = MultivariateNormal(loc=mu_cond_list, covariance_matrix=K_cond_list)
sample = distr.sample()

# Reshape to a regular grid.
grid_sample = my_grid.isotopic_vector_to_grid(sample, n_out)
plot_krig_slice(grid_sample, S_y, L_y)

# Now compute and plot coverage function.
# Need only cross-covariances at fixed locations.
K_cond_diag = torch.diagonal(K_cond_iso, dim1=0, dim2=1).T
lower = torch.tensor([-1.0, -1.0]).double()

coverage = coverage_fct_fixed_location(mu_cond_iso, K_cond_diag, lower, upper=None)
plot_2d_slice(coverage.reshape(my_grid.shape), title="Excursion Probability",
        cmin=0, cmax=1.0)
