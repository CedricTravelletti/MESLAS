""" Scripts for producing figure 4.
"""
import numpy as np
import torch
from meslas.means import LinearMean
from meslas.covariance.covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.geometry.grid import SquareGrid
from meslas.random_fields import GRF
from meslas.excursion import coverage_fct_fixed_location

# Conversion factor for bringing back to 1x1 grid.
conv = 0.4

# Dimension of the response.
n_out = 2

# Spatial Covariance.
# lmbda = (np.sqrt(3) / 0.3) / conv
lmbda = 0.5
matern_cov = Matern32(lmbda=lmbda, sigma=1.0)

# Cross covariance.
cross_cov = UniformMixing(gamma0=0.2, sigmas=[2.25, 2.25])

covariance = FactorCovariance(matern_cov, cross_cov, n_out=n_out)

# Specify mean function, here it is a linear trend that decreases with the
# horizontal coordinate.
beta0s = np.array([5.8, 24.0])
beta1s = np.array([
        [0, -7.0],
        [0, -5.0]])
mean = LinearMean(beta0s, beta1s)

# Create the GRF.
myGRF = GRF(mean, covariance)

# Create a regular square grid in 2 dims.
# Number of repsones.
dim = 2
my_grid = SquareGrid(100, dim)

# Sample all components at all locations.
sample, sample_list = myGRF.sample_grid(my_grid)

# Plot.
from meslas.plotting import plot_2d_slice
plot_2d_slice(sample)
np.save("./uncond_sample_grid.npy", sample.numpy())

# Measure some data on the middle of the grid.
S_y = torch.tensor([[0.1, 0.5], [0.2, 0.5], [0.3, 0.5],
        [0.4, 0.5], [0.5, 0.5], [0.6, 0.5],
        [0.7, 0.5], [0.8, 0.5], [0.9, 0.5]])
# Get the corresponding indices.
S_inds = my_grid.get_closest(S_y)

# Get the data and flatten to a list.
y = sample_list[S_inds].reshape(-1) 

# Get corresponding coordinates, dupplicated so we have one instance per
# response.
S_y_simple = my_grid.points[S_inds]
S_y = torch.repeat_interleave(S_y_simple, myGRF.n_out, dim=0)

# Response index vector.
L_y = torch.tensor(range(myGRF.n_out)).repeat(S_y_simple.shape[0])

mu_cond_grid, mu_cond_list, mu_cond_iso , K_cond_list, K_cond_iso = myGRF.krig_grid(
        my_grid, S_y, L_y, y,
        noise_std=0.05,
        compute_post_cov=True)

# Plot.
from meslas.plotting import plot_2d_slice, plot_krig_slice, plot_proba
plot_krig_slice(mu_cond_grid, S_y, L_y)

# Now compute and plot coverage function.
# Need only cross-covariances at fixed locations.
K_cond_diag = torch.diagonal(K_cond_iso, dim1=0, dim2=1).T
lower = torch.tensor([3.5, 24.0]).double()

coverage = coverage_fct_fixed_location(mu_cond_iso, K_cond_diag, lower, upper=None)
plot_proba(coverage.reshape(my_grid.shape), title="Joint Excursion Probability")

# Compute the univariate excursion probability.
lower = torch.tensor([3.5, -np.Inf]).double()
coverage = coverage_fct_fixed_location(mu_cond_iso, K_cond_diag, lower, upper=None)
plot_proba(coverage.reshape(my_grid.shape), title="One sided Excursion Probability: Salinity")

lower = torch.tensor([-np.Inf, 23.5]).double()
coverage = coverage_fct_fixed_location(mu_cond_iso, K_cond_diag, lower, upper=None)
plot_proba(coverage.reshape(my_grid.shape), title="One sided Excursion Probability: Temperature")
